from asct.src.Optimizer import Optimizer
from asct.src.Cell import Cell
from asct.src.SummaryNet import SummaryCNN
from neuron import h
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import math
import torch
import platform
from tqdm import tqdm

#This class uses the Cell and Optimizer classes to
#optimize a set of current injections based on
#either some user set target statistics or a 
#target model.
#Takes 7 parameters:
# 1) A list of the current injection levels (nA) to match.
# 2) The directory of the modfiles for used.
# 3) The directory for the hoc template file.
# 4) The name of the target template in the template file.
# 5) A list of each parameter to optimize. These should be names of parameters which can be accessed in 
#    the hoc object.
# 6) A tuple containing two arrays, the lows and highs for each parameter specified in the above argument.
# 7) A bool which tells the optimizer to use a CNN to learn the summary statistics or not. If not set then 
#    the tuner uses a user defined summary stats function.
# 8) Some optional kwargs which contain values used for the summary stats function if the parameters
#    arent being learned and aditional parameters need to be passed to the summary stats function.


class CellTuner:
    def __init__(self, modfiles_dir, template_dir, template_name, current_injections, optimization_parameters, parameter_range, architecture='summary', summary_funct = None, features=8,*args, **kwargs):
        
        #Store the current injection list and the spike threshold.
        self.__current_injections = current_injections

  
        self.__embedding_net = None
        if architecture == 'convolution' or architecture == 'hybrid':
            self.__embedding_net = SummaryCNN(len(current_injections), features, hybrid= False if architecture == 'convolution' else True)
            summary_funct = None

            args = ()
            kwargs = {}
        

        if modfiles_dir != None:
            if platform.system() == 'Linux':
                #First lets try to compile our modfiles.
                if os.system('nrnivmodl %s' % modfiles_dir) == 0:
                    print("Compiled Successfully.")
                else:
                    print("Could not compile the modfiles at %s" % modfiles_dir)
            elif platform.system() == 'Windows':
                print('Automatic modfile compilation has not yet been implemented for windows, included modfiles directory will not be used.')
                


        #Now load in the standard run hoc.
        h.load_file('stdrun.hoc')

        #Set up the cell and the optimizer. This will be responsible for optimizing the given cell
        #at each current injection value.
        self.__to_optimize = Cell(template_dir, template_name)

        self.__optimizer = Optimizer(self.__to_optimize, optimization_parameters, parameter_range, summary_funct, *args, **kwargs)

        self.__parameter_samples = []
        self.__target_responses = None

        self.NUM_SAMPLES = 1000

    def run_forward_pass(self, input):
        out = self.__embedding_net(input)
        return out
    
    #Set the simulation parameters.
    def set_simulation_params(self, sim_run_time = 1500, delay = 400, inj_time = 500, v_init = -75):
        self.__sim_run_time = sim_run_time
        self.__delay = delay
        self.__inj_time = inj_time
        self.__v_init = v_init

    #Calculate the target summary statistics based on a provided HOC model.
    #If CNN is activated and not yet trained, train it.
    def calculate_target_stats_from_model(self, template_dir, template_name):    
        self.__target_cell = Cell(template_dir, template_name)
        
        #Create a temporary optimizer which is used to get the target
        #statistics from the target cell at each current injection.
        self.__sim_environ = Optimizer(self.__target_cell, 
                                      None, 
                                      None,
                                      self.__optimizer.summary_funct,
                                      *self.__optimizer.summary_stat_args,
                                      **self.__optimizer.summary_stat_kwargs)

        self.__sim_environ.set_current_injection_list(self.__current_injections)
        
        if self.__embedding_net == None:
            self.__optimizer.set_observed_stats(self.__sim_environ.multi_channel_wrapper_summary())
        else:
            self.__optimizer.set_observed_stats(self.__sim_environ.multi_channel_wrapper_CNN().flatten())

    #Given the current injection responses directly, save it.
    def set_target_responses(self, target_responses):
        self.__time = np.linspace(0,self.__sim_run_time * 1e-3,1024)
        self.__target_responses = target_responses

    #Generates the target statitcs from the data. Target responses must be set.
    def calculate_target_stats_from_data(self):
        if self.__embedding_net == None:
            
            #We just need to get the statistics for each current injection.
            stats = np.array([])

            for trace in self.__target_responses:
                stat = self.__optimizer.summary_funct(trace, self.__time, *self.__optimizer.summary_stat_args, **self.__optimizer.summary_stat_kwargs)
                stats = np.concatenate((stats, stat), axis=0)

            self.__optimizer.set_observed_stats(stats)
        else:
            data = torch.empty((len(self.__current_injections),1024))
            for index,response in enumerate(self.__target_responses):
                data[index] = torch.from_numpy(response).float()

            self.__optimizer.set_observed_stats(data.flatten())

    #Wrapper function which automatically chooses which function to call based on
    #set parameters.
    def optimize_current_injections(self, num_simulations = 500, num_rounds=1, inference_workers=1 ,sample_threshold = 10):
        self.__top_n = sample_threshold
        
        if self.__embedding_net == None:
            self.optimize_current_injections_summary(num_simulations=num_simulations, 
                                                     num_rounds=num_rounds, 
                                                     inference_workers=inference_workers)
        else:
            self.optimize_current_injections_cnn(num_simulations=num_simulations, 
                                                 inference_workers=inference_workers,
                                                 num_rounds=num_rounds)
    
    def optimize_current_injections_cnn(self, num_simulations = 500, num_rounds = 1, inference_workers=1):        
        self.__optimizer.set_current_injection_list(self.__current_injections)
        self.__optimizer.run_inference_learned_stats(self.__embedding_net, num_simulations=num_simulations, num_rounds=num_rounds, workers=inference_workers)
        self.__parameter_samples = self.__optimizer.get_samples(-1, sample_threshold=self.NUM_SAMPLES)

    def optimize_current_injections_summary(self, num_simulations = 500, num_rounds=1, inference_workers=1):
        self.__optimizer.set_current_injection_list(self.__current_injections)
        self.__optimizer.run_inference_multiround(num_simulations=num_simulations, workers=inference_workers, num_rounds=num_rounds)
        self.__parameter_samples = self.__optimizer.get_samples(-1, sample_threshold=self.NUM_SAMPLES)
            

    def compute_correlation_for_parameter_set(self, parameter_set):
        #generate the found voltage responses.
        found_responses = []
        for i_inj in self.__current_injections:
            self.__optimizer.set_simulation_params(sim_run_time=self.__sim_run_time, delay=self.__delay, inj_time=self.__inj_time, v_init=self.__v_init, i_inj=i_inj)
            self.__optimizer.simulation_wrapper(parameter_set)

            voltage,_ = self.__to_optimize.resample()
            found_responses.append(voltage)
        
       

        #Compute average cosine similarity accross current injections.
        num_injections = len(self.__current_injections)
        mean_similarity = 0
        for i in range(num_injections):
            mean_similarity += np.dot(found_responses[i], self.__target_responses[i]) / (np.linalg.norm(found_responses[i]) * np.linalg.norm(self.__target_responses[i]))
        mean_similarity /= num_injections
        
        return mean_similarity


    #Find the best parameter set based on the calulated posterior distributions.
    #This is done by getting the top 'x' solutions from each posterior distribution
    #and finding the closest match in each set of parameters.
    #Takes one parameter:
    #   1) Displays the best found set and its matching ratio.
    #
    #NOTE: as of now there is nothing which takes into account the rank of the parameters,
    #i.e. all parameter sets from out of the top 'x' are treated equally when in reality the
    #first ones are really the best solutions, this could be taken into account in a better
    #function
    #Find the "top_n best paramter sets from the posterior"
    def find_best_parameter_sets(self, SHOW_TOP_CORRELATION=True):
                 
        #Make sure the target is generated if it has not been set.
        if self.__target_responses == [] or self.__target_responses == None:
            self.generate_target_from_model()

        #sort all the paramter samples based on performance (cosine correlation.)
        parameter_ranking = []
        for pset in tqdm(self.__parameter_samples,desc='Generating sample performance for %d samples' % self.NUM_SAMPLES):
            parameter_ranking.append((self.compute_correlation_for_parameter_set(pset), pset))
        
        #Now sort the parameter sets.
        parameter_ranking.sort(key = lambda x:x[0], reverse=True)

        #Now get the top_n samples.
        self.__found_parameters_correlation = [elem[0] for elem in parameter_ranking[:self.__top_n]]
        self.__found_parameters = [elem[1] for elem in parameter_ranking[:self.__top_n]]
        
        if SHOW_TOP_CORRELATION:
            print('Top sample correlation performance: ', self.__found_parameters_correlation)

    #Compare the found solution to the model using the target cell.
    #Takes 2 parameters:
    #   1) If a graph should be displayed which shows the voltage trace of
    #      found parameter set overlayed on the target cell.
    #   2) The directory the above image should be saved to. None means dont
    #      save the image.
    def compare_found_solution_to_target(self, top_n=1, display=False, save_dir=None):
        #Get the time vector.
        #create a gid of subplots to display the results. the grid is NxN where
        #n = ceil(sqrt(n))
        n_plots = math.ceil(math.sqrt(top_n))
        fig = plt.figure(figsize=(10,10))
        outer = gridspec.GridSpec(n_plots,n_plots)

        for i in range(top_n):
            #Now get the real cell with our current injections.
            found_responses = []
            for i_inj in self.__current_injections:
                self.__optimizer.set_simulation_params(sim_run_time=self.__sim_run_time, delay=self.__delay, inj_time=self.__inj_time, v_init=self.__v_init, i_inj=i_inj)
                self.__optimizer.simulation_wrapper(self.__found_parameters[i])
                voltage,_ = self.__to_optimize.resample()
                found_responses.append(voltage)
            
            #Now plot everything.
            current_injection_length = len(self.__current_injections)
            inner = gridspec.GridSpecFromSubplotSpec(current_injection_length, 1, subplot_spec=outer[i])
            if current_injection_length > 1:
                for index in range(current_injection_length):
                    ax = plt.Subplot(fig,inner[index])
                    if index == 0:
                        ax.set_title("Solution: %d" % (i+1))
                    if index != current_injection_length - 1:
                        ax.xaxis.set_visible(False)
                    ax.plot(self.__time, self.__target_responses[index], label='Target')
                    ax.plot(self.__time, found_responses[index], label='Found')
                    ax.set_ylabel(self.__current_injections[index])
                    fig.add_subplot(ax)
            else:
                ax = plt.Subplot(fig,inner[0])
                ax.plot(self.__time, self.__target_responses[0], label='Target')
                ax.plot(self.__time, found_responses[0], label='Found')
                ax.set_ylabel(self.__current_injections[0])
                fig.add_subplot(ax)

        handles, labels = plt.gca().get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right')
        fig.tight_layout()

        if display:  
            fig.show()
            plt.show()
        
        if save_dir != None:
            fig.savefig(save_dir,bbox_inches='tight')

    def generate_target_from_model(self):
        #Find the target voltage traces.
        self.__target_responses = []
        for i_inj in self.__current_injections:
            self.__sim_environ.set_simulation_params(sim_run_time=self.__sim_run_time, delay=self.__delay, inj_time=self.__inj_time, v_init=self.__v_init, i_inj=i_inj)
            self.__sim_environ.simulation_wrapper()
            voltage,_ = self.__target_cell.resample()
            self.__target_responses.append(voltage)

        #Store this for other functions which need the time vector.
        _,time = self.__to_optimize.resample()
        self.__time = time

    #def generate_found_FI_curve(self, display=False, save_dir=None):

    #Return a string containing the parameter names as keys and the found parmeter as the values
    #for the top n values.
    def get_optimial_parameter_sets(self, top_n=1):        
        p_list = []
        for index in range(top_n):
            p_list.append(dict(zip(self.__optimizer.get_simulation_optimization_params(), self.__found_parameters[index])))

        formated_output = ''
        
        for index, solution in enumerate(p_list):
            formated_output = formated_output + 'Solution %d: ' % (index + 1) + str(solution) + '\n'
        
        return formated_output

    #Compare the best found solution to the target voltage trace and return the error.
    #This returns the average cosine similiarity between all found and target voltage traces.
    def get_best_trace_error(self):
        return 1.0 - self.compute_correlation_for_parameter_set(self.__found_parameters[0])

