import scipy as sp
from Optimizer import Optimizer
from Cell import Cell
from SummaryNet import SummaryCNN
from neuron import h
import itertools
import os
from scipy.stats.stats import pearsonr 
import numpy as np
import matplotlib.pyplot as plt


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
    def __init__(self, current_injections, modfiles_dir, template_dir, template_name, optimization_parameters, parameter_range, learn_stats=False, summary_funct = None, *args, **kwargs):
        
        #Store the current injection list and the spike threshold.
        #Store if learn_stats is true or false.
        self.__current_injections = current_injections

        #Check if learn_stats is set to true, if it is then the summary funct and all args and kwargs
        #will be ignored.
        if learn_stats and (summary_funct != None or args != None or kwargs != None):
            print('learn_stats is set to true, disregarding summary statistic function arguments.')
        
        self.__embedding_net = None
        if learn_stats:
            self.__embedding_net = SummaryCNN()
            summary_funct = self.run_forward_pass

            args = ()
            kwargs = {}
        
         #First lets try to compile our modfiles.
        if os.system('nrnivmodl %s' % modfiles_dir) == 0:
            print("Compiled Successfully.")
        else:
            print("Could not compile the modfiles at %s" % modfiles_dir)

        #Now load in the standard run hoc.
        h.load_file('stdrun.hoc')

        #Set up the cell and the optimizer. This will be responsible for optimizing the given cell
        #at each current injection value.
        self.__to_optimize = Cell(template_dir, template_name, summary_funct)

        self.__optimizer = Optimizer(self.__to_optimize, optimization_parameters, parameter_range, summary_funct, *args, **kwargs)

    
    def run_forward_pass(self, input):
        out = self.__embedding_net(input)
        return out

    #This function trains the model based on pre-simulated data.
    def train_summary_cnn(self):
        return 0
    
    #Sets the target statistics to be matched. This should be a list of target statistics
    #where each entry matches a given current injection level.
    def add_target_statistics(self, target_stats):
        self.__target_stats = target_stats
    
    #Set the simulation parameters.
    def set_simulation_params(self, sim_run_time = 1500, delay = 400, inj_time = 500, v_init = -75):
        self.__sim_run_time = sim_run_time
        self.__delay = delay
        self.__inj_time = inj_time
        self.__v_init = v_init

    #Calculate the target summary statistics based on a provided HOC model.
    def calculate_target_stats_from_model(self, template_dir, template_name):    
        self.__target_cell = Cell(template_dir, template_name, self.__optimizer.summary_funct)
        
        #Create a temporary optimizer which is used to get the target
        #statistics from the target cell at each current injection.
        self.__sim_environ = Optimizer(self.__target_cell, 
                                      None, 
                                      None,
                                      self.__optimizer.summary_funct,
                                      *self.__optimizer.summary_stat_args,
                                      **self.__optimizer.summary_stat_kwargs)
                
        #Loop through each current injection and calculate the target statitics.
        self.__target_stats = []
        for i_inj in self.__current_injections:
            self.__sim_environ.set_simulation_params(sim_run_time=self.__sim_run_time, delay=self.__delay, inj_time=self.__inj_time, v_init=self.__v_init, i_inj=i_inj)
            self.__target_stats.append(self.__sim_environ.simulation_wrapper())

    
    # #This pre-simulates the data for SBI. Can be sped up here. 
    # def pre_simulate_data(self, save_dir, num_rounds):
    #     self.__parameter_samples = []

    #     for target_stat, current_injection in zip(self.__target_stats, self.__current_injections):
    #         self.__optimizer.set_target_statistics(target_stat)
    #         self.__optimizer.set_simulation_params(i_inj=current_injection)
    #         self.__optimizer.run_inference(num_simulations=num_simulations, workers=inference_workers, num_rounds=num_rounds)
    #         self.__parameter_samples.append(self.__optimizer.get_samples(sample_threshold=sample_threshold))
    
    
    #Actually calculate the posterior distribution for each current injection.
    def optimize_current_injections(self, num_simulations = 500, num_rounds=1, inference_workers=1, sample_threshold=10):
        #This could be parallelized for speedups.
        self.__parameter_samples = []

        for target_stat, current_injection in zip(self.__target_stats, self.__current_injections):
            self.__optimizer.set_target_statistics(target_stat)
            self.__optimizer.set_simulation_params(i_inj=current_injection)
            self.__optimizer.run_inference(num_simulations=num_simulations, workers=inference_workers, num_rounds=num_rounds)
            self.__parameter_samples.append(self.__optimizer.get_samples(sample_threshold=sample_threshold))
            

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
    def find_best_parameter_set(self, SHOW_BEST_SET=False):
        #What we need to do is find the parameter sets that are the most similar within our threshold.
        #First generate a list of all possible parameter perumutations.
        all_permulations = list(itertools.product(*self.__parameter_samples))
        # print(all_permulations)

        # print('first pair')
        # print(all_permulations[0])

        #Now lets calculate the correlation coeffient sums for each set of parameters.
        correlation_sums = [0] * len(all_permulations)
        
        for index,parameter_set in enumerate(all_permulations):
            pairs = itertools.combinations(parameter_set,2)
            
            num_combinations = 0
            for (a,b) in pairs:
                correlation_sums[index] += pearsonr(a,b)[0]
                num_combinations += 1


        #get the parameter set with the highest total score.
        closest_match = max(correlation_sums)
        best_set = all_permulations[correlation_sums.index(closest_match)]

        self.__final_parameter_matching_ratio = closest_match / num_combinations if num_combinations != 0 else 1

        if SHOW_BEST_SET:
            print('Best parameter set found.')
            print(best_set)
            print(self.__final_parameter_matching_ratio)

        num_parameters = len(best_set[0])
        #Return the average of each value in all parameter sets.
        final = np.zeros(num_parameters)

        for index in range(num_parameters):
            for p_set in best_set:
                final[index] += p_set[index]
            final[index] /= len(best_set)

        final = list(final)
        self.__found_parameters = final

        return final

    #Compare the found solution to the model using the target cell.
    #Takes 2 parameters:
    #   1) If a graph should be displayed which shows the voltage trace of
    #      found parameter set overlayed on the target cell.
    #   2) The directory the above image should be saved to. None means dont
    #      save the image.
    def compare_found_solution_to_model(self, display=False, save_dir=None):
        
        #Find the target voltage traces.
        target_responses = []
        for i_inj in self.__current_injections:
            self.__sim_environ.set_simulation_params(sim_run_time=self.__sim_run_time, delay=self.__delay, inj_time=self.__inj_time, v_init=self.__v_init, i_inj=i_inj)
            self.__sim_environ.simulation_wrapper()
            target_responses.append(np.copy(self.__target_cell.get_potential_as_numpy()))
        

        #Now get the real cell with our current injections.
        found_responses = []
        for i_inj in self.__current_injections:
            self.__optimizer.set_simulation_params(sim_run_time=self.__sim_run_time, delay=self.__delay, inj_time=self.__inj_time, v_init=self.__v_init, i_inj=i_inj)
            self.__optimizer.simulation_wrapper(self.__found_parameters)
            found_responses.append(np.copy(self.__to_optimize.get_potential_as_numpy()))
        
        #Get the time vector.
        time = self.__target_cell.get_time_as_numpy()

        #Now plot everything.
        current_injection_length = len(self.__current_injections)
        fig, axs = plt.subplots(current_injection_length)
        
        if current_injection_length > 1:
            for index, ax in enumerate(axs):
                ax.plot(time, target_responses[index], label='Target')
                ax.plot(time, found_responses[index], label='Found')
                ax.legend()
        else:
            axs.plot(time, target_responses[0], label='Target')
            axs.plot(time, found_responses[0], label='Found')
            axs.legend()


        fig.tight_layout()

        if display:  
            plt.show()
        
        if save_dir != None:
            plt.savefig(os.path.join(save_dir, 'Model_Comparison.png'))

    #Return a dictionary containing the parameter names as keys and the found parmeter as the values.
    def get_optimial_parameter_set(self):
        return dict(zip(self.__optimizer.get_simulation_optimization_params(), self.__found_parameters))

    #Return the best found matching ratio from the find_best_parameter_set function.
    def get_matching_ratio(self):
        return self.__final_parameter_matching_ratio