from matplotlib import axes
from numpy.lib.function_base import average
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
import matplotlib.gridspec as gridspec
from scipy.signal import find_peaks
import math


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
    def __init__(self, current_injections, modfiles_dir, template_dir, template_name, optimization_parameters, parameter_range, architecture='summary', summary_funct = None, features=8,*args, **kwargs):
        
        #Store the current injection list and the spike threshold.
        self.__current_injections = current_injections

  
        self.__embedding_net = None
        if architecture == 'convolution':
            self.__embedding_net = SummaryCNN(len(current_injections), features)
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

        self.__parameter_samples = []

    def run_forward_pass(self, input):
        out = self.__embedding_net(input)
        return out
    
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
    #If CNN is activated and not yet trained, train it.
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

        if self.__embedding_net == None:
            #Loop through each current injection and calculate the target statitics.
            self.__target_stats = []
            for i_inj in self.__current_injections:
                self.__sim_environ.set_simulation_params(sim_run_time=self.__sim_run_time, delay=self.__delay, inj_time=self.__inj_time, v_init=self.__v_init, i_inj=i_inj)
                self.__target_stats.append(self.__sim_environ.simulation_wrapper())
        else:
            self.__sim_environ.set_current_injection_list(self.__current_injections)

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
        
        #Run inference by passing in the current injections in as independent channels.
        
        # self.__optimizer.set_target_statistics(target_stat)
        # self.__optimizer.set_simulation_params(i_inj=current_injection)
        self.__optimizer.set_current_injection_list(self.__current_injections)
        self.__optimizer.run_inference_learned_stats(self.__embedding_net, self.__sim_environ, num_simulations=num_simulations, num_rounds=num_rounds, workers=inference_workers)
        self.__parameter_samples.append(self.__optimizer.get_samples(-1, sample_threshold=self.__top_n))
        # self.__optimizer.clear_posterior()

    def optimize_current_injections_summary(self, num_simulations = 500, num_rounds=1, inference_workers=1):
        #This could be parallelized for speedups.
        for target_stat, current_injection in zip(self.__target_stats, self.__current_injections):
            self.__optimizer.set_target_statistics(target_stat)
            self.__optimizer.set_simulation_params(i_inj=current_injection)
            self.__optimizer.run_inference_multiround(num_simulations=num_simulations, workers=inference_workers, num_rounds=num_rounds)
            self.__parameter_samples.append(self.__optimizer.get_samples(-1, sample_threshold=self.__top_n))
            
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
    def find_best_parameter_sets(self, SHOW_BEST_SET=False):
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

        self.__found_parameters = []

        for _ in range(self.__top_n):
            #get the parameter set with the highest total score.
            closest_match = max(correlation_sums)
            closest_match_index = correlation_sums.index(closest_match)
            best_set = all_permulations[closest_match_index]
            del correlation_sums[closest_match_index]

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
            self.__found_parameters.append(final)

        return final[0]

    #Compare the found solution to the model using the target cell.
    #Takes 2 parameters:
    #   1) If a graph should be displayed which shows the voltage trace of
    #      found parameter set overlayed on the target cell.
    #   2) The directory the above image should be saved to. None means dont
    #      save the image.
    def compare_found_solution_to_target(self, top_n=1, display=False, save_dir=None):
        #Get the time vector.
        time = self.__target_cell.get_time_as_numpy()

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
                found_responses.append(np.copy(self.__to_optimize.get_potential_as_numpy()))
            
            #Now plot everything.
            current_injection_length = len(self.__current_injections)
            inner = gridspec.GridSpecFromSubplotSpec(current_injection_length, 1, subplot_spec=outer[i])
            if current_injection_length > 1:
                for index in range(current_injection_length):
                    ax = plt.Subplot(fig,inner[index])
                    ax.plot(time, self.__target_responses[index], label='Target')
                    ax.plot(time, found_responses[index], label='Found')
                    fig.add_subplot(ax)
            else:
                ax = plt.Subplot(fig,inner[0])
                ax.plot(time, self.__target_responses[0], label='Target')
                ax.plot(time, found_responses[0], label='Found')
                fig.add_subplot(ax)

        handles, labels = plt.gca().get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right')
        fig.tight_layout()

        if display:  
            fig.show()
        
        if save_dir != None:
            fig.savefig(save_dir,bbox_inches='tight')

    def generate_target_from_model(self):
        #Find the target voltage traces.
        self.__target_responses = []
        for i_inj in self.__current_injections:
            self.__sim_environ.set_simulation_params(sim_run_time=self.__sim_run_time, delay=self.__delay, inj_time=self.__inj_time, v_init=self.__v_init, i_inj=i_inj)
            self.__sim_environ.simulation_wrapper()
            self.__target_responses.append(np.copy(self.__target_cell.get_potential_as_numpy()))

    def generate_found_FI_curve(self, display=False, save_dir=None):

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


        #Create two graphs of both transient and average FI.


        # #Resting Membrane Potential.

        # #We need to calculate the resting membrane potential,
        # #to do this we need to find a part of the simmulation where it is at rest.
        # #preferably we get this from the end after the current injection, however if
        # #the current injection ends at the end of the simulation then we can take it from the
        # #beginning with some padding.
        # padding = 50
        # if sim_run_time == delay + inj_time:
        #     start_injection = np.where(np.isclose(time, sim_run_time))[0][0]
        #     start_point = np.where(np.isclose(time, sim_run_time - padding))[0][0]
        #     resting = np.mean(trace[start_point:start_injection])
        # else:
        #     end_injection = np.where(np.isclose(time,delay + inj_time,0.99))[0][0]
        #     end_point = len(time) - 1
        #     resting = np.mean(trace[end_injection:end_point])
        
        # #Average spike and trough voltage.
        # spike_times = np.asarray(find_peaks(trace,height=spike_height_threshold)[0])
        
        # #Take care of the case where nothing spikes.
        # if len(spike_times) == 0:
        #     return np.concatenate((resting, resting, resting, 0, 0, 0),axis=None) 

        # average_peak = np.mean(np.take(trace, spike_times))
        # average_trough = np.mean(np.take(trace, np.asarray(find_peaks(-trace,height=spike_height_threshold)[0])))

        # #Take care of the case where there is only one spike.
        # if len(spike_times) == 1:
        #     return np.concatenate((resting, average_peak, average_trough, 0, 1, 1),axis=None) 

        # #Adaptation ratio        
        # f_max = 1.0 / (spike_times[1] - spike_times[0])
        # f_min = 1.0 / (spike_times[-1] - spike_times[-2])

        # adaptation_index = (f_max - f_min) / f_max

        # #Adaptation speed.
        # instantaneous_freq = 1.0 / np.diff(spike_times)
        # adaptation_speed = np.where(np.isclose(instantaneous_freq, f_min, spike_adaptation_threshold))[0][0]

        # #Number of spikes
        # spike_num = len(spike_times)    
        
        # if DEBUG:
        #     print('Calculated resting membrane potential: %f' % resting)
        #     print('Average peak voltage: %f' % average_peak)
        #     print('Average trough voltage: %f' % average_trough)
        #     print('Adaptation ratio: %f' % adaptation_index)
        #     print('Adaptation speed: %d' % adaptation_speed)
        #     print('Number of spikes: %d' % spike_num)


        # fig.tight_layout()

        # if display:  
        #     plt.show()
        
        # if save_dir != None:
        #     plt.savefig(os.path.join(save_dir, 'Model_Comparison.png'))

    #Return a list of dictionaries containing the parameter names as keys and the found parmeter as the values
    #for the top n values.
    def get_optimial_parameter_sets(self, top_n=1):        
        p_list = []
        for index in range(top_n):
            p_list.append(dict(zip(self.__optimizer.get_simulation_optimization_params(), self.__found_parameters[index])))
        
        return p_list

    #Compare the found parameter sets to the ground truth parameters.
    def compare_found_parameters_to_ground_truth(self, ground_truth, top_n=1):
        found_parameters = self.get_optimial_parameter_sets(top_n=top_n)
        
        error_list = []

        for i in range(len(found_parameters)):
            rpe = {}
            for index, parameter_name in enumerate(found_parameters[i]):
                rpe[parameter_name] = ((found_parameters[i][parameter_name] - ground_truth[index]) / ground_truth[index]) * 100
            
            average_rpe = sum([abs(rpe[key]) for key in rpe])/ len(rpe)  #Calculate relative percent error.

            error_list.append((average_rpe, rpe))

        return error_list