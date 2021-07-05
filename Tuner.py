import scipy as sp
from Optimizer import Optimizer
from Cell import Cell
from neuron import h
import itertools
import os
from scipy.stats.stats import pearsonr 
import numpy as np
import matplotlib.pyplot as plt


class CellTuner:
    def __init__(self, current_injections, modfiles_dir, template_dir, template_name, parameter_range, optimization_parameters, spike_height_threshold=0, spike_adaptation_threshold=0.95, learn_stats=False):
        self.__current_injections = current_injections
        self.__spike_threshold = spike_height_threshold

         #First lets try to compile our modfiles.
        if os.system('nrnivmodl %s' % modfiles_dir) == 0:
            print("Compiled Successfully.")
        else:
            print("Could not compile the modfiles at %s" % modfiles_dir)

        #Now load in the standard run hoc.
        h.load_file('stdrun.hoc')

        self.__learn_stats = learn_stats
        self.__to_optimize = Cell(template_dir, template_name)
        self.__optimizer = Optimizer(self.__to_optimize.get_cell(), parameter_range, self.__to_optimize.calculate_adapting_statistics if not learn_stats else self.__to_optimize.generate_simulation_image, spike_height_threshold=spike_height_threshold,spike_adaptation_threshold=spike_adaptation_threshold,learn_stats=learn_stats)
        self.__optimizer.set_simulation_optimization_params(optimization_parameters)

    def add_target_statistics(self, target_stats):
        self.__target_stats = target_stats
    
    def set_simulation_params(self, sim_run_time = 1500, delay = 400, inj_time = 500, v_init = -75):
        self.__sim_run_time = sim_run_time
        self.__delay = delay
        self.__inj_time = inj_time
        self.__v_init = v_init

    def calculate_target_stats_from_model(self, template_dir, template_name):    
        self.__target_cell = Cell(template_dir, template_name)
        target = self.__target_cell.generate_simulation_image if self.__learn_stats else self.__target_cell.calculate_adapting_statistics
    
        _sim_environ = Optimizer(self.__target_cell.get_cell(), None, target,learn_stats=self.__learn_stats)
        
        self.__target_stats = []
        for i_inj in self.__current_injections:
            _sim_environ.set_simulation_params(sim_run_time=self.__sim_run_time, delay=self.__delay, inj_time=self.__inj_time, v_init=self.__v_init, i_inj=i_inj)
            self.__target_stats.append(_sim_environ.simulation_wrapper())

    def optimize_current_injections(self, num_simulations = 500, num_rounds=1, inference_workers=1, sample_threshold=10):
        #This could be parallelized for speedups.
        self.__parameter_samples = []
        
        for target_stat, current_injection in zip(self.__target_stats, self.__current_injections):
            self.__optimizer.set_target_statistics(target_stat)
            self.__optimizer.set_simulation_params(i_inj=current_injection)
            self.__optimizer.run_inference(num_simulations=num_simulations, workers=inference_workers, num_rounds=num_rounds)
            self.__parameter_samples.append(self.__optimizer.get_samples(sample_threshold=sample_threshold))
            


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

    def compare_found_solution_to_model(self, display=False, save_dir=None):
        _sim_environ = Optimizer(self.__target_cell.get_cell(), None, self.__target_cell.calculate_adapting_statistics)

        target_responses = []
        for i_inj in self.__current_injections:
            _sim_environ.set_simulation_params(sim_run_time=self.__sim_run_time, delay=self.__delay, inj_time=self.__inj_time, v_init=self.__v_init, i_inj=i_inj)
            _sim_environ.simulation_wrapper()
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

    def get_optimial_parameter_set(self):
        return dict(zip(self.__optimizer.get_simulation_optimization_params(), self.__found_parameters))

    def get_matching_ratio(self):
        return self.__final_parameter_matching_ratio