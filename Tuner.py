from Optimizer import Optimizer
from Cell import Cell
from neuron import h
import itertools
import os
from scipy.stats.stats import pearsonr 
import numpy as np

class CellTuner:
    def __init__(self, current_injections, modfiles_dir, template_dir, template_name, parameter_range, optimization_parameters):
        self.__current_injections = current_injections

         #First lets try to compile our modfiles.
        if os.system('nrnivmodl %s' % modfiles_dir) == 0:
            print("Compiled Successfully.")
        else:
            print("Could not compile the modfiles at %s" % modfiles_dir)

        #Now load in the standard run hoc.
        h.load_file('stdrun.hoc')

        self.__to_optimize = Cell(template_dir, template_name)
        self.__optimizer = Optimizer(self.__to_optimize.get_cell(), parameter_range, self.__to_optimize.calculate_adapting_statistics)
        self.__optimizer.set_simulation_optimization_params(optimization_parameters)

    def add_target_statistics(self, target_stats):
        self.__target_stats = target_stats
    
    def set_simulation_params(self, sim_run_time = 1500, delay = 400, inj_time = 50, v_init = -75):
        self.__sim_run_time = sim_run_time
        self.__delay = delay
        self.__inj_time = inj_time
        self.__v_init = v_init

    def calculate_target_stats_from_model(self, template_dir, template_name):
        target_cell = Cell(template_dir, template_name)
        _sim_environ = Optimizer(target_cell.get_cell(), None, target_cell.calculate_adapting_statistics)
        
        self.__target_stats = []
        for i_inj in self.__current_injections:
            _sim_environ.set_simulation_params(sim_run_time=self.__sim_run_time, delay=self.__delay, inj_time=self.__inj_time, v_init=self.__v_init, i_inj=i_inj)
            _sim_environ.simulation_wrapper()
            self.__target_stats.append(target_cell.calculate_adapting_statistics(_sim_environ.get_simulation_time_varibles()))

    def optimize_current_injections(self, num_simulations = 500, inference_workers=1, sample_threshold=10):
        


        #This could be parallelized for speedups.
        self.__parameter_samples = []
        
        for target_stat, current_injection in zip(self.__target_stats, self.__current_injections):
            self.__optimizer.set_target_statistics(target_stat)
            self.__optimizer.set_simulation_params(i_inj=current_injection)
            self.__optimizer.run_inference(num_simulations=num_simulations, workers=inference_workers)
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
            for (a,b) in pairs:
                correlation_sums[index] += pearsonr(a,b)[0]


        #get the parameter set with the highest total score.
        best_set = all_permulations[correlation_sums.index(max(correlation_sums))]

        if SHOW_BEST_SET:
            print('Best parameter set found.')
            print(best_set)

        num_parameters = len(best_set[0])
        #Return the average of each value in all parameter sets.
        final = np.zeros(num_parameters)

        for index in range(num_parameters):
            for p_set in best_set:
                final[index] += p_set[index]
            final[index] /= len(best_set)

        return final