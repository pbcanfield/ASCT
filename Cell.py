from os import get_terminal_size
from neuron import h

import matplotlib.pyplot as plt
import numpy as np

from scipy.signal import find_peaks

class Cell:
    def __init__(self, template_dir, template_name):
        h.load_file(template_dir)
        invoke_cell = getattr(h, template_name)
        self.__cell = invoke_cell()

        #Initialize vectors to track membrane voltage.
        self.__mem_potential = h.Vector()
        self.__time = h.Vector()
        self.__time.record(h._ref_t)
        self.__mem_potential.record(self.__cell.soma[0](0.5)._ref_v) #This line means that all cell templates must
                                                                     #have a soma array with at least one soma
                                                                     # element in it.

    def get_cell(self):
        return self.__cell

    def get_potential_as_numpy(self):
        return self.__mem_potential.as_numpy()
    
    def get_time_as_numpy(self):
        return self.__time.as_numpy()

    def graph_potential(self):
        plt.close()
        plt.figure(figsize = (20,5))
        plt.plot(self.__time, self.__mem_potential)
        plt.xlabel('Time')
        plt.ylabel('Membrane Potential')
        plt.show()

    
    #Important statistcs for an adapting cell
    #Resting membrane potential.
    #Average spike peak?
    #Average trough value?
    #Adaptation ratio: This is defined as a_r = (f_max - f_steadystate)/f_max
    #                  where f_max is the maximum instantaneous frequency  (first spike probably) 
    #                  f_steadystate is the steady state instaneous frequency (last spike probably)
    #Adapation speed: Some sort of metric which captures how fast it adapts.
    #Number of spikes. 
    def calculate_adapting_statistics(self, sim_variables, spike_adaption_threshold = 0.99, DEBUG=False):
        
        sim_run_time = sim_variables[0]
        delay = sim_variables[1]
        inj_time = sim_variables[2]
        
        trace = self.get_potential_as_numpy()
        time = self.get_time_as_numpy()
        
        #Resting Membrane Potential.

        #We need to calculate the resting membrane potential,
        #to do this we need to find a part of the simmulation where it is at rest.
        #preferably we get this from the end after the current injection, however if
        #the current injection ends at the end of the simulation then we can take it from the
        #beginning with some padding.
        padding = 50
        if sim_run_time == delay + inj_time:
            start_injection = np.where(np.isclose(time, sim_run_time))[0][0]
            start_point = np.where(np.isclose(time, sim_run_time - padding))[0][0]
            resting = np.mean(trace[start_point:start_injection])
        else:
            end_injection = np.where(np.isclose(time,delay + inj_time))[0][0]
            end_point = np.where(np.isclose(time,delay + inj_time + padding))[0][0]
            resting = np.mean(trace[end_injection:end_point])
        
        #Average spike and trough voltage.
        spike_times = np.asarray(find_peaks(trace,height=0)[0])
        
        #Take care of the case where nothing spikes.
        if len(spike_times) == 0:
            return np.concatenate((resting, resting, resting, 0, 0, 0),axis=None) 


        

        average_peak = np.mean(np.take(trace, spike_times))
        average_trough = np.mean(np.take(trace, np.asarray(find_peaks(-trace,height=0)[0])))

        #Take care of the case where there is only one spike.
        if len(spike_times) == 1:
            return np.concatenate((resting, average_peak, average_trough, 0, 1, 1),axis=None) 

        #Adaptation ratio        
        f_max = 1.0 / (spike_times[1] - spike_times[0])
        f_min = 1.0 / (spike_times[-1] - spike_times[-2])

        adaptation_index = (f_max - f_min) / f_max

        #Adaptation speed.
        instantaneous_freq = 1.0 / np.diff(spike_times)
        adaptation_speed = np.where(np.isclose(instantaneous_freq, f_min, spike_adaption_threshold))[0][0]

        #Number of spikes
        spike_num = len(spike_times)    

        
        if DEBUG:
            print('Calculated resting membrane potential: %f' % resting)
            print('Average peak voltage: %f' % average_peak)
            print('Average trough voltage: %f' % average_trough)
            print('Adaptation ratio: %f' % adaptation_index)
            print('Adaptation speed: %d' % adaptation_speed)
            print('Number of spikes: %d' % spike_num)

        return np.concatenate((resting, average_peak, average_trough, adaptation_index, adaptation_speed, spike_num), axis=None)
            
        
