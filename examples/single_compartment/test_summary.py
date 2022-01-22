import numpy as np
from scipy.signal import find_peaks

#Important statistcs for an adapting cell
#Resting membrane potential.
#Average spike peak?
#Average trough value?
#Adaptation ratio: This is defined as a_r = (f_max - f_steadystate)/f_max
#                  where f_max is the maximum instantaneous frequency  (first spike probably) 
#                  f_steadystate is the steady state instaneous frequency (last spike probably)
#Adapation speed: Some sort of metric which captures how fast it adapts.
#Number of spikes. 
def calculate_adapting_statistics(v,t, DEBUG=False, *args, **kwargs):
    
    trace = v
    time = t
    
    sim_run_time               = kwargs['sim_run_time']
    delay                      = kwargs['delay']
    inj_time                   = kwargs['inj_time']
    spike_height_threshold     = kwargs['spike_height_threshold']
    spike_adaptation_threshold = kwargs['spike_adaptation_threshold']


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
        end_injection = np.where(np.isclose(time,(delay + inj_time)*1e-3,0.99))[0][0]
        end_point = len(time) - 1
        resting = np.mean(trace[end_injection:end_point])
    
    #Average spike and trough voltage.
    spike_times = np.asarray(find_peaks(trace,height=spike_height_threshold)[0])
    
    #Take care of the case where nothing spikes.
    if len(spike_times) == 0:
        return np.concatenate((resting, resting, resting, 0, 0, 0),axis=None) 

    average_peak = np.mean(np.take(trace, spike_times))
    average_trough = np.mean(np.take(trace, np.asarray(find_peaks(-trace,height=spike_height_threshold)[0])))

    #Take care of the case where there is only one spike.
    if len(spike_times) == 1:
        return np.concatenate((resting, average_peak, average_trough, 0, 1, 1),axis=None) 

    #Adaptation ratio        
    f_max = 1.0 / (spike_times[1] - spike_times[0])
    f_min = 1.0 / (spike_times[-1] - spike_times[-2])

    adaptation_index = (f_max - f_min) / f_max

    #Adaptation speed.
    instantaneous_freq = 1.0 / np.diff(spike_times)
    adaptation_speed = np.where(np.isclose(instantaneous_freq, f_min, spike_adaptation_threshold))[0][0]

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
