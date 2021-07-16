import os
from neuron import h
import scipy as sp
from Cell import Cell
from Optimizer import Optimizer
from Tuner import CellTuner
import argparse
import json
from scipy.signal import find_peaks
import numpy as np
from SummaryNet import SummaryCNN


#Important statistcs for an adapting cell
#Resting membrane potential.
#Average spike peak?
#Average trough value?
#Adaptation ratio: This is defined as a_r = (f_max - f_steadystate)/f_max
#                  where f_max is the maximum instantaneous frequency  (first spike probably) 
#                  f_steadystate is the steady state instaneous frequency (last spike probably)
#Adapation speed: Some sort of metric which captures how fast it adapts.
#Number of spikes. 
def calculate_adapting_statistics(cell,sim_variables=(), spike_height_threshold=0, spike_adaptation_threshold=0.99, DEBUG=False):
    sim_run_time = sim_variables[0]
    delay = sim_variables[1]
    inj_time = sim_variables[2]
    
    trace = cell.get_potential_as_numpy()
    time = cell.get_time_as_numpy()
    
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
        end_injection = np.where(np.isclose(time,delay + inj_time,0.99))[0][0]
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

def test_optimizer(template_name, template_dir='cells', modfiles_dir=None):
    if modfiles_dir == None:
        modfiles_dir = os.path.join(os.path.dirname(template_dir),'modfiles')
    
    if os.path.exists('x86_64'):
        os.system('rm -rf x86_64')

    #First lets try to compile our modfiles.
    if os.system('nrnivmodl %s' % modfiles_dir) == 0:
        print("Compiled Successfully.")
    else:
        print("Could not compile the modfiles at %s" % modfiles_dir)

    #Now load in the standard run hoc.
    h.load_file('stdrun.hoc')
    

    #Summary Stats Test
    # cell_obj = Cell(template_dir, template_name, calculate_adapting_statistics)
    
    # optimizer = Optimizer(cell_obj, 
    #                       None,                                                        #Parameter list, none for now.
    #                       ([0.001, 0.001, 0.00001, 0.001,], [0.1, 0.1, 0.001, 0.1]),   #Parameter range.
    #                       calculate_adapting_statistics,                               #Summary function.
    #                       (600,50,500),                                                #args for summary function
    #                       spike_adaptation_threshold=0.95,                             #kwargs for summary function
    #                       spike_height_threshold=0)

    # optimizer.set_simulation_params()
    # optimizer.simulation_wrapper()

    # print(cell_obj.generate_simulation_image(save_img_dir='generated1.png'))
    # cell_obj.graph_potential(save_img_dir='original.png')

    #CNN Test
    cell_obj = Cell(template_dir, template_name, None)
    optimizer = Optimizer(cell_obj, 
                          ['gbar_natCA3', 'gbar_kdrCA3'], #Parameter list.
                          ([0.001, 0.001], [0.1, 0.1]),   #Parameter range.
                          None)   
                        

    embedding_net = SummaryCNN()
    optimizer.run_inference_learned_stats(embedding_net, num_simulations=1000)


def tune_with_template(current_injections, low, high, 
                       parameter_list, num_simulations, num_rounds,
                       sim_run_time, delay, inj_time, v_init, spike_height, spike_adaptation,
                       template_name, target_template_name,
                       target_template_dir, template_dir, modfiles_dir,
                       threshold_sample_size, workers, display,save_dir, learn_stats):

    
    if os.path.exists('x86_64'):
        os.system('rm -rf x86_64')


    
    tuner = CellTuner(current_injections, 
                      modfiles_dir, 
                      template_dir, 
                      template_name, 
                      parameter_list, 
                      (low, high), 
                      learn_stats=learn_stats,
                      summary_funct=calculate_adapting_statistics,
                      sim_variables=(600,50,500), 
                      spike_height_threshold=spike_height,
                      spike_adaptation_threshold=spike_adaptation)

    tuner.set_simulation_params(sim_run_time=sim_run_time, delay=delay,inj_time=inj_time,v_init=v_init)
    tuner.calculate_target_stats_from_model(target_template_dir, target_template_name)

    
    tuner.optimize_current_injections(num_simulations=num_simulations,inference_workers=workers, sample_threshold=threshold_sample_size, num_rounds=num_rounds)
    tuner.find_best_parameter_set()

    #print('The optimizer found the following parameter set:')
    #print(tuner.get_optimial_parameter_set())

    #print('The matching ratio is: %f (closer to 1 is better)' % tuner.get_matching_ratio())

    tuner.compare_found_solution_to_model(display,save_dir)

def parse_config(config_directory):
    data = None

    try:
        file = open(config_directory)
    except FileNotFoundError:
        print('The config file %s could not be found.' % config_directory)
    else:
        data = json.load(file)

    return data

if __name__ == '__main__':
    #test_optimizer('CA3PyramidalCell', template_dir='cells/CA3Cell_Qian/CA3.hoc')
    #tune_with_template('CA3PyramidalCell', template_dir='cells/CA3Cell_Qian/CA3.hoc')

    
    
    argument_parser = argparse.ArgumentParser(description='Uses SBI to find optimal parameter sets for biologically realistic neuron simulations.')


    argument_parser.add_argument('config_dir', type=str, help='the optimization config file directory')
    argument_parser.add_argument('save_dir', nargs='?', type=str, default=None, help='[optional] the directory to save figures to')
    argument_parser.add_argument('-g', default=False, action='store_true', help='displays graphics')
    

    args = argument_parser.parse_args()

    config_data = parse_config(args.config_dir)

    #Now lets extract the data.
    manifest = config_data['manifest']
    conditions = config_data['conditions']
    run = config_data['run']
    optimization_parameters = config_data['optimization_parameters']

    
    tune_with_template(current_injections=optimization_parameters['current_injections'], 
                       low=optimization_parameters['lows'],
                       high=optimization_parameters['highs'],
                       parameter_list=optimization_parameters['parameters'],
                       num_simulations=run['num_simulations'],
                       num_rounds = run['num_rounds'],
                       sim_run_time=run['tstop'],
                       delay=run['delay'],
                       inj_time=run['duration'],
                       v_init=conditions['v_init'],
                       spike_height=run['spike_threshold'],
                       spike_adaptation=run['spike_adaptation'],
                       template_name=manifest['template_name'],
                       target_template_name=manifest['target_template_name'],
                       target_template_dir=manifest['target_template_dir'],
                       template_dir=manifest['template_dir'],
                       modfiles_dir=manifest['modfiles_dir'],
                       threshold_sample_size=run['threshold_sample_size'],
                       workers=run['workers'],
                       display=args.g,
                       save_dir=args.save_dir,
                       learn_stats=manifest['learn_stats'])
    