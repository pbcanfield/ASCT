import os
from neuron import h
from numpy import nan_to_num
import scipy as sp
from Cell import Cell
from Optimizer import Optimizer
from Tuner import CellTuner
import argparse
import json

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

    
    # cell_obj = Cell(template_dir, template_name)
    # cell_obj.get_cell().all.printnames()

    #TODO:
    # Define a plotting function in the cell object

    
    cell_obj = Cell(template_dir, template_name)
    
    
    optimizer = Optimizer(cell_obj.get_cell(), ([0.001, 0.001, 0.00001, 0.001,], [0.1, 0.1, 0.001, 0.1]), cell_obj.calculate_adapting_statistics)
    optimizer.set_simulation_params()
    optimizer.simulation_wrapper()
    target = cell_obj.calculate_adapting_statistics(sim_variables=optimizer.get_simulation_time_varibles(),DEBUG=True)
    #cell_obj.graph_potential()

    optimizer.set_target_statistics(target)
    optimizer.set_simulation_optimization_params(['gbar_natCA3', 'gbar_kdrCA3', 'gbar_napCA3', 'gbar_imCA3'])
    optimizer.run_inference(num_simulations=100,workers=1) #Cant parallelize because of NEURON?
    obtained = optimizer.get_sample()

    print(obtained)
    optimizer.simulation_wrapper(obtained)
    cell_obj.calculate_adapting_statistics(sim_variables=optimizer.get_simulation_time_varibles(),DEBUG=True)
    #cell_obj.graph_potential()
    
def tune_with_template(current_injections, low, high, 
                       parameter_list, num_simulations, num_rounds,
                       sim_run_time, delay, inj_time, v_init, spike_height, spike_adaptation,
                       template_name, target_template_name,
                       target_template_dir, template_dir, modfiles_dir,
                       threshold_sample_size, workers, display,save_dir):

    
    if os.path.exists('x86_64'):
        os.system('rm -rf x86_64')

    tuner = CellTuner(current_injections, modfiles_dir, template_dir, template_name, (low, high), parameter_list,spike_height_threshold=spike_height,spike_adaptation_threshold=spike_adaptation)
    tuner.set_simulation_params(sim_run_time=sim_run_time, delay=delay,inj_time=inj_time,v_init=v_init)
    tuner.calculate_target_stats_from_model(target_template_dir, target_template_name)
    tuner.optimize_current_injections(num_simulations=num_simulations,inference_workers=workers, sample_threshold=threshold_sample_size, num_rounds=num_rounds)

    tuner.find_best_parameter_set()

    print('The optimizer found the following parameter set:')
    print(tuner.get_optimial_parameter_set())

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
    # argument_parser.add_argument('--modfiles', nargs='?', type=str, default=None, help='[optional] the directory containing the modfiles, otherwise assumed to be the same as template_dir')
    # argument_parser.add_argument('--workers', nargs= '?', type=int, default=1, help='number of workers to spawn')
    # argument_parser.add_argument('--simulations', nargs= '?', type=int, default=1, help='number of simulations')

    
    # argument_parser.add_argument('template_dir', type=str, help='the directory containing the cell HOC template')
    # argument_parser.add_argument('template_name', type=str, help='the name of the HOC template')

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
                       save_dir=args.save_dir)


