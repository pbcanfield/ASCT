import numpy as np
from numpy.core.fromnumeric import reshape
from src.Tuner import CellTuner
import argparse
import json
import csv
from datetime import datetime
import importlib

def tune_with_template(config_data, *args, **kwargs):


    manifest = config_data['manifest']
    conditions = config_data['conditions']
    run = config_data['run']
    settings = config_data['optimization_settings']
    parameters = config_data['optimization_parameters']
    summary = config_data['summary']

    #Lets load the summary function into memory.
    summary['summary_file'] = summary['summary_file'].replace('.py','').replace('/','.')

    summary_module = importlib.import_module(summary['summary_file'])
    summary_function = getattr(summary_module, summary['function_name'])

    #Now remove the file and function name from the dictionary.
    del summary['summary_file']
    del summary['function_name']

    if not kwargs['c_mod']:
        modfiles_dir = None
    
    tuner = CellTuner( 
                      manifest['modfiles_dir'], 
                      manifest['template_dir'], 
                      manifest['template_name'],
                      parameters['current_injections'],
                      parameters['parameters'], 
                      (parameters['lows'], parameters['highs']), 
                      architecture=manifest['architecture'],
                      summary_funct=summary_function,
                      features=settings['features'],
                      kwargs=summary)       #Pass in the summary function variables as kwargs

    tuner.set_simulation_params(sim_run_time=run['tstop'], delay=run['delay'],inj_time=run['duration'],v_init=conditions['v_init'])
    
    #We need to check if we are validating against a ground truth model or if we are tuning from input data.
    if manifest['job_type'] == 'ground_truth':
        tuner.calculate_target_stats_from_model(manifest['target_template_dir'], manifest['target_template_name'])
        
        tuner.optimize_current_injections(num_simulations=settings['num_simulations'],
                                          inference_workers=settings['workers'], 
                                          sample_threshold=kwargs['result_threshold'], 
                                          num_rounds=settings['num_rounds'])

        tuner.find_best_parameter_sets()
        
        print('The optimizer found the following parameter set:')
        print(tuner.get_optimial_parameter_sets(kwargs['result_threshold']))

        print('Relative Percent Error from ground truth:')
        print(tuner.compare_found_parameters_to_ground_truth(parameters['ground_truth'],kwargs['result_threshold']))

        #print('The matching ratio is: %f (closer to 1 is better)' % tuner.get_matching_ratio())

        tuner.generate_target_from_model()
        tuner.compare_found_solution_to_target(kwargs['result_threshold'],kwargs['display'],kwargs['save_dir'])
    
    elif manifest['job_type'] == 'from_data':
        #Load in the input_data
        responses = load_current_injections_from_csv(manifest['input_data'])
        tuner.set_target_responses(responses)

        tuner.calculate_target_stats_from_data()
        
        tuner.optimize_current_injections(num_simulations=settings['num_simulations'],
                                          inference_workers=settings['workers'], 
                                          sample_threshold=kwargs['result_threshold'], 
                                          num_rounds=settings['num_rounds'])

        tuner.find_best_parameter_sets()
        
        print('The optimizer found the following parameter set:')
        print(tuner.get_optimial_parameter_sets(kwargs['result_threshold']))

        tuner.compare_found_solution_to_target(kwargs['result_threshold'],kwargs['display'],kwargs['save_dir'])

def parse_config(config_directory):
    data = None

    try:
        file = open(config_directory)
    except FileNotFoundError:
        print('The config file %s could not be found.' % config_directory)
    else:
        data = json.load(file)

    return data

def load_current_injections_from_csv(file_dir):
    file = open(file_dir, 'r')
    csv_file =  csv.reader(file,delimiter=',')

    responses = [[] for _ in range(len(next(csv_file)))]
    file.seek(0)
    for row in csv_file:
        for i, column in enumerate(row):
            responses[i].append(column)
    
    responses = [np.array(e,dtype=float) for e in responses]
    return responses
    

if __name__ == '__main__':
    argument_parser = argparse.ArgumentParser(description='Uses SBI to find optimal parameter sets for biologically realistic neuron simulations.')

    argument_parser.add_argument('config_dir', type=str, help='the optimization config file directory')
    argument_parser.add_argument('save_dir', nargs='?', type=str, default=None, help='[optional] the directory to save figures to')
    argument_parser.add_argument('-g', default=False, action='store_true', help='displays graphics')
    argument_parser.add_argument('-c', default=False, action='store_true', help='compiles modfiles')
    argument_parser.add_argument('-l', default=False, action='store_true', help='store log files')
    argument_parser.add_argument('-n', default=1, type=int, help='the number of found parameters to show (must be less than the threshold sample size in the optimization config file)')

    args = argument_parser.parse_args()

    config_data = parse_config(args.config_dir)

    tune_with_template(config_data, save_dir=args.save_dir, display=args.g, result_threshold=args.n, c_mod=args.c, log=args.l)
    