from  src.Cell import Cell
from src.Optimizer import Optimizer
from neuron import h
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sys

def parse_config(config_directory):
    data = None

    try:
        file = open(config_directory)
    except FileNotFoundError:
        print('The config file %s could not be found.' % config_directory)
    else:
        data = json.load(file)

    return data


def main(config_file, outfile, SHOW_GENERATED=True):
    data = parse_config(config_file) 

    template_dir = data['manifest']['template_dir']
    modfiles_dir = data['manifest']['modfiles_dir']
    template_name = data['manifest']['template_name']
        
        
    sim_run_time = data['run']['tstop']
    delay = data['run']['delay']
    inj_time = data['run']['duration']
    v_init = data['conditions']['v_init']


    if os.system('nrnivmodl %s' % modfiles_dir) == 0:
        print("Compiled Successfully.")
    else:
        print("Could not compile the modfiles at %s" % modfiles_dir)

    h.load_file('stdrun.hoc')

    target_cell = Cell(template_dir, template_name)
    sim_environ = Optimizer(target_cell,None, None, None, *(), **{})

    responses = []
    for i_inj in data['optimization_parameters']['current_injections']:
        sim_environ.set_simulation_params(sim_run_time=sim_run_time, delay=delay, inj_time=inj_time, v_init=v_init,i_inj=i_inj)
        sim_environ.simulation_wrapper()

        responses.append(np.copy(target_cell.get_potential_as_numpy()))
    

    if SHOW_GENERATED:
        time = np.linspace(0,sim_run_time * 1e-3,1024)

      
        #Now plot everything.
        current_injection_length = len(responses)
        fig,axes = plt.subplots(current_injection_length)
        if current_injection_length > 1:
            for index,ax in enumerate(axes):
                ax.plot(time, responses[index])
        else:
            axes[0].plot(time, responses[0])


        handles, labels = plt.gca().get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right')
        fig.tight_layout()        
        plt.show()


    responses = np.stack(responses, axis=0).T

    #Save the csv
    np.savetxt(outfile, responses, delimiter=',')
    
    
if __name__ == "__main__":
    main(sys.argv[1],sys.argv[2])