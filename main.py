import os
from neuron import h
from Cell import Cell
from Optimizer import Optimizer

def summary():
    return 0

def main(template_name, template_dir='cells', modfiles_dir=None):
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
    cell_obj.graph_potential()

    optimizer.set_target_statistics(target)
    optimizer.set_simulation_optimization_params(['gbar_natCA3', 'gbar_kdrCA3', 'gbar_napCA3', 'gbar_imCA3'])
    optimizer.run_inference(workers=1)
    obtained = optimizer.get_sample()

    print(obtained)
    optimizer.simulation_wrapper(obtained)
    cell_obj.calculate_adapting_statistics(sim_variables=optimizer.get_simulation_time_varibles(),DEBUG=True)
    cell_obj.graph_potential()
    

    

if __name__ == '__main__':
    main('CA3PyramidalCell', template_dir='cells/CA3Cell_Qian/CA3.hoc')