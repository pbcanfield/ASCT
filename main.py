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
    optimizer = Optimizer(cell_obj.get_cell(), ([0.001,0.001,1e-6,1e-6], [0.1,0.1,1e-4,1e-3]), None, None)
    
    optimizer.set_simulation_params(i_inj=0)
    optimizer.simmulation_wrapper()
    cell_obj.graph_potential()

    optimizer.set_simulation_params()
    optimizer.simmulation_wrapper()
    cell_obj.graph_potential()


if __name__ == '__main__':
    main('CA3PyramidalCell', template_dir='cells/CA3Cell_Qian/CA3.hoc')