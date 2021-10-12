from neuron import h
import matplotlib.pyplot as plt
from scipy import signal


#This class takes a NEURON HOC file as an input creates a wrapper
#which can be run by sbi for simulation data.
class Cell:

    #Constructs a Cell object.
    #Takes:
    #   1) The HOC template directory.
    #   2) The HOC object name for the given cell.
    #   3) The summary statistics function.
    def __init__(self, template_dir, template_name):
        
        #Load the template directory.
        h.load_file(template_dir)

        #Get the cell from the h object.
        invoke_cell = getattr(h, template_name)
       
        #Exctract the cell object itself.
        self.__cell = invoke_cell()

        #Initialize vectors to track membrane voltage.
        self.__mem_potential = h.Vector()
        self.__time = h.Vector()

        #Record time and membrane potential.
        self.__time.record(h._ref_t)
        self.__mem_potential.record(self.__cell.soma[0](0.5)._ref_v) #This line means that all cell templates must
                                                                     #have a soma array with at least one soma
                                                                     # element in it.

    #Return the neuron cell onject.
    def get_cell(self):
        return self.__cell

    #Return the membrane potential neuron vector as a numpy array.
    def get_potential_as_numpy(self):
        return signal.resample(self.__mem_potential.as_numpy(),32**2)
    
    #Return the time vector as a numpy array.
    def get_time_as_numpy(self):
        return signal.resample(self.__time.as_numpy(),32**2)

    #Graph the membrane vs time graph based on whatever is in membrane voltage 
    #and time vectors.
    def graph_potential(self, save_img_dir = None):
        plt.close()
        plt.figure(figsize = (20,5))
        plt.plot(self.__time, self.__mem_potential)
        plt.xlabel('Time')
        plt.ylabel('Membrane Potential')
        
        if save_img_dir != None:
            plt.savefig(save_img_dir)
        
        plt.show()