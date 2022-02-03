from abc import abstractmethod
from neuron import h
import matplotlib.pyplot as plt
from scipy import signal

#This is the base class for all cell wrapper objects reqiured for ASCT to run.
class Cell:

    #Constructs a Cell object.
    #Takes:
    #   1) The HOC template directory.
    #   2) The HOC object name for the given cell.
    #   3) The summary statistics function.
    def __init__(self):
        #Initialize vectors to track membrane voltage.
        self.__mem_potential = h.Vector()
        self.__time = h.Vector()

        #Record time and membrane potential.
        self.__time.record(h._ref_t)
        self.__mem_potential.record(self.get_recording_section()._ref_v)


    #Resamples the membrane voltage and time vectors
    def resample(self):
        return signal.resample(self.__mem_potential.as_numpy(),32**2, t=self.__time.as_numpy())


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

    #Required abstract method which implements how parameters are set
    #in the cell wrapper.
    @abstractmethod
    def set_parameters(self,parameter_list,parameter_values):
        pass

    #Required abstract method which defines what section is being recorded
    #for voltages.
    @abstractmethod
    def get_recording_section(self):
            pass
