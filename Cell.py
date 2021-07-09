from os import get_terminal_size
from neuron import h

import matplotlib.pyplot as plt
import numpy as np

from scipy.signal import find_peaks
from matplotlib.backends.backend_agg import FigureCanvasAgg

import torch


#This class takes a NEURON HOC file as an input creates a wrapper
#which can be run by sbi for simulation data.
class Cell:

    #Constructs a Cell object.
    #Takes:
    #   1) The HOC template directory.
    #   2) The HOC object name for the given cell.
    #   3) The summary statistics function.
    def __init__(self, template_dir, template_name, summary_stats_funct):
        
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
        self.summary_funct = summary_stats_funct

    #Return the neuron cell onject.
    def get_cell(self):
        return self.__cell

    #Return the membrane potential neuron vector as a numpy array.
    def get_potential_as_numpy(self):
        return np.resize(self.__mem_potential.as_numpy(),96**2)
    
    #Return the time vector as a numpy array.
    def get_time_as_numpy(self):
        return np.resize(self.__time.as_numpy(),96**2)

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

    
    #Generates an image of given dimensionality for the embedded CNN.
    def generate_simulation_image(self, save_img_dir=None, dimensionality=128 , dpi=1):
        trace = self.get_potential_as_numpy()
        time = self.get_time_as_numpy()
        
        plt.style.use('grayscale')
        plt.figure(figsize=(dimensionality/dpi, dimensionality/dpi), dpi=dpi)
        plt.plot(time, trace)
        plt.axis('off')

        canvas = plt.gcf().canvas
        agg = canvas.switch_backends(FigureCanvasAgg)
        agg.draw()
        raw_data = np.mean(np.asarray(agg.buffer_rgba()) / 255, axis=2)

        im = plt.imshow(raw_data)
        im.set_cmap('Greys')
        if save_img_dir != None:
            plt.savefig(save_img_dir, bbox_inches='tight')

        plt.axis('on')
        raw_tensor = torch.tensor(raw_data.flatten(), dtype=torch.get_default_dtype())
        return raw_tensor