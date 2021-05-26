from os import get_terminal_size
from neuron import h

import matplotlib.pyplot as plt
import numpy as np

class Cell:
    def __init__(self, template_dir, template_name):
        h.load_file(template_dir)
        invoke_cell = getattr(h, template_name)
        self.__cell = invoke_cell()

        #Initialize vectors to track membrane voltage.
        self.__mem_potential = h.Vector()
        self.__time = h.Vector()
        self.__time.record(h._ref_t)
        self.__mem_potential.record(self.__cell.soma[0](0.5)._ref_v) #This line means that all cell templates must
                                                                     #have a soma array with at least one soma
                                                                     # element in it.

    def get_cell(self):
        return self.__cell

    def get_potential_as_numpy(self):
        return self.__mem_potential.as_numpy()
    
    def get_time_as_numpy(self):
        return self.__time.as_numpy()

    def graph_potential(self):
        plt.close()
        plt.figure(figsize = (20,5))
        plt.plot(self.__time, self.__mem_potential)
        plt.xlabel('Time')
        plt.ylabel('Membrane Potential')
        plt.show()


