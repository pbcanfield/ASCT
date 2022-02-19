from asct.src.Cell import Cell
from activecell import ActiveCell
from neuron import h
import pandas as pd
import pdb



class CellToOptimize(Cell):
    def __init__(self):
        geometry_file = 'geom_standard.csv'
        
        self.cell = ActiveCell(geometry=pd.read_csv(geometry_file))

        super(CellToOptimize, self).__init__()

    #REQUIRED FUNCTION
    def set_parameters(self,parameter_list,parameter_values):
        for index, key in enumerate(parameter_list):
            setattr(self.cell.soma, key, parameter_values[index])

    #REQUIRED FUNCTION
    def get_recording_section(self):
        return self.cell.soma(0.5)     


class ModelCell(Cell):
    def __init__(self):
        geometry_file = 'geom_standard.csv'
        
        self.cell = ActiveCell(geometry=pd.read_csv(geometry_file))

        super(ModelCell, self).__init__()

    #REQUIRED FUNCTION
    def set_parameters(self,parameter_list,parameter_values):
        for index, key in enumerate(parameter_list):
            setattr(self.cell.soma, key, parameter_values[index])

    #REQUIRED FUNCTION
    def get_recording_section(self):
        return self.cell.soma(0.5)  