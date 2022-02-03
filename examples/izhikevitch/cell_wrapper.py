from asct.src.Cell import Cell
from neuron import h

#This class takes a NEURON HOC file as an input creates a wrapper
#which can be run by sbi for simulation data.

class CellToOptimize(Cell):
    def __init__(self):
        
        
        #Load in the cell via hoc file.
        template_name = "CA3Cell"
        template_dir = "CA3Cell_Izh/CA3.hoc"
        #Load the template directory.
        h.load_file(template_dir)
        #Get the cell from the h object. 
        invoke_cell = getattr(h, template_name) 
        #Exctract the neuron cell object itself. This also inserts the cell into the neuron simulator.
        self.__cell = invoke_cell()

        #Required_line, this cell wrapper must tell the super class what to record for voltage.
        super(CellToOptimize, self).__init__()

    #REQUIRED FUNCTION
    def set_parameters(self,parameter_list,parameter_values):
        sec = self.__cell.IzhiSoma
        for index, key in enumerate(parameter_list):
            setattr(sec, key, parameter_values[index])

    #REQUIRED FUNCTION
    def get_recording_section(self):
        return self.__cell.soma[0](0.5)

class ModelCell(Cell):
    def __init__(self):
        
        
        #Load in the cell via hoc file.
        template_name = "CA3Cell"
        template_dir = "CA3Cell_Izh/CA3.hoc"
        #Load the template directory.
        h.load_file(template_dir)
        #Get the cell from the h object. 
        invoke_cell = getattr(h, template_name) 
        #Exctract the neuron cell object itself. This also inserts the cell into the neuron simulator.
        self.__cell = invoke_cell()

        #Required_line, this cell wrapper must tell the super class what to record for voltage.
        super(ModelCell, self).__init__()

    #REQUIRED FUNCTION
    def set_parameters(self,parameter_list,parameter_values):
        sec = self.__cell.IzhiSoma
        for index, key in enumerate(parameter_list):
            setattr(sec, key, parameter_values[index])
    
    #REQUIRED FUNCTION
    def get_recording_section(self):
        return self.__cell.soma[0](0.5)