# Dependencies
from neuron import h
import pandas as pd
import numpy as np
from typing import Optional, Union
import pdb

# Project Imports
from stylizedcell import StylizedCell

class ActiveCell(StylizedCell):
    """Define single cell model using parent class Stylized_Cell"""

    def __init__(self, geometry: Optional[pd.DataFrame] = None,
                 biophys: Optional[np.ndarray] = None, **kwargs) -> None:
        """
        Initialize cell model
        geometry: pandas dataframe of cell morphology properties
        biophys: vector of biophysical parameters corresponding to "biophys_entries". Use -1 for default value.
        dL: maximum segment length
        vrest: reversal potential for leak channels
        """
        self.grp_ids = []
        self.biophys = biophys
        self.v_rec = None
        self.biophys_entries = [
            (0, 'g_pas'), (1, 'g_pas'), (2, 'g_pas'),  # g_pas of soma, basal, apical
            (0, 'gNaTa_tbar_NaTa_t'), (2, 'gNaTa_tbar_NaTa_t'),  # gNaTa_t of soma, apical
            (0, 'gSKv3_1bar_SKv3_1'), (2, 'gSKv3_1bar_SKv3_1')  # gSKv3_1 of soma, apical
        ]
        
        super(ActiveCell, self).__init__(geometry, **kwargs)
        
#         self.set_channels()

    # PRIVATE METHODS
    def __create_biophys_entries(self, biophys: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Define list of entries of biophysical parameters.
        Each entry is a pair of group id and parameter reference string.
        Define default values and set parameters in "biophys".
        """
        grp_sec_type_ids = [[0], [1, 2], [3, 4]]  # select section id's for each group
        for ids in grp_sec_type_ids:
            secs = []
            for i in ids:
                secs.extend(self.sec_id_lookup[i])
            self.grp_ids.append(secs)
        default_biophys = np.array([3.3e-5, 6.3e-5, 8.8e-5, 2.43, 0.0252, 0.983, 0.0112])
        #default_biophys = np.array([0.0000338, 0.0000467, 0.0000589, 2.04, 0.0213, 0.693, 0.000261])
        if biophys is not None:
            for i in range(len(biophys)):
                if biophys[i] >= 0:
                    default_biophys[i] = biophys[i]
        return default_biophys


    # PUBLIC METHODS
    def set_channels(self) -> None:
        if not self.grp_ids:
            self.biophys = self.__create_biophys_entries(self.biophys)
        # common parameters
        for sec in self.all:
            sec.cm = 2.0
            sec.Ra = 100
            sec.insert('pas')
            sec.e_pas = self._vrest
        # fixed parameters
        soma = self.soma
        soma.cm = 1.0
        soma.insert('NaTa_t')  # Sodium channel
        soma.insert('SKv3_1')  # Potassium channel
        soma.ena = 50
        soma.ek = -85
        for isec in self.grp_ids[2]:
            sec = self.get_sec_by_id(isec)  # apical dendrites
            sec.insert('NaTa_t')
            sec.insert('SKv3_1')
            sec.ena = 50
            sec.ek = -85
        # variable parameters
        for i, entry in enumerate(self.biophys_entries):
            for sec in self.get_sec_by_id(self.grp_ids[entry[0]]):
                setattr(sec, entry[1], self.biophys[i])
        h.v_init = self._vrest