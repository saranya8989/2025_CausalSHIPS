import tigramite
from tigramite import data_processing as pp
from tigramite.toymodels import structural_causal_processes as toys
from tigramite.models import Models, Prediction
from tigramite import plotting as tp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr
from copy import deepcopy
import pandas as pd
import numpy as np
import xarray as xr
import numpy as np
import matplotlib   
import glob,os
from tqdm.auto import tqdm
import math

class Pipeline:
    """
    Tigramite and Linear Regression Pipeline
    """
    def __init__(self,data,pc_alpha,pc_type='run_pcstable',tau_min0=None,tau_max0=None,
                 var_name=None,cond_ind_test=ParCorr(),link_assumptions=None):
        self.pc_alpha = pc_alpha
        self.data = data
        self.pc_type = pc_type
        self.tau_min0 = tau_min0
        self.tau_max0 = tau_max0
        self.var_name = var_name
        self.cond_ind_test = cond_ind_test
        self.link_assumptions = link_assumptions
    
    #################################################################################
    # Step 1: Tigramite
    #################################################################################
    def run_tigramite(self,ships_links=None):
        #assert len(self.data)==lengthtrain,"Wrong shape!"
        datae = self.data.copy()
        dataframe =  pp.DataFrame(datae,analysis_mode ='multiple', var_names=self.var_name,missing_flag=-999.)
        #################################################################################
        # Sel links
        #################################################################################
        for member in dataframe.values.keys():
            ships_links = self.link_assumptions
        #################################################################################
        # Run Tigramite
        #################################################################################        
        pcmci = PCMCI(dataframe = dataframe, cond_ind_test = self.cond_ind_test)
        if self.pc_type=='run_pcstable':
            results = pcmci.run_pc_stable(link_assumptions=ships_links,\
                                          tau_min=self.tau_min0, tau_max=self.tau_max0,\
                                          pc_alpha=self.pc_alpha)

        pcmci.verbosity = 2
        return results