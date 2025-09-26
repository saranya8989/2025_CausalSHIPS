import collections
from copy import deepcopy
import pandas as pd
import numpy as np
import xarray as xr
import numpy as np
import matplotlib   
import glob,os
from tqdm.auto import tqdm

def link_onlyships(numvar=None,lag=None,target_ind=None):
    # Creating the causal arrows towards the target
    ships_index = np.arange(0,numvar)
    super_dict = collections.defaultdict(list)
    for d in [{(obj,-lag):"-?>"} for obj in ships_index]:
        for k, v in d.items():
            super_dict[k].append(str(v))
    tdict = dict(super_dict)
    
    # Assign causal settings to each target
    result = {key:str(value[0]) for key,value in tdict.items()}
    ships_links= {obj:result for obj in target_ind}
    for vartargetindex in target_ind:
        ships_links[vartargetindex].update({(var, -lagz): '-?>' for var in list(np.arange(0,numvar)) for lagz in range(lag,lag+1)})
        
    empty_list = {}
    for vartargetindex in range(target_ind[-1]+1,numvar):
        empty_list[vartargetindex] = {}
    all_links = ships_links.copy()
    all_links.update(empty_list)
    return all_links

def link_shipsera5(numvar=None,lag=None,target_ind=None,ships_ind=23):
    # Creating the causal arrows towards the target
    ships_index = np.arange(0,ships_ind)
    super_dict = collections.defaultdict(list)
    for d in [{(obj,-lag):"-?>"} for obj in ships_index]:
        for k, v in d.items():
            super_dict[k].append(str(v))
    tdict = dict(super_dict)
    
    # Assign causal settings to each target
    result = {key:str(value[0]) for key,value in tdict.items()}
    ships_links= {obj:result for obj in target_ind}
    for vartargetindex in target_ind:
        ships_links[vartargetindex].update({(var, -lagz): '-?>' for var in list(np.arange(0,numvar)) for lagz in range(lag,lag+1)})
        
    empty_list = {}
    for vartargetindex in range(target_ind[-1]+1,numvar):
        empty_list[vartargetindex] = {}
    all_links = ships_links.copy()
    all_links.update(empty_list)
    return all_links


def link_shipsera5_withassum(numvar=None,lag=None,target_ind=None,ships_ind=21):
    # Creating the causal arrows towards the target
    ships_index = np.arange(0,ships_ind)
    super_dict = collections.defaultdict(list)
    for d in [{(obj,-lag):"-->"} for obj in ships_index]:
        for k, v in d.items():
            super_dict[k].append(str(v))
    tdict = dict(super_dict)
    
    # Assign causal settings to each target
    result = {key:str(value[0]) for key,value in tdict.items()}
    ships_links= {obj:result for obj in target_ind}
    for vartargetindex in target_ind:
        ships_links[vartargetindex].update({(var, -lagz): '-?>' for var in list(np.arange(ships_ind,numvar)) for lagz in range(lag,lag+1)})
        
    empty_list = {}
    for vartargetindex in range(target_ind[-1]+1,numvar):
        empty_list[vartargetindex] = {}
    all_links = ships_links.copy()
    all_links.update(empty_list)
    return all_links