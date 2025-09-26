import pandas as pd
from tqdm.auto import tqdm
import numpy as np
import glob
import ast
import pickle
import xarray as xr

def remove_storms(trackpath=None,basinID='NA',yearmin=None,yearmax=None,remove_set=None):
    """
    Hard-code storm removal.
    """
    filt_stormnames = []
    for year in tqdm([int(obj) for obj in np.linspace(yearmin,yearmax,yearmax-yearmin+1)]):
        # Read processed track files
        track = sorted(glob.glob(trackpath+basinID+'_'+str(year)+'.csv'))
        tracksDF = pd.read_csv(track[0])
        # Find unique TCs in the track file
        tracksDF['name'].unique()
        stormnames = list(tracksDF['name'].unique())

        # Use the dictionary "remove_set" to filter TCs
        for obj in remove_set:
            yearFILT,stormFILT=obj[0],obj[1]
            if year==yearFILT:
                stormnames.remove(stormFILT)
        filt_stormnames.append(stormnames)
    return filt_stormnames

        
def read_processed_vars_ships(loc=None,foldernames=['newships_dev_POT'],year=None,stormname=None):
    store = []
    for foldername in foldernames:
        tmp = pd.read_csv(glob.glob(loc+str(foldername)+'/'+\
                                 str(year)+'/'+str(year)+'*shipsdev*'+str(stormname)+'.csv')[0],delimiter=r",").fillna(0)
        store.append(tmp.iloc[:,:])
    return store

################################################################################################################################################################
# SHIPS routine
################################################################################################################################################################
def read_SHIPS_csv(startyear=None,endyear=None,vars_path=None,filted_TCnames=None,suffixlist=None):
    storeyear = []
    for ind,year in tqdm(enumerate([int(obj) for obj in np.linspace(startyear,endyear,int(endyear)-int(startyear)+1)])):
        filt_TCyear = filted_TCnames[ind]
        storestorms = {}
        for i in range(len(filt_TCyear)):
            storestorms[filt_TCyear[i]] = read_processed_vars_ships(vars_path,suffixlist,year,filt_TCyear[i])
        storeyear.append(storestorms)
    return storeyear

def create_SHIPS_df(startyear=None,endyear=None,SHIPSdict=None,wantvarnames=None,targetname=None,filted_TCnames=None,lagnum=None,withshift='Yes' or 'No'):
    store_dfstorms = {}
    for inddd,year in tqdm(enumerate([int(obj) for obj in np.linspace(int(startyear),int(endyear),int(endyear)-int(startyear)+1)])):
        filt_TCyear = filted_TCnames[inddd]
        want_varnames = ast.literal_eval(wantvarnames)
        df_storms = {}
        for stormname in filt_TCyear:
            temp = pd.concat([SHIPSdict[inddd][stormname][i] for i in range(len(SHIPSdict[inddd][stormname]))], axis=1, join='inner')
            if withshift=='Yes':
                tempv = temp[targetname][lagnum:].reset_index(drop=True)
                tempd = temp[want_varnames][:-lagnum].reset_index(drop=True)
            elif withshift=='No':
                tempv = temp[targetname].reset_index(drop=True)
                tempd = temp[want_varnames].reset_index(drop=True)
            df_storms[stormname] = pd.concat([tempv,tempd], axis=1, join='inner')
        store_dfstorms[year]=df_storms
    return store_dfstorms

def add_derive_df(startyear=None,endyear=None,SHIPSdict=None,addfilepath=None,addvarname=None,filted_TCnames=None,lagnum=None,withshift='Yes' or 'No'):
    with open(addfilepath, 'rb') as f:
        ships_df = pickle.load(f)   
    store_dfstorms = {}
    for inddd,year in tqdm(enumerate([int(obj) for obj in np.linspace(int(startyear),int(endyear),int(endyear)-int(startyear)+1)])):
        filt_TCyear = filted_TCnames[inddd]
        df_storms = {}
        for stormname in filt_TCyear:
            temp = ships_df[year][stormname]
            if withshift=='Yes':
                tempd = temp[addvarname][:-lagnum].reset_index(drop=True)
            elif withshift=='No':
                tempd = temp[addvarname].reset_index(drop=True)                
            df_storms[stormname] = pd.concat([SHIPSdict[year][stormname],tempd], axis=1, join='inner')
        store_dfstorms[year]=df_storms
    return store_dfstorms

################################################################################################################################################################
# ERA5 routine
################################################################################################################################################################
def read_processed_vars_era5(loc=None,foldernames=['era5_wmaxdelv','tigramite_6hr'],year=None,stormname=None,era5_dropvar=None):
    """
    era5_dropvar: Variables dropped due to inconsistency with SHIPS 
    """
    def _clean_up(tmp):
        tmp.rename({"Unnamed: 0":"a"}, axis="columns", inplace=True)
        tmp=tmp.drop('a', axis=1)
        #tmp=tmp.drop('index', axis=1)
        if "Unnamed: 0.1" in tmp.keys():
            tmp.rename({"Unnamed: 0.1":"b"}, axis="columns", inplace=True)
            tmp=tmp.drop('b', axis=1)
        if "index" in tmp.keys():
            tmp=tmp.drop('index', axis=1)
        return tmp
        
    store = []
    for foldername in foldernames:
        tmp = pd.read_csv(glob.glob(loc+str(foldername)+'/'+str(year)+'/'+str(year)+'*'+str(stormname)+'.csv')[0],delimiter=r",").fillna(0)
        tmp = _clean_up(tmp)
        
        if foldername=='era5_wmaxdelv':
            tmp = tmp.iloc[:,:]
            tmp = tmp[['delv24']] ##change the target name along with config.py here, 
        if foldername=='tigramite_6hr':
            tmp = tmp.iloc[:,:]
            #here according to CIRA suggestion we only keep the vertical levels present in GFS and also remove the single level variables
            tmp = tmp.drop(era5_dropvar, axis=1)
        store.append(tmp)
    return store

def read_ERA5_csv(startyear=None,endyear=None,vars_path=None,filted_TCnames=None,suffixlist=None,era5_dropvar=None):
    storeyear = []
    for ind,year in tqdm(enumerate([int(obj) for obj in np.linspace(startyear,endyear,int(endyear)-int(startyear)+1)])):
        filt_TCyear = filted_TCnames[ind]
        storestorms = {}
        for i in range(len(filt_TCyear)):
            storestorms[filt_TCyear[i]] = read_processed_vars_era5(vars_path,suffixlist,year,filt_TCyear[i],era5_dropvar)
        storeyear.append(storestorms)
    return storeyear

def read_TCPRIMED_df(startyear=None,endyear=None,ERA5dict=None,filted_TCnames=None,PRIMEDpath=None,PRIMEDlevels=None):
    store_dfstorms = {}
    for inddd,year in tqdm(enumerate([int(obj) for obj in np.linspace(startyear,endyear,int(endyear)-int(startyear)+1)])):
        filt_TCyear = filted_TCnames[inddd]
        df_storms = {}
        for stormname in filt_TCyear:
            product=pd.concat([ERA5dict[inddd][stormname][i] for i in range(len(ERA5dict[inddd][stormname]))], axis=1, join='inner')
            # PRIMED
            tcdat=xr.open_dataset(PRIMEDpath+'/'+str(year)+'_ships_natl_'+str(stormname)+'.nc')
            divstore,vortstore,geopstore,rhstore1,tanostore = {},{},{},{},{}
            for indz,obj in enumerate(PRIMEDlevels):
                tmp = tcdat.divergence[:,indz,0].data
                tmp2 = tcdat.vorticity[:,indz,0].data
                tmp3 = tcdat.geopotential_height[:,indz,0].data
                tmp4a = tcdat.relative_humidity[:,indz,0].data
                tmp5 = tcdat.temperature_anomaly[:,indz,0].data
                divstore['div_0_1000_'+str(obj)] = tmp
                vortstore['vort_0_1000_'+str(obj)] = tmp2
                geopstore['geop_0_1000_'+str(obj)] = tmp3
                rhstore1['rh_0_500_'+str(obj)] = tmp4a
                tanostore['tanom'+str(obj)] = tmp5
                
            tcprimed1 = pd.DataFrame.from_dict({'tgrad_0_500':tcdat.temperature_gradient[:,0].data,'tgrad_200_800':tcdat.temperature_gradient[:,1].data,\
                                                'pwat_0_200':tcdat.precipitable_water[:,0].data,'pwat_200_400':tcdat.precipitable_water[:,1].data,\
                                                'pwat_400_600':tcdat.precipitable_water[:,2].data,'pwat_600_800':tcdat.precipitable_water[:,3].data,\
                                                'pwat_800_1000':tcdat.precipitable_water[:,4].data})
            
            tcprimed_div = pd.DataFrame.from_dict(divstore)
            tcprimed_vort = pd.DataFrame.from_dict(vortstore)
            tcprimed_geop = pd.DataFrame.from_dict(geopstore)
            tcprimed_rh1 = pd.DataFrame.from_dict(rhstore1)
            tcprimed_tano = pd.DataFrame.from_dict(tanostore)
            df_storms[stormname] = pd.concat([product,tcprimed1,tcprimed_div,tcprimed_vort,tcprimed_geop,tcprimed_rh1,tcprimed_tano], axis=1, join='inner')
        store_dfstorms[year] = df_storms
    return store_dfstorms

def create_ERA5_df(startyear=None,endyear=None,ERA5SPS_path=None,ERA5SPS_suffix='all_storms_ships23vars_era5only.pkl',
                   ERA5dict=None,wantvarnames=None,targetname=None,filted_TCnames=None,lagnum=None,withshift='Yes' or 'No'):
    with open(ERA5SPS_path+ERA5SPS_suffix, 'rb') as f:
        ships_df = pickle.load(f)
        store_dfstorms_era5ships = {}
    for inddd,year in tqdm(enumerate([int(obj) for obj in np.linspace(int(startyear),int(endyear),int(endyear)-int(startyear)+1)])):
        stormnames = filted_TCnames[inddd]
        df_storms = {}
        for stormname in stormnames:
            temps = ships_df[year][stormname]
            temp = ERA5dict[year][stormname]
            if withshift=='Yes':
                tempv = temp[targetname][lagnum:].reset_index(drop=True)
                tempq = temp.drop([targetname], axis=1)[:-lagnum].reset_index(drop=True)
                tempd = temps[wantvarnames][:-lagnum].reset_index(drop=True)
            elif withshift=='No':
                tempv = temp[targetname].reset_index(drop=True)
                tempq = temp.drop([targetname], axis=1).reset_index(drop=True)
                tempd = temps[wantvarnames].reset_index(drop=True)
            df_storms[stormname] = pd.concat([tempv,tempd,tempq], axis=1, join='inner')
        store_dfstorms_era5ships[year] = df_storms
    return store_dfstorms_era5ships
