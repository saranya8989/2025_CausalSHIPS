from copy import deepcopy
# Imports
import pandas as pd
import numpy as np
import xarray as xr
import numpy as np
import matplotlib   
import glob,os
from tqdm.auto import tqdm
import math
from scipy.ndimage import gaussian_filter1d

class proc_data:
    def __init__(self,df=None,seed=42):
    # Initialize with dataframe dictionary and random seed
        self.df = df
        self.seed = seed

    def random_testindex(self,totalexp=None,testexp=None):
    # Randomly pick indices for test storms
        from numpy.random import default_rng
        rng = default_rng(self.seed)
        np.random.seed(self.seed)
        seed = rng.choice(totalexp, testexp, replace=False)
        return seed

    def random_validindex(self,totalexp=None,testindex=None,validexp=None):
    # Randomly pick validation indices, excluding test indices
        from numpy.random import default_rng
        rng = default_rng(self.seed)
        np.random.seed(self.seed)
        listt = [int(obj) for obj in np.linspace(0,totalexp-1,totalexp)]
        seed = np.random.choice([obj for obj in listt if obj not in testindex],validexp,replace=False)
        return seed

    def random_splitdata(self,validindex=None,newtestindex=None):
    # Split dataset into train/valid/test dictionaries based on indices
        datae = self.df.copy()
        traindata = {}
        testdata = {}
        validdata = {}
        stormnames_fulllist = list(datae.keys())
        
        for ind,obj in enumerate(datae.keys()):
            if ind in list(newtestindex):
                testdata[stormnames_fulllist[ind]] = datae[stormnames_fulllist[ind]]
            elif ind in list(validindex):
                validdata[stormnames_fulllist[ind]] = datae[stormnames_fulllist[ind]]
            else:
                traindata[stormnames_fulllist[ind]] = datae[stormnames_fulllist[ind]]
        return traindata,validdata,testdata

    def year_splitdata(self,testyears=None,config=None):
    # Split storms into train/valid/test by year
        datae = self.df.copy()
        traindata = {}
        validdata = {}

        # Separate test data from others
        testdata = {}
        trainvaliddata = {}
        for i in testyears:
            TC_keys = list(datae.keys())
            for obj in TC_keys:
                if obj[:4]==str(i):
                    testdata[obj] = datae[obj]
                else:
                    trainvaliddata[obj] = datae[obj]
                    
        # Randomly select validation storms from remaining pool
        validindex = self.random_testindex(totalexp=len(trainvaliddata.keys()),testexp=int(len(trainvaliddata.keys())*float(config['splitratio'])))
        stormnames_fulllist = list(trainvaliddata.keys())
        
        for ind,obj in enumerate(trainvaliddata.keys()):
            if ind in list(validindex):
                validdata[stormnames_fulllist[ind]] = trainvaliddata[stormnames_fulllist[ind]]
            else:
                traindata[stormnames_fulllist[ind]] = trainvaliddata[stormnames_fulllist[ind]]
        return traindata,validdata,testdata,validindex
        
    def smooth_and_minindices(self,varname='MSLP',sigma=3):
     # Smooth time series using Gaussian filter and find index of minimum
        smoothed_set,pmin = {},{}
        for key in self.df['train'].keys():
            temp = gaussian_filter1d(self.df['train'][key][varname],3)
            smoothed_set[key] = temp
            pmin[key] = temp.argmin()
            
        smoothed_set_valid,pmin_valid = {},{}
        for key in self.df['valid'].keys():
            temp = gaussian_filter1d(self.df['valid'][key][varname],3)
            smoothed_set_valid[key] = temp
            pmin_valid[key] = temp.argmin()
            
        smoothed_set_test,pmin_test = {},{}
        for key in self.df['test'].keys():
            temp = gaussian_filter1d(self.df['test'][key][varname],3)
            smoothed_set_test[key] = temp
            pmin_test[key] = temp.argmin()
        return {'train':smoothed_set,'valid':smoothed_set_valid,'test':smoothed_set_test}, {'train':pmin,'valid':pmin_valid,'test':pmin_test}

    def align_data(self,refpoint=None,individualpoint=None,data=None):
    # Shift storm data so that all align to the same reference index
        newtraincyclone2 = np.zeros((data.shape[0]+(refpoint-individualpoint),data.shape[1]))
        for i in range((data.shape[1])):
        # Pad before data with -999 values
            newtraincyclone2[:,i] = np.concatenate([np.ones((refpoint-individualpoint))*(-999.),data[:,i]])
        return newtraincyclone2
    
    def do_data_align(self,newddwp=None,indices=None,var_names=None):
    # Align each storm and replace -999 placeholders with NaN
        aligned_newddwp = {}
        for intt,obj in enumerate(newddwp.keys()):
            temp = pd.DataFrame(self.align_data(np.asarray(list(indices.values())).max(),indices[obj],np.asarray(newddwp[obj])),columns=var_names)
            aligned_newddwp[obj] = temp.replace(-999.0,np.nan)
        return aligned_newddwp
        
    def combine_for_PC1(self,Xdataset=None,ydataset=None,target=None):
    # Insert target variable back into predictors (for PCA or analysis)
        X_forPC1 = deepcopy(Xdataset)
        X_forPC1['train'].insert(loc=0, column=target, value=ydataset['train'])
        X_forPC1['valid'].insert(loc=0, column=target, value=ydataset['valid'])
        X_forPC1['test'].insert(loc=0, column=target, value=ydataset['test'])
        return X_forPC1
        
# Split data into three subsets
def splitdata_handler(df=None,method='random',seed=None,config=None,testyears=[2020,2021]):
    if method=='random':
        testindex = proc_data(df=df,seed=42).random_testindex(totalexp=len(df.keys()),\
                                                                            testexp=int(len(df.keys())*float(config['splitratio'])))
        validindex = proc_data(df=df,seed=seed).random_validindex(totalexp=len(df.keys()),
                                                                              testindex=testindex,
                                                                              validexp=int(len(df.keys())*float(config['splitratio'])))
        traindata,validdata,testdata = proc_data(df=df,seed=seed).random_splitdata(validindex=validindex,newtestindex=testindex)
        return {'train':traindata,'valid':validdata,'test':testdata,'validindex':validindex,'testindex':testindex}
    elif method=='year':
        traindata,validdata,testdata,validindex = proc_data(df=df,seed=seed).year_splitdata(testyears=testyears,config=config)
        return {'train':traindata,'valid':validdata,'test':testdata,'validindex':validindex}
# Normalize predictors using training mean/std, keep target unchanged
def normalized_TCs_handler(train=None,valid=None,test=None,trainmean=None,trainstd=None,dropcol=['DELV24'],target=None):
    train_norml = {key: normalize_data(train[key].drop(columns=dropcol),trainmean,trainstd) for key in train.keys()}
    valid_norml = {key: normalize_data(valid[key].drop(columns=dropcol),trainmean,trainstd) for key in valid.keys()}
    test_norml = {key: normalize_data(test[key].drop(columns=dropcol),trainmean,trainstd) for key in test.keys()}

    sss_train,sss_valid,sss_test = {},{},{}
    for key in train_norml.keys():
        train_norml[key][target] = train[key][target]
        cols = [train_norml[key].columns[-1]] + train_norml[key].columns[:-1].tolist()  # move last column to front
        sss_train[key] = train_norml[key][cols]

    for key in valid_norml.keys():
        valid_norml[key][target] = valid[key][target]
        cols = [valid_norml[key].columns[-1]] + valid_norml[key].columns[:-1].tolist()  # move last column to front
        sss_valid[key] = valid_norml[key][cols]

    for key in test_norml.keys():
        test_norml[key][target] = test[key][target]
        cols = [test_norml[key].columns[-1]] + test_norml[key].columns[:-1].tolist()  # move last column to front
        sss_test[key] = test_norml[key][cols]
    
    return {'train':sss_train,'valid':sss_valid,'test':sss_test}
# Combine all storms into one predictor and target DataFrame    
def combine_df_storms(datastore=None,targetname=None):
    storepred,storetarget,outsize = [],[],[]
    for stormname in datastore.keys():
        storetarget.append(datastore[stormname][targetname])
        storepred.append(datastore[stormname].drop(columns=[targetname]))
        outsize.append(np.asarray(datastore[stormname][targetname].shape[0]))
    predictors = pd.concat(storepred).reset_index(drop=True)
    targets = pd.concat(storetarget).reset_index(drop=True)
    return targets,predictors,outsize
# Prepare X and y for train/valid/test sets        
def df_proc_separate(trainstore=None,validstore=None,teststore=None,target='DELV24'):
    ytrain,Xtrain,trainsize = combine_df_storms(datastore=trainstore,targetname=target)
    yvalid,Xvalid,validsize = combine_df_storms(datastore=validstore,targetname=target)
    ytest,Xtest,testsize = combine_df_storms(datastore=teststore,targetname=target)
    return {'train':Xtrain,'valid':Xvalid,'test':Xtest},{'train':ytrain,'valid':yvalid,'test':ytest},{'train':trainsize,'valid':validsize,'test':testsize}
# Standardize data using provided mean and std
def normalize_data(X, mean, std):
    return (X-mean)/std



