import configparser

def read_config():
    config = configparser.ConfigParser()
    
    # Read config file
    config.read('config.ini')
    
    # Access values
    start_year = config.get('General','start_year')
    end_year = config.get('General','end_year')
    target_lag = config.get('General','target_lag')
    basin = config.get('General','basin')
    splitratio = config.get('General','splitratio')
    PRIMED_levels = config.get('General','PRIMED_levels')
    filt_TCs = config.get('Dataset', 'TCfilt')
    SHIPSops_varname = config.get('Dataset','SHIPSop_varname')
    ERA5_dropvarname = config.get('Dataset','ERA5_dropvarname')
    ERA5SPS_varname = config.get('Dataset','ERA5SPS_varname')
    track_path = config.get('paths','tracks_path')
    vars_path = config.get('paths','vars_path')
    PRIMED_path = config.get('paths','PRIMED_path')
    ERA5SPS_path = config.get('paths','ERA5_SHIPSemul_path')
    tau_min = config.get('causal','tau_min')
    tau_max = config.get('causal','tau_max')
    alpha_totest = config.get('causal','alpha_totest')
    
    
    # Return dictionary
    config_values = {
        'TCfilt': filt_TCs,
        'track_path':track_path,
        'vars_path':vars_path,
        'basin':basin,
        'PRIMED_path':PRIMED_path,
        'ERA5SPS_path':ERA5SPS_path,
        'PRIMED_levels':PRIMED_levels,
        'start_year':start_year,
        'end_year':end_year,
        'target_lag':target_lag,
        'splitratio':splitratio,
        'SHIPSops_varname':SHIPSops_varname,
        'ERA5_dropvarname':ERA5_dropvarname,
        'ERA5SPS_varname':ERA5SPS_varname,
        'tau_min':tau_min,
        'tau_max':tau_max,
        'alpha_totest':alpha_totest
    }
    return config_values
    
