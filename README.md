[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17242382.svg)](https://doi.org/10.5281/zenodo.17242382)

Welcome to the Causal SHIPS Project Repository
===============================================

Main Contributors: Dr. Saranya Ganesh S., Frederick Iat Hin Tam

Repository Overview:

Part 1: Tutorial1 -3 ERA5 replication experiments — serves as a tutorial with step-by-step Jupyter notebooks. Users can follow this to learn how the causal experiments are implemented.

Part 2: Tutorial 1- 4 Analysis of SHIPS developmental data (baseline and new predictors) and MLP/SHAP experiments.

Part 3: Case study using MLP and SHAP

The "util" directory contains reusable helper modules used by the main scripts from Part 1 to 3.


Prerequisites
=========================================
Install the Python packages listed in the provided conda environment file (`ships.yml`):

- Core: numpy, pandas, xarray, netCDF4, glob, ast, gc, pickle, copy, os
- Visualization: matplotlib, seaborn
- Utilities: tqdm
- Causal inference: tigramite
- Jupyter: (notebook/lab; `%matplotlib inline` used in notebooks)
- Custom modules: read_config, util.data_process, util.models

The environment YAML (`ships.yml`) contains the package versions used in the script.

Note on CONFIG.py:
- The CONFIG.py script handles creation/preprocessing of the pickle dataframe.  
- Its only role here is to set the time lag threshold. Since the data provided is for 24-hour intensity prediction, the threshold is fixed at 4 and already set.  
- You still need to run it once before starting the notebooks, but no modifications are required.

Instructions to Run the Scripts
=========================================

1. Clone the repository to your local folder or cluster:
   git clone <repo_url>
   cd <repo_name>

2. Create and activate the environment from the provided yml file:
   conda env create -f ships.yml
   pip install -r requirements.txt
   conda activate ships

4. Run the configuration file to set paths and defaults (run once, no edits needed):
   python CONFIG.py

5. Launch Jupyter Lab and run the notebooks step by step:
   jupyter lab

   Begin from Part 1.Tutorial 1 and continue sequentially.

Data Notes
=========================================
- For brevity, Part 1 provides only a sample dataset for 24-hour intensity-change prediction.
- #### For other time lags (48, 72, 96, 120 hrs), all data are in tigramite_ready_dataset_part1.zip and can be accessed at https://doi.org/10.5281/zenodo.17241222
- Part 2 & 3 use the full SHIPS developmental data provided as proc/pickle/dev24.zip.
- In the causal experiments the time-lag threshold is set to **4**.
- Data resolution: **6-hourly**. 
- See the Supplementary Information (SI) for the full list of variables.

------------------------------ 
PART 1 Replication Experiment Dataset (Tutorial)
 ------------------------------ 

Dataset for Part 1 is the input to the causal discovery alogorithm and is provided as multiple zip files in data_for_tigramite/: ]tigramite_ready_dataset_delv24_*.zip 


Unzipping Instructions: 
unzip -j data_for_tigramite/tigramite_ready_dataset_delv24_0.zip -d data_for_tigramite/ 

 Replace 0 with 1..6 or use * 

⚠️ Warning: If unzipping creates subfolders (e.g., data_for_tigramite/tigramite_ready_dataset_delv24_0/), 

move the files up one level: mv data_for_tigramite/tigramite_ready_dataset_delv24_0/* data_for_tigramite/ rmdir data_for_tigramite/tigramite_ready_dataset_delv24_0 

Now you can test run the tutorial notebooks on jupyter lab starting from Part1_Tutorial.1.*. ; 

Part 1.1 lets you create the results from the causal discover experiments; To reproduce the full results, run all 7 folds in the loop given. This may take ~7 minutes per fold depending on your system. 



--------------------------------------------
Tests with SHIPS Developmental Data (Part 2)
---------------------------------------------
Compressed dataset: proc/pickle/dev24.zip

After unzipping, the folder dev24/ will appear containing:

Dataset with SHIPS Baseline predictors: olddict_split0..6.pkl

Dataset with SHIPS+ 6 new predictors: dict_split0..6.pkl

All files are preprocessed and ready to use in the scripts named Part2*. Scripts are similar to Part1, but for SHIPS developmental dataset.

--------------------------------------------
Processed SHAP values and trained MLPs (Part 3)
---------------------------------------------
We calculated the SHAP values for both the 24-hr lead time SHIPS (MLP) and SHIPS+ (MLP, MLR) models for both the case study Larry (2021) and the last 300 test samples.

This data is processed and can be accessed from individual pickle files in ./MLP_data_for_part3/shap_results

SHIPS+ SHAP: plus_shap_values_4_*.pkl
SHIPS SHAP: ships_shap_values_4_*.pkl

The trained MLP models for the SHIPS and SHIPS+ datasets can be downloaded as compressed zip files from a publicly-available Zenodo repository ( https://doi.org/10.5281/zenodo.17241222).

We recommend to unzip these files in the proc/ folder, after which you will see two new folders (mlp_models_causal_oldships_, mlp_models_causal_plus_) and a new file 
(mlp_models_causal_no_fs.pkl).

The MLP files are relatively large and have been shown to cause kernel crashing in older Python environments. To ensure easy reproduction of the main results in Part3, 

we provide processed version of the partial dependence for selected variables (part3_pdp_*.joblib) and MLP test predictions (part3_case_ships_*.joblib)

in the ./MLP_data_for_part3 folder. The scripts to generate these files are attached at the end of the relevant Part 3 notebooks for users who wish to generate this data themselves.



Notes & Contact
=========================================
- Output files (results* `.pkl`, figures) are written to the `results/` and `figures/` subdirectories used in the scripts.
- If you encounter issues with `tqdm` widgets in notebooks, use `from tqdm import tqdm` (text-mode) or install `ipywidgets`.
- For questions or contributions, contact: saranyaganesh.s@gmail.com or iathin.tam@unil.ch
