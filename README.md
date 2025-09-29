
Welcome to the Causal SHIPS Project Repository
===============================================

Main Contributors: Dr. Saranya Ganesh S., Frederick Iat Hin Tam

Repository Overview:

Part 1: ERA5 replication experiments — serves as a tutorial with step-by-step Jupyter notebooks. Users can follow this to learn how the causal experiments are implemented.

Part 2 & 3: Analysis of SHIPS developmental data (baseline and new predictors) and MLP/SHAP experiments — these scripts were used directly for the manuscript and are more “production-style” than the tutorial.

The "util" directory contains reusable helper modules used by the main scripts.

Data Notes
=========================================
- For brevity, Part 1 provides only a sample dataset for 24-hour intensity-change prediction.
- Part 2 & 3 use the full SHIPS developmental data provided as proc/pickle/dev24.zip.
- In the causal experiments the time-lag threshold is set to **4**.
- Data resolution: **6-hourly**. 
- See the Supplementary Information (SI) for the full list of variables.

------------------------------
PART 1 Replication Experiment Dataset (Tutorial)
------------------------------
Sample data for 24-hour intensity-change prediction only.

Time-lag threshold for causal experiments: 4.

Data resolution: 6-hourly.

See the Supplementary Information (SI) for the full variable list.

Dataset is provided as multiple zip files in data_for_tigramite/:
tigramite_ready_dataset_delv24_*.zip

Unzipping Instructions:

  unzip -j data_for_tigramite/tigramite_ready_dataset_delv24_0.zip -d data_for_tigramite/
  # Replace 0 with 1..6 or use * 

⚠️ Warning: If unzipping creates subfolders (e.g., data_for_tigramite/tigramite_ready_dataset_delv24_0/), move the files up one level:

  mv data_for_tigramite/tigramite_ready_dataset_delv24_0/* data_for_tigramite/
  rmdir data_for_tigramite/tigramite_ready_dataset_delv24_0
  # Repeat for other folds

--------------------------------------------
SHIPS Developmental Data (Part 2 & 3)
---------------------------------------------
Compressed dataset: proc/pickle/dev24.zip

After unzipping, the folder dev24/ will appear containing:

Baseline predictors: olddict_split0..6.pkl

New predictors: dict_split0..6.pkl

All files are preprocessed and ready to use in the scripts named Part2*.


Prerequisites
=========================================
Install the Python packages listed in the provided environment file (`ships-2024.yml`):

- Core: numpy, pandas, xarray, netCDF4, glob, ast, gc, pickle, copy, os
- Visualization: matplotlib, seaborn
- Utilities: tqdm
- Causal inference: tigramite
- Jupyter: (notebook/lab; `%matplotlib inline` used in notebooks)
- Custom modules: read_config, util.data_process, util.models

The environment YAML (`ships-2024.yml`) contains exact package versions. Use the Python version specified in that file.

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
   conda env create -f environment.yml
   conda activate ships

3. Run the configuration file to set paths and defaults (run once, no edits needed):
   python CONFIG.py

4. Launch Jupyter Lab and run the notebooks step by step:
   jupyter lab

   Begin from Part 1 and continue sequentially for the analysis of the result*.pkl files.


Notes & Contact
=========================================
- Output files (results* `.pkl`, figures) are written to the `results/` and `figures/` subdirectories used in the scripts.
- If you encounter issues with `tqdm` widgets in notebooks, use `from tqdm import tqdm` (text-mode) or install `ipywidgets`.
- For questions or contributions, contact: saranyaganesh.s@gmail.com or iathin.tam@unil.ch
