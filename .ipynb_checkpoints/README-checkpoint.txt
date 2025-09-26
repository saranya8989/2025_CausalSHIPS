===============================================
Welcome to the Causal SHIPS Project Repository
===============================================

Main Contributors: Dr. Saranya Ganesh S., Frederick Iat Hin Tam

Repository overview:
- Part 1: ERA5 replication experiments (with and without causal assumptions)
- Part 2: Analysis of SHIPS developmental data (with and without causal predictors)
- Part 3: Experiments with MLP models, SHAP analysis, and a case study

The "util" directory contains reusable helper modules used by the main scripts.

=========================================
Data Notes
=========================================
- For brevity, we provide sample data only for 24-hour intensity-change prediction.
- In the causal experiments the time-lag threshold is set to **4**.
- Data resolution: **6-hourly**.
- See the Supplementary Information (SI) for the full list of variables.
------------------------------
PART 1
------------------------------
The sample dataset for Part 1 is zipped (~548 MB) and included in data_for_tigramite/.  
After cloning the repository, unzip the files into the same directory using:

   unzip -j data_for_tigramite/tigramite_ready_dataset_delv24.zip -d data_for_tigramite/

⚠️ Note: If unzipping creates a subfolder (e.g., `data_for_tigramite/tigramite_ready_dataset_delv24/`), move the files up one level:

   mv data_for_tigramite/tigramite_ready_dataset_delv24/* data_for_tigramite/  
   rmdir data_for_tigramite/tigramite_ready_dataset_delv24
--------------------------------------------
SHIPS Developmental Data (Part 2 & 3)
---------------------------------------------
Compressed dataset: proc/pickle/dev24.zip

After unzipping, the folder dev24/ will appear containing:

Baseline predictors: olddict_split0..6.pkl

New predictors: dict_split0..6.pkl

All files are preprocessed and ready to use in the scripts named Part2*.

=========================================
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

=========================================
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

   Begin from Part 1 and continue sequentially.

=========================================
Notes & Contact
=========================================
- Output files (results `.pkl`, figures) are written to the `results/` and `figures/` subdirectories used in the scripts.
- If you encounter issues with `tqdm` widgets in notebooks, use `from tqdm import tqdm` (text-mode) or install `ipywidgets`.
- For questions or contributions, contact: saranyaganesh.s@gmail.com or iathin.tam@unil.ch


