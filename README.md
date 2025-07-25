### Decoding filenames

There are 12 different conditions following a Stimulus Modality (**Dig**its or **Num**ber **Wo**rds) x Stimulus Font (1 font, 20 **S**tandard fonts or 20 **A**typical fonts) x Number Set Type (**P**arity or **C**ontrol) structure

The data in the folder is not raw data and it has already been preprocessed up to segmentation of the data by condition. 
Filenames reflect the processing stages that have been applied to each file, from right to left:
- "chanlabels"/"chansel"/"chanlocs" steps are for renaming channels to fit the 68 channels template and for attributing coordinates to channels for plotting response topographies
- "but" step applies a Butterworth band pass filter for 0.1 to 80 Hz (filter order 4) 
- "ep1" step is a first segmentation step where we select only parts of the signal corresponding to sequences of interest, i.e., we keep signal from -1 before the start trigger (tells when the first stimulus is presented and indicates which condition is administered) and for a duration of 65 seconds; at this stage, all sequences are stored into one file (not separated by condition)
- "ica" step means we ran an independent component analysis(ICA) on the data: this is for finding a component reflecting eye blinks and removing it from data if there are too many of them
- "icfilt" step means the dataset needed an ICA and it was applied, i.e., the component reflecting eye blinks was removed from the data (not all participants needed ICA)
- "fix" followed by a channel name means this electrode was interpolated due to noisy signal; interpolation is performed with 3 neighboring channels
- "rr" step means data was re-referenced to the average of all EEG channels
- "epbin" followed by abbreviation of condition and its corresponding trigger number means that data was further segmented to include only the exact number of bins we are interested in. This number of bins is calculated using the frequency of interest (3.75Hz), the sampling rate (512) and the sequence duration at full contrast (60).

### Explanation of the files

- `data/` directory stores all raw samples in csv format: time,channel,condition,value
- `data/mat_files_cleaned.txt` all mat files that were converted to csv
- `pr_fe.py` module with preprocessing and feature extraction
- `environment.yml` lists all dependencies of env
- `h5_load.ipynb` loads data in hdf5 format for hierarchal structure
- `h5_sep/` stores files generated in `h5_load.ipynb`
- `ML.ipynb` includes Machine Learning pipeline for all combinations
- `modality_eval.py`, `diagnostics.py` and `load_data.py` are helper functions for Machine Learning pipeline

### Goal

Train a model on digit sequences and test it on number word sequences (and vice versa) to investigate to what extent digit and word representations are shared.

### Methodology

1. Filter out mat files from the initial dataset and convert them to csv
2. Use pre-processing (i.e. STFT) and feature extraction (i.e. GLCM and frequency bands) from module `pr_fe.py` to convert everything in hierarchal h5 format through `h5_load.ipynb`, separating conditions
3. Do cross-modality, within-modality and mixed-modality predictions through `ML.ipynb`, employing XGBoost classifier. Jupyter notebook allows to interactively change parameters (e.g. parity or control) and helper functions such as `modality_eval.py`, `diagnostics.py` and `load_data.py`

### Instructions

1. Install dependencies from `environment.yml`
```
conda env create -f environment.yml
conda activate env
```
2. Run notebooks