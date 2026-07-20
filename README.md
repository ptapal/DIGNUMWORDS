# DIGNUMWORDS

Unsupervised EEG analysis of numerical parity processing using FPVS (Frequency-tagging / Fast Periodic Visual Stimulation).

Two independent datasets are analysed:
- **D1** (n=30): digits and number words in German, 3 font conditions (1F, 20F/A, 20F/S), ~60 s sequences
- **D2** (n=15): digits only, 4 font conditions (1F, 10F/S, 10F/M, 20F/HD), ~50 s sequences

In both datasets, stimuli alternate at F_2 = 3.75 Hz between two groups whose composition defines the condition (Parity: odd vs even; Control: {2,3,6,7} vs {4,5,8,9}). The EEG discrimination response appears at F₂ and its harmonics.

---

## Repository structure

```
data/                         In .gitignore, available upon request
  angelique_v2/               D1 source data — preprocessed .mat files 
  talia/                      D2 source data — preprocessed .mat files
  angelique_v1/               Earlier D1 version, kept for reference

h5_new/                       Intermediate HDF5 cache built by rebuild_seqlevel.py (in .gitignore, available upon request)
  angelique_v2_seqlevel.h5
  talia_seqlevel.h5

results/                      All output figures and CSVs, organised by script
  seqlevel/                   Diagnostic plots from the H5 build step
  within_subject/             Clustering results (histograms, UMAP grids, score CSV)
  font_effects/               Forest plots, delta heatmap
  harmonic_comparison/        Violin/bar comparison across harmonic subsets
  epoch_analysis/             diff-ERP diagnostic plots

electrodes.py                 Channel name list (biosemi_68_order), region map
epoch_analysis.py             diff-ERP feature construction from raw CSV sequences
rebuild_seqlevel.py           SNR feature extraction; builds h5_new/ cache from .mat source files
within_subject_analysis.py    Main within-subject clustering pipeline (PCA -> UMAP -> GMM/KMeans)
font_effect_analysis.py       Per-participant font effect sizes (Cohen's d on SNR delta)
harmonic_comparison.py        Compares discrimination vs stimulation harmonics on all electrodes

environment.yml               Conda environment specification
```

---

## Pipeline overview

```
data/ (.mat files)
       |
       v
rebuild_seqlevel.py -> h5_new/*.h5   (SNR features, 748-D per sequence)
       │
       |-> within_subject_analysis.py -> results/within_subject/
       |-> font_effect_analysis.py    -> results/font_effects/
       |-> harmonic_comparison.py     -> results/harmonic_comparison/

data/ (CSV sequences, via epoch_analysis.py)
       |
       |-> within_subject_analysis.py  (diff-ERP branch)
            font_effect_analysis.py
```

`rebuild_seqlevel.py` must be run first. All other scripts read from `h5_new/` and can be run independently afterwards.

---

## How to run

### 1. Set up the environment

```bash
conda env create -f environment.yml
conda activate myenv
```

### 2. Build the H5 feature cache

This reads `.mat` files from `data/`, extracts per-sequence SNR features (748-D: 68 electrodes × 11 features), and writes `h5_new/angelique_v2_seqlevel.h5` and `h5_new/talia_seqlevel.h5`.

```bash
python rebuild_seqlevel.py
```

Expected output: progress per subject, then diagnostic UMAP plots saved to `results/seqlevel/`.

### 3. Run the analyses

Each script is self-contained and reads from `h5_new/`. Run in any order.

**Within-subject clustering** (main result):
```bash
python within_subject_analysis.py
```
Runs PCA -> UMAP -> GMM/KMeans within each participant. Outputs purity scores, silhouette scores, ARI, UMAP grid plots, and a summary CSV to `results/within_subject/`.

**Font effect analysis**:
```bash
python font_effect_analysis.py
```
Computes per-participant Cohen's d (Par − Ctrl SNR delta) for each font condition. Outputs forest plots and a delta heatmap to `results/font_effects/`.

**Harmonic comparison**:
```bash
python harmonic_comparison.py
```
Compares within-subject clustering purity for discrimination harmonics (3.75 + 11.25 + 18.75 Hz) vs stimulation harmonics (7.5 + 15 Hz), both using all 68 electrodes. Outputs to `results/harmonic_comparison/`.