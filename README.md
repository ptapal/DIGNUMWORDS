# DIGNUMWORDS

Do spatial filters from SSVEP and BCI research extract more from frequency-tagging (FPVS) EEG
than the fixed occipitotemporal region of interest (ROI) that FPVS studies normally use?

Test case: automatic discrimination of numerical parity, within each participant, in two datasets.

Poster: CuttingGardens 2026, Groningen.
Tapal, Volfart, Ansarinia & Schiltz, *Do SSVEP spatial filters outperform the FPVS ROI?
A within-participant test on numerical parity*.


## Question

FPVS quantifies the discrimination response as amplitude or signal-to-noise ratio over 8 fixed
occipitotemporal channels (PO7, PO9, O1, I1, PO8, PO10, O2, I2; Retter et al., 2024). The ROI is
assumed from prior work, and the response is summarised by its size.

This repository asks two things:

1. **How well.** Does a spatial filter learned from the data separate Parity from Control better
   than that standard read-out?
2. **Where.** Do learned filters place their weights on the ROI without being told where it is?


## Design

Stimuli appear at 7.5 Hz and alternate between two sets of four numbers, so a response that
differs between the sets appears at 3.75 Hz and its harmonics (asymmetry response). A response
common to both sets appears at 7.5 Hz and its harmonics (symmetry response).

- **Parity**: set A = 2 4 6 8, set B = 3 5 7 9
- **Control**: set A = 2 3 6 7, set B = 4 5 8 9 (same eight numerals, different grouping)

| | D1 | D2 |
|---|---|---|
| participants | 30 | 15 |
| formats | Arabic digits and German number words | Arabic digits |
| stimulus sets | 1 font, 20 standard fonts, 20 atypical fonts | 1 font, 10 fonts, 10 mixed, hand-drawn |
| condition files | 12 | 8 |
| sequences per condition | 2 x 60 s | 3 x ~50 s |
| EEG | 68 channels, 512 Hz | 68 channels, 512 Hz |

D2 is the dataset of Retter, Eraßmy & Schiltz (2024). D1 is from a manuscript in preparation
(Volfart, Retter & Schiltz).


## Data

**Raw and preprocessed EEG are not included in this repository.** They belong to the two studies
above and are available from the authors on request. 

Everything the analyses produced is included: per-participant statistics and group summaries in
`results/within_subject/`, and figures in `results/figures/`. Those CSVs can be inspected and
re-analysed without the recordings.


## Analysis in brief

**Two signals per condition file (referred to as a block below).**
- *discrimination*: average of odd epochs minus average of even epochs, where one epoch is one
  7.5 Hz cycle (64 samples after resampling to 480 Hz). Frequency-domain equivalent: the
  asymmetry harmonics 3.75, 11.25, 18.75 Hz.
- *control*: average of all epochs, i.e. the general visual response. Frequency-domain
  equivalent: the symmetry harmonics 7.5, 15, ... Hz.

**Summaries compared (each turns 68 channels into one score per block).**

| operator | labels used | scores | file |
|---|---|---|---|
| standard FPVS read-out over the fixed ROI (baseline-corrected amplitude) | no | amplitude | `standard_fpvs.py` |
| ROI waveform, template score (no spatial learning) | yes | waveform | `tdca_ablation.py` (variant A1) |
| maximum signal-to-noise ratio filter | no | amplitude | `spatial_filter.py` |
| rhythmic entrainment source separation (RESS) | no | amplitude | `ress.py` |
| common spatial patterns (CSP) | yes | power | `csp2.py` |
| task-discriminant component analysis (TDCA) | yes | waveform | `tdca.py` |
| TDCA filter scored by amplitude | yes | amplitude | `tdca_rms.py` |

**Inference (`block_level_test.py`).**
- The unit is the block, not the sequence: sequences inside one file share a label and are not
  independent.
- Parity and Control labels are permuted only within each font and format cell, which is the
  randomisation the experiments performed. All assignments are enumerated: 64 in D1, 16 in D2.
- Each participant gets `z`, the position of the true labelling in their own permutation
  distribution, and a tie-corrected mid-p.
- Filters that use labels are refitted inside every permutation, leaving one block out, so labels
  never leak into the block being scored.
- Because of the design, the smallest attainable p per participant is 0.0078 (D1) or 0.031 (D2)
  for signed statistics, and 0.016 or 0.0625 for statistics that are invariant to swapping every
  label. **No individual-level claims are possible; all results are group level.**
- Group level: one-sample t on per-participant z. D1 is primary, D2 the replication.

**Pre-specification and validation.** Each analysis file starts with a docstring written before
the file was run: question, statistic, and how the outcome is to be read. Methods with tunable
settings were calibrated on shuffled labels and checked against a planted synthetic effect
before any real-label run. Settings were never tuned on real labels. Deviations, including
changes made during validation, are declared in the same docstrings.


## Main results

Per-participant z, discrimination arm, group mean.

- **TDCA gives the strongest parity response in both datasets.** Against the standard FPVS
  read-out: D2 +1.05 (p = .003); D1 +0.38 (p = .053).
- **The other learned filters do not beat the standard read-out.** The maximum signal-to-noise
  ratio filter is worse in D1; RESS and CSP show no difference.
- **Scoring matters more than electrode choice.** The ROI waveform scored with a template, with
  no spatial learning, recovers most of TDCA's advantage. TDCA restricted to the 8 ROI channels
  does as well as TDCA on all 68.
- **Specificity.** In the standard read-out, Parity raises the general visual response almost as
  much as the discrimination response (D1), leaving a parity-specific effect of +0.09 (n.s.).
  TDCA keeps a parity-specific response in both datasets (D1 +0.81, D2 +1.87) and exceeds the
  standard read-out on it (p = .036, p = .046).
- **Where.** The pre-specified test of whether learned patterns concentrate on the ROI failed its
  validation at the observed effect size (planted sources recovered in fewer than half the
  participants). Learned topographies are therefore descriptive only.

Caveat: Parity and Control regroup the same eight numerals, so item-level visual differences are
not excluded. Cross-format transfer (train on digits, test on number words) is the planned test.


## Where to look first

| you want | file |
|---|---|
| the comparison of every operator against the standard read-out | `results/within_subject/standard_fpvs_summary.csv` |
| per-participant z and p for each operator | `*_exact.csv` in `results/within_subject/` |
| how the permutation test works | `block_level_test.py`, `enumerate_stratified` |
| the standard FPVS read-out, reproducing the published analysis | `standard_fpvs.py` |
| what TDCA does here and how it is scored | `tdca.py` |
| what TDCA's advantage comes from | `tdca_ablation.py`, `tdca_rms.py` |
| whether learned filters find the ROI | `pattern_convergence.py`, `power_curve.py` |
| digits versus number words | `modality_split.py` |


## Repository layout

```
block_level_test.py       exact stratified permutation test; fixed-ROI and whole-pattern operators
standard_fpvs.py          standard FPVS read-out (baseline-corrected amplitude over the ROI)
spatial_filter.py         maximum signal-to-noise ratio filter (unsupervised)
ress.py                   rhythmic entrainment source separation (unsupervised, spectral)
csp2.py                   common spatial patterns (supervised)
tdca.py                   task-discriminant component analysis (supervised, spatiotemporal)
tdca_ablation.py          TDCA variants: ROI channels only, no delays, no discriminant, etc.
tdca_rms.py               TDCA filter scored by amplitude instead of waveform template
pattern_convergence.py    do learned patterns concentrate on the ROI (validation failed)
power_curve.py            size of the observed effect on the synthetic plant scale
modality_split.py         digits and number words analysed separately
operator_vs_roi.py        operator versus fixed ROI comparisons on existing CSVs
summarise_results.py      group tables from the per-participant CSVs

electrodes.py             channel order, ROI definition, montage
epoch_analysis.py         epoching and diff-ERP construction
rebuild_seqlevel.py       CSV loaders and per-sequence spectral features

within_subject_analysis.py  earlier exploratory clustering work, superseded
font_effect_analysis.py     earlier exploratory work
harmonic_comparison.py      earlier exploratory work

results/within_subject/   per-participant statistics and group summaries (CSV)
results/figures/          figures
environment.yml           conda environment
```


## Running the code

```bash
conda env create -f environment.yml
conda activate env
```

Every analysis needs the EEG data, which is not in this repository. With the data in place,
caches are built first, then the analyses run from the caches:

```bash
python rebuild_seqlevel.py        # per-sequence spectral features
python csp2.py --build            # epoch sub-average cache, used by CSP and TDCA
python ress.py --build            # cross-spectral cache
python standard_fpvs.py --build   # spectra for the standard read-out
```

Then, for example:

```bash
python block_level_test.py erp    # whole pattern, exact test
python standard_fpvs.py           # standard read-out and the comparisons against it
python tdca.py --validate         # calibration and planted-effect check
python tdca.py --run              # real labels; refuses to run before validation passes
python tdca_ablation.py
python summarise_results.py --csv
```

Without the data, the CSVs in `results/within_subject/` still allow the group-level analyses to
be reproduced: `operator_vs_roi.py` and `summarise_results.py` read only those files.


## Feedback wanted, and co-authorship offered

This work is at the stage where outside eyes are worth more than another analysis by me.
If you work on frequency tagging, spatial filtering, or numerical cognition, I would like your
criticism, especially on:

- the inference: block-level exact permutation, and what the design ceiling does allow;
- whether the standard FPVS read-out is implemented as your lab would implement it;
- the adaptation of TDCA to a two-class contrast at one frequency, and its scoring;
- the failed localisation test: what would make "where the filter looks" answerable at this
  effect size;
- the item-level confound, and whether cross-format transfer is the right test of it;
- anything in the code that is wrong.

**Substantial contributions earn authorship.** If your feedback materially changes the analysis
or the manuscript and you take part in preparing the submission, I will include you as a
co-author, with the agreement of my co-authors. Smaller comments will be acknowledged.

How to reach me:

- open an issue or a pull request in this repository, which keeps the discussion attached to the
  code;
- or email me: **polina.tapal.001@student.uni.lu**.

I present this work at CuttingGardens 2026 in Groningen, where the poster is on display for the
week. Come and find me there, or write at any time.


## References

Cohen, M. X., & Gulbinaite, R. (2017). Rhythmic entrainment source separation. *NeuroImage, 147*, 43–56.

de Cheveigné, A., & Simon, J. Z. (2008). Denoising based on spatial filtering. *Journal of Neuroscience Methods, 171*(2), 331–339.


Liu, B., Chen, X., Shi, N., Wang, Y., Gao, S., & Gao, X. (2021). Improving the performance of individually calibrated SSVEP-BCI by task-discriminant component analysis. *IEEE Transactions on Neural Systems and Rehabilitation Engineering, 29*, 1998–2007.

Ramoser, H., Müller-Gerking, J., & Pfurtscheller, G. (2000). Optimal spatial filtering of single trial EEG during imagined hand movement. *IEEE Transactions on Rehabilitation Engineering, 8*(4), 441–446.

Retter, T. L., Eraßmy, L., & Schiltz, C. (2024). Identifying conceptual neural responses to symbolic numerals. *Proceedings of the Royal Society B, 291*(2025), 20240589.
