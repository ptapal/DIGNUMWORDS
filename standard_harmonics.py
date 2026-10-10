"""
PRE-SPECIFIED 27-09-2026, before any result from this file was computed.

QUESTION
harmonic_ablation.py varied bandwidth for the WAVEFORM operators, using the
time-domain twin of the standard read-out (ROI RMS). This file does the same
for the published read-out itself: baseline-corrected amplitude (BCA) on the
sequence spectrum, over the fixed ROI, summed over M asymmetry harmonics.
Does the standard read-out gain anything from harmonics beyond the three it
uses?

CACHE (h5_new/standard_harmonics_cache.npz)
One pass over the raw CSVs, identical to standard_fpvs.py except that the
45-bin amplitude segments are stored PER HARMONIC instead of summed:
  asymmetry 3.75 (2m+1) Hz, m = 0..10  -> 3.75 ... 78.75 Hz
  symmetry  7.5 k Hz,       k = 1..10  -> 7.5 ... 75 Hz
Both stop below the 80 Hz low-pass of the recordings. Scaling 2|X|/N and the
segment width are those of standard_fpvs.py, so M = 3 reproduces it exactly.

SCORE (per block, per arm, per bandwidth)
sum the first M per-harmonic segments, baseline-correct exactly as
standard_fpvs.bca (centre bin minus the mean of 20 neighbours, adjacent bin
excluded, local maximum and minimum of those 20 removed), average the 8 ROI
channels. Per-channel BCA then ROI mean, as published.

GRID
asymmetry (discrimination arm)  M = 3 (published), 6, 11
symmetry  (control arm)         K = 3, 6 (published), 10

TEST
Unchanged: signed Parity - Control per participant, exact stratified
permutation, per-participant z and tie-corrected mid-p. Group: mean z, 95% CI,
one-sample t (two-sided). D1 primary, D2 replication. Bandwidth contrasts are
paired within participant against M = 3.

CHECK
M = 3 (disc) and K = 6 (ctrl) must reproduce standard_fpvs_exact.csv to
numerical tolerance; correlation and maximum difference are printed.

GATE
None: no new method, no tuning, only the number of harmonics summed.

READING
z rises with M              the published read-out loses signal by stopping at
                            three harmonics; report the gain
z flat with M               the three harmonics already carry the amplitude
                            information, and the operator comparisons in
                            fig 9 / fig 9a are not a bandwidth artefact

    python standard_harmonics.py --build    one CSV pass, ~1 h
    python standard_harmonics.py
"""
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import sys
import numpy as np
import pandas as pd
from io import StringIO
from scipy.stats import ttest_1samp, t as tdist

from rebuild_seqlevel import load_sequences, parse_angelique_fname, parse_talia_fname
from epoch_analysis import SR, ANG_CSV_DIR, TAL_CSV_DIR
from ress import _get_talia_files
from block_level_test import enumerate_stratified, signed_stat, EXCLUDE, OUT_DIR
from electrodes import RETTER_ROI_IDX
from standard_fpvs import bca, HALF

CACHE = 'h5_new/standard_harmonics_cache.npz'
N_ELEC = 68
ASYM_HZ = tuple(3.75 * (2 * m + 1) for m in range(11)) # 3.75 ... 78.75
SYM_HZ = tuple(7.5 * (k + 1) for k in range(10)) # 7.5 ... 75
M_DISC = (3, 6, 11)
M_CTRL = (3, 6, 10)
DS = [('Angelique', 'D1'), ('Talia', 'D2')]


def per_harmonic_segments(x, freqs):
    """(N, C) block average -> (n_freq, 45, C) amplitude segments, 2|X|/N."""
    N = x.shape[0]
    amp = 2 * np.abs(np.fft.rfft(x, axis=0)) / N
    return np.stack([amp[int(round(f * N / SR)) - HALF:int(round(f * N / SR)) + HALF + 1]
                     for f in freqs])


def build_cache(path=CACHE):
    rows, sa, ss = [], [], []
    for csv_dir, parse_fn, get_files, ds in [
            (ANG_CSV_DIR, parse_angelique_fname, None, 'Angelique'),
            (TAL_CSV_DIR, parse_talia_fname, _get_talia_files, 'Talia')]:
        files = get_files(csv_dir) if get_files else sorted(
            f for f in os.listdir(csv_dir) if f.endswith('.csv'))
        print(f'[{ds}] {len(files)} files', flush=True)
        for i, fname in enumerate(sorted(files)):
            try:
                subj, mod, ftype, sfont, cond = parse_fn(fname)
                if None in (subj, mod, ftype, cond):
                    continue
                seqs = load_sequences(os.path.join(csv_dir, fname), ds)
            except Exception as e:
                print(f'  SKIP {fname}: {e}')
                continue
            x = np.mean([seqs[s][:, :N_ELEC].astype(np.float64) for s in sorted(seqs)], axis=0)
            rows.append(dict(dataset=ds, subject=f'S{subj}', modality=mod,
                             font=f'{ftype}/{sfont}' if sfont else ftype,
                             condition=cond, block_id=fname, n_samples=x.shape[0]))
            sa.append(per_harmonic_segments(x, ASYM_HZ))
            ss.append(per_harmonic_segments(x, SYM_HZ))
            if (i + 1) % 40 == 0:
                print(f'  [{i+1}/{len(files)}]', flush=True)
    np.savez_compressed(path, seg_asym=np.stack(sa).astype(np.float32),
                        seg_sym=np.stack(ss).astype(np.float32),
                        meta=pd.DataFrame(rows).to_csv(index=False))
    print(f'wrote {path}  ({len(rows)} blocks)')


def describe(v):
    v = np.asarray(v, dtype=float); n = len(v)
    ci = tdist.ppf(.975, n - 1) * v.std(ddof=1) / np.sqrt(n)
    t, p = ttest_1samp(v, 0.0)
    return dict(n=n, mean=v.mean(), lo=v.mean() - ci, hi=v.mean() + ci, t=t, p=p,
                pos=int((v > 0).sum()))


def fmt(label, s):
    return (f'  {label:36}{s["n"]:>4}{s["mean"]:>+8.3f}  [{s["lo"]:+.2f},{s["hi"]:+.2f}]'
            f'{s["t"]:>+7.2f}{s["p"]:>9.4f}{s["pos"]:>4}/{s["n"]}')


def main():
    if '--build' in sys.argv:
        return build_cache()

    z = np.load(CACHE, allow_pickle=True)
    meta = pd.read_csv(StringIO(str(z['meta'])))
    SA, SS = z['seg_asym'].astype(np.float64), z['seg_sym'].astype(np.float64)
    keep = ~meta.subject.isin(EXCLUDE)
    meta, SA, SS = meta[keep].reset_index(drop=True), SA[keep.values], SS[keep.values]

    for arm, seg, grid in (('disc', SA, M_DISC), ('ctrl', SS, M_CTRL)):
        for M in grid:
            meta[f'{arm}_{M}'] = [bca(s[:M].sum(0))[RETTER_ROI_IDX].mean() for s in seg]
    meta.to_csv(f'{OUT_DIR}/standard_harmonics_blocks.csv', index=False)

    rows = []
    for (ds, subj), g in meta.groupby(['dataset', 'subject']):
        y = (g.condition == 'Par').values.astype(int)
        st = (g.modality + '|' + g.font).values
        for arm, grid in (('disc', M_DISC), ('ctrl', M_CTRL)):
            for M in grid:
                v = g[f'{arm}_{M}'].values
                _, p, _, zz, n_all, _ = enumerate_stratified(
                    lambda m, v=v: signed_stat(v, m), y, st)
                rows.append(dict(dataset=ds, subject=subj, arm=arm, harmonics=M,
                                 z=zz, p=p, n_assignments=n_all))
    r = pd.DataFrame(rows)
    r.to_csv(f'{OUT_DIR}/standard_harmonics_exact.csv', index=False)

    print('\nCHECK against standard_fpvs_exact.csv (disc M=3, ctrl K=6)')
    old = pd.read_csv(f'{OUT_DIR}/standard_fpvs_exact.csv')
    for arm, M, oarm in (('disc', 3, 'asym'), ('ctrl', 6, 'sym')):
        a = old[old.arm == oarm].set_index(['dataset', 'subject']).z
        b = r[(r.arm == arm) & (r.harmonics == M)].set_index(['dataset', 'subject']).z
        i = a.index.intersection(b.index)
        print(f'  {arm} M={M}: r = {np.corrcoef(a[i], b[i])[0, 1]:.4f}   '
              f'max |diff| = {np.abs(a[i] - b[i]).max():.4f}   n = {len(i)}')

    head = f'  {"":36}{"n":>4}{"mean z":>8}  {"95% CI":^15}{"t":>7}{"p":>9}{"pos":>7}'
    out = []
    for ds, lab in DS:
        for arm, grid in (('disc', M_DISC), ('ctrl', M_CTRL)):
            print(f'\n{lab} {arm}: standard read-out at each bandwidth\n' + head)
            for M in grid:
                s = describe(r[(r.dataset == ds) & (r.arm == arm)
                               & (r.harmonics == M)].z.values)
                print(fmt(f'M = {M}', s))
                out.append(dict(table='alone', dataset=lab, arm=arm, harmonics=M, **s))
            base = r[(r.dataset == ds) & (r.arm == arm)
                     & (r.harmonics == grid[0])].set_index('subject').z
            for M in grid[1:]:
                a = r[(r.dataset == ds) & (r.arm == arm)
                      & (r.harmonics == M)].set_index('subject').z
                i = a.index.intersection(base.index)
                s = describe((a[i] - base[i]).values)
                print(fmt(f'M = {M} - M = {grid[0]}', s))
                out.append(dict(table='bandwidth', dataset=lab, arm=arm,
                                harmonics=f'{M} - {grid[0]}', **s))
    pd.DataFrame(out).to_csv(f'{OUT_DIR}/standard_harmonics_summary.csv', index=False)
    print(f'\nwrote {OUT_DIR}/standard_harmonics_exact.csv, '
          f'standard_harmonics_summary.csv, standard_harmonics_blocks.csv')


if __name__ == '__main__':
    main()
