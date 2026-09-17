"""
PRE-SPECIFIED 14-09-2026, before any result from this file was computed.

The standard FPVS read-out, implemented as in Volfart, Retter & Schiltz
(manuscript, section 2.5) and Retter et al. (2024), used as the fixed-ROI
comparator for the learned filters.

PER BLOCK (= one participant x number set x font x format cell, the unit the
manuscript averages over)
1. average the block's sequences in the time domain (preprocessed exports:
   D1 30,720 samples = 60 s, integer number of 3.75 Hz cycles; D2 25,532
   samples = 187.00 cycles)
2. FFT, amplitude spectrum (2|X|/N)
3. asymmetry: 45-bin segments centred on 3.75, 11.25, 18.75 Hz, summed
   symmetry:  45-bin segments centred on 7.5, 15, ..., 45 Hz, summed
4. baseline correction on the summed segment: centre bin minus the mean of
   20 neighbours (10 per side, immediately adjacent bin excluded), after
   removing the local maximum and minimum of those 20
5. average across the 8 ROI channels (PO7, PO9, O1, I1, PO8, PO10, O2, I2)

REPRODUCTION CHECK (D1, must match the manuscript before anything else is read)
participant means over Parity cells and over Control cells, paired t
manuscript: asymmetry Parity 0.174, Control 0.141, F(1,29) = 7.4 (t = 2.72)
            symmetry  Parity 1.66,  Control 1.62,  F(1,29) = 4.6 (t = 2.15)
plus the six planned Parity > Control cell t-tests.

WITHIN-PARTICIPANT TEST (same as every other operator)
signed Parity - Control of the block ROI BCA, exact stratified permutation
arms: asymmetry (discrimination), symmetry (control)
COMPARISONS (discrimination arm, paired z, per dataset)
TDCA - standard ROI BCA; ROI waveform template (A1) - standard ROI BCA
D1 primary, D2 replication.

    python standard_fpvs.py --build     (one pass over the raw CSVs, ~1 h)
    python standard_fpvs.py
"""
import os
os.chdir(os.path.dirname(os.path.abspath(__file__))) # repo-relative paths
import sys
import numpy as np
import pandas as pd
from io import StringIO
from scipy.stats import ttest_rel, ttest_1samp, t as tdist

from rebuild_seqlevel import load_sequences, parse_angelique_fname, parse_talia_fname
from epoch_analysis import SR, ANG_CSV_DIR, TAL_CSV_DIR
from ress import _get_talia_files
from block_level_test import enumerate_stratified, signed_stat, EXCLUDE, OUT_DIR
from electrodes import RETTER_ROI_IDX

CACHE = 'h5_new/standard_fpvs_cache.npz'
N_ELEC = 68
ASYM_HZ = (3.75, 11.25, 18.75)
SYM_HZ = (7.5, 15.0, 22.5, 30.0, 37.5, 45.0)
HALF = 22 # 45-bin segment
DS = [('Angelique', 'D1'), ('Talia', 'D2')]


def summed_segment(x, freqs):
    """(N, C) time-domain block average -> (45, C) amplitude segments summed over freqs."""
    N = x.shape[0]
    amp = 2 * np.abs(np.fft.rfft(x, axis=0)) / N
    seg = np.zeros((2 * HALF + 1, x.shape[1]))
    for f in freqs:
        k = int(round(f * N / SR))
        seg += amp[k - HALF:k + HALF + 1]
    return seg


def bca(seg):
    """(45, C) -> (C,) centre minus mean of 20 neighbours without adjacent bins, max and min."""
    c = HALF
    nb = np.concatenate([seg[c - 11:c - 1], seg[c + 2:c + 12]]) # 20 bins
    nb = np.sort(nb, axis=0)[1:-1] # drop min and max
    return seg[c] - nb.mean(0)


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
                             font=f'{ftype}/{sfont}' if sfont else ftype, condition=cond,
                             block_id=fname, n_seq=len(seqs), n_samples=x.shape[0]))
            sa.append(summed_segment(x, ASYM_HZ))
            ss.append(summed_segment(x, SYM_HZ))
            if (i + 1) % 40 == 0:
                print(f'  [{i+1}/{len(files)}]', flush=True)
    meta = pd.DataFrame(rows)
    np.savez_compressed(path, seg_asym=np.stack(sa).astype(np.float32),
                        seg_sym=np.stack(ss).astype(np.float32), meta=meta.to_csv(index=False))
    print(f'wrote {path}  ({len(meta)} blocks)')


def describe(v):
    v = np.asarray(v, dtype=float); n = len(v)
    ci = tdist.ppf(.975, n - 1) * v.std(ddof=1) / np.sqrt(n)
    t, p = ttest_1samp(v, 0.0)
    return dict(n=n, mean=v.mean(), lo=v.mean() - ci, hi=v.mean() + ci, t=t, p=p, pos=int((v > 0).sum()))


def fmt(label, s):
    return (f'  {label:40}{s["n"]:>4}{s["mean"]:>+8.3f}  [{s["lo"]:+.2f},{s["hi"]:+.2f}]'
            f'{s["t"]:>+7.2f}{s["p"]:>9.4f}{s["pos"]:>4}/{s["n"]}')


def main():
    if '--build' in sys.argv:
        return build_cache()
    z = np.load(CACHE, allow_pickle=True)
    meta = pd.read_csv(StringIO(str(z['meta'])))
    roi = {'asym': np.array([bca(s)[RETTER_ROI_IDX].mean() for s in z['seg_asym'].astype(np.float64)]),
           'sym': np.array([bca(s)[RETTER_ROI_IDX].mean() for s in z['seg_sym'].astype(np.float64)])}
    meta['asym'], meta['sym'] = roi['asym'], roi['sym']
    meta = meta[~meta.subject.isin(EXCLUDE)]
    meta.to_csv(f'{OUT_DIR}/standard_fpvs_blocks.csv', index=False)

    print('REPRODUCTION CHECK, D1 (manuscript: asym 0.174 vs 0.141, t = 2.72; sym 1.66 vs 1.62, t = 2.15)')
    d1 = meta[meta.dataset == 'Angelique']
    print(f'  participants: {d1.subject.nunique()}, sequences per block: {sorted(d1.n_seq.unique())}')
    for arm in ('asym', 'sym'):
        m = d1.groupby(['subject', 'condition'])[arm].mean().unstack()
        t, p = ttest_rel(m['Par'], m['Control'])
        print(f'  {arm:4}  Parity {m["Par"].mean():.3f}  Control {m["Control"].mean():.3f}  t({len(m)-1}) = {t:.2f}  p = {p:.3f}')
    print('  planned cell t-tests, asymmetry, Parity > Control (manuscript: Dig 1F 0.58, 20S 1.60, 20A 2.14;'
          ' Words 1F -0.71, 20S 2.10, 20A 1.29)')
    for (mod, font), g in d1.groupby(['modality', 'font']):
        m = g.pivot_table(index='subject', columns='condition', values='asym').dropna()
        t, _ = ttest_rel(m['Par'], m['Control'])
        print(f'    {mod:8} {font:6} t = {t:+.2f}')

    rows = []
    for (ds, subj), g in meta.groupby(['dataset', 'subject']):
        y = (g.condition == 'Par').values.astype(int)
        st = (g.modality + '|' + g.font).values
        for arm in ('asym', 'sym'):
            v = g[arm].values
            _, p, _, zz, n_all, _ = enumerate_stratified(lambda m, v=v: signed_stat(v, m), y, st)
            rows.append(dict(dataset=ds, subject=subj, arm=arm, z=zz, p=p, n_assignments=n_all))
    r = pd.DataFrame(rows)
    r.to_csv(f'{OUT_DIR}/standard_fpvs_exact.csv', index=False)

    td = pd.read_csv(f'{OUT_DIR}/tdca_exact.csv')
    td = td[td.measure == 'tdca_diff'].set_index(['dataset', 'subject']).z
    ab = pd.read_csv(f'{OUT_DIR}/tdca_ablation_exact.csv')
    a1 = ab[(ab.variant == 'A1_roimean') & (ab.arm == 'disc')].set_index(['dataset', 'subject']).z
    std = r[r.arm == 'asym'].set_index(['dataset', 'subject']).z
    sym = r[r.arm == 'sym'].set_index(['dataset', 'subject']).z

    out = []
    head = f'  {"":40}{"n":>4}{"mean z":>8}  {"95% CI":^15}{"t":>7}{"p":>9}{"pos":>7}'
    print('\nWITHIN-PARTICIPANT EXACT TEST AND COMPARISONS\n' + head)
    for ds, lab in DS:
        i = std.xs(ds).index
        lines = [('standard ROI BCA, asymmetry', std.xs(ds)),
                 ('standard ROI BCA, symmetry', sym.xs(ds)),
                 ('standard, asymmetry - symmetry', std.xs(ds) - sym.xs(ds)[i]),
                 ('TDCA - standard ROI BCA', td.xs(ds)[i] - std.xs(ds)),
                 ('ROI template - standard ROI BCA', a1.xs(ds)[i] - std.xs(ds))]
        for name, v in lines:
            s = describe(v.values)
            print(fmt(f'{lab}  {name}', s))
            out.append(dict(dataset=lab, contrast=name, **s))
        print()
    pd.DataFrame(out).to_csv(f'{OUT_DIR}/standard_fpvs_summary.csv', index=False)
    print(f'wrote {OUT_DIR}/standard_fpvs_blocks.csv, standard_fpvs_exact.csv, standard_fpvs_summary.csv')


if __name__ == '__main__':
    main()
