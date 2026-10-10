"""
PRE-SPECIFIED 27-09-2026, before any result from this file was computed.

QUESTION
The standard FPVS read-out sums 3 asymmetry harmonics (3.75, 11.25, 18.75 Hz).
The waveform operators (ROI template, TDCA) use the whole cycle, i.e. every
odd harmonic of 3.75 Hz up to the recording's 80 Hz low-pass. Their advantage
may therefore be bandwidth, not scoring or spatial learning. This file crosses
the three factors:
    frequency content  x  scoring  x  spatial summary
so each can be read on its own.

BAND-LIMITING (exact, no filter design)
Over one 7.5 Hz cycle (64 samples at 480 Hz, window T = 133.3 ms) the
antisymmetric (odd minus even) part of the response is a sum of the odd
harmonics of 3.75 Hz. Those basis functions, exp(i 2 pi (2m+1) 3.75 t), differ
in frequency by multiples of 7.5 Hz = 1/T, so they are ORTHOGONAL over the
window: keeping the first M of them is an exact projection, not a filter.
    d(t)   -> demodulate by exp(-i 2 pi 3.75 t)
           -> DFT over the 64 samples, keep bins 0..M-1
           -> back, take the real part
The symmetric (all-epoch average) part is a sum of harmonics of 7.5 Hz, which
are the DFT bins of the same window; keeping the first K is bins 1..K.
Applied to every sub-average of a block before any operator sees it, so every
operator is compared at matched bandwidth.

GRID
M (discrimination arm) 3 (the standard read-out's harmonics), 6, 11
                       (3.75..78.75 Hz, the last below the 80 Hz low-pass),
                       all (32, the untouched waveform)
K (control arm)        3, 6, 11, all, matched to M by position
operators (locked, nothing tuned)
  roi_amp    RMS over time of the ROI-mean waveform, signed Parity - Control
  roi_tmpl   ROI-mean waveform, template projection (tdca_ablation A1)
  tdca_amp   locked TDCA filter, RMS of the held-out projection (tdca_rms)
  tdca       locked TDCA, midpoint template projection (tdca.py)
TDCA settings from h5_new/tdca_validated.json (k=15, L=4, shrink .02, 3 comp),
leave-one-block-out, everything refitted inside every label assignment.

READING (fixed here)
tdca advantage disappears at M=3      the advantage is bandwidth: the standard
                                      read-out's 3 harmonics are the limit
tdca advantage survives at M=3        the advantage is scoring and/or spatial
                                      learning, not bandwidth
roi_amp rises with M                  the standard read-out loses signal by
                                      stopping at 3 harmonics, independently of
                                      how it is scored
roi_tmpl > roi_amp at every M         scoring effect, bandwidth held constant
tdca > roi_tmpl at every M            spatial/temporal learning effect,
                                      bandwidth held constant

INFERENCE
Unchanged: signed or swap-invariant statistic per operator, exact stratified
permutation over the 64 (D1) / 16 (D2) assignments, per-participant z and
tie-corrected mid-p, group mean z with one-sample t (two-sided). D1 primary,
D2 replication. Contrasts are paired within participant and reported per M.

GATE
None. No new inferential method and no tuning: band-limiting is a projection
applied identically to every operator and arm, and all four operators are
already validated. Sanity check printed instead: at M = all the scores must
reproduce the existing CSVs (tdca_exact, tdca_ablation_exact, tdca_rms_exact,
block_exact_erp_roi) to numerical tolerance.

    python harmonic_ablation.py            run and summarise
    python harmonic_ablation.py --summary  re-issue tables from the CSV
"""
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import sys
import json
import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp, t as tdist

from csp2 import load_cache
from tdca import prepare, fold_stats, fold_decision
from tdca_ablation import make_stat as ablation_stat
from tdca_rms import make_stat as rms_stat
from block_level_test import enumerate_stratified, signed_stat, EXCLUDE, OUT_DIR
from electrodes import RETTER_ROI_IDX
from operator_vs_roi import holm

CFG = json.load(open('h5_new/tdca_validated.json'))
DS = [('Angelique', 'D1'), ('Talia', 'D2')]
SR_EPOCH = 480.0
N_SAMP = 64
M_GRID = (3, 6, 11, None) # None = untouched waveform (all harmonics)
OPS = ('roi_amp', 'roi_tmpl', 'tdca_amp', 'tdca')


# band-limiting

def band_limit(x, M, arm):
    """(..., 64, C) -> same shape, keeping the first M harmonics of the arm.

    disc: odd harmonics of 3.75 Hz, orthogonal over the window after
          demodulation by exp(-i 2 pi 3.75 t).
    ctrl: harmonics of 7.5 Hz, i.e. DFT bins 1..M of the same window.
    """
    if M is None:
        return x
    t = np.arange(N_SAMP) / SR_EPOCH
    if arm == 'disc':
        dem = np.exp(-2j * np.pi * 3.75 * t)[:, None]
        X = np.fft.fft(x * dem, axis=-2)
        # harmonic m sits at bin m after demodulation; its negative-frequency
        # mirror sits at bin N-(m+1). Both are needed to rebuild a real signal.
        keep = np.zeros(N_SAMP, bool); keep[:M] = True; keep[N_SAMP - M:] = True
        X[..., ~keep, :] = 0
        return np.real(np.fft.ifft(X, axis=-2) / dem)
    X = np.fft.fft(x, axis=-2)
    keep = np.zeros(N_SAMP, bool)
    keep[1:M + 1] = True; keep[-M:] = True # harmonics and their mirrors
    X[..., ~keep, :] = 0
    return np.real(np.fft.ifft(X, axis=-2))


# operators

def roi_amp_stat(B):
    v = np.array([np.sqrt(np.mean(b[:, :, RETTER_ROI_IDX].mean(2).mean(0) ** 2)) for b in B])
    return lambda m: signed_stat(v, m)


def tdca_stat(B, k=CFG['k'], L=CFG['L'], shrink=CFG['shrink']):
    folds = [fold_stats(B, h, k, L) for h in range(len(B))]
    memo = {}

    def stat(m):
        key = m.tobytes()
        if key not in memo:
            memo[key] = float(np.mean([fold_decision(f, m, shrink) for f in folds]))
        return memo[key]
    return stat


def make_stat(B, op):
    if op == 'roi_amp':
        return roi_amp_stat(B)
    if op == 'roi_tmpl':
        return ablation_stat(B, 'A1_roimean')
    if op == 'tdca_amp':
        return rms_stat(B)
    return tdca_stat(B)


# run

def describe(v):
    v = np.asarray(v, dtype=float); n = len(v)
    ci = tdist.ppf(.975, n - 1) * v.std(ddof=1) / np.sqrt(n)
    t, p = ttest_1samp(v, 0.0)
    return dict(n=n, mean=v.mean(), lo=v.mean() - ci, hi=v.mean() + ci, t=t, p=p,
                pos=int((v > 0).sum()))


def fmt(label, s, extra=''):
    return (f'  {label:40}{s["n"]:>4}{s["mean"]:>+8.3f}  [{s["lo"]:+.2f},{s["hi"]:+.2f}]'
            f'{s["t"]:>+7.2f}{s["p"]:>9.4f}{s["pos"]:>4}/{s["n"]}{extra}')


def run(prep):
    rows = []
    for arm in prep:
        for (ds, subj), (B, y, st) in sorted(prep[arm].items()):
            for M in M_GRID:
                BM = [band_limit(b, M, arm) for b in B]
                for op in OPS:
                    _, p, _, z, n_all, _ = enumerate_stratified(make_stat(BM, op), y, st)
                    rows.append(dict(dataset=ds, subject=subj, arm=arm,
                                     harmonics=('all' if M is None else M), op=op,
                                     z=z, p=p, n_assignments=n_all))
            print(f'  {arm} {ds} {subj}', flush=True)
    return pd.DataFrame(rows)


def check(r):
    """At M = all, reproduce the existing CSVs."""
    refs = [('tdca', 'disc', 'tdca_exact.csv', dict(measure='tdca_diff')),
            ('roi_tmpl', 'disc', 'tdca_ablation_exact.csv',
             dict(variant='A1_roimean', arm='disc')),
            ('tdca_amp', 'disc', 'tdca_rms_exact.csv', dict(arm='disc')),
            ('roi_amp', 'disc', 'block_exact_erp_roi.csv', {})]
    print('\nSANITY CHECK at M = all (correlation and max difference vs existing CSVs)')
    for op, arm, fn, sel in refs:
        path = f'{OUT_DIR}/{fn}'
        if not os.path.exists(path):
            print(f'  {op:9} {fn}: missing, skipped')
            continue
        d = pd.read_csv(path)
        for c, v in sel.items():
            d = d[d[c] == v]
        a = d.set_index(['dataset', 'subject']).z
        b = r[(r.op == op) & (r.arm == arm) & (r.harmonics == 'all')
              ].set_index(['dataset', 'subject']).z
        i = a.index.intersection(b.index)
        print(f'  {op:9} vs {fn:28} r = {np.corrcoef(a[i], b[i])[0, 1]:.4f}   '
              f'max |diff| = {np.abs(a[i] - b[i]).max():.4f}   n = {len(i)}')


def summarise(r=None):
    if r is None:
        r = pd.read_csv(f'{OUT_DIR}/harmonic_ablation_exact.csv')
    r['harmonics'] = r.harmonics.astype(str)
    order = [str(m) if m is not None else 'all' for m in M_GRID]
    head = f'  {"":40}{"n":>4}{"mean z":>8}  {"95% CI":^15}{"t":>7}{"p":>9}{"pos":>7}'
    out = []

    for ds, lab in DS:
        for arm in ('disc', 'ctrl'):
            print(f'\n{lab} {arm}: each operator at each bandwidth\n' + head)
            for op in OPS:
                for M in order:
                    g = r[(r.dataset == ds) & (r.arm == arm) & (r.op == op)
                          & (r.harmonics == M)]
                    if g.empty:
                        continue
                    s = describe(g.z.values)
                    print(fmt(f'{op:9} M = {M}', s))
                    out.append(dict(table='alone', dataset=lab, arm=arm, op=op,
                                    harmonics=M, **s))

    print('\nCONTRASTS at matched bandwidth (paired per participant, disc arm)\n' + head)
    for ds, lab in DS:
        for M in order:
            g = r[(r.dataset == ds) & (r.arm == 'disc') & (r.harmonics == M)]
            if g.empty:
                continue
            z = {op: g[g.op == op].set_index('subject').z for op in OPS}
            pairs = [('roi_tmpl', 'roi_amp'), # scoring, fixed ROI
                     ('tdca', 'roi_tmpl'), # spatial + delays, template scoring
                     ('tdca_amp', 'roi_amp'), # spatial, amplitude scoring
                     ('tdca', 'tdca_amp')] # scoring, learned filter
            res = []
            for a, b in pairs:
                i = z[a].index.intersection(z[b].index)
                res.append((f'{a} - {b}', describe((z[a][i] - z[b][i]).values)))
            for (name, s), h in zip(res, holm([s['p'] for _, s in res])):
                print(fmt(f'{lab} M = {M:3}  {name}', s, f'   Holm {h:.4f}'))
                out.append(dict(table='contrast', dataset=lab, arm='disc',
                                harmonics=M, op=name, p_holm=h, **s))

    print('\nBANDWIDTH EFFECT within operator (paired, disc arm, vs M = 3)\n' + head)
    for ds, lab in DS:
        for op in OPS:
            base = r[(r.dataset == ds) & (r.arm == 'disc') & (r.op == op)
                     & (r.harmonics == '3')].set_index('subject').z
            for M in order[1:]:
                a = r[(r.dataset == ds) & (r.arm == 'disc') & (r.op == op)
                      & (r.harmonics == M)].set_index('subject').z
                i = a.index.intersection(base.index)
                if len(i) < 3:
                    continue
                s = describe((a[i] - base[i]).values)
                print(fmt(f'{lab} {op:9} M = {M:3} - M = 3', s))
                out.append(dict(table='bandwidth', dataset=lab, arm='disc', op=op,
                                harmonics=f'{M} - 3', **s))

    pd.DataFrame(out).to_csv(f'{OUT_DIR}/harmonic_ablation_summary.csv', index=False)
    print(f'\nwrote {OUT_DIR}/harmonic_ablation_summary.csv')


def main():
    if '--summary' in sys.argv:
        return summarise()
    meta, D, M = load_cache()
    prep = {'disc': prepare(meta, D, set(EXCLUDE)), 'ctrl': prepare(meta, M, set(EXCLUDE))}
    r = run(prep)
    r.to_csv(f'{OUT_DIR}/harmonic_ablation_exact.csv', index=False)
    print(f'wrote {OUT_DIR}/harmonic_ablation_exact.csv')
    check(r)
    summarise(r)


if __name__ == '__main__':
    main()
