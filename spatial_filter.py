"""
PRE-SPECIFIED 06-08-2026, before any result from this file was computed.

QUESTION: Does the field-standard fixed occipitotemporal ROI discard 
discrimination-specific signal that a learned spatial filter recovers?

PRIMARY COMPARISON
Per participant, signal minus control (diff-ERP minus mean-ERP), paired,
computed twice: once with the fixed ROI, once with the learned filter.
The fixed-ROI answer is already known from block_level_test.py, it cannot tell 
the discrimination response apart from the general onset response. The only 
open question is whether the learned filter can. (roi_diff here recomputes 
erp_roi there, so those two should agree; if they do not, there is a bug.)

EXPECTATIONS
The learned filter separates signal from control in D1, and more weakly in D2,
since the whole-pattern multivariate stat already does while the ROI average 
does not. That combination implies the information is spatially present but not 
captured by eight equally weighted channels, which is exactly what a learned
weighting should fix.

RULES OF READING THE OUTCOME
learned yes, ROI no         the ROI summary is the bottleneck. Report as the
                            constructive form of the claim the draft already
                            makes negatively.
both null                   the effect is genuinely distributed and no linear
                            spatial filter concentrates it. Report; strengthens 
                            "pattern, not amplitude".
both significant            the ROI was adequate all along and the filter only
                            adds power. Report plainly.
ROI yes, learned no         treat as a bug, not as a finding.

LOCKED
shrinkage on Sigma_noise   0.05   (pattern verified stable over 0.02-0.4)
components retained        1, the leading one
filter fitting             leave-one-block-out; labels never used
statistic                  signed, mean(Parity) - mean(Control)
null                       stratified exact permutation, font x modality
status                     planned secondary analysis, not the primary"""
import os
import sys
import numpy as np
import pandas as pd
from math import gcd
from scipy.linalg import eigh
from scipy.signal import resample_poly
from scipy.stats import ttest_rel, ttest_1samp, wilcoxon, binomtest, t as tdist

from rebuild_seqlevel import (load_sequences, parse_angelique_fname,
                              parse_talia_fname, _get_talia_files, N_ELEC)
from epoch_analysis import (SR, TARGET_SR, EPOCH_SAMPLES,
                            ANG_CSV_DIR, TAL_CSV_DIR)
from block_level_test import (enumerate_stratified, signed_stat, bh_fdr,
                              EXCLUDE, OUT_DIR)
from electrodes import RETTER_ROI_IDX, biosemi_68_order

CACHE  = 'h5_new/spatial_cache.npz'
SHRINK = 0.05 


# epoching
def _epochs(eeg):
    """(N, 68) at 512 Hz -> (n_epochs, 64, 68) at 480 Hz.
    """
    g = gcd(int(SR), int(TARGET_SR))
    rs = resample_poly(eeg, TARGET_SR // g, SR // g).astype(np.float64)
    n = rs.shape[0] // EPOCH_SAMPLES
    return rs[:n * EPOCH_SAMPLES].reshape(n, EPOCH_SAMPLES, -1)


def _sequence_stats(eeg):
    """Per sequence: the two averaged responses and their noise covariances."""
    ep = _epochs(eeg)[:, :, :N_ELEC]
    T  = ep.shape[1]

    # discrimination contrast: pairs of adjacent epochs, set A minus set B
    n_pair = ep.shape[0] // 2
    d = ep[1:2 * n_pair:2] - ep[0:2 * n_pair:2]           # (n_pair, T, 68)
    D = d.mean(axis=0)                                    # the diff-ERP
    Cn_d = np.einsum('itc,itd->cd', d, d) / (len(d) * T)  # single-pair covariance

    # general onset response: the epochs themselves
    M = ep.mean(axis=0)
    Cn_m = np.einsum('itc,itd->cd', ep, ep) / (len(ep) * T)

    return (D.astype(np.float32), M.astype(np.float32),
            Cn_d.astype(np.float32), Cn_m.astype(np.float32))


# the filter
def max_snr_filter(D_train, Cn_train, shrink=SHRINK):
    """Leading max-SNR spatial filter and its interpretable pattern.
    """
    Cs = D_train.T @ D_train / D_train.shape[0]
    Cn = Cn_train + shrink * np.trace(Cn_train) / N_ELEC * np.eye(N_ELEC)
    vals, vecs = eigh(Cs, Cn) # ascending
    w = vecs[:, -1]
    w = w / (np.linalg.norm(w) + 1e-12)
    pattern = Cn @ w # Haufe et al. 2014
    return w, pattern, float(vals[-1])


def _rms(x):
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)))


# the cache
def build_cache(path=CACHE):
    """One CSV pass. Stores per-sequence responses and covariances."""
    rows, D, M, Cd, Cm = [], [], [], [], []
    for csv_dir, parse_fn, get_files, ds in [
            (ANG_CSV_DIR, parse_angelique_fname, None, 'Angelique'),
            (TAL_CSV_DIR, parse_talia_fname, _get_talia_files, 'Talia')]:
        files = get_files(csv_dir) if get_files else sorted(
            f for f in os.listdir(csv_dir) if f.endswith('.csv'))
        print(f'[{ds}] {len(files)} files')
        for i, fname in enumerate(sorted(files)):
            try:
                subj, mod, ftype, sfont, cond = parse_fn(fname)
            except Exception as e:
                print(f'  SKIP {fname}: {e}');  continue
            if None in (subj, mod, ftype, cond):
                continue
            try:
                seqs = load_sequences(os.path.join(csv_dir, fname), ds)
            except Exception as e:
                print(f'  SKIP {fname}: {e}');  continue

            font = f'{ftype}/{sfont}' if sfont else ftype
            for s in sorted(seqs):
                d, m, cd, cm = _sequence_stats(seqs[s])
                rows.append(dict(dataset=ds, subject=f'S{subj}', modality=mod,
                                 font=font, condition=cond, block_id=fname,
                                 seq=s))
                D.append(d);  M.append(m);  Cd.append(cd);  Cm.append(cm)
            if (i + 1) % 40 == 0:
                print(f'  [{i+1}/{len(files)}] {len(rows)} sequences')

    meta = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, D=np.stack(D), M=np.stack(M),
                        Cd=np.stack(Cd), Cm=np.stack(Cm),
                        meta=meta.to_csv(index=False))
    print(f'wrote {path}  ({len(meta)} sequences)')


def load_cache(path=CACHE):
    z = np.load(path, allow_pickle=True)
    from io import StringIO
    meta = pd.read_csv(StringIO(str(z['meta'])))
    return meta, z['D'], z['M'], z['Cd'], z['Cm']


# analysis
def block_scores(meta, D, M, Cd, Cm, exclude, pooled=False):
    """Reduce every block to four numbers: {learned, ROI} x {diff, mean}."""
    meta = meta[~meta.subject.isin(exclude)]
    out, patterns = [], []

    for (ds, subj), g in meta.groupby(['dataset', 'subject']):
        blocks = list(g.groupby('block_id'))
        # average the sequences inside each block first
        bD = {b: D[gg.index].mean(0) for b, gg in blocks}
        bM = {b: M[gg.index].mean(0) for b, gg in blocks}
        bCd = {b: Cd[gg.index].mean(0) for b, gg in blocks}
        bCm = {b: Cm[gg.index].mean(0) for b, gg in blocks}
        names = [b for b, _ in blocks]

        for b, gg in blocks:
            train = names if pooled else [x for x in names if x != b]
            wd, pat_d, snr_d = max_snr_filter(
                np.mean([bD[t] for t in train], axis=0),
                np.mean([bCd[t] for t in train], axis=0))
            wm, _, _ = max_snr_filter(
                np.mean([bM[t] for t in train], axis=0),
                np.mean([bCm[t] for t in train], axis=0))

            r = gg.iloc[0]
            out.append(dict(
                dataset=ds, subject=subj, block_id=b,
                stratum=f'{r.modality}|{r.font}',
                y=1 if r.condition == 'Par' else 0,
                learned_diff=_rms(bD[b] @ wd),
                learned_mean=_rms(bM[b] @ wm),
                # average the 8 ROI channels into one timecourse
                roi_diff=_rms(bD[b][:, RETTER_ROI_IDX].mean(axis=1)),
                roi_mean=_rms(bM[b][:, RETTER_ROI_IDX].mean(axis=1)),
                eig=snr_d))
            patterns.append(dict(dataset=ds, subject=subj, block_id=b,
                                 **{biosemi_68_order[i]: pat_d[i]
                                    for i in range(N_ELEC)}))
    return pd.DataFrame(out), pd.DataFrame(patterns)


def run_test(scores, column):
    """Signed Parity - Control on one measure, stratified exact permutation."""
    rows = []
    for (ds, subj), g in scores.groupby(['dataset', 'subject']):
        y = g['y'].values
        if y.sum() < 1 or (len(y) - y.sum()) < 1:
            continue
        v = g[column].values
        strata = g['stratum'].values
        _, p, _, z, n_all, exact = enumerate_stratified(
            lambda m, v=v: signed_stat(v, m), y, strata)
        rows.append(dict(dataset=ds, subject=subj, measure=column,
                         z=z, p=p, n_assignments=n_all))
    r = pd.DataFrame(rows)
    for ds, g in r.groupby('dataset'):
        r.loc[g.index, 'q'] = bh_fdr(g['p'].values)
    return r


def _line(label, v):
    v = np.asarray(v, dtype=float)
    n = len(v)
    ci = tdist.ppf(.975, n - 1) * v.std(ddof=1) / np.sqrt(n)
    t, p = ttest_1samp(v, 0.0)
    k = int((v > 0).sum())
    bt = binomtest(k, n, 0.5, alternative='greater').pvalue
    return (f'{label:34}{n:>4}{v.mean():>+9.3f}'
            f'{f"  [{v.mean()-ci:+.2f},{v.mean()+ci:+.2f}]":>18}'
            f'{t:>+8.2f}{p/2:>9.4f}{f"{k}/{n}":>8}{bt:>10.4f}')


def main():
    pooled = '--pooled' in sys.argv
    if not os.path.exists(CACHE):
        print('building cache (one full CSV pass)')
        build_cache()

    meta, D, M, Cd, Cm = load_cache()
    print(f'{len(meta)} sequences cached')
    scores, patterns = block_scores(meta, D, M, Cd, Cm, set(EXCLUDE),
                                    pooled=pooled)
    tag = '_pooled' if pooled else ''
    scores.to_csv(f'{OUT_DIR}/spatial_block_scores{tag}.csv', index=False)
    patterns.to_csv(f'{OUT_DIR}/spatial_patterns{tag}.csv', index=False)

    res = {c: run_test(scores, c) for c in
           ('learned_diff', 'learned_mean', 'roi_diff', 'roi_mean')}
    pd.concat(res.values()).to_csv(f'{OUT_DIR}/spatial_exact{tag}.csv',
                                   index=False)

    print(f'\n{"=" * 96}')
    print(f'LEARNED SPATIAL FILTER vs FIXED ROI  '
          f'({"pooled" if pooled else "leave-one-block-out"} filters)')
    print('=' * 96)
    print(f'{"":34}{"n":>4}{"z mean":>9}{"  95% CI":>18}{"t":>8}'
          f'{"p (1t)":>9}{"pos":>8}{"sign p":>10}')

    for ds, lab in [('Angelique', 'D1'), ('Talia', 'D2')]:
        print(f'\n  {lab}')
        sub = {c: r[r.dataset == ds] for c, r in res.items()}
        if sub['learned_diff'].empty:
            continue
        for c, name in [('learned_diff', 'learned filter, discrimination'),
                        ('learned_mean', 'learned filter, control'),
                        ('roi_diff', 'fixed ROI, discrimination'),
                        ('roi_mean', 'fixed ROI, control')]:
            print('    ' + _line(name, sub[c]['z'].values))

        print()
        for a, b, name in [('learned_diff', 'learned_mean', 'LEARNED  signal - control'),
                           ('roi_diff', 'roi_mean', 'FIXED ROI  signal - control')]:
            x = sub[a].set_index('subject')['z']
            y = sub[b].set_index('subject')['z']
            i = x.index.intersection(y.index)
            d = (x[i] - y[i]).values
            t, p = ttest_rel(x[i], y[i])
            _, pw = wilcoxon(d, alternative='greater')
            print('    ' + _line(name, d) + f'   Wilcoxon p={pw:.4f}')

        # how much discrimination signal the fixed ROI throws away
        s = scores[scores.dataset == ds]
        gain = (s['learned_diff'] / s['roi_diff']).median()
        print(f'\n    median RMS gain, learned filter over fixed ROI: {gain:.2f}x')

    print(f'\nwrote {OUT_DIR}/spatial_exact{tag}.csv, '
          f'spatial_block_scores{tag}.csv, spatial_patterns{tag}.csv')


if __name__ == '__main__':
    main()
