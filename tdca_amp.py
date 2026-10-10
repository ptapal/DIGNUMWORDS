"""
PRE-SPECIFIED 23-09-2026, before any result from this file was computed.

QUESTION 
Does a learned TDCA filter help when the held-out block is scored by the
standard FPVS baseline-corrected amplitude, as the ROI is? Together with A1
(fixed ROI, template scoring) this separates the filter from the scoring.
The earlier version of this cell, scored by signed RMS of the filtered epoch,
is tdca_rms.py.

METHOD 
locked TDCA from h5_new/tdca_validated.json: k=15, L=4, shrink .02, 3 comp
per LOO fold and per label assignment: PCA basis P and W refitted on the
training blocks, on the same sub-average epochs as tdca.py
held-out block score = standard_fpvs read-out, computed on the filtered block
1. the held-out block's complex spectrum (2X/N of its sequence-averaged time
   series, standard_fpvs.py cache format, 45-bin segments around each harmonic)
   is passed through the learned filter: P, then the L+1 delay taps at the
   epoch sampling rate (480 Hz), as a frequency response
       Y(f) = sum_d W_d' P' X(f) exp(+i 2 pi f d / 480)
   which is exactly the time-domain filter applied to the block average
2. amplitude |Y|, summed over the harmonics of the arm, per bin
3. BCA as standard_fpvs.bca (centre minus trimmed mean of 20 neighbours)
4. averaged over the 3 components, as the ROI averages its 8 channels
arms: disc = asymmetry harmonics (3.75, 11.25, 18.75 Hz), W from diff-ERP
sub-averages; ctrl = symmetry harmonics (7.5 ... 45 Hz), W from mean-ERP
sub-averages. D1 primary.
statistic = mean(score | Parity) - mean(score | Control) over held-out
blocks. Signed (not swap-invariant: W is swap-invariant, the difference
flips). mid-p upper = Parity > Control; two-sided p reported.

GATE (calibration, as tdca_ablation.py)
20 stratified label shuffles per participant, rng seed 0, sorted order,
disc arm; pooled null mid-p median in [0.40, 0.60], fraction < .05 in
[0.02, 0.10]. Fail -> reported, not run on real labels. Pass writes
h5_new/tdca_amp_validated.json.

CONTRASTS (z scale and rank scale as dual_scale.py: rank = Phi^-1(1 - mid-p);
ROI BCA and learned x BCA are signed, so the one-sided rank is primary and
the two-sided rank |.| is reported)
PRIMARY    D1 disc: learned x BCA - ROI BCA, two-sided one-sample t.
           Stands only if p < .05 on BOTH z and rank scales.
secondary  TDCA - learned x BCA   (scoring effect, learned filter)
           A1 - ROI BCA           (scoring effect, fixed ROI)
           learned x BCA - A1
           interaction (TDCA - A1) - (learned x BCA - ROI BCA)
           all on disc and ctrl, D1 and D2 (D2 flagged NOT INTERPRETED if
           more than half the participants are tied on the rank scale)
READING
learned x BCA ~ ROI BCA   scoring main effect, no learning benefit
learned x BCA > ROI BCA   spatial learning helps amplitude scoring
TDCA - learned x BCA large  the template scoring, not the filter, carries TDCA

    python tdca_amp.py --build     
    python tdca_amp.py
    python tdca_amp.py --summary
"""
import os
os.chdir(os.path.dirname(os.path.abspath(__file__))) # repo-relative paths; imported modules makedirs at import
import sys
import json
import numpy as np
import pandas as pd
from io import StringIO
from scipy.linalg import eigh
from scipy.stats import norm, ttest_1samp, wilcoxon

from csp2 import load_cache
from tdca import prepare, fold_stats
from block_level_test import enumerate_stratified, EXCLUDE, OUT_DIR
from pattern_convergence import _tdca_W
from standard_fpvs import (bca, ASYM_HZ, SYM_HZ, HALF, SR, N_ELEC, ANG_CSV_DIR, TAL_CSV_DIR,
                           parse_angelique_fname, parse_talia_fname, _get_talia_files, load_sequences)
from epoch_analysis import TARGET_SR

DS = [('Angelique', 'D1'), ('Talia', 'D2')]
CFG = json.load(open('h5_new/tdca_validated.json'))
N_SHUF = 20
CAL_MEDIAN = (0.40, 0.60)
CAL_TAIL = (0.02, 0.10)
PASSED = 'h5_new/tdca_amp_validated.json'
CACHE = 'h5_new/tdca_amp_cache.npz'
HARM = {'disc': ASYM_HZ, 'ctrl': SYM_HZ}


def complex_segments(x, freqs):
    """(N, C) time-domain block average -> (len(freqs), 45, C) complex 2X/N around each harmonic."""
    N = x.shape[0]
    X = 2 * np.fft.rfft(x, axis=0) / N
    return np.stack([X[int(round(f * N / SR)) - HALF:int(round(f * N / SR)) + HALF + 1] for f in freqs])


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
            rows.append(dict(dataset=ds, subject=f'S{subj}', block_id=fname, n_samples=x.shape[0]))
            sa.append(complex_segments(x, ASYM_HZ))
            ss.append(complex_segments(x, SYM_HZ))
            if (i + 1) % 40 == 0:
                print(f'  [{i+1}/{len(files)}]', flush=True)
    np.savez_compressed(path, seg_disc=np.stack(sa).astype(np.complex64),
                        seg_ctrl=np.stack(ss).astype(np.complex64), meta=pd.DataFrame(rows).to_csv(index=False))
    print(f'wrote {path}  ({len(rows)} blocks)')


def load_spectra(meta, path=CACHE):
    """{arm: {(ds, subj): (segments (n_blocks, n_harm, 45, C), bin Hz (n_blocks, n_harm, 45))}},
    blocks in sorted block_id order, as tdca.prepare orders them."""
    z = np.load(path, allow_pickle=True)
    sm = pd.read_csv(StringIO(str(z['meta'])))
    out = {'disc': {}, 'ctrl': {}}
    for (ds, subj), g in sm[~sm.subject.isin(EXCLUDE)].groupby(['dataset', 'subject']):
        g = g.sort_values('block_id')
        want = sorted(meta[(meta.dataset == ds) & (meta.subject == subj)].block_id.unique())
        assert list(g.block_id) == want, f'{ds} {subj}: spectra cache and csp2 cache blocks differ'
        for arm in out:
            hz = np.stack([[(np.round(f * N / SR) + np.arange(-HALF, HALF + 1)) * SR / N for f in HARM[arm]]
                           for N in g.n_samples])
            out[arm][(ds, subj)] = (z[f'seg_{arm}'][g.index.values].astype(np.complex128), hz)
    return out


def _pca_basis(B, held, k):
    """P of tdca.fold_stats (its first two lines, verbatim)."""
    tr = [i for i in range(len(B)) if i != held]
    C = sum(np.einsum('itc,itd->cd', B[i], B[i]) for i in tr)
    return eigh(C)[1][:, -k:]


def make_stat(B, spec, k=CFG['k'], L=CFG['L'], shrink=CFG['shrink']):
    """spec = (segments, bin Hz) of this participant's blocks, one arm."""
    seg, hz = spec
    folds, held_Z = [], []
    for h in range(len(B)):
        folds.append(fold_stats(B, h, k, L))
        # held-out block through P and the L+1 delays: (n_harm, 45, (L+1)*k), d-major as tdca._embed
        Zs = seg[h] @ _pca_basis(B, h, k)
        ph = np.exp(2j * np.pi * hz[h][..., None] * np.arange(L + 1) / TARGET_SR)
        held_Z.append((Zs[..., None, :] * ph[..., :, None]).reshape(*Zs.shape[:2], -1))
    memo = {}

    def stat(m):
        key = m.tobytes()
        if key not in memo:
            s = np.array([bca(np.abs(Z @ _tdca_W(f, m, shrink)).sum(0)).mean() for f, Z in zip(folds, held_Z)])
            memo[key] = float(s[m].mean() - s[~m].mean())
        return memo[key]
    return stat


def _shuffle(y, st, rng):
    yy = y.copy()
    for u in np.unique(st):
        i = np.where(st == u)[0]
        yy[i] = rng.permutation(y[i])
    return yy


def _desc(d):
    d = np.asarray(d, dtype=float)
    t, p = ttest_1samp(d, 0.0)
    ties = int((np.abs(d) < 1e-9).sum())
    w = wilcoxon(d).pvalue if ties < len(d) else np.nan
    return dict(n=len(d), mean=d.mean(), t=t, p=p, wilcoxon_p=w, ties=ties, pos=int((d > 1e-9).sum()))


def main():
    if '--build' in sys.argv:
        return build_cache()
    if '--summary' in sys.argv: # re-issue tables from tdca_amp_exact.csv, no permutations
        return summarise()
    meta, D, M = load_cache()
    prep = {'disc': prepare(meta, D, set(EXCLUDE)), 'ctrl': prepare(meta, M, set(EXCLUDE))}
    spec = load_spectra(meta)
    stats = {arm: {key: make_stat(B, spec[arm][key]) for key, (B, _, _) in prep[arm].items()} for arm in prep}

    rng = np.random.default_rng(0)
    cal = []
    for key in sorted(prep['disc']):
        _, y, st = prep['disc'][key]
        for _ in range(N_SHUF):
            cal.append(dict(ds=dict(DS)[key[0]],
                            p=enumerate_stratified(stats['disc'][key], _shuffle(y, st, rng), st)[1]))
    cal = pd.DataFrame(cal)
    med, tail = cal.p.median(), (cal.p < .05).mean()
    ok = CAL_MEDIAN[0] <= med <= CAL_MEDIAN[1] and CAL_TAIL[0] <= tail <= CAL_TAIL[1]
    print(f'CALIBRATION n_null={len(cal)}  median {med:.3f}  frac<.05 {tail:.3f}  {"PASS" if ok else "FAIL"}')
    for d, g in cal.groupby('ds'):
        print(f'  {d}: median {g.p.median():.3f}  frac<.05 {(g.p < .05).mean():.3f}')
    if not ok:
        print('failed calibration: NOT run on real labels')
        return
    json.dump(dict(cfg=CFG, n_null=len(cal), null_median=med, null_tail=tail,
                   shuffles_per_subject=N_SHUF, seed=0, passed=True), open(PASSED, 'w'), indent=1)

    rows = []
    for arm in prep:
        for key, (B, y, st) in sorted(prep[arm].items()):
            obs, p, p2, z, n_all, _ = enumerate_stratified(stats[arm][key], y, st)
            rows.append(dict(dataset=key[0], subject=key[1], arm=arm, stat=obs, z=z, p=p,
                             p_two=p2, n_assignments=n_all))
    r = pd.DataFrame(rows)
    r.to_csv(f'{OUT_DIR}/tdca_amp_exact.csv', index=False)
    summarise()


def summarise():
    # the 2x2 cells, z and ranks
    def cell(fn, filt, signed):
        d = pd.read_csv(f'{OUT_DIR}/{fn}')
        out = {}
        for arm, f in filt.items():
            g = f(d).set_index(['dataset', 'subject'])
            p = g['p'].clip(1e-12, 1 - 1e-12)
            rk = pd.Series(norm.isf(p), index=g.index)
            # |rank| folded the scale (null mean +0.8); Phi^-1(1-p_two) has null mean 0
            rk2 = pd.Series(norm.isf((2 * np.minimum(p, 1 - p)).clip(1e-12, 1 - 1e-12)), index=g.index)
            out[arm] = {'z': g['z'], 'rank': rk, 'rank2': rk2 if signed else rk}
        return out
    C = {
        'ROI_AMP': cell('standard_fpvs_exact.csv', {'disc': lambda d: d[d.arm == 'asym'],
                                                    'ctrl': lambda d: d[d.arm == 'sym']}, True),
        'A1':      cell('tdca_ablation_exact.csv',
                        {a: (lambda d, a=a: d[(d.variant == 'A1_roimean') & (d.arm == a)])
                         for a in ('disc', 'ctrl')}, False),
        'TDCA':    cell('tdca_exact.csv', {'disc': lambda d: d[d.measure == 'tdca_diff'],
                                           'ctrl': lambda d: d[d.measure == 'tdca_mean']}, False),
        'L_AMP':   cell('tdca_amp_exact.csv', {a: (lambda d, a=a: d[d.arm == a]) for a in ('disc', 'ctrl')},
                        True),
    }
    CONTR = [('L_AMP - ROI_AMP', [('L_AMP', 1), ('ROI_AMP', -1)]),
             ('TDCA - L_AMP', [('TDCA', 1), ('L_AMP', -1)]),
             ('A1 - ROI_AMP', [('A1', 1), ('ROI_AMP', -1)]),
             ('L_AMP - A1', [('L_AMP', 1), ('A1', -1)]),
             ('interaction (TDCA-A1)-(L_AMP-ROI_AMP)',
              [('TDCA', 1), ('A1', -1), ('L_AMP', -1), ('ROI_AMP', 1)])]

    out = []
    print('\n' + '=' * 118)
    print('LEARNED x AMP, and the 2x2 (per-participant z / rank; two-sided t)')
    print('=' * 118)
    for ds, lab in DS:
        for arm in ('disc', 'ctrl'):
            print(f'\n  {lab} {arm}' + ('   PRIMARY: L_AMP - ROI_AMP' if lab == 'D1' and arm == 'disc' else ''))
            print(f'  {"":40}{"scale":7}{"mean":>8}{"t":>8}{"p":>8}{"Wilc p":>8}{"ties":>6}{"pos":>7}')
            for cn in ('L_AMP', 'ROI_AMP', 'A1', 'TDCA'):
                for scale in ('z', 'rank'):
                    s = _desc(C[cn][arm][scale].xs(ds).values)
                    print(f'  {cn + " alone" if scale == "z" else "":40}{scale:7}{s["mean"]:>+8.3f}{s["t"]:>+8.2f}'
                          f'{s["p"]:>8.4f}{s["wilcoxon_p"]:>8.4f}{s["ties"]:>6}{str(s["pos"]) + "/" + str(s["n"]):>7}')
                    out.append(dict(dataset=lab, arm=arm, contrast=f'{cn} alone', scale=scale, **s))
            for name, terms in CONTR:
                res = {}
                for scale in ('z', 'rank', 'rank2'):
                    ser = [C[c][arm][scale].xs(ds) * w for c, w in terms]
                    idx = ser[0].index
                    for x in ser[1:]:
                        idx = idx.intersection(x.index)
                    s = _desc(sum(x[idx] for x in ser).values)
                    res[scale] = s
                flag = (lab == 'D2' and res['rank']['ties'] / res['rank']['n'] > 0.5)
                for scale, s in res.items():
                    print(f'  {name if scale == "z" else "":40}{scale:7}{s["mean"]:>+8.3f}{s["t"]:>+8.2f}'
                          f'{s["p"]:>8.4f}{s["wilcoxon_p"]:>8.4f}{s["ties"]:>6}{str(s["pos"]) + "/" + str(s["n"]):>7}'
                          f'{"  NOT INTERPRETED (>50% tied)" if flag else ""}')
                    out.append(dict(dataset=lab, arm=arm, contrast=name, scale=scale,
                                    not_interpreted=flag, **s))
                if lab == 'D1' and arm == 'disc' and name == 'L_AMP - ROI_AMP':
                    both = res['z']['p'] < .05 and res['rank']['p'] < .05
                    print(f'  {"":40}-> primary {"HOLDS on both scales" if both else "does not hold on both scales"}')
    pd.DataFrame(out).to_csv(f'{OUT_DIR}/tdca_amp_summary.csv', index=False)
    print(f'\nwrote {OUT_DIR}/tdca_amp_exact.csv\nwrote {OUT_DIR}/tdca_amp_summary.csv')


if __name__ == '__main__':
    main()
