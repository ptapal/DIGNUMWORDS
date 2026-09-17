"""
PRE-SPECIFIED 13-09-2026, before any result from this file was computed.

QUESTION (reviewer round 2, request 2: the missing cell of the 2x2)
                amplitude (signed RMS)        template (midpoint projection)
fixed ROI       ROI RMS  (block_exact_erp_roi) A1   (tdca_ablation_exact)
learned filter  THIS FILE: learned x RMS       TDCA (tdca_exact)
Does a learned TDCA filter help when the held-out block is scored by signed
amplitude, as the ROI is?

METHOD (tdca.py imported, not modified; GED step via pattern_convergence._tdca_W,
a verbatim copy of tdca.fold_decision's GED lines)
locked TDCA from h5_new/tdca_validated.json: k=15, L=4, shrink .02, 3 comp
per LOO fold and per label assignment: W refitted on the training blocks
held-out block score = RMS over time and the 3 components of
    H @ W,   H = held-out block's embedded PCA mean (tdca.fold_stats 'H')
statistic = mean(score | Parity) - mean(score | Control) over held-out
blocks. Signed (not swap-invariant: W is swap-invariant, the difference
flips). mid-p upper = Parity > Control; two-sided p reported.
arms: disc (diff-ERP sub-averages), ctrl (mean-ERP); D1 primary.

GATE (calibration, as tdca_ablation.py)
20 stratified label shuffles per participant, rng seed 0, sorted order,
disc arm; pooled null mid-p median in [0.40, 0.60], fraction < .05 in
[0.02, 0.10]. Fail -> reported, not run on real labels. Pass writes
h5_new/tdca_rms_validated.json.

CONTRASTS (z scale and rank scale as dual_scale.py: rank = Phi^-1(1 - mid-p);
ROI RMS and learned x RMS are signed, so the one-sided rank is primary and
the two-sided rank |.| is reported)
PRIMARY    D1 disc: learned x RMS - ROI RMS, two-sided one-sample t.
           Stands only if p < .05 on BOTH z and rank scales.
secondary  TDCA - learned x RMS   (scoring effect, learned filter)
           A1 - ROI RMS           (scoring effect, fixed ROI)
           learned x RMS - A1
           interaction (TDCA - A1) - (learned x RMS - ROI RMS)
           all on disc and ctrl, D1 and D2 (D2 flagged NOT INTERPRETED if
           more than half the participants are tied on the rank scale)
READING (reviewer)
learned x RMS ~ ROI RMS   scoring main effect, no learning benefit
learned x RMS > ROI RMS   spatial learning helps amplitude scoring

    python tdca_rms.py
"""
import os
os.chdir(os.path.dirname(os.path.abspath(__file__))) # repo-relative paths; imported modules makedirs at import
import sys
import json
import numpy as np
import pandas as pd
from scipy.stats import norm, ttest_1samp, wilcoxon

from csp2 import load_cache
from tdca import prepare, fold_stats
from block_level_test import enumerate_stratified, EXCLUDE, OUT_DIR
from pattern_convergence import _tdca_W

DS = [('Angelique', 'D1'), ('Talia', 'D2')]
CFG = json.load(open('h5_new/tdca_validated.json'))
N_SHUF = 20
CAL_MEDIAN = (0.40, 0.60)
CAL_TAIL = (0.02, 0.10)
PASSED = 'h5_new/tdca_rms_validated.json'


def make_stat(B, k=CFG['k'], L=CFG['L'], shrink=CFG['shrink']):
    folds = [fold_stats(B, h, k, L) for h in range(len(B))]
    memo = {}

    def stat(m):
        key = m.tobytes()
        if key not in memo:
            s = np.array([np.sqrt(np.mean((f['H'] @ _tdca_W(f, m, shrink)) ** 2)) for f in folds])
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
    if '--summary' in sys.argv: # re-issue tables from tdca_rms_exact.csv, no permutations
        return summarise()
    meta, D, M = load_cache()
    prep = {'disc': prepare(meta, D, set(EXCLUDE)), 'ctrl': prepare(meta, M, set(EXCLUDE))}
    stats = {arm: {key: make_stat(B) for key, (B, _, _) in prep[arm].items()} for arm in prep}

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
    r.to_csv(f'{OUT_DIR}/tdca_rms_exact.csv', index=False)
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
        'ROI_RMS': cell('block_exact_erp_roi.csv', {'disc': lambda d: d}, True),
        'A1':      cell('tdca_ablation_exact.csv',
                        {a: (lambda d, a=a: d[(d.variant == 'A1_roimean') & (d.arm == a)])
                         for a in ('disc', 'ctrl')}, False),
        'TDCA':    cell('tdca_exact.csv', {'disc': lambda d: d[d.measure == 'tdca_diff'],
                                           'ctrl': lambda d: d[d.measure == 'tdca_mean']}, False),
        'L_RMS':   cell('tdca_rms_exact.csv', {a: (lambda d, a=a: d[d.arm == a]) for a in ('disc', 'ctrl')},
                        True),
    }
    C['ROI_RMS']['ctrl'] = cell('block_exact_erpmean_roi.csv', {'ctrl': lambda d: d}, True)['ctrl']
    CONTR = [('L_RMS - ROI_RMS', [('L_RMS', 1), ('ROI_RMS', -1)]),
             ('TDCA - L_RMS', [('TDCA', 1), ('L_RMS', -1)]),
             ('A1 - ROI_RMS', [('A1', 1), ('ROI_RMS', -1)]),
             ('L_RMS - A1', [('L_RMS', 1), ('A1', -1)]),
             ('interaction (TDCA-A1)-(L_RMS-ROI_RMS)',
              [('TDCA', 1), ('A1', -1), ('L_RMS', -1), ('ROI_RMS', 1)])]

    out = []
    print('\n' + '=' * 118)
    print('LEARNED x RMS, and the 2x2 (per-participant z / rank; two-sided t)')
    print('=' * 118)
    for ds, lab in DS:
        for arm in ('disc', 'ctrl'):
            print(f'\n  {lab} {arm}' + ('   PRIMARY: L_RMS - ROI_RMS' if lab == 'D1' and arm == 'disc' else ''))
            print(f'  {"":40}{"scale":7}{"mean":>8}{"t":>8}{"p":>8}{"Wilc p":>8}{"ties":>6}{"pos":>7}')
            for cn in ('L_RMS', 'ROI_RMS', 'A1', 'TDCA'):
                for scale in ('z', 'rank'):
                    s = _desc(C[cn][arm][scale].xs(ds).values)
                    print(f'  {cn + " alone" if scale == "z" else "":40}{scale:7}{s["mean"]:>+8.3f}{s["t"]:>+8.2f}'
                          f'{s["p"]:>8.4f}{s["wilcoxon_p"]:>8.4f}{s["ties"]:>6}{f"{s["pos"]}/{s["n"]}":>7}')
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
                          f'{s["p"]:>8.4f}{s["wilcoxon_p"]:>8.4f}{s["ties"]:>6}{f"{s["pos"]}/{s["n"]}":>7}'
                          f'{"  NOT INTERPRETED (>50% tied)" if flag else ""}')
                    out.append(dict(dataset=lab, arm=arm, contrast=name, scale=scale,
                                    not_interpreted=flag, **s))
                if lab == 'D1' and arm == 'disc' and name == 'L_RMS - ROI_RMS':
                    both = res['z']['p'] < .05 and res['rank']['p'] < .05
                    print(f'  {"":40}-> primary {"HOLDS on both scales" if both else "does not hold on both scales"}')
    pd.DataFrame(out).to_csv(f'{OUT_DIR}/tdca_rms_summary.csv', index=False)
    print(f'\nwrote {OUT_DIR}/tdca_rms_exact.csv\nwrote {OUT_DIR}/tdca_rms_summary.csv')


if __name__ == '__main__':
    main()
