"""
PRE-SPECIFIED 13-09-2026, before any result from this file was computed.

QUESTION
Where does TDCA's separation of Parity from Control come from: spatial
learning, the delay embedding, the dimension, the supervised discriminant,
or simply the signed template scoring of the full waveform?

VARIANTS (reviewer round 1, R3). tdca.py is imported, never modified. Shared
by all: csp2 cache sub-averages (8 per sequence), leave-one-block-out CV,
midpoint projection scoring from tdca.fold_decision, every step refitted
inside every stratified assignment, same 64 / 16 assignments.
A0  locked TDCA                    68 ch, k=15, L=4, GED 3 comp, shrink .02
                                   (reproduction check against tdca_exact.csv,
                                   not a variant)
A1  ROI-mean channel               1 ch (mean of the 8 Retter channels),
                                   L=0, no GED: template projection only
A2  ROI 8 channels                 k=8, L=4, GED 3 comp, shrink .02
A3  68 channels, matched dim       k=8, L=4, GED 3 comp, shrink .02
A4  no delays                      68 ch, k=15, L=0, GED 3 comp, shrink .02
A5  no discriminant                68 ch, k=15, L=0, W = I (the 15 PCs)
"No GED" = the same midpoint projection with W = identity in the PC space:
the held-out block's (PCA, embedded) mean minus the class-template midpoint,
projected on the template difference, sign-flipped for Control blocks.
For A1 the 1-channel PCA is +-1, so it is the ROI-mean waveform itself.

GATE (calibration half only; the variants are fixed, nothing is tuned)
shuffled labels  20 stratified shuffles per participant (within-stratum
                 permutation of the real design labels), rng seed 0,
                 participants in sorted order, discrimination sub-averages
pass             pooled null mid-p median in [0.40, 0.60] and
                 fraction < .05 in [0.02, 0.10]  (tdca.py thresholds)
Deviation from tdca.py --validate: 20 shuffles per participant instead of
one, because with the tie-corrected mid-p a swap-invariant statistic has
null p in {1,3,...}/64 (D1) and {1,3,...,15}/16 (D2): one draw per
participant leaves the tail fraction at 0-4 of 45, failing ~15% of the
time by chance alone. Exact enumeration makes within-stratum shuffles
uniform by construction, so this gate checks the implementation (ties,
incomplete strata), not power.
A variant that fails is reported as failed and not run on real labels.
Pass writes h5_new/tdca_ablation_<variant>_validated.json.

REAL LABELS
arms       discrimination (diff-ERP sub-averages), control (mean-ERP)
per subj   z, mid-p (tie-corrected enumerate_stratified)
statistic  d = z_TDCA - z_Ax per participant, discrimination arm; one-sample
           t, two-sided; D1 primary, D2 calibration. z_TDCA from
           tdca_exact.csv. Control arm and each variant's own arms
           (disc, ctrl, disc - ctrl) reported. Holm over A1-A5 in D1 disc
           reported alongside uncorrected p (the reviewer asked for none;
           added, declared).

READING (reviewer's, unchanged)
A1 ~ TDCA         the gain is template scoring, not spatial learning
A2 ~ A3 ~ TDCA    the ROI holds the information
A3 > A2           information outside the ROI
A4 ~ TDCA         delays irrelevant
A5 ~ TDCA         supervision irrelevant
"~" means TDCA - Ax not significant; absence of a difference is not
equivalence, so these are read as "no evidence of a loss".

    python tdca_ablation.py
"""
import os
os.chdir(os.path.dirname(os.path.abspath(__file__))) # repo-relative paths; imported modules makedirs at import
import json
import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp, t as tdist

from csp2 import load_cache
from tdca import prepare, fold_stats, fold_decision
from block_level_test import enumerate_stratified, EXCLUDE, OUT_DIR
from electrodes import RETTER_ROI_IDX
from operator_vs_roi import holm

DS = [('Angelique', 'D1'), ('Talia', 'D2')]
SHRINK = 0.02
N_SHUF = 20
CAL_MEDIAN = (0.40, 0.60)
CAL_TAIL = (0.02, 0.10)
ALL = slice(None)

# name: (channels, k, L, ged)
VARIANTS = {
    'A0_tdca':     (ALL, 15, 4, True),
    'A1_roimean':  ('roimean', 1, 0, False),
    'A2_roi8':     (RETTER_ROI_IDX, 8, 4, True),
    'A3_all68_k8': (ALL, 8, 4, True),
    'A4_L0':       (ALL, 15, 0, True),
    'A5_noged':    (ALL, 15, 0, False),
}
PASSED = 'h5_new/tdca_ablation_{}_validated.json'


def fold_template(fs, mask):
    """tdca.fold_decision with W = I: no discriminant step."""
    tr = fs['tr']
    m1, m0 = tr & mask, tr & ~mask
    M1 = fs['sumZ'][m1].sum(0) / fs['n'][m1].sum()
    M0 = fs['sumZ'][m0].sum(0) / fs['n'][m0].sum()
    diff = (M1 - M0).ravel()
    dec = float((fs['H'] - 0.5 * (M1 + M0)).ravel() @ diff / (np.linalg.norm(diff) + 1e-300))
    return dec if mask[fs['held']] else -dec


def _channels(B, ch):
    if isinstance(ch, str):
        return [b[:, :, RETTER_ROI_IDX].mean(2, keepdims=True) for b in B]
    return [b[:, :, ch] for b in B]


def make_stat(B, variant):
    """Memoised cross-validated statistic over label assignments."""
    ch, k, L, ged = VARIANTS[variant]
    X = _channels(B, ch)
    folds = [fold_stats(X, h, k, L) for h in range(len(X))]
    memo = {}

    def stat(m):
        key = m.tobytes()
        if key not in memo:
            memo[key] = float(np.mean([fold_decision(f, m, SHRINK) if ged
                                       else fold_template(f, m) for f in folds]))
        return memo[key]
    return stat


def _shuffle(y, st, rng):
    yy = y.copy()
    for u in np.unique(st):
        i = np.where(st == u)[0]
        yy[i] = rng.permutation(y[i])
    return yy


def calibrate(variant, prep, stats):
    rng = np.random.default_rng(0)
    rows = []
    for key in sorted(prep):
        _, y, st = prep[key]
        for s in range(N_SHUF):
            p = enumerate_stratified(stats[key], _shuffle(y, st, rng), st)[1]
            rows.append(dict(dataset=key[0], p=p))
    r = pd.DataFrame(rows)
    med, tail = r.p.median(), (r.p < .05).mean()
    ok = CAL_MEDIAN[0] <= med <= CAL_MEDIAN[1] and CAL_TAIL[0] <= tail <= CAL_TAIL[1]
    per = '  '.join(f'{lab}: med {g.p.median():.3f} tail {(g.p < .05).mean():.3f}'
                    for ds, lab in DS for _, g in [(ds, r[r.dataset == ds])])
    print(f'  {variant:13} n_null={len(r)}  median {med:.3f}  frac<.05 {tail:.3f}  '
          f'{"PASS" if ok else "FAIL"}   ({per})', flush=True)
    return ok, dict(variant=variant, spec=str(VARIANTS[variant]), n_null=len(r),
                    null_median=med, null_tail=tail, shuffles_per_subject=N_SHUF,
                    seed=0, passed=bool(ok))


def describe(v):
    v = np.asarray(v, dtype=float); n = len(v)
    ci = tdist.ppf(.975, n - 1) * v.std(ddof=1) / np.sqrt(n)
    t, p = ttest_1samp(v, 0.0)
    return dict(n=n, mean=v.mean(), lo=v.mean() - ci, hi=v.mean() + ci, t=t,
                p_two=p, k_pos=int((v > 0).sum()))


def _fmt(label, s, extra=''):
    return (f'{label:32}{s["n"]:>4}{s["mean"]:>+9.3f}'
            f'{f"  [{s["lo"]:+.2f},{s["hi"]:+.2f}]":>18}{s["t"]:>+8.2f}'
            f'{s["p_two"]:>9.4f}{f"{s["k_pos"]}/{s["n"]}":>8}{extra}')


def main():
    meta, D, M = load_cache()
    prep = {'disc': prepare(meta, D, set(EXCLUDE)), 'ctrl': prepare(meta, M, set(EXCLUDE))}

    print('CALIBRATION GATE (shuffled labels, discrimination sub-averages)')
    stats, passed = {}, {}
    for v in VARIANTS:
        stats[v] = {arm: {key: make_stat(B, v) for key, (B, _, _) in prep[arm].items()}
                    for arm in prep}
        ok, info = calibrate(v, prep['disc'], stats[v]['disc'])
        passed[v] = ok
        if ok:
            json.dump(info, open(PASSED.format(v), 'w'), indent=1)

    rows = []
    for v in VARIANTS:
        if not passed[v]:
            print(f'  {v}: failed calibration, NOT run on real labels')
            continue
        for arm in prep:
            for key, (B, y, st) in sorted(prep[arm].items()):
                obs, p, _, z, n_all, _ = enumerate_stratified(stats[v][arm][key], y, st)
                rows.append(dict(dataset=key[0], subject=key[1], variant=v, arm=arm,
                                 stat=obs, z=z, p=p, n_assignments=n_all))
    r = pd.DataFrame(rows)
    out = f'{OUT_DIR}/tdca_ablation_exact.csv'
    r.to_csv(out, index=False)

    ref = pd.read_csv(f'{OUT_DIR}/tdca_exact.csv')
    ref['arm'] = ref.measure.map({'tdca_diff': 'disc', 'tdca_mean': 'ctrl'})
    head = f'{"":32}{"n":>4}{"mean":>9}{"  95% CI":>18}{"t":>8}{"p (2t)":>9}{"pos":>8}'
    summ = []
    if 'A0_tdca' in set(r.variant):
        a0 = r[r.variant == 'A0_tdca'].merge(ref, on=['dataset', 'subject', 'arm'])
        print(f'\nREPRODUCTION A0 vs tdca_exact.csv: n={len(a0)}  '
              f'max|dz|={np.abs(a0.z_x - a0.z_y).max():.2e}  max|dp|={np.abs(a0.p_x - a0.p_y).max():.2e}')

    for ds, lab in DS:
        g = r[r.dataset == ds]
        zt = {arm: ref[(ref.dataset == ds) & (ref.arm == arm)].set_index('subject')['z']
              for arm in ('disc', 'ctrl')}
        print(f'\n{"=" * 92}\n  {lab}: each variant alone\n  {head}')
        for v in g.variant.unique():
            za = {arm: g[(g.variant == v) & (g.arm == arm)].set_index('subject')['z']
                  for arm in ('disc', 'ctrl')}
            i = za['disc'].index.intersection(za['ctrl'].index)
            for arm, x in [('disc', za['disc']), ('ctrl', za['ctrl']),
                           ('disc-ctrl', za['disc'][i] - za['ctrl'][i])]:
                s = describe(x.values)
                print('    ' + _fmt(f'{v}, {arm}', s))
                summ.append(dict(table='variant', dataset=lab, comparison=v, arm=arm, **s))
        for arm in ('disc', 'ctrl'):
            print(f'\n  {lab}: TDCA - variant, {arm} arm'
                  f'{"   (PRIMARY: D1 disc)" if lab == "D1" and arm == "disc" else ""}\n  {head}{"Holm p":>9}')
            res = []
            for v in [x for x in g.variant.unique() if x != 'A0_tdca']:
                za = g[(g.variant == v) & (g.arm == arm)].set_index('subject')['z']
                i = zt[arm].index.intersection(za.index)
                res.append((v, describe((zt[arm][i] - za[i]).values)))
            ph = holm([s['p_two'] for _, s in res]) if res else []
            for (v, s), hp in zip(res, ph):
                print('    ' + _fmt(f'TDCA - {v}', s, f'{hp:>9.4f}'))
                summ.append(dict(table='tdca_minus_variant', dataset=lab, comparison=f'TDCA - {v}',
                                 arm=arm, p_holm=hp, **s))
            if arm == 'disc':
                za2 = {v: g[(g.variant == v) & (g.arm == arm)].set_index('subject')['z']
                       for v in ('A2_roi8', 'A3_all68_k8')}
                if all(len(x) for x in za2.values()):
                    i = za2['A3_all68_k8'].index.intersection(za2['A2_roi8'].index)
                    s = describe((za2['A3_all68_k8'][i] - za2['A2_roi8'][i]).values)
                    print('    ' + _fmt('A3_all68_k8 - A2_roi8', s, '  (not in Holm)'))
                    summ.append(dict(table='A3_minus_A2', dataset=lab, comparison='A3 - A2',
                                     arm=arm, **s))
    pd.DataFrame(summ).to_csv(f'{OUT_DIR}/tdca_ablation_summary.csv', index=False)
    print(f'\nwrote {out}\nwrote {OUT_DIR}/tdca_ablation_summary.csv')


if __name__ == '__main__':
    main()
