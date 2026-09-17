"""
PRE-SPECIFIED 13-09-2026, before any result from this file was computed.
Synthetic plants only; the observed-data statistic enters solely as the fixed
target value z_obs below.

QUESTION (reviewer round 2, request 3)
Where does the observed D1 TDCA effect sit on the plant scale of the pattern-
convergence gate, and could the localisation method have found the ROI at an
effect of that size?

DATA / PLANTS
discrimination sub-averages (csp2 cache), both datasets, S20/S30 excluded
labels: ONE stratified shuffle per participant, rng = default_rng(1),
        participants in sorted order (identical to pattern_convergence gate)
plant:  pattern_convergence.plant(B, shuffled labels, 'ROI', boost):
        1 on the 8 ROI channels + 0.15 N(0,1) (seed 0), unit norm, x a
        3-cycle sine per 64-sample epoch (22.5 Hz), scaled to boost x
        block-average RMS, added to every sub-average of the blocks the
        shuffle labels Parity. Tested against the SAME shuffled labels.
boosts: 0 (added, declared: baseline incl. any real-effect leakage), 0.02,
        0.05, 0.1, 0.2, 0.3, 0.5
MEASURES per participant, per boost (all locked, no tuning)
TDCA     tdca.test_subject, h5_new/tdca_validated.json          z, p
A1       tdca_ablation.make_stat('A1_roimean')                   z, p
ROI RMS  ROI-mean waveform of the block average, RMS, signed     z, p
loc hit  TDCA pattern ROI top-10% among 67 candidate sets,
         pattern_convergence.set_table / zp / top10               bool
group per dataset: mean z, mean rank Phi^-1(1 - mid-p), hit rate.
b* (FIXED RULE)
z_obs  = mean per-participant z of tdca_diff, D1, tdca_exact.csv
curve  = D1 synthetic TDCA group-mean z at boosts in ascending order
b*     = linear interpolation of boost between the FIRST adjacent pair
         (b_i, b_i+1) with z_i <= z_obs < z_i+1.
         If z(0) >= z_obs, b* = 0. If z(0.5) < z_obs, b* > 0.5 (undefined).
hit(b*) = localisation hit rate in D1 from a NEW run at exactly b*
         (rounded to 4 decimals), same shuffles and plant; the linearly
         interpolated hit rate is reported alongside.
Descriptive only: the same for D2 with z_obs = D2 tdca_diff mean.
READING (reviewer)
hit(b*) < 0.80   "not validated at the observed effect size" may be said
hit(b*) >= 0.80  it may not; a real-label pattern run would be a post-failure
                 gate amendment (paper stage, disclosed)
CAVEATS (to be stated with any use)
the plant is a stationary 22.5 Hz sine on the ROI, not a realistic
Parity/Control shape difference; shuffled labels coincide with the real
labels or their complement for 2/64 (D1) or 2/16 (D2) assignments, so the
real effect can leak into some participants (boost 0 row shows the size).

    python power_curve.py
"""
import os
os.chdir(os.path.dirname(os.path.abspath(__file__))) # repo-relative paths; imported modules makedirs at import
import json
import time
import numpy as np
import pandas as pd
from scipy.stats import norm

from csp2 import load_cache
from tdca import prepare, test_subject
from block_level_test import enumerate_stratified, signed_stat, EXCLUDE, OUT_DIR
from electrodes import RETTER_ROI_IDX
import pattern_convergence as pc
from tdca_ablation import make_stat as a_stat

DS = [('Angelique', 'D1'), ('Talia', 'D2')]
CFG = json.load(open('h5_new/tdca_validated.json'))
BOOSTS = (0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5)


def roi_rms_test(B, y, st):
    v = np.array([np.sqrt((b.mean(0)[:, RETTER_ROI_IDX].mean(1) ** 2).mean()) for b in B])
    return enumerate_stratified(lambda m: signed_stat(v, m), y, st)


def run_boost(prep, shuf, boost, measures=('tdca', 'a1', 'roi', 'loc')):
    rows = []
    for key in sorted(prep):
        B, y, st = prep[key]
        yy = shuf[key]
        Bp = pc.plant(B, yy, 'ROI', boost) if boost > 0 else B
        row = dict(dataset=key[0], subject=key[1], boost=boost)
        if 'tdca' in measures:
            _, p, _, z, _, _ = test_subject(Bp, yy, st, CFG['L'], CFG['k'], CFG['shrink'])
            row.update(tdca_z=z, tdca_p=p)
        if 'a1' in measures:
            _, p, _, z, _, _ = enumerate_stratified(a_stat(Bp, 'A1_roimean'), yy, st)
            row.update(a1_z=z, a1_p=p)
        if 'roi' in measures:
            _, p, _, z, _, _ = roi_rms_test(Bp, yy, st)
            row.update(roi_z=z, roi_p=p)
        if 'loc' in measures:
            idx, M, _ = pc.set_table(pc.tdca_energy(Bp), yy, st)
            zz, _ = pc.zp(M, idx[yy.astype(bool).tobytes()])
            row.update(loc_hit=pc.top10(zz, 'ROI'), loc_z=zz[pc.NAMES.index('ROI')])
        rows.append(row)
    return pd.DataFrame(rows)


def group(r):
    g = r.copy()
    for m in ('tdca', 'a1', 'roi'):
        if f'{m}_p' in g:
            g[f'{m}_rank'] = norm.isf(g[f'{m}_p'].clip(1e-12, 1 - 1e-12))
    cols = [c for c in g.columns if c.endswith(('_z', '_rank', '_hit'))]
    g['dataset'] = g.dataset.map(dict(DS))
    return g.groupby(['dataset', 'boost'])[cols].mean().reset_index()


def bstar(curve, z_obs):
    b, z = curve.boost.values, curve.tdca_z.values
    if z[0] >= z_obs:
        return 0.0, 0
    for i in range(len(b) - 1):
        if z[i] <= z_obs < z[i + 1]:
            return float(b[i] + (z_obs - z[i]) * (b[i + 1] - b[i]) / (z[i + 1] - z[i])), i
    return np.nan, None


def main():
    meta, D, M = load_cache()
    prep = prepare(meta, D, set(EXCLUDE))
    rng = np.random.default_rng(1)
    shuf = {key: pc._shuffle(prep[key][1], prep[key][2], rng) for key in sorted(prep)}

    res = []
    for boost in BOOSTS:
        t0 = time.time()
        res.append(run_boost(prep, shuf, boost))
        print(f'boost {boost}: {time.time() - t0:.0f}s', flush=True)
    r = pd.concat(res, ignore_index=True)
    r.to_csv(f'{OUT_DIR}/power_curve_subjects.csv', index=False)
    G = group(r)

    obs = pd.read_csv(f'{OUT_DIR}/tdca_exact.csv')
    obs = obs[obs.measure == 'tdca_diff'].groupby('dataset')['z'].mean()
    print('\nGROUP MEANS (synthetic, ROI plant, shuffled labels)')
    print(G.round(3).to_string(index=False))

    extra, summ = [], []
    for ds, lab in DS:
        cur = G[G.dataset == lab].sort_values('boost')
        z_obs = float(obs[ds])
        b, i = bstar(cur, z_obs)
        if i is None and np.isnan(b):
            line = f'{lab}: z_obs {z_obs:+.3f} above z(0.5) {cur.tdca_z.iloc[-1]:+.3f}: b* > 0.5, undefined'
            hit_interp = hit_run = np.nan
        else:
            hb = cur.loc_hit.values
            hit_interp = (hb[i] if b == 0.0 else
                          hb[i] + (b - cur.boost.values[i]) * (hb[i + 1] - hb[i]) /
                          (cur.boost.values[i + 1] - cur.boost.values[i]))
            bb = round(b, 4)
            sub = {k: v for k, v in prep.items() if k[0] == ds}
            t0 = time.time()
            rb = run_boost(sub, shuf, bb, measures=('tdca', 'loc'))
            extra.append(rb)
            hit_run = rb.loc_hit.mean()
            line = (f'{lab}: z_obs {z_obs:+.3f}  b* = {b:.4f}  run at {bb}: synthetic TDCA z '
                    f'{rb.tdca_z.mean():+.3f}, hit rate {hit_run:.3f} '
                    f'(interpolated {hit_interp:.3f})  [{time.time() - t0:.0f}s]')
        verdict = ('' if lab != 'D1' or np.isnan(hit_run) else
                   ('   -> hit(b*) < 0.80: "not validated at the observed effect size" MAY be said'
                    if hit_run < 0.80 else
                    '   -> hit(b*) >= 0.80: "not validated at the observed effect size" may NOT be said'))
        print(line + verdict)
        summ.append(dict(dataset=lab, z_obs=z_obs, b_star=b, hit_run=hit_run, hit_interp=hit_interp,
                         role='decision' if lab == 'D1' else 'descriptive'))
    if extra:
        pd.concat(extra).to_csv(f'{OUT_DIR}/power_curve_bstar_subjects.csv', index=False)
    G.to_csv(f'{OUT_DIR}/power_curve_group.csv', index=False)
    pd.DataFrame(summ).to_csv(f'{OUT_DIR}/power_curve_bstar.csv', index=False)
    print(f'\nwrote {OUT_DIR}/power_curve_{{subjects,group,bstar,bstar_subjects}}.csv')


if __name__ == '__main__':
    main()
