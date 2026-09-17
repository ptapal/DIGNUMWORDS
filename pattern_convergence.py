"""
PRE-SPECIFIED 13-09-2026, before any result from this file was computed.
Replaces the brief's proposed ROI-mass-vs-8/68 analysis (reviewer round 1, M5/R4).

QUESTION (research question part 1)
Do supervised learned filters place their spatial pattern on the Retter ROI
more than label-shuffled fits of the same filter on the same data would?

SCOPE
TDCA and CSP (the supervised operators) only. max-SNR and RESS (unsupervised,
reviewer: paired disc-filter minus control-filter mass) are NOT implemented
in this round: they need a different null (no permutation) and a separate
label-free plant model for the gate. Declared as deferred.

PATTERNS (locked configurations, tdca_validated.json / csp2_validated.json)
TDCA  per LOO fold: PCA basis P (68 x 15, training blocks), embedded input
      covariance Sigma_x (training blocks, all sub-averages and samples,
      centred), W = 3 GED components exactly as tdca.fold_decision.
      Haufe A = Sigma_x W (W' Sigma_x W)^-1, reshaped (L+1) x 15 x 3 and
      back-projected through P -> (L+1) x 68 x 3.
CSP   in-sample (csp2 has no CV): P = top-15 PCs of the mean block
      covariance, Sigma_x = mean projected block covariance, generalised
      eigenvectors of the two shrunk class covariances; keep the 3 with the
      largest lambda + 1/lambda - 2 (their contribution to the KL statistic;
      invariant to a label swap). A = Sigma_x W (W' Sigma_x W)^-1, P A.
energy  each component normalised to unit norm over (delays x) channels,
      squared, summed over components and delays -> 68 energies summing
      to 3; TDCA averages energies over the 12 (8 in D2) folds.
set mass  mass_S = energy on S / total energy.
Reference: as exported by the collectors; nothing is re-referenced in the
loaders, so patterns are reference-dependent.
Patterns of TDCA live in the top-15 PC subspace of total variance.

CANDIDATE SETS (contiguity)
3-D positions from electrodes.get_biosemi68_mne_montage(). For each of the
68 channels, the channel plus its 7 nearest neighbours (Euclidean) = one
8-channel cluster; duplicates dropped (66 unique). The Retter ROI is not one
of them (bilateral, wider), so it is added as a 67th candidate. The non-ROI
plant cluster is the nearest-neighbour cluster centred on Pz
(Pz P2 P1 CPz POz CP2 CP1 P4; contains the reviewer's CP1/CPz/CP2/P1/Pz/P2,
no ROI channel).

STATISTIC
per participant, per set S: mass_S at the observed labels referenced to
mass_S under all stratified assignments (64 / 16); z and tie-corrected
mid-p exactly as block_level_test.enumerate_stratified (ROI values are
cross-checked against a direct call). This null keeps channel variance,
smoothness and volume conduction; 8/68 and the hypergeometric are dropped.
per-participant top-10%: at most 6 of the other 66 candidates have z >= the
target's z.
group: mean z over participants, one-sample t, ONE-sided (z > 0); group
rank of the ROI = fraction of the 66 NN clusters with lower group-mean z.

GATE (must pass before real labels; per method, per dataset)
calibration  real discrimination data, 20 stratified label shuffles per
             participant (rng seed 0): ROI-mass mid-p pooled over both
             datasets, median in [0.40, 0.60], fraction < .05 in [0.02, 0.10]
             (existing thresholds). ROI top-10% rate reported, not gated.
power/loc    plants added to every sub-average of the blocks labelled 1 by
             ONE stratified shuffle per participant (rng seed 1), tested with
             those shuffled labels, so no real-label information enters.
             Plant = spatial vector (1 on the 8 target channels + 0.15 N(0,1),
             seed 0, unit norm) x sin(2 pi 3 t / 64), scaled to boost x
             block-average RMS (as tdca._plant). Targets: ROI, Pz cluster.
             Boost 0.5 and 0.1.
pass (dataset) for all 4 plants the target is top-10% in >= 80% of that
             dataset's participants, AND for both Pz plants the ROI is top-10%
             in <= 10% of participants, AND calibration passes.
A method/dataset that fails is reported as failed and not run on real
labels. Pass writes h5_new/pattern_convergence_<method>_validated.json.
Deviation from the reviewer: gate applied per dataset (reviewer did not
say); plants on shuffled rather than real labels; plant cluster is the Pz
NN cluster (8 channels) rather than the 6 named channels.

REAL LABELS
arms: discrimination (diff-ERP sub-averages) and control (mean-ERP).
CONVERGES (pre-declared, reviewer): D1 discrimination, group ROI z > 0 at
one-sided p < .05 AND ROI group rank in the top 10% of the 66 clusters.
Reported also: D2, control arm, per-participant top-10% counts, the five
best candidate sets by group-mean z, paired disc - ctrl ROI z.
READING
converges           the supervised pattern concentrates on the ROI beyond
                    what the data's spatial structure gives any fit
z > 0, rank fails   ROI-weighted, but some other cluster more so
neither             no evidence the discriminant pattern favours the ROI

    python pattern_convergence.py
"""
import os
os.chdir(os.path.dirname(os.path.abspath(__file__))) # repo-relative paths; imported modules makedirs at import
import sys
import json
import time
import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scipy.stats import ttest_1samp

from csp2 import load_cache, block_covs, reduce_basis
from tdca import prepare, fold_stats, N_COMP
from block_level_test import enumerate_stratified, EXCLUDE, OUT_DIR
from electrodes import (biosemi_68_order as ORDER, RETTER_ROI_IDX,
                        get_biosemi68_mne_montage)
from rebuild_seqlevel import N_ELEC

DS = [('Angelique', 'D1'), ('Talia', 'D2')]
TD = json.load(open('h5_new/tdca_validated.json'))
CS = json.load(open('h5_new/csp2_validated.json'))
N_SHUF = 20
CAL_MEDIAN = (0.40, 0.60)
CAL_TAIL = (0.02, 0.10)
BOOSTS = (0.5, 0.1)
RECOVER = 0.80
FALSE_ROI = 0.10
PASSED = 'h5_new/pattern_convergence_{}_validated.json'


def candidate_sets():
    pos = get_biosemi68_mne_montage().get_positions()['ch_pos']
    X = np.array([pos[c] for c in ORDER])
    nn = np.argsort(np.linalg.norm(X[:, None] - X[None], axis=2), axis=1)[:, :8]
    sets, seen = {}, set()
    for c in range(N_ELEC):
        f = frozenset(nn[c].tolist())
        if f not in seen:
            seen.add(f)
            sets[f'nn:{ORDER[c]}'] = np.array(sorted(f))
    sets['ROI'] = np.array(sorted(RETTER_ROI_IDX))
    return sets


SETS = candidate_sets()
NAMES = list(SETS)
PZ = 'nn:Pz'
NN = [n for n in NAMES if n != 'ROI']


def _tdca_W(fs, mask, shrink):
    """The GED step of tdca.fold_decision, unchanged."""
    tr = fs['tr']
    m1, m0 = tr & mask, tr & ~mask
    n1, n0 = fs['n'][m1].sum(), fs['n'][m0].sum()
    S1, S0 = fs['sumZ'][m1].sum(0), fs['sumZ'][m0].sum(0)
    M1, M0 = S1 / n1, S0 / n0
    Ma = (S1 + S0) / (n1 + n0)
    Sw = (fs['sumZZ'][m1].sum(0) - n1 * M1.T @ M1
          + fs['sumZZ'][m0].sum(0) - n0 * M0.T @ M0)
    Sb = n1 * (M1 - Ma).T @ (M1 - Ma) + n0 * (M0 - Ma).T @ (M0 - Ma)
    D = Sw.shape[0]
    Sw = Sw + shrink * np.trace(Sw) / D * np.eye(D)
    return eigh(Sb, Sw)[1][:, -N_COMP:]


def _haufe(Sx, W):
    return Sx @ W @ np.linalg.inv(W.T @ Sx @ W)


def tdca_energy(B, k=TD['k'], L=TD['L'], shrink=TD['shrink']):
    folds = []
    for h in range(len(B)):
        C = sum(np.einsum('itc,itd->cd', B[i], B[i]) for i in range(len(B)) if i != h)
        P = eigh(C)[1][:, -k:] # same basis as tdca.fold_stats
        fs = fold_stats(B, h, k, L)
        tr = fs['tr']
        N = fs['n'][tr].sum() * fs['sumZ'].shape[1]
        mu = fs['sumZ'][tr].sum((0, 1)) / N
        Sx = fs['sumZZ'][tr].sum(0) / N - np.outer(mu, mu)
        folds.append((fs, P, Sx))

    def energy(mask):
        e = np.zeros(N_ELEC)
        for fs, P, Sx in folds:
            A = _haufe(Sx, _tdca_W(fs, mask, shrink)).reshape(L + 1, k, -1)
            A = np.einsum('ck,dkm->dcm', P, A)
            A /= np.linalg.norm(A, axis=(0, 1), keepdims=True)
            e += (A ** 2).sum((0, 2))
        return e / len(folds)
    return energy


def csp_energy(B, k=CS['k'], shrink=CS['shrink']):
    C = np.array([block_covs(b) for b in B]) # s_keep = 8 = cache
    P = reduce_basis(C, k)
    Ck = np.einsum('ba,nbc,cd->nad', P, C, P)
    Sx = Ck.mean(0)

    def energy(mask):
        A_, B_ = Ck[mask].mean(0), Ck[~mask].mean(0)
        A_ = A_ + shrink * np.trace(A_) / k * np.eye(k)
        B_ = B_ + shrink * np.trace(B_) / k * np.eye(k)
        lam, V = eigh(A_, B_)
        lam = np.clip(lam, 1e-12, None)
        W = V[:, np.argsort(lam + 1 / lam)[-N_COMP:]]
        A = P @ _haufe(Sx, W)
        A /= np.linalg.norm(A, axis=0, keepdims=True)
        return (A ** 2).sum(1)
    return energy


ENERGY = {'tdca': tdca_energy, 'csp': csp_energy}


def set_table(energy, y, st):
    """All assignments -> (masks index, mass matrix n_assign x n_sets).
    Enumeration order comes from enumerate_stratified itself."""
    masks = []

    def rec(m):
        masks.append(m.copy())
        return 0.0
    enumerate_stratified(rec, y, st)
    masks = masks[:-1] # last call is the observed
    E = np.array([energy(m) for m in masks])
    M = np.stack([E[:, SETS[n]].sum(1) / E.sum(1) for n in NAMES], axis=1)
    return {m.tobytes(): i for i, m in enumerate(masks)}, M, E


def zp(M, j):
    """z and tie-corrected mid-p for observed row j, as enumerate_stratified."""
    obs = M[j]
    tie = np.isclose(M, obs, rtol=1e-9, atol=0.0)
    p = (((M > obs) & ~tie).sum(0) + 0.5 * tie.sum(0)) / len(M)
    z = (obs - M.mean(0)) / (M.std(0, ddof=1) + 1e-12)
    return z, p


def top10(z, target):
    t = NAMES.index(target)
    others = np.delete(z, t)
    return int((others >= z[t]).sum()) <= int(np.floor(0.10 * len(others)))


def _shuffle(y, st, rng):
    yy = y.copy()
    for u in np.unique(st):
        i = np.where(st == u)[0]
        yy[i] = rng.permutation(y[i])
    return yy


def plant(B, yy, target, boost, seed=0):
    rng = np.random.default_rng(seed)
    v = np.zeros(N_ELEC)
    v[SETS[target]] = 1.0
    v += 0.15 * rng.normal(size=N_ELEC)
    v /= np.linalg.norm(v)
    T = B[0].shape[1]
    t = np.sin(2 * np.pi * 3 * np.arange(T) / T)
    scale = np.sqrt(np.mean([np.mean(b.mean(0) ** 2) for b in B]))
    pat = np.outer(t, v)
    pat /= np.sqrt(np.mean(pat ** 2))
    comp = boost * scale * pat
    return [b + comp if yy[i] else b for i, b in enumerate(B)]


def gate(method, prep, tables):
    lab = dict(DS)
    rows = []
    rng = np.random.default_rng(0)
    cal = []
    for key in sorted(prep):
        B, y, st = prep[key]
        idx, M, _ = tables[key]
        for s in range(N_SHUF):
            yy = _shuffle(y, st, rng)
            z, p = zp(M, idx[yy.astype(bool).tobytes()])
            if s == 0: # cross-check the re-implementation
                chk = enumerate_stratified(lambda m: M[idx[m.tobytes()], NAMES.index('ROI')], yy, st)
                assert np.isclose(chk[1], p[NAMES.index('ROI')]) and np.isclose(chk[3], z[NAMES.index('ROI')])
            cal.append(dict(ds=lab[key[0]], p=p[NAMES.index('ROI')], top=top10(z, 'ROI')))
    cal = pd.DataFrame(cal)
    med, tail = cal.p.median(), (cal.p < .05).mean()
    cal_ok = CAL_MEDIAN[0] <= med <= CAL_MEDIAN[1] and CAL_TAIL[0] <= tail <= CAL_TAIL[1]
    print(f'  [{method}] calibration n={len(cal)}  median {med:.3f}  frac<.05 {tail:.3f}  '
          f'ROI top-10% rate {cal.top.mean():.3f}  {"PASS" if cal_ok else "FAIL"}', flush=True)
    for d, g in cal.groupby('ds'):
        print(f'           {d}: median {g.p.median():.3f}  frac<.05 {(g.p < .05).mean():.3f}  '
              f'ROI top-10% {g.top.mean():.3f}')
        rows.append(dict(method=method, dataset=d, check='calibration', median=g.p.median(),
                         tail=(g.p < .05).mean(), roi_top10=g.top.mean(), n=len(g)))
    rows.append(dict(method=method, dataset='pooled', check='calibration', median=med,
                     tail=tail, roi_top10=cal.top.mean(), n=len(cal), passed=cal_ok))

    rng = np.random.default_rng(1)
    shuf = {key: _shuffle(prep[key][1], prep[key][2], rng) for key in sorted(prep)}
    res = []
    for target in ('ROI', PZ):
        for boost in BOOSTS:
            t0 = time.time()
            for key in sorted(prep):
                B, y, st = prep[key]
                yy = shuf[key]
                idx, M, _ = set_table(ENERGY[method](plant(B, yy, target, boost)), yy, st)
                z, p = zp(M, idx[yy.astype(bool).tobytes()])
                res.append(dict(ds=lab[key[0]], target=target, boost=boost,
                                hit=top10(z, target), roi_top=top10(z, 'ROI'),
                                z_target=z[NAMES.index(target)]))
            print(f'  [{method}] plant {target:6} boost {boost}: {time.time() - t0:.0f}s', flush=True)
    res = pd.DataFrame(res)
    ok = {}
    for d in ('D1', 'D2'):
        good = cal_ok
        for (target, boost), g in res[res.ds == d].groupby(['target', 'boost']):
            hit, fr = g.hit.mean(), g.roi_top.mean()
            this = hit >= RECOVER and (target == 'ROI' or fr <= FALSE_ROI)
            good &= this
            print(f'  [{method}] {d} plant {target:6} x{boost}: target top-10% in {hit:.2f}  '
                  f'ROI top-10% {fr:.2f}  mean z_target {g.z_target.mean():+.2f}  '
                  f'{"ok" if this else "FAIL"}')
            rows.append(dict(method=method, dataset=d, check=f'plant {target} x{boost}',
                             target_top10=hit, roi_top10=fr, z_target=g.z_target.mean(),
                             n=len(g), passed=this))
        ok[d] = bool(good)
        print(f'  [{method}] {d} GATE {"PASS" if good else "FAIL"}', flush=True)
    return ok, rows


def main():
    methods = [m for m in ('tdca', 'csp') if m in sys.argv] or ['tdca', 'csp']
    meta, Dsub, Msub = load_cache()
    prep = {'disc': prepare(meta, Dsub, set(EXCLUDE)), 'ctrl': prepare(meta, Msub, set(EXCLUDE))}
    print(f'{len(NN)} unique nearest-neighbour clusters + ROI; {PZ} = '
          f'{[ORDER[i] for i in SETS[PZ]]}')

    gate_rows, rows, energies = [], [], []
    for method in methods:
        t0 = time.time()
        tables = {key: set_table(ENERGY[method](B), y, st)
                  for key, (B, y, st) in prep['disc'].items()}
        print(f'\n[{method}] disc tables {time.time() - t0:.0f}s', flush=True)
        ok, gr = gate(method, prep['disc'], tables)
        gate_rows += gr
        # bookkeeping fix after the first run (declared): that run wrote this json
        # although both datasets failed; it now matches the docstring (pass only)
        if any(ok.values()):
            json.dump(dict(method=method, passed=ok, rows=gr), open(PASSED.format(method), 'w'),
                      indent=1, default=float)
        run_ds = [ds for ds, lab in DS if ok[lab]]
        for ds, lab in DS:
            if not ok[lab]:
                print(f'  [{method}] {lab}: gate failed, NOT run on real labels')
        if not run_ds:
            continue
        for arm in ('disc', 'ctrl'):
            for key, (B, y, st) in sorted(prep[arm].items()):
                if key[0] not in run_ds:
                    continue
                idx, M, E = tables[key] if arm == 'disc' else set_table(ENERGY[method](B), y, st)
                j = idx[y.astype(bool).tobytes()]
                z, p = zp(M, j)
                for n, zz, pp, mm in zip(NAMES, z, p, M[j]):
                    rows.append(dict(method=method, arm=arm, dataset=key[0], subject=key[1],
                                     set=n, mass=mm, z=zz, p=pp, top10=top10(z, n)))
                energies.append(dict(method=method, arm=arm, dataset=key[0], subject=key[1],
                                     **dict(zip(ORDER, E[j]))))
    pd.DataFrame(gate_rows).to_csv(f'{OUT_DIR}/pattern_convergence_gate.csv', index=False)
    if not rows:
        print('\nnothing passed the gate; no real-label output')
        return
    r = pd.DataFrame(rows)
    r.to_csv(f'{OUT_DIR}/pattern_convergence_exact.csv', index=False)
    pd.DataFrame(energies).to_csv(f'{OUT_DIR}/pattern_convergence_energy.csv', index=False)

    print('\n' + '=' * 96)
    print('PATTERN CONVERGENCE ON THE ROI  (observed-label ROI mass vs all stratified assignments)')
    print('=' * 96)
    summ = []
    for (method, ds, arm), g in r.groupby(['method', 'dataset', 'arm'], sort=False):
        lab = dict(DS)[ds]
        gz = g.groupby('set')['z'].mean()
        roi = g[g.set == 'ROI']
        t, p2 = ttest_1samp(roi.z, 0.0)
        p1 = p2 / 2 if t > 0 else 1 - p2 / 2
        pct = float((gz[NN] < gz['ROI']).mean())
        conv = p1 < .05 and pct >= 0.90
        best = gz.sort_values(ascending=False).head(5)
        print(f'\n  {method.upper()} {lab} {arm}: n={len(roi)}  ROI z mean {roi.z.mean():+.3f}  '
              f't={t:+.2f}  p(1t)={p1:.4f}  pos {int((roi.z > 0).sum())}/{len(roi)}  '
              f'mean mass {roi.mass.mean():.3f}')
        print(f'    ROI group-mean z beats {pct:.3f} of the {len(NN)} clusters; '
              f'participants with ROI top-10%: {int(roi.top10.sum())}/{len(roi)}'
              f'{"   CONVERGES" if conv and lab == "D1" and arm == "disc" else ""}')
        print('    best sets by group-mean z: ' + ', '.join(f'{n} {v:+.2f}' for n, v in best.items()))
        summ.append(dict(method=method, dataset=lab, arm=arm, n=len(roi), roi_z=roi.z.mean(),
                         t=t, p_one=p1, k_pos=int((roi.z > 0).sum()), roi_mass=roi.mass.mean(),
                         roi_rank_pct=pct, roi_top10_n=int(roi.top10.sum()), criterion=conv))
    for (method, ds), g in r[r.set == 'ROI'].groupby(['method', 'dataset'], sort=False):
        a = g[g.arm == 'disc'].set_index('subject')['z']
        b = g[g.arm == 'ctrl'].set_index('subject')['z']
        i = a.index.intersection(b.index)
        t, p2 = ttest_1samp((a[i] - b[i]).values, 0.0)
        print(f'\n  {method.upper()} {dict(DS)[ds]} ROI z disc - ctrl: mean {(a[i] - b[i]).mean():+.3f}  '
              f't={t:+.2f}  p(2t)={p2:.4f}')
    pd.DataFrame(summ).to_csv(f'{OUT_DIR}/pattern_convergence_summary.csv', index=False)
    print(f'\nwrote {OUT_DIR}/pattern_convergence_{{gate,exact,energy,summary}}.csv')


if __name__ == '__main__':
    main()
