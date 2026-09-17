"""
PRE-SPECIFIED 12-09-2026, before any result from this file was computed.

TDCA (Liu et al., 2021, IEEE TNSRE 29:1998-2007), adapted to a two-class
contrast at one frequency.
1. PCA to k spatial components            fitted on training blocks only
2. time-delay embedding with L delays      so the filter can weight time
3. discriminant GED: between-class scatter vs within-class scatter
   (shrinkage on the latter), keep 3 components
4. score the held-out block by its signed projection onto
   (Parity template - Control template), measured from their midpoint
Deviations from the original:
- no sine/cosine reference projection. It helps separate stimulus
  frequencies; both classes here share one frequency.
- scoring. TDCA correlates a trial with each class template, which suits
  many-class spellers. With two classes the templates become mirror images
  in the discriminant space and correlation discards magnitude; this was
  found on synthetic data before any real-label run and replaced with the
  standard two-class discriminant score.

Inference: leave-one-block-out CV, every step fitted within fold. The whole
cross-validated pipeline is rerun under each of the 2^k stratified label
assignments, so label use is exact. Signal = diff-ERP sub-averages;
control = mean-ERP sub-averages; primary result is paired signal - control.

GATE (identical to csp2.py, fixed before any sweep)
calibration  null p median in [0.40, 0.60], fraction < .05 in [0.02, 0.10]
power        planted spatiotemporal effect at 0.5 x block-average RMS
             detected at GROUP-level p < .05 in at least one dataset
sweep        L in {2, 4, 8}, k in {10, 15}, shrinkage in {0.02, 0.15}
locked       3 components

READING
works in both datasets  a spatiotemporal discriminant recovers what every
                        spatial summary lost
works in D1 only        matches CSP; time adds nothing beyond space
fails                   not learnable at six blocks per condition, even with
                        temporal information

    python tdca.py --validate
    python tdca.py --run
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scipy.stats import norm, ttest_1samp, t as tdist

from csp2 import load_cache
from block_level_test import enumerate_stratified, bh_fdr, EXCLUDE, OUT_DIR
from rebuild_seqlevel import N_ELEC

PASSED = 'h5_new/tdca_validated.json'
N_COMP = 3
DS = [('Angelique', 'D1'), ('Talia', 'D2')]

CAL_MEDIAN = (0.40, 0.60)
CAL_TAIL = (0.02, 0.10)
POWER_BOOST = 0.5
POWER_ALPHA = 0.05
GRID = [(L, k, sh) for L in (2, 4, 8) for k in (10, 15) for sh in (0.02, 0.15)]


def prepare(meta, sub, exclude):
    """{(ds, subj): (list of (n_sub, T, C) block arrays, y, strata)}."""
    meta = meta[~meta.subject.isin(exclude)] # index stays aligned
    out = {}
    for (ds, subj), g in meta.groupby(['dataset', 'subject']):
        B, y, st = [], [], []
        for _, gg in g.groupby('block_id'):
            b = sub[gg.index].astype(np.float64) # (nseq, S, T, C)
            B.append(b.reshape(-1, b.shape[2], b.shape[3]))
            r = gg.iloc[0]
            y.append(1 if r.condition == 'Par' else 0)
            st.append(f'{r.modality}|{r.font}')
        out[(ds, subj)] = (B, np.array(y), np.array(st))
    return out


def _embed(Z, L):
    T = Z.shape[1]
    return np.concatenate([Z[:, d:T - L + d, :] for d in range(L + 1)], axis=2)


def fold_stats(B, held, k, L):
    """Everything label-independent for one fold, fitted on training blocks."""
    tr = [i for i in range(len(B)) if i != held]
    C = sum(np.einsum('itc,itd->cd', B[i], B[i]) for i in tr)
    P = eigh(C)[1][:, -k:]
    sumZ, sumZZ, n = [], [], []
    for i in range(len(B)):
        Zt = _embed(B[i] @ P, L)
        sumZ.append(Zt.sum(0))
        sumZZ.append(np.einsum('itd,ite->de', Zt, Zt))
        n.append(len(Zt))
    H = _embed(B[held] @ P, L).mean(0)
    return dict(tr=np.array([i != held for i in range(len(B))]),
                sumZ=np.array(sumZ), sumZZ=np.array(sumZZ), n=np.array(n),
                H=H, held=held)


def _corr(a, b):
    a = a.ravel() - a.mean(); b = b.ravel() - b.mean()
    return float(a @ b / (np.sqrt((a @ a) * (b @ b)) + 1e-300))


def fold_decision(fs, mask, shrink):
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
    W = eigh(Sb, Sw)[1][:, -N_COMP:]
    h, p1, p0 = fs['H'] @ W, M1 @ W, M0 @ W
    # Signed projection onto the template difference, from the midpoint.
    # TDCA's template correlation is built for many classes; with two, the
    # discriminant space makes the templates mirror images, so correlation
    # reduces to the sign of one noisy value and a Control block (defined by
    # the ABSENCE of a component) correlates randomly with its own template.
    # Found on synthetic data before real labels: a 30x planted effect was
    # recovered in only 8/12 folds.
    diff = (p1 - p0).ravel()
    dec = float((h - 0.5 * (p1 + p0)).ravel() @ diff / (np.linalg.norm(diff) + 1e-300))
    return dec if mask[fs['held']] else -dec


def test_subject(B, y, st, L, k, shrink):
    folds = [fold_stats(B, h, k, L) for h in range(len(B))]
    stat = lambda m: float(np.mean([fold_decision(f, m, shrink) for f in folds]))
    return enumerate_stratified(stat, y, st)


def _plant(B, y, boost, seed=0):
    rng = np.random.default_rng(seed)
    v = np.zeros(N_ELEC)
    v[[24, 26, 61, 63]] = 1.0 # PO7, O1, PO8, O2
    v += 0.15 * rng.normal(size=N_ELEC)
    v /= np.linalg.norm(v)
    T = B[0].shape[1]
    t = np.sin(2 * np.pi * 3 * np.arange(T) / T)
    scale = np.sqrt(np.mean([np.mean(b.mean(0) ** 2) for b in B]))
    pattern = np.outer(t, v)
    pattern /= np.sqrt(np.mean(pattern ** 2)) # unit RMS, so boost is in
    comp = boost * scale * pattern # block-average-RMS units
    return [b + comp if y[i] else b for i, b in enumerate(B)]


def _group_p(ps):
    pc = np.clip(np.array(ps), 1e-9, 1 - 1e-9)
    return float(norm.sf(norm.isf(pc).sum() / np.sqrt(len(pc))))


def validate(prep):
    rng = np.random.default_rng(0)
    print(f'acceptance: null median in {CAL_MEDIAN}, tail in {CAL_TAIL}; '
          f'group p < {POWER_ALPHA} at boost {POWER_BOOST}\n')
    print(f'{"L":>3}{"k":>4}{"shrink":>8}{"null med":>10}{"null<.05":>10}'
          f'{"D1 grp p":>10}{"D2 grp p":>10}   verdict', flush=True)
    best = None
    for L, k, sh in GRID:
        nulls, planted = [], {}
        for (ds, subj), (B, y, st) in prep.items():
            yy = y.copy()
            for u in np.unique(st):
                i = np.where(st == u)[0]
                yy[i] = rng.permutation(y[i])
            nulls.append(test_subject(B, yy, st, L, k, sh)[1])
            planted.setdefault(ds, []).append(
                test_subject(_plant(B, y, POWER_BOOST), y, st, L, k, sh)[1])
        nulls = np.array(nulls)
        med, tail = np.median(nulls), np.mean(nulls < .05)
        gp = {ds: _group_p(ps) for ds, ps in planted.items()}
        ok = (CAL_MEDIAN[0] <= med <= CAL_MEDIAN[1]
              and CAL_TAIL[0] <= tail <= CAL_TAIL[1]
              and min(gp.values()) < POWER_ALPHA)
        print(f'{L:>3}{k:>4}{sh:>8.2f}{med:>10.3f}{tail:>10.3f}'
              f'{gp.get("Angelique", np.nan):>10.4f}{gp.get("Talia", np.nan):>10.4f}'
              f'   {"PASS" if ok else "fail"}', flush=True)
        if ok and (best is None or min(gp.values()) < best[3]):
            best = (L, k, sh, min(gp.values()))
    return best


def _line(label, v):
    v = np.asarray(v, dtype=float); n = len(v)
    ci = tdist.ppf(.975, n - 1) * v.std(ddof=1) / np.sqrt(n)
    t, p = ttest_1samp(v, 0.0)
    k = int((v > 0).sum())
    return (f'{label:30}{n:>4}{v.mean():>+9.3f}'
            f'{f"  [{v.mean()-ci:+.2f},{v.mean()+ci:+.2f}]":>18}'
            f'{t:>+8.2f}{p/2:>9.4f}{f"{k}/{n}":>8}')


def main():
    meta, D, M = load_cache()
    if '--validate' in sys.argv:
        best = validate(prepare(meta, D, set(EXCLUDE)))
        if best is None:
            print('\nNOTHING PASSED. Not fit to run on real labels.')
            if os.path.exists(PASSED):
                os.remove(PASSED)
            return
        L, k, sh, gp = best
        json.dump(dict(L=L, k=k, shrink=sh, group_p=gp), open(PASSED, 'w'))
        print(f'\nPASSED: L={L} k={k} shrink={sh}  -> {PASSED}')
        return

    if '--run' not in sys.argv:
        print('use --validate, then --run')
        return
    if not os.path.exists(PASSED):
        print('REFUSED: run --validate first.')
        return
    cfg = json.load(open(PASSED))
    L, k, sh = cfg['L'], cfg['k'], cfg['shrink']
    print(f'validated L={L} k={k} shrink={sh}')

    rows = []
    for name, sub in [('tdca_diff', D), ('tdca_mean', M)]:
        for (ds, subj), (B, y, st) in sorted(prepare(meta, sub, set(EXCLUDE)).items()):
            _, p, _, z, n_all, _ = test_subject(B, y, st, L, k, sh)
            rows.append(dict(dataset=ds, subject=subj, measure=name, z=z, p=p,
                             n_assignments=n_all))
    r = pd.DataFrame(rows)
    r.to_csv(f'{OUT_DIR}/tdca_exact.csv', index=False)

    print('\n' + '=' * 84)
    print('TDCA (leave-one-block-out, refit inside every permutation)')
    print('=' * 84)
    for ds, lab in DS:
        g = r[r.dataset == ds]
        if g.empty:
            continue
        a = g[g.measure == 'tdca_diff'].set_index('subject')['z']
        b = g[g.measure == 'tdca_mean'].set_index('subject')['z']
        i = a.index.intersection(b.index)
        print(f'\n  {lab}')
        print('    ' + _line('TDCA, discrimination', a.values))
        print('    ' + _line('TDCA, control (mean-ERP)', b.values))
        print('    ' + _line('TDCA  signal - control', (a[i] - b[i]).values))
    print(f'\nwrote {OUT_DIR}/tdca_exact.csv')


if __name__ == '__main__':
    main()
