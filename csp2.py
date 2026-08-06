"""
python csp2.py --build       # one CSV pass, sub-average cache
python csp2.py --validate    # calibration + power sweep on synthetic
python csp2.py --run         # real labels; only after validate passes
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from math import gcd
from scipy.linalg import eigh
from scipy.signal import resample_poly
from scipy.stats import (norm, ttest_rel, ttest_1samp, wilcoxon,
                         binomtest, t as tdist)

from rebuild_seqlevel import (load_sequences, parse_angelique_fname,
                              parse_talia_fname, _get_talia_files, N_ELEC)
from epoch_analysis import SR, TARGET_SR, EPOCH_SAMPLES, ANG_CSV_DIR, TAL_CSV_DIR
from block_level_test import enumerate_stratified, bh_fdr, EXCLUDE, OUT_DIR

CACHE   = 'h5_new/csp2_cache.npz'
PASSED  = 'h5_new/csp2_validated.json'
S_SUB   = 8 # sub-averages stored per sequence; validation may merge
DS      = [('Angelique', 'D1'), ('Talia', 'D2')]

# acceptance thresholds, fixed before any sweep
CAL_MEDIAN = (0.40, 0.60)
CAL_TAIL   = (0.02, 0.10)
POWER_BOOST = 0.5
POWER_ALPHA = 0.05 # group-level


# the cache
def _subaverages(eeg, n_sub=S_SUB):
    """One sequence -> (n_sub, T, 68) sub-averages of the diff-ERP and mean-ERP.

    Pairs are assigned to sub-averages round-robin, so each sub-average spans
    the whole sequence and none of them sits in a single stretch of drift.
    """
    g = gcd(int(SR), int(TARGET_SR))
    rs = resample_poly(eeg, TARGET_SR // g, SR // g).astype(np.float64)
    n = rs.shape[0] // EPOCH_SAMPLES
    ep = rs[:n * EPOCH_SAMPLES].reshape(n, EPOCH_SAMPLES, -1)[:, :, :N_ELEC]

    npair = ep.shape[0] // 2
    d = ep[1:2 * npair:2] - ep[0:2 * npair:2]
    m = 0.5 * (ep[1:2 * npair:2] + ep[0:2 * npair:2])

    D = np.stack([d[i::n_sub].mean(0) for i in range(n_sub)])
    M = np.stack([m[i::n_sub].mean(0) for i in range(n_sub)])
    return D.astype(np.float32), M.astype(np.float32)


def build_cache(path=CACHE):
    rows, Ds, Ms = [], [], []
    for csv_dir, parse_fn, get_files, ds in [
            (ANG_CSV_DIR, parse_angelique_fname, None, 'Angelique'),
            (TAL_CSV_DIR, parse_talia_fname, _get_talia_files, 'Talia')]:
        files = get_files(csv_dir) if get_files else sorted(
            f for f in os.listdir(csv_dir) if f.endswith('.csv'))
        print(f'[{ds}] {len(files)} files')
        for i, fname in enumerate(sorted(files)):
            try:
                subj, mod, ftype, sfont, cond = parse_fn(fname)
                if None in (subj, mod, ftype, cond):
                    continue
                seqs = load_sequences(os.path.join(csv_dir, fname), ds)
            except Exception as e:
                print(f'  SKIP {fname}: {e}')
                continue
            font = f'{ftype}/{sfont}' if sfont else ftype
            for s in sorted(seqs):
                D, M = _subaverages(seqs[s])
                rows.append(dict(dataset=ds, subject=f'S{subj}', modality=mod,
                                 font=font, condition=cond, block_id=fname, seq=s))
                Ds.append(D); Ms.append(M)
            if (i + 1) % 40 == 0:
                print(f'  [{i+1}/{len(files)}] {len(rows)} sequences')
    meta = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, D=np.stack(Ds), M=np.stack(Ms),
                        meta=meta.to_csv(index=False))
    print(f'wrote {path}  ({len(meta)} sequences x {S_SUB} sub-averages)')


def load_cache(path=CACHE):
    from io import StringIO
    z = np.load(path, allow_pickle=True)
    return pd.read_csv(StringIO(str(z['meta']))), z['D'], z['M']


# CSP
def _merge(sub, s_keep):
    """(n, S, T, C) -> (n, s_keep, T, C) by averaging adjacent sub-averages."""
    n, S, T, C = sub.shape
    if s_keep == S:
        return sub
    step = S // s_keep
    return np.stack([sub[:, i * step:(i + 1) * step].mean(1)
                     for i in range(s_keep)], axis=1)


def block_covs(sub_block):
    """(n_sub, T, C) sub-averages of one block -> its (C, C) covariance.
    """
    return np.einsum('itc,itd->cd', sub_block, sub_block) / (
        sub_block.shape[0] * sub_block.shape[1])


def prepare(meta, sub, exclude, s_keep):
    """{(ds, subj): (block covariances, labels, strata)} in sensor space."""
    meta = meta[~meta.subject.isin(exclude)] # index stays aligned
    out = {}
    for (ds, subj), g in meta.groupby(['dataset', 'subject']):
        C, y, st = [], [], []
        for _, gg in g.groupby('block_id'):
            b = _merge(sub[gg.index], s_keep)           # (nseq, s, T, Cx)
            b = b.reshape(-1, b.shape[2], b.shape[3])   # pool sequences
            C.append(block_covs(b.astype(np.float64)))
            r = gg.iloc[0]
            y.append(1 if r.condition == 'Par' else 0)
            st.append(f'{r.modality}|{r.font}')
        out[(ds, subj)] = (np.array(C), np.array(y), np.array(st))
    return out


def reduce_basis(C, k):
    """Top-k spatial PCs of the total covariance. Label-free, so it is
    identical under every permutation and may be computed once."""
    tot = C.mean(axis=0)
    vals, vecs = eigh(tot)
    return vecs[:, -k:]


def csp_stat(Ck, mask, shrink, kind='lmax'):
    """Separation between the two groups' covariances, in the reduced space.

    lmax    log of the largest generalised eigenvalue
    logdet  sum of log eigenvalues = log det(A) - log det(B)
    kl      symmetrised KL (Jeffreys) between zero-mean Gaussians
    """
    A = Ck[mask].mean(0)
    B = Ck[~mask].mean(0)
    k = A.shape[0]
    A = A + shrink * np.trace(A) / k * np.eye(k)
    B = B + shrink * np.trace(B) / k * np.eye(k)
    try:
        lam = eigh(A, B, eigvals_only=True)
    except np.linalg.LinAlgError:
        return np.nan
    lam = np.clip(lam, 1e-12, None)
    if kind == 'lmax':
        return float(np.log(lam[-1]))
    if kind == 'logdet':
        return float(np.sum(np.log(lam)))
    if kind == 'kl':
        return float(0.25 * (lam.sum() + (1.0 / lam).sum() - 2 * k))
    raise ValueError(kind)


def test_subject(C, y, st, k, shrink, kind='lmax'):
    P = reduce_basis(C, k)
    Ck = np.einsum('ba,nbc,cd->nad', P, C, P) # project every block covariance
    return enumerate_stratified(lambda m: csp_stat(Ck, m, shrink, kind), y, st)


# validation
def _plant(C, y, boost, seed=0):
    """Add a rank-1 occipital component to the Parity blocks' covariances."""
    rng = np.random.default_rng(seed)
    v = np.zeros(N_ELEC)
    v[[24, 26, 61, 63]] = 1.0 # PO7, O1, PO8, O2
    v += 0.15 * rng.normal(size=N_ELEC)
    v /= np.linalg.norm(v)
    scale = np.trace(C.mean(0)) / N_ELEC
    C2 = C.copy()
    C2[y.astype(bool)] += boost * scale * np.outer(v, v)
    return C2


def validate(prep_by_s, grid):
    rng = np.random.default_rng(0)
    print(f'\nacceptance: null median in {CAL_MEDIAN}, tail in {CAL_TAIL}; '
          f'GROUP-level p < {POWER_ALPHA} at boost {POWER_BOOST}')
    print(f'\n{"stat":>7}{"S":>3}{"k":>4}{"shrink":>8}{"null med":>10}'
          f'{"null<.05":>10}{"power":>8}   verdict')
    print('-' * 68)
    best = None
    for kind, s_keep, k, shrink in grid:
        prep = prep_by_s[s_keep]
        nulls, planted = [], {}
        for (ds, subj), (C, y, st) in prep.items():
            yy = y.copy()
            for u in np.unique(st):
                i = np.where(st == u)[0]
                yy[i] = rng.permutation(y[i])
            _, p, _, _, _, _ = test_subject(C, yy, st, k, shrink, kind)
            nulls.append(p)
            _, pp, _, _, _, _ = test_subject(_plant(C, y, POWER_BOOST), y, st,
                                             k, shrink, kind)
            planted.setdefault(ds, []).append(pp)
        nulls = np.array(nulls)
        med, tail = np.median(nulls), np.mean(nulls < .05)
        # group level: Stouffer over subjects, best dataset
        gps = []
        for ds, ps in planted.items():
            pc = np.clip(np.array(ps), 1e-9, 1 - 1e-9)
            gps.append(float(norm.sf(norm.isf(pc).sum() / np.sqrt(len(pc)))))
        power = min(gps)
        ok = (CAL_MEDIAN[0] <= med <= CAL_MEDIAN[1]
              and CAL_TAIL[0] <= tail <= CAL_TAIL[1]
              and power < POWER_ALPHA)
        print(f'{kind:>7}{s_keep:>3}{k:>4}{shrink:>8.2f}{med:>10.3f}{tail:>10.3f}'
              f'{power:>8.4f}   {"PASS" if ok else "fail"}')
        if ok and (best is None or power < best[4]):
            best = (kind, s_keep, k, shrink, power)
    return best


def _line(label, v):
    v = np.asarray(v, dtype=float); n = len(v)
    ci = tdist.ppf(.975, n - 1) * v.std(ddof=1) / np.sqrt(n)
    t, p = ttest_1samp(v, 0.0)
    k = int((v > 0).sum())
    bt = binomtest(k, n, 0.5, alternative='greater').pvalue
    return (f'{label:30}{n:>4}{v.mean():>+9.3f}'
            f'{f"  [{v.mean()-ci:+.2f},{v.mean()+ci:+.2f}]":>18}'
            f'{t:>+8.2f}{p/2:>9.4f}{f"{k}/{n}":>8}{bt:>10.4f}')


def main():
    if '--build' in sys.argv or not os.path.exists(CACHE):
        build_cache()
        if '--build' in sys.argv:
            return

    meta, D, M = load_cache()
    print(f'{len(meta)} sequences x {D.shape[1]} sub-averages cached')

    if '--validate' in sys.argv:
        s_values = [1, 2, 4, 8]
        prep_by_s = {s: prepare(meta, D, set(EXCLUDE), s) for s in s_values}
        grid = [(kind, s, k, sh) for kind in ('lmax', 'logdet', 'kl')
                for s in s_values for k in (6, 10, 15)
                for sh in (0.02, 0.05, 0.15)]
        best = validate(prep_by_s, grid)
        if best is None:
            print('\nNOTHING PASSED. The design is not fit to run on real '
                  'labels; do not run it.')
            if os.path.exists(PASSED):
                os.remove(PASSED)
            return
        kind, s_keep, k, shrink, power = best
        json.dump(dict(kind=kind, s_keep=s_keep, k=k, shrink=shrink,
                       power=power), open(PASSED, 'w'))
        print(f'\nPASSED: stat={kind} S={s_keep} k={k} shrink={shrink} '
              f'(group p {power:.4f})')
        print('locked to h5_new/csp2_validated.json; now `python csp2.py --run`')
        return

    if '--run' not in sys.argv:
        print('\nnothing to do. use --validate first, then --run')
        return
    if not os.path.exists(PASSED):
        print('REFUSED: no validated configuration. Run --validate first.')
        return

    cfg = json.load(open(PASSED))
    kind, s_keep, k, shrink = cfg['kind'], cfg['s_keep'], cfg['k'], cfg['shrink']
    print(f'\nusing validated config stat={kind} S={s_keep} k={k} '
          f'shrink={shrink}')

    res = {}
    for name, sub in [('diff', D), ('mean', M)]:
        prep = prepare(meta, sub, set(EXCLUDE), s_keep)
        rows = []
        for (ds, subj), (C, y, st) in sorted(prep.items()):
            obs, p, _, z, n_all, _ = test_subject(C, y, st, k, shrink, kind)
            rows.append(dict(dataset=ds, subject=subj, measure=f'csp2_{name}',
                             stat=obs, z=z, p=p, n_assignments=n_all))
        r = pd.DataFrame(rows)
        for ds, g in r.groupby('dataset'):
            r.loc[g.index, 'q'] = bh_fdr(g['p'].values)
        res[name] = r
    pd.concat(res.values()).to_csv(f'{OUT_DIR}/csp2_exact.csv', index=False)

    print('\n' + '=' * 90)
    print('CSP (epoch-pair estimation, PCA-reduced, validated configuration)')
    print('=' * 90)
    print(f'{"":30}{"n":>4}{"z mean":>9}{"  95% CI":>18}{"t":>8}'
          f'{"p (1t)":>9}{"pos":>8}{"sign p":>10}')
    for ds, lab in DS:
        a = res['diff'][res['diff'].dataset == ds]
        b = res['mean'][res['mean'].dataset == ds]
        if a.empty:
            continue
        print(f'\n  {lab}')
        print('    ' + _line('CSP, discrimination', a['z'].values))
        print('    ' + _line('CSP, control (mean-ERP)', b['z'].values))
        x = a.set_index('subject')['z']; yv = b.set_index('subject')['z']
        i = x.index.intersection(yv.index)
        d = (x[i] - yv[i]).values
        t, p = ttest_rel(x[i], yv[i])
        _, pw = wilcoxon(d, alternative='greater')
        print('    ' + _line('CSP  signal - control', d) + f'   Wilcoxon p={pw:.4f}')
    print(f'\nwrote {OUT_DIR}/csp2_exact.csv')


if __name__ == '__main__':
    main()
