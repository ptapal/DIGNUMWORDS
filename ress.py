"""
PRE-SPECIFIED 12-09-2026, before any result from this file was computed.

RESS (Cohen & Gulbinaite, 2017, NeuroImage 147:43-56), frequency-domain form.
Filter = leading generalised eigenvector of
  S  cross-spectral matrix at the discrimination harmonics 3.75/11.25/18.75 Hz
  R  mean cross-spectral matrix over +-20 neighbouring bins, +-2-bin guard
Rectangular window (both datasets are effectively integer-cycle).
Control: the same at the symmetry harmonics 7.5/15 Hz.
Fixed ROI scored with the SAME SNR, using equal weights on the 8 Retter
channels, so learned and fixed spatial weightings are compared like for like.

Filters are fitted leave-one-block-out and never see condition labels, so
per-block scores are fixed under permutation and the stratified test is exact.

GATE (identical to csp2.py, fixed before any sweep)
calibration  null p median in [0.40, 0.60], fraction < .05 in [0.02, 0.10]
power        planted rank-1 occipital component at 0.5 x mean channel power
             detected at GROUP-level p < .05 in at least one dataset
sweep        shrinkage in {0.01, 0.05, 0.15} only

READING
RESS separates signal from control  a frequency-tagging filter beats the ROI
RESS fails                          condition-blind filters target the shared
                                    response, now shown with the field's own
                                    method

    python ress.py --build      one CSV pass
    python ress.py --validate   calibration + power on synthetic
    python ress.py --run        real labels; refused unless validate passed
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from io import StringIO
from scipy.linalg import eigh
from scipy.stats import norm, ttest_rel, ttest_1samp, binomtest, t as tdist

from rebuild_seqlevel import (load_sequences, parse_angelique_fname,
                              parse_talia_fname, _get_talia_files, N_ELEC)
from epoch_analysis import SR, ANG_CSV_DIR, TAL_CSV_DIR
from block_level_test import (enumerate_stratified, signed_stat, bh_fdr,
                              EXCLUDE, OUT_DIR)
from electrodes import RETTER_ROI_IDX

CACHE = 'h5_new/ress_cache.npz'
PASSED = 'h5_new/ress_validated.json'
DISC_HZ = (3.75, 11.25, 18.75)
SYM_HZ = (7.5, 15.0)
GUARD = 2
N_NEIGH = 20
DS = [('Angelique', 'D1'), ('Talia', 'D2')]

CAL_MEDIAN = (0.40, 0.60)
CAL_TAIL = (0.02, 0.10)
POWER_BOOST = 0.5
POWER_ALPHA = 0.05
SHRINKS = (0.01, 0.05, 0.15)


def spectral_mats(eeg, freqs):
    """(N, 68) -> S, R cross-spectral matrices summed over `freqs`."""
    x = np.asarray(eeg, dtype=np.float64)
    N = x.shape[0]
    X = np.fft.rfft(x, axis=0)
    S = np.zeros((N_ELEC, N_ELEC))
    R = np.zeros((N_ELEC, N_ELEC))
    for f in freqs:
        k = int(round(f * N / SR))
        S += np.real(np.outer(X[k], X[k].conj()))
        nb = np.r_[k - GUARD - N_NEIGH:k - GUARD, k + GUARD + 1:k + GUARD + 1 + N_NEIGH]
        Xn = X[nb]
        R += np.real(Xn.T @ Xn.conj()) / len(nb)
    return (S / N).astype(np.float32), (R / N).astype(np.float32)


def build_cache(path=CACHE):
    rows, Sd, Rd, Ss, Rs = [], [], [], [], []
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
            font = f'{ftype}/{sfont}' if sfont else ftype
            for s in sorted(seqs):
                a, b = spectral_mats(seqs[s][:, :N_ELEC], DISC_HZ)
                c, d = spectral_mats(seqs[s][:, :N_ELEC], SYM_HZ)
                rows.append(dict(dataset=ds, subject=f'S{subj}', modality=mod,
                                 font=font, condition=cond, block_id=fname, seq=s))
                Sd.append(a); Rd.append(b); Ss.append(c); Rs.append(d)
            if (i + 1) % 40 == 0:
                print(f'  [{i+1}/{len(files)}] {len(rows)} sequences', flush=True)
    meta = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, Sd=np.stack(Sd), Rd=np.stack(Rd), Ss=np.stack(Ss),
                        Rs=np.stack(Rs), meta=meta.to_csv(index=False))
    print(f'wrote {path}  ({len(meta)} sequences)')


def load_cache(path=CACHE):
    z = np.load(path, allow_pickle=True)
    return (pd.read_csv(StringIO(str(z['meta']))),
            z['Sd'], z['Rd'], z['Ss'], z['Rs'])


def prepare(meta, arrs, exclude):
    """{(ds, subj): (dict of per-block matrices, y, strata)}."""
    meta = meta[~meta.subject.isin(exclude)] # index stays aligned
    out = {}
    for (ds, subj), g in meta.groupby(['dataset', 'subject']):
        blk = {k: [] for k in arrs}
        y, st = [], []
        for _, gg in g.groupby('block_id'):
            for k, a in arrs.items():
                blk[k].append(a[gg.index].astype(np.float64).mean(0))
            r = gg.iloc[0]
            y.append(1 if r.condition == 'Par' else 0)
            st.append(f'{r.modality}|{r.font}')
        out[(ds, subj)] = ({k: np.array(v) for k, v in blk.items()},
                           np.array(y), np.array(st))
    return out


def ress_filter(S, R, shrink):
    R2 = R + shrink * np.trace(R) / N_ELEC * np.eye(N_ELEC)
    _, vecs = eigh(S, R2)
    return vecs[:, -1]


def _snr(w, S, R):
    return float(np.log((w @ S @ w) / (w @ R @ w + 1e-300) + 1e-300))


W_ROI = np.zeros(N_ELEC)
W_ROI[RETTER_ROI_IDX] = 1.0


def block_scores(S, R, shrink):
    """Leave-one-block-out RESS score, and the fixed-ROI score, per block."""
    n = len(S)
    ress, roi = np.empty(n), np.empty(n)
    for b in range(n):
        tr = [i for i in range(n) if i != b]
        w = ress_filter(S[tr].mean(0), R[tr].mean(0), shrink)
        ress[b] = _snr(w, S[b], R[b])
        roi[b] = _snr(W_ROI, S[b], R[b])
    return ress, roi


def _test(v, y, st):
    return enumerate_stratified(lambda m, v=v: signed_stat(v, m), y, st)


def _plant(S, y, boost, seed=0):
    rng = np.random.default_rng(seed)
    v = np.zeros(N_ELEC)
    v[[24, 26, 61, 63]] = 1.0 # PO7, O1, PO8, O2
    v += 0.15 * rng.normal(size=N_ELEC)
    v /= np.linalg.norm(v)
    scale = np.trace(S.mean(0)) / N_ELEC
    S2 = S.copy()
    S2[y.astype(bool)] += boost * scale * np.outer(v, v)
    return S2


def _group_p(ps):
    pc = np.clip(np.array(ps), 1e-9, 1 - 1e-9)
    return float(norm.sf(norm.isf(pc).sum() / np.sqrt(len(pc))))


def validate(prep):
    rng = np.random.default_rng(0)
    print(f'acceptance: null median in {CAL_MEDIAN}, tail in {CAL_TAIL}; '
          f'group p < {POWER_ALPHA} at boost {POWER_BOOST}\n')
    print(f'{"shrink":>8}{"null med":>10}{"null<.05":>10}{"D1 grp p":>10}'
          f'{"D2 grp p":>10}   verdict')
    best = None
    for sh in SHRINKS:
        nulls, planted = [], {}
        for (ds, subj), (B, y, st) in prep.items():
            ress, _ = block_scores(B['Sd'], B['Rd'], sh)
            yy = y.copy()
            for u in np.unique(st):
                i = np.where(st == u)[0]
                yy[i] = rng.permutation(y[i])
            nulls.append(_test(ress, yy, st)[1])
            r2, _ = block_scores(_plant(B['Sd'], y, POWER_BOOST), B['Rd'], sh)
            planted.setdefault(ds, []).append(_test(r2, y, st)[1])
        nulls = np.array(nulls)
        med, tail = np.median(nulls), np.mean(nulls < .05)
        gp = {ds: _group_p(ps) for ds, ps in planted.items()}
        ok = (CAL_MEDIAN[0] <= med <= CAL_MEDIAN[1]
              and CAL_TAIL[0] <= tail <= CAL_TAIL[1]
              and min(gp.values()) < POWER_ALPHA)
        print(f'{sh:>8.2f}{med:>10.3f}{tail:>10.3f}{gp.get("Angelique", np.nan):>10.4f}'
              f'{gp.get("Talia", np.nan):>10.4f}   {"PASS" if ok else "fail"}', flush=True)
        if ok and (best is None or min(gp.values()) < best[1]):
            best = (sh, min(gp.values()))
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
    if '--build' in sys.argv:
        build_cache()
        return
    meta, Sd, Rd, Ss, Rs = load_cache()
    prep = prepare(meta, dict(Sd=Sd, Rd=Rd, Ss=Ss, Rs=Rs), set(EXCLUDE))

    if '--validate' in sys.argv:
        best = validate(prep)
        if best is None:
            print('\nNOTHING PASSED. Not fit to run on real labels.')
            if os.path.exists(PASSED):
                os.remove(PASSED)
            return
        json.dump(dict(shrink=best[0], group_p=best[1]), open(PASSED, 'w'))
        print(f'\nPASSED: shrink={best[0]}  -> {PASSED}')
        return

    if '--run' not in sys.argv:
        print('use --build, --validate, then --run')
        return
    if not os.path.exists(PASSED):
        print('REFUSED: run --validate first.')
        return
    sh = json.load(open(PASSED))['shrink']
    print(f'validated shrink={sh}')

    rows = []
    for (ds, subj), (B, y, st) in sorted(prep.items()):
        rd, od = block_scores(B['Sd'], B['Rd'], sh)
        rs, os_ = block_scores(B['Ss'], B['Rs'], sh)
        for name, v in [('ress_disc', rd), ('ress_sym', rs),
                        ('roi_disc', od), ('roi_sym', os_)]:
            _, p, _, z, n_all, _ = _test(v, y, st)
            rows.append(dict(dataset=ds, subject=subj, measure=name, z=z, p=p,
                             n_assignments=n_all))
    r = pd.DataFrame(rows)
    r.to_csv(f'{OUT_DIR}/ress_exact.csv', index=False)

    print('\n' + '=' * 84)
    print('RESS vs FIXED ROI  (same spectral SNR, leave-one-block-out, label-free)')
    print('=' * 84)
    for ds, lab in DS:
        g = r[r.dataset == ds]
        if g.empty:
            continue
        z = {m: g[g.measure == m].set_index('subject')['z'] for m in g.measure.unique()}
        print(f'\n  {lab}')
        for m, nm in [('ress_disc', 'RESS, discrimination'), ('ress_sym', 'RESS, symmetry control'),
                      ('roi_disc', 'fixed ROI, discrimination'), ('roi_sym', 'fixed ROI, symmetry control')]:
            print('    ' + _line(nm, z[m].values))
        for a, b, nm in [('ress_disc', 'ress_sym', 'RESS  signal - control'),
                         ('roi_disc', 'roi_sym', 'ROI   signal - control')]:
            i = z[a].index.intersection(z[b].index)
            print('    ' + _line(nm, (z[a][i] - z[b][i]).values))
    print(f'\nwrote {OUT_DIR}/ress_exact.csv')


if __name__ == '__main__':
    main()
