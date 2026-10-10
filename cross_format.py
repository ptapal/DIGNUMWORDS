"""
PRE-SPECIFIED 23-09-2026, before any result from this file was computed.

QUESTION
Parity and Control regroup the same eight numerals, so the discrimination
response may carry item-level visual differences. Does a template learned in
one notation predict the condition of blocks in the OTHER notation? Digits
and German number words share parity and share no visual form, so transfer
cannot be explained by the appearance of the items.

DATA
D1 only (the sole dataset with both notations). Per participant 12 blocks:
2 notations x 3 font cells x {Parity, Control}, i.e. 6 digit and 6 word
blocks. Source: csp2 cache sub-averages, discrimination arm (diff-ERP) as
primary, control arm (mean-ERP) reported. D2 is digits only and is not used.

OPERATORS (both already used elsewhere, nothing tuned here)
roi_tmpl  ROI-mean channel waveform (mean of the 8 Retter channels), template
          projection: train templates m1 (Parity) and m0 (Control) on the
          training blocks, score a test block h by
              (h - (m1 + m0)/2) . (m1 - m0) / ||m1 - m0||,
          sign-flipped when the test block is Control. This is tdca_ablation
          A1 with the training set chosen by the analysis below instead of
          leave-one-block-out.
tdca      locked TDCA from h5_new/tdca_validated.json (k=15, L=4, shrink .02,
          3 components): PCA and delay embedding fitted on the training
          blocks, discriminant GED and midpoint projection exactly as
          tdca.fold_decision, with the training set chosen by the analysis.

ANALYSES (per participant, per operator, per arm)
cross_d2w   train on the 6 digit blocks, score each of the 6 word blocks
cross_w2d   train on the 6 word blocks, score each of the 6 digit blocks
within_dig  leave one block out inside digits (train 5, score 1)
within_wrd  leave one block out inside words
font_1to20  train on the 1-font cell, score the 20-font cells, within notation
            (digits and words separately, then pooled)
font_20to1  the reverse
within_* is the matched benchmark for cross_*: same operator, same test
blocks, training set of similar size, but training and test in one notation.
font_* is a positive control for the transfer machinery: same notation,
different visual appearance. If font transfer works and cross-notation
transfer does not, the failure is about notation, not about power.

SCALING (both reported, neither is primary alone)
raw     block representations used as they are
unit    every block representation scaled to unit norm before training and
        scoring, so a notation difference in overall response size cannot
        drive the transfer score. Words subtend a larger visual angle and
        show larger asymmetry responses in D1, so the raw version is
        confounded by amplitude; the unit version is the interpretable one.

INFERENCE
Labels of ALL 12 blocks are permuted within their font x notation cells and
every step (templates, PCA, GED, scores) is refitted, exactly as elsewhere:
2^6 = 64 assignments, tie-corrected mid-p, per-participant z
(block_level_test.enumerate_stratified). The statistic is invariant to
swapping every label (template and score both flip, the sign correction
flips with them), so the attainable minimum p is 1/64 = 0.0156.
Group: mean z over the 30 participants, 95% CI, one-sample t, two-sided.

PRIMARY
D1, discrimination arm, unit scaling, roi_tmpl:
  cross_d2w z > 0 and cross_w2d z > 0, Holm over the two directions.
Secondary, reported in full: tdca; the raw scaling; the control arm; the
within-notation benchmarks; the font-transfer controls; the paired contrast
cross - within per direction; the transfer index (group mean cross / group
mean within, with a bootstrap interval over participants).

GATE (before real labels)
calibration  20 stratified label shuffles per participant (rng seed 0),
             discrimination arm, unit scaling: pooled null mid-p median in
             [0.40, 0.60] and fraction < .05 in [0.02, 0.10], per operator.
recovery     a planted spatiotemporal effect (tdca._plant, boost 0.5) added
             to the blocks labelled Parity by ONE stratified shuffle per
             participant (rng seed 1), in BOTH notations, must be detected by
             cross_d2w against that shuffle at group p < .05, per operator.
specificity  the same plant added to the DIGIT blocks only must NOT be
             detected (group p >= .05). A transfer test that fires on a
             notation-specific effect cannot support an abstraction claim.
             Plants are tested against the shuffled labels, never the real
             ones, so the gate carries no real-label information.

PRECONDITION (must be verified before the real-label run, outside this file)
The epoch parity convention must be the same in digit and word files: set A
must fall on the same epoch index parity in both notations. If it flips
between notations, a shared response appears as failed transfer and a
notation-specific response can appear as transfer. Check the trigger or
stimulus logs.

READING 
both directions > 0            a component of the response is shared across
                               notation; item-level visual differences cannot
                               be the whole account
one direction only             asymmetric generalisation; reported, not read
                               as abstraction
cross ~ 0 while within > 0
  and font transfer > 0        the response is notation-specific
cross ~ 0 and font transfer
  also ~ 0                     the procedure is underpowered here; no claim
                               in either direction

    python cross_format.py --gate      
    python cross_format.py --run       
    python cross_format.py --summary   
"""
import os
os.chdir(os.path.dirname(os.path.abspath(__file__))) # repo-relative paths
import sys
import json
import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scipy.stats import ttest_1samp, t as tdist

from csp2 import load_cache
from tdca import _embed, fold_decision, _plant
from block_level_test import enumerate_stratified, EXCLUDE, OUT_DIR
from electrodes import RETTER_ROI_IDX
from operator_vs_roi import holm

CFG = json.load(open('h5_new/tdca_validated.json'))
PASSED = 'h5_new/cross_format_validated.json'
DATASET = 'Angelique'
DIG, WRD = 'Dig', 'NumWoGE'
N_SHUF = 20
CAL_MEDIAN = (0.40, 0.60)
CAL_TAIL = (0.02, 0.10)
PLANT_BOOST = 0.5
POWER_ALPHA = 0.05
OPS = ('roi_tmpl', 'tdca')
SCALINGS = ('unit', 'raw')


def prepare(meta, sub, exclude=EXCLUDE):
    """{subject: (blocks, y, strata, modality)} for D1, blocks in block_id order."""
    meta = meta[(meta.dataset == DATASET) & (~meta.subject.isin(exclude))]
    out = {}
    for subj, g in meta.groupby('subject'):
        B, y, st, mod = [], [], [], []
        for _, gg in g.groupby('block_id'):
            b = sub[gg.index].astype(np.float64) # (n_seq, S, T, C)
            B.append(b.reshape(-1, b.shape[2], b.shape[3]))
            r = gg.iloc[0]
            y.append(1 if r.condition == 'Par' else 0)
            st.append(f'{r.modality}|{r.font}')
            mod.append(r.modality)
        out[subj] = (B, np.array(y), np.array(st), np.array(mod))
    return out


def _sets(mod, st, analysis):
    """(train indices, test indices) for one analysis; None if not available."""
    dig = np.where(mod == DIG)[0]
    wrd = np.where(mod == WRD)[0]
    one = np.array([i for i in range(len(st)) if st[i].split('|')[1] == '1F'])
    if analysis == 'cross_d2w':
        return (dig, wrd) if len(dig) and len(wrd) else None
    if analysis == 'cross_w2d':
        return (wrd, dig) if len(dig) and len(wrd) else None
    if analysis in ('within_dig', 'within_wrd'):
        idx = dig if analysis == 'within_dig' else wrd
        return (idx, idx) if len(idx) > 2 else None # leave-one-out inside idx
    if analysis in ('font_1to20', 'font_20to1'):
        pairs = []
        for m, idx in ((DIG, dig), (WRD, wrd)):
            a = np.array([i for i in idx if i in one])
            b = np.array([i for i in idx if i not in one])
            if len(a) >= 2 and len(b) >= 2:
                pairs.append((a, b) if analysis == 'font_1to20' else (b, a))
        return pairs or None
    raise ValueError(analysis)



def _roi_waveforms(B, scaling):
    """One 64-sample ROI-mean waveform per block."""
    W = [b[:, :, RETTER_ROI_IDX].mean(2).mean(0) for b in B]
    if scaling == 'unit':
        W = [w / (np.linalg.norm(w) + 1e-300) for w in W]
    return np.array(W)


def roi_stat(B, train, test, scaling):
    """Template projection, training set given. Returns f(mask) -> statistic."""
    W = _roi_waveforms(B, scaling)
    tr_mask = np.zeros(len(B), bool); tr_mask[train] = True

    def stat(m):
        s = []
        for i in test:
            tr = tr_mask.copy(); tr[i] = False # never train on the block being scored
            m1, m0 = tr & m, tr & ~m
            if not m1.any() or not m0.any():
                continue
            t1, t0 = W[m1].mean(0), W[m0].mean(0)
            d = t1 - t0
            nd = np.linalg.norm(d) + 1e-300
            s.append((W[i] - 0.5 * (t1 + t0)) @ d / nd * (1.0 if m[i] else -1.0))
        return float(np.mean(s)) if s else 0.0
    return stat


def _fold_stats(B, train, held, k, L):
    """tdca.fold_stats with an explicit training set (which may exclude the test block)."""
    C = sum(np.einsum('itc,itd->cd', B[i], B[i]) for i in train)
    P = eigh(C)[1][:, -k:]
    sumZ, sumZZ, n = [], [], []
    for i in range(len(B)):
        Zt = _embed(B[i] @ P, L)
        sumZ.append(Zt.sum(0))
        sumZZ.append(np.einsum('itd,ite->de', Zt, Zt))
        n.append(len(Zt))
    tr = np.zeros(len(B), bool); tr[train] = True
    tr[held] = False # never train on the test block
    return dict(tr=tr, sumZ=np.array(sumZ), sumZZ=np.array(sumZZ), n=np.array(n),
                H=_embed(B[held] @ P, L).mean(0), held=held)


def tdca_stat(B, train, test, scaling, k=CFG['k'], L=CFG['L'], shrink=CFG['shrink']):
    """Locked TDCA, training set given. Returns f(mask) -> statistic."""
    X = B
    if scaling == 'unit':
        X = [b / (np.sqrt(np.mean(b.mean(0) ** 2)) + 1e-300) for b in B]
    folds = [_fold_stats(X, train, h, k, L) for h in test]
    memo = {}

    def stat(m):
        key = m.tobytes()
        if key not in memo:
            memo[key] = float(np.mean([fold_decision(f, m, shrink) for f in folds]))
        return memo[key]
    return stat


def make_stat(B, train, test, op, scaling):
    return (roi_stat if op == 'roi_tmpl' else tdca_stat)(B, train, test, scaling)


def analysis_stat(B, st, mod, analysis, op, scaling):
    """One statistic per analysis; font_* averages its notation-specific parts."""
    sets = _sets(mod, st, analysis)
    if sets is None:
        return None
    parts = sets if isinstance(sets, list) else [sets]
    fns = [make_stat(B, tr, te, op, scaling) for tr, te in parts]
    return lambda m: float(np.mean([f(m) for f in fns]))




def _shuffle(y, st, rng):
    yy = y.copy()
    for u in np.unique(st):
        i = np.where(st == u)[0]
        yy[i] = rng.permutation(y[i])
    return yy


def describe(v):
    v = np.asarray(v, dtype=float); n = len(v)
    ci = tdist.ppf(.975, n - 1) * v.std(ddof=1) / np.sqrt(n)
    t, p = ttest_1samp(v, 0.0)
    return dict(n=n, mean=v.mean(), lo=v.mean() - ci, hi=v.mean() + ci, t=t, p=p,
                pos=int((v > 0).sum()))


def fmt(label, s, extra=''):
    return (f'  {label:44}{s["n"]:>4}{s["mean"]:>+8.3f}  [{s["lo"]:+.2f},{s["hi"]:+.2f}]'
            f'{s["t"]:>+7.2f}{s["p"]:>9.4f}{s["pos"]:>4}/{s["n"]}{extra}')


ANALYSES = ('cross_d2w', 'cross_w2d', 'within_dig', 'within_wrd',
            'font_1to20', 'font_20to1')


def run_real(prep):
    rows = []
    for arm in prep:
        for subj, (B, y, st, mod) in sorted(prep[arm].items()):
            for op in OPS:
                for sc in SCALINGS:
                    for an in ANALYSES:
                        f = analysis_stat(B, st, mod, an, op, sc)
                        if f is None:
                            continue
                        _, p, _, z, n_all, _ = enumerate_stratified(f, y, st)
                        rows.append(dict(subject=subj, arm=arm, op=op, scaling=sc,
                                         analysis=an, z=z, p=p, n_assignments=n_all))
            print(f'  {arm} {subj}', flush=True)
    return pd.DataFrame(rows)


def gate(prep):
    """Calibration, recovery of a plant in both notations, specificity to one."""
    rng = np.random.default_rng(0)
    report, ok_all = [], True
    for op in OPS:
        cal = []
        for subj, (B, y, st, mod) in sorted(prep['disc'].items()):
            f = analysis_stat(B, st, mod, 'cross_d2w', op, 'unit')
            for _ in range(N_SHUF):
                cal.append(enumerate_stratified(f, _shuffle(y, st, rng), st)[1])
        cal = np.asarray(cal)
        med, tail = float(np.median(cal)), float((cal < .05).mean())
        cal_ok = CAL_MEDIAN[0] <= med <= CAL_MEDIAN[1] and CAL_TAIL[0] <= tail <= CAL_TAIL[1]

        prng = np.random.default_rng(1)
        z_both, z_dig = [], []
        for subj, (B, y, st, mod) in sorted(prep['disc'].items()):
            ys = _shuffle(y, st, prng)
            both = _plant(B, ys, PLANT_BOOST)
            only = _plant(B, ys * (mod == DIG), PLANT_BOOST)
            z_both.append(enumerate_stratified(
                analysis_stat(both, st, mod, 'cross_d2w', op, 'unit'), ys, st)[3])
            z_dig.append(enumerate_stratified(
                analysis_stat(only, st, mod, 'cross_d2w', op, 'unit'), ys, st)[3])
        p_both = describe(z_both)['p'] / 2
        p_dig = describe(z_dig)['p'] / 2
        rec_ok, spec_ok = p_both < POWER_ALPHA, p_dig >= POWER_ALPHA
        ok = cal_ok and rec_ok and spec_ok
        ok_all &= ok
        print(f'  {op:9} calibration median {med:.3f} tail {tail:.3f} '
              f'{"PASS" if cal_ok else "FAIL"} | plant both notations p {p_both:.4f} '
              f'{"PASS" if rec_ok else "FAIL"} | plant digits only p {p_dig:.4f} '
              f'{"PASS" if spec_ok else "FAIL"}', flush=True)
        report.append(dict(op=op, null_median=float(med), null_tail=float(tail),
                           calibration=bool(cal_ok), plant_both_p=float(p_both),
                           recovery=bool(rec_ok), plant_digits_p=float(p_dig),
                           specificity=bool(spec_ok), passed=bool(ok)))
    pd.DataFrame(report).to_csv(f'{OUT_DIR}/cross_format_gate.csv', index=False)
    return ok_all, report


def summarise(r=None):
    if r is None:
        r = pd.read_csv(f'{OUT_DIR}/cross_format_exact.csv')
    head = (f'  {"":44}{"n":>4}{"mean z":>8}  {"95% CI":^15}{"t":>7}{"p":>9}{"pos":>7}')
    out = []
    for sc in SCALINGS:
        for arm in ('disc', 'ctrl'):
            tag = 'PRIMARY' if (sc == 'unit' and arm == 'disc') else 'secondary'
            print(f'\n{tag}: {sc} scaling, {arm} arm\n' + head)
            for op in OPS:
                g = r[(r.scaling == sc) & (r.arm == arm) & (r.op == op)]
                if g.empty:
                    continue
                cross = []
                for an in ANALYSES:
                    v = g[g.analysis == an].z
                    if v.empty:
                        continue
                    s = describe(v.values)
                    if an.startswith('cross'):
                        cross.append((an, s))
                    print(fmt(f'{op:9} {an}', s))
                    out.append(dict(table='alone', scaling=sc, arm=arm, op=op,
                                    analysis=an, **s))
                if len(cross) == 2:
                    for (an, s), h in zip(cross, holm([s['p'] for _, s in cross])):
                        out.append(dict(table='primary', scaling=sc, arm=arm, op=op,
                                        analysis=an, p_holm=h, **s))
                        print(f'  {op:9} {an:14} Holm p = {h:.4f}')
                for cr, wi in (('cross_d2w', 'within_wrd'), ('cross_w2d', 'within_dig')):
                    a = g[g.analysis == cr].set_index('subject').z
                    b = g[g.analysis == wi].set_index('subject').z
                    i = a.index.intersection(b.index)
                    if len(i) < 3:
                        continue
                    s = describe((a[i] - b[i]).values)
                    print(fmt(f'{op:9} {cr} - {wi}', s))
                    out.append(dict(table='cross_minus_within', scaling=sc, arm=arm,
                                    op=op, analysis=f'{cr} - {wi}', **s))
                    idx = a[i].mean() / b[i].mean() if abs(b[i].mean()) > 1e-9 else np.nan
                    print(f'  {op:9} transfer index {cr} / {wi} = {idx:+.2f}')
                    out.append(dict(table='transfer_index', scaling=sc, arm=arm, op=op,
                                    analysis=f'{cr} / {wi}', mean=idx))
    pd.DataFrame(out).to_csv(f'{OUT_DIR}/cross_format_summary.csv', index=False)
    print(f'\nwrote {OUT_DIR}/cross_format_summary.csv')


def main():
    meta, D, M = load_cache()
    prep = {'disc': prepare(meta, D), 'ctrl': prepare(meta, M)}
    n_sub = len(prep['disc'])
    print(f'D1: {n_sub} participants, '
          f'{sorted({len(v[0]) for v in prep["disc"].values()})} blocks each')

    if '--gate' in sys.argv:
        print('GATE (shuffled labels and planted effects only)')
        ok, report = gate(prep)
        if ok:
            json.dump(dict(report=report, boost=PLANT_BOOST, shuffles=N_SHUF, seed=0,
                           passed=True), open(PASSED, 'w'), indent=1)
            print(f'\nPASSED -> {PASSED}')
        else:
            if os.path.exists(PASSED):
                os.remove(PASSED)
            print('\nFAILED. Not fit to run on real labels.')
        return

    if '--summary' in sys.argv:
        return summarise()

    if '--run' not in sys.argv:
        print(__doc__.strip().splitlines()[-3])
        return
    if not os.path.exists(PASSED):
        print('REFUSED: run --gate first.')
        return
    r = run_real(prep)
    r.to_csv(f'{OUT_DIR}/cross_format_exact.csv', index=False)
    print(f'wrote {OUT_DIR}/cross_format_exact.csv')
    summarise(r)


if __name__ == '__main__':
    main()
