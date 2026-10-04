"""
ablation_case_studies.py
========================

Ablation of the pruning method on the controlled case studies of
``emergentSR.emergent.create_model``.  Each case study is built so that a
specific move must fire, and the ground truth ``f_true`` / ``y_true`` is known,
so every method is scored by its agreement with the truth (not only by
held-out risk):

    nested                                      spurious-term removal
    bessel / competing_saturations              saturation at infinity
    rational_gauge / separable_2d               signed identification / gauges
    perturbative / multidimensional /
    near_alias_fourier                          mixed sloppiness
    buried_transcendentals                      theta -> +/-inf on coefficient-
                                                less terms
    tanh_gauge                                  signed identification

For every (case study x noise level x seed) five methods are run from an
identical fitted starting model:

    full        4 Fisher modes, looped (CV gate + BIC tie-break)
    sloppy_only strict MBAM/BR single smallest-eigenvalue direction, CV gate
    zero_only   Wald-to-zero over all parameters, CV gate
    magnitude   naive floor: relative-scale |theta| threshold, test-free
    no_cv_bic   the 4 Fisher modes with a BIC-only gate (dBIC < 3)

Configuration: see ``make_case_config`` (n=100, beta=0.1, cv_equiv_k=2,
lambda_threshold=-10, lambda_gap_threshold=25, snr_sq=16).

Outputs (under ./results_ablation_cases/)
    ablation_cases_tidy_<seed>.csv              one row per (case x noise x method)
    ablation_cases_<model>_<noise>_<seed>.pkl   per-case raw dict

Usage
-----
    python ablation_case_studies.py                 # all case studies, 30 seeds
    python ablation_case_studies.py nested bessel   # a subset of case studies
"""

import os
os.environ.setdefault("MPLBACKEND", "Agg")

import sys
import time
import pickle
import traceback

import numpy as np
import pandas as pd
import sympy as sp
from joblib import (
    Parallel,
    delayed,
    parallel_config,
    cpu_count,
    effective_n_jobs,
)

from emergentSR.emergent import *
from emergentSR.ablation_variants import ABLATION_VARIANTS


CASE_RESULTS_DIR = "./results_ablation_cases/"

CASE_MODELS = [
    "perturbative", "multidimensional", 
    "near_alias_fourier", "bessel",
    "rational_gauge", "competing_saturations", 
    "separable_2d", "nested",
    'tanh_gauge', 'buried_transcendentals',
]

# noise standard deviations per case study (same values as the tutorial notebook)
HIGH_NOISE = dict(nested=1.5, perturbative=0.2, multidimensional=0.2,
                  near_alias_fourier=0.2, bessel=0.3, rational_gauge=0.2,
                  competing_saturations=0.1, separable_2d=0.1,
                  tanh_gauge=0.15, buried_transcendentals=1.5)
LOW_NOISE = dict(nested=0.5, perturbative=0.05, multidimensional=0.05,
                 near_alias_fourier=0.05, bessel=0.05, rational_gauge=0.01,
                 competing_saturations=0.01, separable_2d=0.01,
                tanh_gauge=0.05, buried_transcendentals=0.20)
NOISE = {"high": HIGH_NOISE, "low": LOW_NOISE}

VARIANTS = ["full", "sloppy_only", "zero_only", "magnitude", "no_cv_bic"]
BASELINES = ["sloppy_only", "zero_only", "magnitude", "no_cv_bic"]


def make_case_config(seed):
    """Hyperparameters of the case-study runs (shared by all five methods)."""
    return dict(
        criterion="CV", second_crit="BIC", seed=seed,
        init_loc=0.0, init_scale=0.1, def_bounds=[(-100, 100)],
        snr_sq=16, n=100,
        lambda_threshold=-10, lambda_gap_threshold=25.0,
        beta=0.1, cv_equiv_k=2,
        n_starts=30, maxiter=5000,
        ident_threshold=0.01, selection_tol=1e-2,
        magnitude_frac=0.01, magnitude_scale="max", bic_accept_band=3.0,
        cancel_removable=False,
    )


# ============================================================
# firing tally (same categories as operon_ablation)
# ============================================================
def _n_nodes(expr):
    """Complexity = number of nodes in the sympy expression tree."""
    try:
        return sum(1 for _ in sp.preorder_traversal(expr))
    except Exception:
        return -1


def _tally_moves(history):
    t = dict(total=0, zero=0, one=0, inf=0, signed_ident=0)
    for h in (history or []):
        if h.get("action") != "removed":
            continue
        t["total"] += 1
        b = str(h.get("boundary") or "")
        if b == "zero":
            t["zero"] += 1
        elif b == "one":
            t["one"] += 1
        elif b in ("+inf", "-inf"):
            t["inf"] += 1
        elif "=" in b:
            t["signed_ident"] += 1
    return t


# ============================================================
# shared per-case setup (fit once)
# ============================================================
def _setup_case(model, noise_level, cfg):
    np.random.seed(cfg["seed"])
    sigma_true = NOISE[noise_level][model]
    x, y, model_sym, xs, params, f_true, y_true = create_model(model, cfg["n"], sigma_true)

    theta_mle, sigma_noise, model_func, jac_funcs = compute_lambdify_and_mle_estimates(
        model_sym, params, xs, x, y, cfg["def_bounds"], cfg["init_loc"], cfg["init_scale"],
        n_starts=cfg["n_starts"], maxiter=cfg["maxiter"],
    )
    guess_cur = [float(t) for t in theta_mle] if isinstance(theta_mle, np.ndarray) else None
    cv0, _ = kfold_cv_mse_for_sympy_model(
        model_sym=model_sym, params=params, xs=xs, x=x, y=y,
        def_bounds=cfg["def_bounds"], init_loc=cfg["init_loc"], init_scale=cfg["init_scale"],
        guess_init=guess_cur, return_se=True, verbose=False,
    )
    bic0 = bic_for_sympy_model(params, model_func, theta_mle, x, y, beta=1)
    df = compute_eigenvecs_eigenvals_and_alignment(
        params, jac_funcs, theta_mle, x, y, sigma_noise, model_func
    ).sort_values(by="ln(lambda)")

    base_truth = float(np.mean((model_func(*(tuple(theta_mle) + tuple(x))) - y_true) ** 2))
    return dict(
        model_sym=model_sym, params=params, model_func=model_func, jac_funcs=jac_funcs,
        theta=theta_mle, sigma=sigma_noise, df=df, xs=xs, x=x, y=y, y_true=y_true,
        f_true=f_true, cv=cv0, bic=bic0, base_truth=base_truth,
    )


def _run_full_cases(st0, cfg, history):
    """Full method: loop the four Fisher modes until a whole pass removes
    nothing, recording every accepted move in ``history``."""
    st = dict(model_sym=st0["model_sym"], params=list(st0["params"]),
              model_func=st0["model_func"], jac_funcs=st0["jac_funcs"],
              theta=np.array(st0["theta"], dtype=float), sigma=st0["sigma"],
              df=st0["df"].copy(), cv=st0["cv"], bic=st0["bic"])

    def _mode(mode):
        return remove_and_recalibrate_sloppy_parameters(
            st["model_sym"], st["params"], st["model_func"], st["jac_funcs"], st["theta"],
            st0["xs"], st0["x"], st0["y"], st["sigma"], st["df"],
            cfg["def_bounds"], cfg["init_loc"], cfg["init_scale"], cfg["beta"],
            cfg["criterion"], st["cv"], cfg["second_crit"], st["bic"],
            None, snr_sq=cfg["snr_sq"], cv_equiv_k=cfg["cv_equiv_k"],
            lambda_threshold=cfg["lambda_threshold"],
            lambda_gap_threshold=cfg["lambda_gap_threshold"],
            cancel_removable=cfg["cancel_removable"],
            pruning_mode=mode, verbose=False, plot=False, history=history,
        )

    effectiveness = True
    while effectiveness:
        eff = 0
        for mode in ("unidentifiability",  "zero_compatibility",
                     "infinity_compatibility",
                     "signed_identifiability"):
            (ms, pr, mf, th, sg, jf, dff, cv, bic, e) = _mode(mode)
            st.update(model_sym=ms, params=pr, model_func=mf, theta=th,
                      sigma=sg, jac_funcs=jf, df=dff, cv=cv, bic=bic)
            eff += e
        effectiveness = bool(eff)

    return (st["model_sym"], st["params"], st["model_func"], st["theta"],
            st["sigma"], st["jac_funcs"], st["df"], st["cv"], st["bic"],
            len(st0["params"]) - len(st["params"]))


def _run_variant_case(name, st0, cfg, history):
    if name == "full":
        return _run_full_cases(st0, cfg, history)
    fn = ABLATION_VARIANTS[name]
    return fn(
        st0["model_sym"], list(st0["params"]), st0["model_func"], st0["jac_funcs"],
        np.array(st0["theta"], dtype=float),
        xs=st0["xs"], x=st0["x"], y=st0["y"], sigma_noise=st0["sigma"],
        df=st0["df"].copy(), def_bounds=cfg["def_bounds"],
        init_loc=cfg["init_loc"], init_scale=cfg["init_scale"],
        beta=cfg["beta"], criterion=cfg["criterion"],
        criterion_value=st0["cv"], second_crit=cfg["second_crit"],
        second_crit_value=st0["bic"], node_states=None, history=history, verbose=False,
        snr_sq=cfg["snr_sq"], cv_equiv_k=cfg["cv_equiv_k"],
        ident_threshold=cfg["ident_threshold"], selection_tol=cfg["selection_tol"],
        cancel_removable=cfg["cancel_removable"],
        magnitude_frac=cfg["magnitude_frac"], scale=cfg["magnitude_scale"],
        bic_accept_band=cfg["bic_accept_band"],
        # no_cv_bic mimics the full method's staged modes: it needs the SAME
        # eigenvalue masks (other variants absorb these via **kwargs)
        lambda_threshold=cfg["lambda_threshold"],
        lambda_gap_threshold=cfg["lambda_gap_threshold"],
    )


def _metrics(st0, out_tuple, history):
    (ms, pr, mf, th, sg, jf, dff, cv, bic, nrem) = out_tuple
    pred = mf(*(tuple(th) + tuple(st0["x"])))
    mse_truth = float(np.mean((pred - st0["y_true"]) ** 2))   # agreement with GROUND TRUTH
    mse_train = float(np.mean((pred - st0["y"]) ** 2))
    return dict(
        status="ok", n_removed=int(nrem), final_nparams=len(pr),
        final_complexity=int(_n_nodes(ms)),
        new_cv=float(cv) if cv is not None else np.nan,
        new_bic=float(bic) if bic is not None else np.nan,
        new_mse_truth=mse_truth, new_mse_train=mse_train,
        fired=_tally_moves(history), model=str(ms),
    )


# ============================================================
# One case study = full + four variants from a shared start
# ============================================================
def run_case(model, noise_level, cfg, keep_history=False):
    st0 = _setup_case(model, noise_level, cfg)
    out = dict(status="ok", model=model, noise=noise_level,
               f_true=str(st0["f_true"]), old_model=str(st0["model_sym"]),
               start_nparams=len(st0["params"]),
               old_complexity=int(_n_nodes(st0["model_sym"])),
               old_cv=float(st0["cv"]), old_bic=float(st0["bic"]),
               old_mse_truth=st0["base_truth"], variants={})
    for name in VARIANTS:
        t0 = time.time()
        history = init_pruning_history()
        try:
            res = _run_variant_case(name, st0, cfg, history)
            m = _metrics(st0, res, history)
        except Exception as e:
            m = dict(status="error", error=repr(e), traceback=traceback.format_exc())
        m["walltime_s"] = round(time.time() - t0, 2)
        if keep_history:
            m["history"] = history
        out["variants"][name] = m
    return out


# ============================================================
# Sweep + tidy table
# ============================================================
def run_all_cases(cfg, models=None, noises=("high", "low"), save=True, keep_history=False):
    models = models or CASE_MODELS
    os.makedirs(CASE_RESULTS_DIR, exist_ok=True)
    seed=cfg['seed']
    rows = []
    for model in models:
        for noise in noises:
            t0 = time.time()
            try:
                r = run_case(model, noise, cfg, keep_history=keep_history)
            except Exception as e:
                print(f"[{model}/{noise}] SETUP ERROR: {e!r}", flush=True)
                continue
            if save:
                fp = os.path.join(CASE_RESULTS_DIR, f"ablation_cases_{model}_{noise}_{seed}.pkl")
                with open(fp, "wb") as fh:
                    pickle.dump(r, fh)
            for name, m in r["variants"].items():
                if m.get("status") != "ok":
                    print(f"[{model}/{noise}] {name} ERROR: {m.get('error')}", flush=True)
                    continue
                f = m["fired"]
                rows.append(dict(
                    model=model, noise=noise, method=name,
                    start_nparams=r["start_nparams"], old_complexity=r["old_complexity"],
                    old_cv=r["old_cv"], old_mse_truth=r["old_mse_truth"],
                    n_removed=m["n_removed"], final_nparams=m["final_nparams"],
                    final_complexity=m["final_complexity"], new_cv=m["new_cv"],
                    new_bic=m["new_bic"], new_mse_truth=m["new_mse_truth"],
                    d_mse_truth=m["new_mse_truth"] - r["old_mse_truth"],
                    fired_zero=f["zero"], fired_one=f["one"], fired_inf=f["inf"],
                    fired_signed=f["signed_ident"], walltime_s=m["walltime_s"],
                    final_model=m["model"], f_true=r["f_true"],
                ))
            print(f"[{model}/{noise}] done in {time.time() - t0:.0f}s", flush=True)

    tidy = pd.DataFrame(rows)
    if save and not tidy.empty:
        fp = os.path.join(CASE_RESULTS_DIR, f"ablation_cases_tidy_{seed}.csv")
        tidy.to_csv(fp, index=False)
        print(f"\nSaved {fp}", flush=True)
    return tidy


def print_case_report(tidy):
    """Compact per-case comparison: ground-truth MSE and complexity by method."""
    if tidy.empty:
        print("no successful cases.")
        return
    for (model, noise), g in tidy.groupby(["model", "noise"]):
        print(f"\n===== {model} / {noise} =====", flush=True)
        show = g.set_index("method")[["n_removed", "final_complexity",
                                       "new_mse_truth", "new_cv",
                                       "fired_inf", "fired_signed"]].reindex(VARIANTS)
        print(show.round(4).to_string(), flush=True)



def run_seed(seed: int, models: list[str]):
    print(
        f"[seed={seed}] pid={os.getpid()} starting",
        flush=True,
    )

    cfg = make_case_config(seed)
    tidy = run_all_cases(cfg, models=models)

    print(
        f"[seed={seed}] pid={os.getpid()} finished",
        flush=True,
    )

    return seed, tidy


if __name__ == "__main__":
    models = [m for m in sys.argv[1:] if m in CASE_MODELS] or CASE_MODELS

    print(f"Detected CPUs: {cpu_count()}", flush=True)
    print(f"Effective workers: {effective_n_jobs(-2)}", flush=True)

    with parallel_config(
        backend="loky",
        n_jobs=-2,
        inner_max_num_threads=1,
    ):
        results = Parallel(
            verbose=100,
            batch_size=1,
            pre_dispatch="all",
        )(
            delayed(run_seed)(seed, models)
            for seed in range(0,30)
        )

    for seed, tidy in sorted(results):
        print(f"\n{'=' * 20} seed={seed} {'=' * 20}")
        print_case_report(tidy)