"""
operon_ablation.py
==================

Paired ablation on real symbolic-regression populations (Operon expressions
fitted to the Friedman benchmarks in ``SR_population_models/``).  For every
expression the fitted starting model is set up ONCE and pruned with five
methods from that identical starting point:

    full        the four Fisher modes, looped (CV gate + BIC tie-break)
    sloppy_only strict MBAM/BR: single smallest-eigenvalue direction, CV gate
    zero_only   Wald-to-zero over all parameters, CV gate
    magnitude   naive floor: relative-scale |theta| threshold, test-free
    no_cv_bic   the four Fisher modes with a BIC-only gate (dBIC < 3)

Sharing the starting model gives a PAIRED design: for each comparison the N
expressions of a population give N matched differences (variant - full),
tested with the two-sided Wilcoxon signed-rank test.  The per-move firing rate
of the full method over the population (fraction of expressions that ever
triggered a saturation at infinity, a signed identification, ...) is also
reported, so that a null ablation result can be told apart from an
underpowered one (a move that never fires).

Usage
-----
    # single population, optionally restricted to the first max_rows rows
    python operon_ablation.py friedman_sigma-1.0 [max_rows]

    # or shard the datasets of ``dataset_list`` across nodes
    python operon_ablation.py <shard_id> <n_shards>

Outputs (under ./results_ablation/)
    <dataset>/row_<i>.pkl           per-expression raw result (resumable)
    results_ablation_<dataset>.pkl
    ablation_tidy_<dataset>.csv     one row per (expression x variant)
    ablation_summary_<dataset>.csv  paired Wilcoxon vs full
"""

import os
# ---- single-thread everything BEFORE importing numpy / joblib ----
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "LOKY_MAX_CPU_COUNT"):
    os.environ[_v] = "1"
os.environ.setdefault("MPLBACKEND", "Agg")

import sys
import time
import socket
import signal
import pickle
import traceback
from contextlib import contextmanager

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from emergentSR.emergent import *
from emergentSR.operon_formatting import *
from emergentSR.ablation_variants import ABLATION_VARIANTS


# ============================================================
# Paths and population table
# ============================================================
DATA_DIR = "./SR_population_models/"
ABL_RESULTS_DIR = "./results_ablation/"

runs = read_sr_population_file(DATA_DIR + "expressions.txt")
groups_dict = runs.groupby("filename").groups
files_column = list(groups_dict.keys())
dataset_name = [el.split("/")[0] for el in files_column]
files_name = [el + "_train.csv" for el in dataset_name]
data_dict = {d: {"file": f, "col": c} for d, f, c in zip(dataset_name, files_name, files_column)}

dataset_list = [
    "friedman_sigma-0.5",
    "friedman_sigma-1.0",
    "friedman_sigma-13.6",
    #"friedman_sigma-2.0",
    "friedman_additive_n-1000_sigma-1.0",
    "friedman_additive_n-100_sigma-1.0",
    #"friedman_additive_n-200_sigma-1.0",
    #"friedman_additive_n-500_sigma-1.0",
    "friedman_additive_n-50_sigma-1.0",
]


def bic_from_MSE(mse, n, p):
    """Gaussian BIC from the training MSE; equals bic_for_sympy_model (beta=1)."""
    return n * (1.0 + np.log(2 * np.pi * mse)) + len(p) * np.log(n)


# full method + the four variants, in table order
VARIANTS = ["full", "sloppy_only", "zero_only", "magnitude", "no_cv_bic"]
BASELINES = ["sloppy_only", "zero_only", "magnitude", "no_cv_bic"]  # compared vs full


def make_config():
    """All run hyperparameters.  Everything the variants share with the full
    method is passed to all of them, so each ablation changes one thing."""
    return dict(
        criterion="CV",
        second_crit="BIC",
        seed=42,
        init_loc=0.0,
        init_scale=0.1,
        def_bounds=[(-1e8, 1e8)],   # used by CV refit and the pruning modes
        lambda_threshold=-10,
        lambda_gap_threshold=25.0,
        beta=0.2,
        snr_sq=16,
        cv_equiv_k=2,
        n_starts=30,
        maxiter=5000,
        cancel_removable=False,   # opt-in: cancel removable 0/0 created by a saturation
        ident_threshold=0.01,
        selection_tol=1e-2,
        # magnitude (naive floor) knobs
        magnitude_frac=0.01,      # tau = frac * scale(|theta|)
        magnitude_scale="max",    # "max" or "std"
        # no-CV / BIC-only knob
        bic_accept_band=3.0,      # accept while dBIC < 3
        # per-equation wall-clock budget (seconds) shared across the 5 variants;
        # on breach the row is saved with NaN entries and the sweep continues.
        # Set to 0 to disable.
        per_equation_timeout_s=22 * 3600,
    )


# ============================================================
# Per-equation wall-clock timeout (SIGALRM; Unix, worker main thread)
# ============================================================
class _EquationTimeout(BaseException):
    """Raised when an equation exceeds its wall-clock budget. Subclasses
    BaseException (not Exception) so inner ``except Exception`` handlers -- in
    the pruning code and the per-variant loop -- do NOT swallow it."""


@contextmanager
def _time_limit(seconds):
    """Raise _EquationTimeout after `seconds`. No-op if the budget is falsy or
    signals are unavailable here (e.g. not the main thread on this platform)."""
    usable = bool(seconds) and seconds > 0 and hasattr(signal, "SIGALRM")
    old = None
    if usable:
        def _handler(signum, frame):
            raise _EquationTimeout(f"exceeded {seconds:.0f}s wall-clock budget")
        try:
            old = signal.signal(signal.SIGALRM, _handler)
            signal.setitimer(signal.ITIMER_REAL, float(seconds))
        except (ValueError, OSError):
            usable = False
    try:
        yield
    finally:
        if usable:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
            if old is not None:
                signal.signal(signal.SIGALRM, old)


def _nan_variant(reason):
    return dict(status=reason, n_removed=np.nan, final_nparams=np.nan,
                final_complexity=np.nan, new_cv=np.nan, new_bic=np.nan,
                new_mse_train=np.nan, new_mse_test=np.nan,
                fired=dict(total=0, zero=0, one=0, inf=0, signed_ident=0),
                model=None, walltime_s=np.nan)


def _nan_result(expr_str, reason):
    return dict(status=reason, expression=expr_str,
                old_cv=np.nan, old_bic=np.nan, old_mse_train=np.nan,
                old_mse_test=np.nan, old_complexity=np.nan, start_nparams=np.nan,
                old_model=None,
                variants={name: _nan_variant(reason) for name in VARIANTS})


# ============================================================
# Move-firing tally from a pruning history
# ============================================================
def _tally_moves(history):
    t = dict(total=0, zero=0, one=0, inf=0, signed_ident=0,
             mode_unident=0, mode_infinity=0, mode_zero=0, mode_signed=0)
    if not history:
        return t
    for h in history:
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
        m = h.get("pruning_mode") or ""
        t["mode_unident"] += (m == "unidentifiability")
        t["mode_infinity"] += (m == "infinity_compatibility")
        t["mode_zero"] += (m == "zero_compatibility")
        t["mode_signed"] += (m == "signed_identifiability")
    return t


# ============================================================
# Shared per-expression setup (fit once) + the full method
# ============================================================
def _setup_start(expr_str, train, test, true_model_kind, cfg):
    """Fit the starting model once; return the shared starting state + test arrays."""
    np.random.seed(cfg["seed"])
    x, y, model_sym, xs, params, f_true, y_true, model_func, init_map = (
        create_friedman_operon_model_from_string(
            dataset=train, expr_str=expr_str, n_features=10, y_col="target",
            reparametrize_constants=True, const_prefix="C",
            true_model_kind=true_model_kind, simplify_before_extraction=False,
        )
    )
    guess_init = init_from_map(params, init_map)
    theta_mle, sigma_noise, model_func, jac_funcs, _diag = compute_lambdify_and_mle_estimates(
        model_sym=model_sym, params=params, xs=xs, x=x, y=y,
        def_bounds=[(-100, 100)], init_loc=0.0, init_scale=0.1, guess_init=guess_init,
        n_starts=cfg["n_starts"], maxiter=cfg["maxiter"], return_diagnostics=True,
    )

    guess_cur = [float(t) for t in theta_mle] if isinstance(theta_mle, np.ndarray) else None
    current_cv_mse, _ = kfold_cv_mse_for_sympy_model(
        model_sym=model_sym, params=params, xs=xs, x=x, y=y,
        def_bounds=cfg["def_bounds"], init_loc=cfg["init_loc"], init_scale=cfg["init_scale"],
        guess_init=guess_cur, return_se=True, verbose=False,
    )
    df = compute_eigenvecs_eigenvals_and_alignment(
        params, jac_funcs, theta_mle, x, y, sigma_noise, model_func
    ).sort_values(by="ln(lambda)")

    # test-design arrays and baseline (refit) train/test MSE, shared by all variants
    X_test = test[[f"x{i}" for i in range(1, 11)]].values
    xtest = [X_test[:, i] for i in range(X_test.shape[1])]
    y_test = test["target"].values
    base_train = float(np.mean((model_func(*(tuple(theta_mle) + tuple(x))) - y) ** 2))
    base_test = float(np.mean((model_func(*(tuple(theta_mle) + tuple(xtest))) - y_test) ** 2))

    return dict(
        model_sym=model_sym, params=params, model_func=model_func,
        jac_funcs=jac_funcs, theta=theta_mle, sigma=sigma_noise, df=df,
        xs=xs, x=x, y=y, xtest=xtest, y_test=y_test,
        cv=current_cv_mse, bic=bic_from_MSE(base_train, len(y), params),
        base_train=base_train, base_test=base_test,
    )


def _run_full(st0, cfg, history):
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
        eff_total = 0
        for mode in ("unidentifiability",  "zero_compatibility",
                     "infinity_compatibility",
                     "signed_identifiability"):
            (ms, pr, mf, th, sg, jf, df, cv, bic, eff) = _mode(mode)
            st.update(model_sym=ms, params=pr, model_func=mf, theta=th,
                      sigma=sg, jac_funcs=jf, df=df, cv=cv, bic=bic)
            eff_total += eff
        effectiveness = bool(eff_total)

    return (st["model_sym"], st["params"], st["model_func"], st["theta"],
            st["sigma"], st["jac_funcs"], st["df"], st["cv"], st["bic"],
            len(st0["params"]) - len(st["params"]))


def _run_variant(name, st0, cfg, history):
    """Run one variant from the shared starting state. A single unified kwargs
    bundle is passed to every variant; each absorbs the extras it does not use
    via **kwargs, so dispatch is uniform."""
    if name == "full":
        return _run_full(st0, cfg, history)

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
        # unified extras (absorbed by **kwargs where unused):
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


def _variant_metrics(st0, out_tuple, history):
    """Compute the comparable metrics for one finished variant run."""
    (ms, pr, mf, th, sg, jf, df, cv, bic, nrem) = out_tuple
    call_tr = tuple(th) + tuple(st0["x"])
    call_te = tuple(th) + tuple(st0["xtest"])
    mse_tr = float(np.mean((mf(*call_tr) - st0["y"]) ** 2))
    mse_te = float(np.mean((mf(*call_te) - st0["y_test"]) ** 2))
    return dict(
        n_removed=int(nrem),
        final_nparams=len(pr),
        final_complexity=int(count_sympy_nodes(ms)),
        new_cv=float(cv) if cv is not None else np.nan,
        new_bic=float(bic) if bic is not None else np.nan,
        new_mse_train=mse_tr,
        new_mse_test=mse_te,
        fired=_tally_moves(history),
        model=str(ms),
    )


# ============================================================
# One expression = full + all four variants from a shared start
# ============================================================
def prune_one_ablation(expr_str, train, test, true_model_kind, cfg):
    timeout_s = float(cfg.get("per_equation_timeout_s", 0) or 0)
    deadline = (time.time() + timeout_s) if timeout_s > 0 else None

    # ---- initial fit (setup) under the full budget ----
    try:
        with _time_limit(timeout_s):
            st0 = _setup_start(expr_str, train, test, true_model_kind, cfg)
    except _EquationTimeout as e:
        r = _nan_result(expr_str, "timeout")
        r["error"] = f"setup {e}"
        return r

    out = dict(
        status="ok", expression=expr_str,
        old_cv=float(st0["cv"]), old_bic=float(st0["bic"]),
        old_mse_train=st0["base_train"], old_mse_test=st0["base_test"],
        old_complexity=int(count_sympy_nodes(st0["model_sym"])),
        start_nparams=len(st0["params"]), old_model=str(st0["model_sym"]),
        variants={},
    )
    # ---- variants share the REMAINING budget; finished ones are kept ----
    for name in VARIANTS:
        remaining = (deadline - time.time()) if deadline is not None else None
        if deadline is not None and remaining <= 0:
            out["variants"][name] = _nan_variant("timeout")
            out["status"] = "timeout"
            continue
        t0 = time.time()
        history = init_pruning_history()
        try:
            with _time_limit(remaining):
                res = _run_variant(name, st0, cfg, history)
            m = _variant_metrics(st0, res, history)
            m["status"] = "ok"
        except _EquationTimeout:
            m = _nan_variant("timeout")
            out["status"] = "timeout"
        except Exception as e:
            m = dict(status="error", error=repr(e),
                     traceback=traceback.format_exc())
        m["walltime_s"] = round(time.time() - t0, 2)
        out["variants"][name] = m
    return out


def _run_and_save(idx, expr_str, out_fp, train, test, true_model_kind, cfg):
    """Never raises: always persists a pickle so the row counts as done
    (resumable) and the sweep proceeds to the next equation."""
    try:
        res = prune_one_ablation(expr_str, train, test, true_model_kind, cfg)
    except _EquationTimeout:
        res = _nan_result(expr_str, "timeout")
        print('Equation timeout', out_fp)
    except Exception as e:
        res = _nan_result(expr_str, "error")
        res["error"] = repr(e)
        res["traceback"] = traceback.format_exc()
    with open(out_fp, "wb") as fh:
        pickle.dump(res, fh)
    return idx, res.get("status", "error")


# ============================================================
# One dataset: parallel sweep over population rows
# ============================================================
def run_dataset(dataset, n_jobs, cfg, max_rows=0):
    if "friedman_additive" in dataset:
        true_model_kind = "additive"
    elif "friedman" in dataset:
        true_model_kind = "standard"
    else:
        true_model_kind = None

    train = pd.read_csv(DATA_DIR +'/'+ data_dict[dataset]["file"])
    
    if "additive" in dataset:
        test = generate_friedman_nonlin_additive_test(n=1000, noiselvl=0)
    else:
        test = generate_friedman_test(n=1000, noiselvl=0)

    population = (runs.loc[runs["filename"] == data_dict[dataset]["col"]]
                  .sort_values(by="MSE_train", ascending=False).reset_index())
    if max_rows and max_rows > 0:
        population = population.iloc[:max_rows]

    outdir = os.path.join(ABL_RESULTS_DIR, dataset)
    os.makedirs(outdir, exist_ok=True)

    tasks = []
    for i, row in population.iterrows():
        fp = os.path.join(outdir, f"row_{i}.pkl")
        if os.path.exists(fp):
            continue
        tasks.append((i, row["expression"], fp))

    print(f"  [{dataset}] {len(population)} rows, {len(tasks)} to run, "
          f"{len(VARIANTS)} variants each, n_jobs={n_jobs}", flush=True)

    if tasks:
        Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_run_and_save)(i, expr, fp, train, test, true_model_kind, cfg)
            for (i, expr, fp) in tasks
        )

    results = {}
    for i, row in population.iterrows():
        fp = os.path.join(outdir, f"row_{i}.pkl")
        if os.path.exists(fp):
            with open(fp, "rb") as fh:
                results[i] = pickle.load(fh)

    with open(os.path.join(ABL_RESULTS_DIR, f"results_ablation_{dataset}.pkl"), "wb") as fh:
        pickle.dump(results, fh)

    summarize_ablation(results, dataset)
    n_err = sum(1 for r in results.values() if r.get("status") != "ok")
    print(f"  [{dataset}] aggregated {len(results)} rows ({n_err} setup errors)", flush=True)
    return results


# ============================================================
# Aggregation: tidy table + paired Wilcoxon + firing rates
# ============================================================
def _tidy_rows(results, dataset):
    """One row per (equation x variant). Timed-out / errored equations and
    variants are emitted with NaN metrics (and a 'status' column) rather than
    dropped, so every attempted equation is represented."""
    def _num(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return np.nan

    rows = []
    for idx, r in results.items():
        oc = _num(r.get("old_complexity"))
        ocv = _num(r.get("old_cv"))
        omt = _num(r.get("old_mse_test"))
        base = dict(dataset=dataset, row=idx,
                    old_cv=ocv, old_bic=_num(r.get("old_bic")),
                    old_mse_train=_num(r.get("old_mse_train")), old_mse_test=omt,
                    old_complexity=oc, start_nparams=_num(r.get("start_nparams")))
        variants = r.get("variants", {}) or {}
        for name in VARIANTS:
            m = variants.get(name, {})
            ok = (m.get("status") == "ok")
            f = m.get("fired", {}) if ok else {}
            fc = _num(m.get("final_complexity")) if ok else np.nan
            ncv = _num(m.get("new_cv")) if ok else np.nan
            nmt = _num(m.get("new_mse_test")) if ok else np.nan
            rows.append(dict(
                base, variant=name,
                status=m.get("status", r.get("status", "missing")),
                n_removed=_num(m.get("n_removed")) if ok else np.nan,
                final_nparams=_num(m.get("final_nparams")) if ok else np.nan,
                final_complexity=fc, new_cv=ncv,
                new_bic=_num(m.get("new_bic")) if ok else np.nan,
                new_mse_train=_num(m.get("new_mse_train")) if ok else np.nan,
                new_mse_test=nmt, walltime_s=_num(m.get("walltime_s")),
                fired_zero=f.get("zero", np.nan), fired_one=f.get("one", np.nan),
                fired_inf=f.get("inf", np.nan), fired_signed=f.get("signed_ident", np.nan),
                d_complexity=(fc - oc) if (ok and np.isfinite(fc) and np.isfinite(oc)) else np.nan,
                d_cv=(ncv - ocv) if (ok and np.isfinite(ncv) and np.isfinite(ocv)) else np.nan,
                d_mse_test=(nmt - omt) if (ok and np.isfinite(nmt) and np.isfinite(omt)) else np.nan,
                model=m.get("model") if ok else None,
            ))
    return pd.DataFrame(rows)


def _paired_wilcoxon(tidy, metric):
    """For each baseline variant, Wilcoxon signed-rank on (variant - full) paired
    over rows present in both. Returns a list of summary dicts."""
    try:
        from scipy.stats import wilcoxon
    except Exception:
        wilcoxon = None
    out = []
    full = tidy[tidy.variant == "full"].set_index("row")[metric]
    for name in BASELINES:
        v = tidy[tidy.variant == name].set_index("row")[metric]
        common = full.index.intersection(v.index)
        if len(common) == 0:
            continue
        diff = (v.loc[common] - full.loc[common]).astype(float).dropna()
        med = float(np.median(diff)) if len(diff) else np.nan
        p = np.nan
        if wilcoxon is not None and len(diff) >= 1 and np.any(diff != 0):
            try:
                p = float(wilcoxon(diff, zero_method="wilcox",
                                   alternative="two-sided").pvalue)
            except Exception:
                p = np.nan
        out.append(dict(metric=metric, variant=name, n_pairs=int(len(diff)),
                        median_diff_vs_full=med, wilcoxon_p=p))
    return out


def summarize_ablation(results, dataset):
    tidy = _tidy_rows(results, dataset)
    if tidy.empty:
        print(f"  [{dataset}] no successful rows to summarise.", flush=True)
        return tidy, None

    tidy_fp = os.path.join(ABL_RESULTS_DIR, f"ablation_tidy_{dataset}.csv")
    tidy.to_csv(tidy_fp, index=False)

    # ---- per-variant central tendencies ----
    agg = (tidy.groupby("variant")
           .agg(n=("row", "count"),
                med_final_complexity=("final_complexity", "median"),
                med_n_removed=("n_removed", "median"),
                med_new_cv=("new_cv", "median"),
                med_new_mse_test=("new_mse_test", "median"),
                med_walltime_s=("walltime_s", "median"))
           .reindex(VARIANTS))

    # ---- firing rates of the FULL method over the population ----
    full_rows = tidy[tidy.variant == "full"]
    n_full = len(full_rows)
    firing = {}
    if n_full:
        firing = dict(
            n_expressions=n_full,
            frac_fired_zero=float((full_rows.fired_zero > 0).mean()),
            frac_fired_one=float((full_rows.fired_one > 0).mean()),
            frac_fired_inf=float((full_rows.fired_inf > 0).mean()),
            frac_fired_signed=float((full_rows.fired_signed > 0).mean()),
        )

    # ---- paired Wilcoxon vs full, on three metrics ----
    stats_rows = []
    for metric in ("final_complexity", "new_mse_test", "new_cv"):
        stats_rows.extend(_paired_wilcoxon(tidy, metric))
    stats = pd.DataFrame(stats_rows)
    stats_fp = os.path.join(ABL_RESULTS_DIR, f"ablation_summary_{dataset}.csv")
    stats.to_csv(stats_fp, index=False)

    # ---- console report ----
    print(f"\n===== ABLATION SUMMARY: {dataset} =====", flush=True)
    print("Per-variant medians:\n", agg.round(4).to_string(), flush=True)
    if firing:
        print("\nFull-method per-move firing rate over the population "
              f"(n={firing['n_expressions']}):", flush=True)
        print(f"  zero={firing['frac_fired_zero']:.2f}  one={firing['frac_fired_one']:.2f}  "
              f"inf={firing['frac_fired_inf']:.2f}  signed-ident={firing['frac_fired_signed']:.2f}",
              flush=True)
    print("\nPaired Wilcoxon (variant - full), two-sided:\n",
          stats.round(4).to_string(index=False), flush=True)
    print(f"\nSaved: {tidy_fp}\n       {stats_fp}\n", flush=True)
    return tidy, stats


# ============================================================
# Entry point
# ============================================================
if __name__ == "__main__":
    n_jobs = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    os.makedirs(ABL_RESULTS_DIR, exist_ok=True)
    cfg = make_config()

    arg1 = sys.argv[1] if len(sys.argv) > 1 else "0"

    if arg1 in dataset_list:
        # single-dataset mode: python operon_ablation.py <dataset> [max_rows]
        max_rows = int(sys.argv[2]) if len(sys.argv) > 2 else 0
        my_datasets = [arg1]
        print(f"[single] host={socket.gethostname()} n_jobs={n_jobs} "
              f"dataset={arg1} max_rows={max_rows or 'all'}", flush=True)
        for ds in my_datasets:
            t0 = time.time()
            run_dataset(ds, n_jobs, cfg, max_rows=max_rows)
            print(f"[single] {ds} done in {time.time() - t0:.0f}s", flush=True)
    else:
        # shard mode: python operon_ablation.py <shard_id> <n_shards>
        shard = int(arg1)
        n_shards = int(sys.argv[2]) if len(sys.argv) > 2 else 4
        my_datasets = dataset_list[shard::n_shards]
        print(f"[shard {shard}/{n_shards}] host={socket.gethostname()} "
              f"n_jobs={n_jobs} datasets={my_datasets}", flush=True)
        for ds in my_datasets:
            t0 = time.time()
            run_dataset(ds, n_jobs, cfg)
            print(f"[shard {shard}] {ds} done in {time.time() - t0:.0f}s", flush=True)
