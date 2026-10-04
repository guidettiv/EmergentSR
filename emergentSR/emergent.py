"""
emergentSR.emergent
===================

Fisher-geometry pruning of over-parametrised symbolic-regression models.

Given a fitted symbolic model f(x; theta), the library looks for parameter
reductions supported by the data and applies them one at a time:

* theta_k -> 0, 1, +inf or -inf   (scalar saturations; infinities are handled
  through the compactification phi_k = arctan(theta_k)), and
* theta_i = +/- theta_j            (signed identification).

Candidate moves are nominated from the local Fisher information (numerically
unidentifiable directions, and Wald-type distances to each boundary), every
candidate is refitted, and a move is accepted only if the cross-validated MSE
stays within a tolerance band; BIC is used solely to break ties between moves
that cross-validation cannot distinguish.  The accepted model is refitted with
tight tolerances before the next iteration.

Main entry points
-----------------
compute_lambdify_and_mle_estimates      multi-start, ridge-continued least squares
compute_eigenvecs_eigenvals_and_alignment
                                        Fisher spectrum + boundary distances
remove_and_recalibrate_sloppy_parameters
                                        one pruning mode iterated to exhaustion
select_best_pruning_result              CV-band acceptance with BIC tie-break
kfold_cv_mse_for_sympy_model            K-fold CV-MSE of a symbolic model
create_model                            controlled case studies
init_pruning_history / record_snapshot  pruning trajectory for plots
"""

import os
import re
import warnings

import numpy as np
import pandas as pd
import sympy as sp
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from scipy.optimize import least_squares
from scipy.stats import chi2

try:
    from sympy.printing.printer import PrintMethodNotImplementedError
except ImportError:  # older SymPy versions
    from sympy.printing.codeprinter import PrintMethodNotImplementedError

# Refitting many candidate models on singular / overflowing landscapes produces
# a flood of harmless floating-point warnings; silence them globally.
os.environ["PYTHONWARNINGS"] = "ignore::RuntimeWarning"
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore", RuntimeWarning)
np.seterr(all="ignore")

# Print full diagnostic tables (verbose mode prints the Fisher spectrum).
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.width", None)

# Lambdify with a NumPy-compatible absolute value.
LAMBDA_MODULES = [{"Abs": np.abs, "abs": np.abs}, "numpy"]


# ============================================================================
# Utilities for symbolic manipulation
# ============================================================================

def _as_len_n(vec, n):
    """Coerce a lambdify output to shape (n,). Scalars/constants get broadcast."""
    arr = np.asarray(vec)
    if arr.shape == ():              # scalar
        return np.full(n, arr.item(), dtype=float)
    arr = arr.ravel()
    if arr.size == 1:                # e.g. shape (1,)
        return np.full(n, float(arr[0]))
    if arr.size != n:
        raise ValueError(f"Expected length {n}, got {arr.size}")
    return arr.astype(float, copy=False)

def add_next_param(params, **assumptions):
    """
    Append the next s{i} symbol (smallest i >= 0 not in params).
    `assumptions` are passed to sp.Symbol (e.g., real=True).
    Returns (updated_params, new_symbol).
    """
    params = list(params)
    used = {
        int(m.group(1))
        for p in params
        for m in [re.match(r'^s(\d+)$', str(p))]
        if m
    }
    i = 0
    while i in used:
        i += 1
    s = sp.Symbol(f's{i}', **assumptions)
    return params + [s]

def add_k_params(params, k, **assumptions):
    out = list(params)
    for _ in range(k):
        out = add_next_param(out, **assumptions)
    return out

def contract_param_only_subtrees(expr, params, xs, theta_map=None):
    """
    Collapse parameter-only subexpressions, but only when their parameters
    are local to that subtree.

    In particular, do NOT collapse a parameter-only subtree if one of its
    parameters also appears elsewhere in the full expression.

    This prevents cases such as

        -s2 + s2/(c*exp(b*x0) + 1)

    from being contracted back to

        s0 + s2/(c*exp(b*x0) + 1)

    after imposing a signed-identification constraint.

    Returns
    -------
    (new_expr, init_override) where ``init_override`` maps every parameter
    reused as the representative of a collapsed cluster to the cluster's value
    at ``theta_map`` (a warm start for the refit).
    """
    params = list(params)
    xs = list(xs)
    P = set(params)
    X = set(xs)

    # Substitutions (param -> current MLE value) used to warm-start a reused
    # parameter at the collapsed cluster's value. Empty if no theta_map given.
    theta_subs = None
    if theta_map is not None:
        theta_subs = {}
        for _p, _v in theta_map.items():
            try:
                theta_subs[_p] = sp.Float(float(_v))
            except (TypeError, ValueError):
                pass

    init_override = {}   # reused/created param -> warm-start value

    def contains_special(e):
        return e.has(sp.AccumBounds, sp.oo, -sp.oo, sp.zoo, sp.nan)

    def has_number(e):
        return bool(e.atoms(sp.Number))

    def is_param_only(e):
        fs = e.free_symbols
        return fs and fs.issubset(P) and fs.isdisjoint(X)

    def symbol_occurrence_counts(e):
        counts = {p: 0 for p in P}

        def walk(node):
            if node in P:
                counts[node] += 1
            for a in getattr(node, "args", ()):
                walk(a)

        walk(e)
        return counts

    def subtree_is_param_local(e, global_counts):
        local_counts = symbol_occurrence_counts(e)
        for p in e.free_symbols & P:
            if local_counts.get(p, 0) != global_counts.get(p, 0):
                return False
        return True

    def qualifies(e, global_counts):
        # do not collapse bare symbol/number
        if e.is_Atom or e.is_Symbol:
            return False
        if contains_special(e):
            return False
        if not is_param_only(e):
            return False

        # only collapse if all involved parameters are local to this subtree
        if not subtree_is_param_local(e, global_counts):
            return False

        fs = e.free_symbols
        return (len(fs) >= 2) or (len(fs) >= 1 and has_number(e))

    def choose_repl(cluster, pool):
        # Prefer reusing a parameter that already lives inside the cluster:
        # qualifies() guarantees every such parameter is local to the cluster,
        # so reuse cannot alias an occurrence elsewhere. Its warm start is the
        # cluster's value at the current MLE (recorded in init_override).
        local_ps = sorted(cluster.free_symbols & P, key=str)
        if local_ps:
            repl = local_ps[0]
            if theta_subs is not None:
                try:
                    val = complex(cluster.xreplace(theta_subs))
                    if val.imag == 0 and np.isfinite(val.real):
                        init_override[repl] = float(val.real)
                except (TypeError, ValueError):
                    pass
            return repl
        # fallback: a fresh symbol from the pool (no meaningful warm start)
        return pool.pop(0) if pool else None

    def available_params(cur_expr):
        used = cur_expr.free_symbols
        return [p for p in params if p not in used]

    def transform(node, pool, global_counts):
        if contains_special(node):
            return node
        if node.is_Atom:
            return node

        if isinstance(node, sp.Add):
            new_args = [transform(a, pool, global_counts) for a in node.args]
            if any(contains_special(a) for a in new_args):
                return node.func(*new_args)

            param_args, other_args = [], []
            for a in new_args:
                if is_param_only(a) or (a.is_Number and a != 0):
                    param_args.append(a)
                else:
                    other_args.append(a)

            if param_args:
                cluster = sp.Add(*param_args)
                if contains_special(cluster):
                    return node.func(*new_args)
                if qualifies(cluster, global_counts):
                    repl = choose_repl(cluster, pool)
                    if repl is not None:
                        return sp.Add(*(other_args + [repl]))

            return node.func(*new_args)

        if isinstance(node, sp.Mul):
            new_args = [transform(a, pool, global_counts) for a in node.args]
            if any(contains_special(a) for a in new_args):
                return node.func(*new_args)

            param_factors, other_factors = [], []
            for a in new_args:
                if is_param_only(a) or a.is_Number:
                    param_factors.append(a)
                else:
                    other_factors.append(a)

            if param_factors:
                cluster = sp.Mul(*param_factors)
                if contains_special(cluster):
                    return node.func(*new_args)
                if qualifies(cluster, global_counts):
                    repl = choose_repl(cluster, pool)
                    if repl is not None:
                        return sp.Mul(*(other_factors + [repl]))

            return node.func(*new_args)

        new_args = [transform(a, pool, global_counts) for a in node.args]
        rebuilt = node.func(*new_args)

        if contains_special(rebuilt):
            return rebuilt
        if qualifies(rebuilt, global_counts):
            repl = choose_repl(rebuilt, pool)
            if repl is not None:
                return repl

        return rebuilt

    out = expr
    while True:
        pool = available_params(out)
        global_counts = symbol_occurrence_counts(out)

        new_out = transform(out, pool[:], global_counts)
        if new_out == out:
            break
        out = new_out

    return out, init_override


# ============================================================================
# Fitting and Fisher geometry
# ============================================================================


def compute_lambdify_and_mle_estimates(
    model_sym, params, xs, x, y,
    def_bounds, init_loc, init_scale,
    n_starts=30,
    maxiter=5000,
    guess_init=None,
    ridge_path=(1e-2, 1e-4),
    final_polish=True,
    polish_ridge=1e-8,
    xtol=1e-10,
    ftol=1e-10,
    gtol=1e-10,
    jac_clip=None,
    param_scale_mode="jac",
    return_diagnostics=False,
):
    """
    Stabilized nonlinear least-squares fit of a symbolic model.

      - scipy.optimize.least_squares with method='trf' and analytic Jacobian
      - penalized residuals [data residuals ; sqrt(ridge)*theta]
      - continuation over a decreasing ridge path
      - optional final, nearly unpenalized polish (kept only if it does not
        worsen SSE or blow up the conditioning of J^T J)
      - multi-start: warm start (if given), bound midpoint, zero, then
        ``n_starts`` random starts; the best start is chosen by
        (SSE, gradient norm, condition number)

    Parameters
    ----------
    model_sym : sympy.Expr
    params : list[sympy.Symbol]
    xs : list[sympy.Symbol]
    x : list[np.ndarray]
    y : np.ndarray
    def_bounds : list[tuple]
        One bound tuple or one per parameter.
    init_loc, init_scale : float
        Random init parameters.
    n_starts : int
        Number of random starts after deterministic starts.
    maxiter : int
        Max function evaluations per least_squares call (max_nfev).
    guess_init : array-like or None
        Warm start candidate.
    ridge_path : tuple[float]
        Decreasing ridge penalties.
    final_polish : bool
        Whether to do final near-unpenalized polish.
    polish_ridge : float
        Tiny ridge used during polish for numerical safety.
    jac_clip : float or None
        If not None, clip Jacobian entries to [-jac_clip, jac_clip].
    param_scale_mode : {"jac", "ones"}
        x_scale for least_squares.
    return_diagnostics : bool
        If True, also returns a diagnostics dict.

    Returns
    -------
    theta_best : np.ndarray or None
    sigma_hat : float
    model_func : callable
    jac_funcs : list[callable]
    diagnostics : dict   (only if return_diagnostics=True)
    """

    xs = list(xs) if isinstance(xs, (list, tuple)) else [xs]
    if not isinstance(x, (list, tuple)):
        raise ValueError("x must be a list/tuple of arrays, one per xs symbol.")
    if len(x) != len(xs):
        raise ValueError(f"len(x)={len(x)} must match len(xs)={len(xs)}.")

    y = np.asarray(y, dtype=float).reshape(-1)
    n = len(y)
    for i, xi in enumerate(x):
        if len(xi) != n:
            raise ValueError(f"x[{i}] has length {len(xi)} but y has length {n}.")

    p = len(params)
    if p == 0:
        raise ValueError("No parameters to fit.")

    # Symbolic derivatives wrt parameters
    jac_sym = [sp.diff(model_sym, par) for par in params]

    # Lambdified model and jacobians
    arg_order = tuple(params + xs)
    model_func = sp.lambdify(arg_order, model_sym, modules=LAMBDA_MODULES)
    jac_funcs = [sp.lambdify(arg_order, j, modules=LAMBDA_MODULES) for j in jac_sym]    

    def _as_len_n(vec, n):
        arr = np.asarray(vec)
        if arr.shape == ():
            return np.full(n, float(arr), dtype=float)
        arr = arr.ravel()
        if arr.size == 1:
            return np.full(n, float(arr[0]), dtype=float)
        if arr.size != n:
            raise ValueError(f"Expected length {n}, got {arr.size}")
        return arr.astype(float, copy=False)

    def _call_args(theta, x_use):
        return tuple(theta) + tuple(x_use)

    # Bounds handling
    if len(def_bounds) == 1:
        bounds = def_bounds * p
    else:
        if len(def_bounds) != p:
            raise ValueError("def_bounds length must match number of params.")
        bounds = def_bounds

    lb = np.array([b[0] for b in bounds], dtype=float)
    ub = np.array([b[1] for b in bounds], dtype=float)

    # Safer initial point projection into bounds
    def _project_to_bounds(theta):
        theta = np.asarray(theta, dtype=float).copy()
        epsb = 1e-12
        theta = np.maximum(theta, lb + epsb)
        theta = np.minimum(theta, ub - epsb)
        return theta

    # Model prediction
    def _predict(theta, x_use):
        yhat = model_func(*_call_args(theta, x_use))
        return _as_len_n(yhat, len(x_use[0]))

    # Data residuals
    def _data_residuals(theta, x_use, y_use):
        return y_use - _predict(theta, x_use)

    # Data Jacobian of residuals = -df/dtheta
    def _data_jacobian(theta, x_use):
        n_use = len(x_use[0])
        J = np.zeros((n_use, p), dtype=float)
        args = _call_args(theta, x_use)
        for j, jf in enumerate(jac_funcs):
            col = _as_len_n(jf(*args), n_use)
            if jac_clip is not None:
                col = np.clip(col, -jac_clip, jac_clip)
            J[:, j] = -col
        return J

    # Penalized residual stack
    def _penalized_residuals(theta, x_use, y_use, ridge):
        r_data = _data_residuals(theta, x_use, y_use)
        if ridge > 0:
            r_pen = np.sqrt(ridge) * theta
            return np.concatenate([r_data, r_pen])
        return r_data

    def _penalized_jacobian(theta, x_use, ridge):
        J_data = _data_jacobian(theta, x_use)
        if ridge > 0:
            J_pen = np.sqrt(ridge) * np.eye(p)
            return np.vstack([J_data, J_pen])
        return J_data

    def _make_x_scale(theta0):
        if param_scale_mode == "ones":
            return np.ones(p, dtype=float)
        try:
            J0 = _data_jacobian(theta0, x)
            coln = np.linalg.norm(J0, axis=0)
            coln = np.where(np.isfinite(coln) & (coln > 1e-12), coln, 1.0)
            return 1.0 / coln
        except Exception:
            return np.ones(p, dtype=float)

    def _sse(theta):
        r = _data_residuals(theta, x, y)
        return 0.5 * float(np.dot(r, r))

    def _sigma_hat(theta):
        r = _data_residuals(theta, x, y)
        return float(np.sqrt(np.mean(r**2)))

    def _grad_inf_norm(theta, ridge=0.0):
        try:
            r = _data_residuals(theta, x, y)
            J = _data_jacobian(theta, x)
            g = J.T @ r
            if ridge > 0:
                g = g + ridge * theta
            return float(np.max(np.abs(g)))
        except Exception:
            return np.inf

    def _jtj_cond(theta):
        try:
            J = _data_jacobian(theta, x)
            JTJ = J.T @ J
            s = np.linalg.svd(JTJ, compute_uv=False)
            smax = np.max(s)
            smin = np.min(s)
            if not np.isfinite(smax) or not np.isfinite(smin) or smin <= 1e-16:
                return np.inf
            return float(smax / smin)
        except Exception:
            return np.inf

    def _fit_one_start(theta_start):
        """
        Continuation fit from one start.
        """
        theta_cur = _project_to_bounds(theta_start)
        path_info = []

        try:
            x_scale = _make_x_scale(theta_cur)

            # Continuation over ridge path
            for ridge in ridge_path:
                res = least_squares(
                    fun=lambda th: _penalized_residuals(th, x, y, ridge),
                    x0=theta_cur,
                    jac=lambda th: _penalized_jacobian(th, x, ridge),
                    bounds=(lb, ub),
                    method="trf",
                    x_scale=x_scale,
                    loss="linear",
                    ftol=ftol,
                    xtol=xtol,
                    gtol=gtol,
                    max_nfev=maxiter,
                )
                theta_cur = _project_to_bounds(res.x)
                path_info.append({
                    "ridge": float(ridge),
                    "cost": float(res.cost),
                    "status": int(res.status),
                    "success": bool(res.success),
                    "nfev": int(res.nfev),
                })

            # Optional final polish
            polish_attempted = False
            polish_success = False
            polish_kept = False
            if final_polish:
                polish_attempted = True
                ridge = float(polish_ridge)
                res_polish = least_squares(
                    fun=lambda th: _penalized_residuals(th, x, y, ridge),
                    x0=theta_cur,
                    jac=lambda th: _penalized_jacobian(th, x, ridge),
                    bounds=(lb, ub),
                    method="trf",
                    x_scale=x_scale,
                    loss="linear",
                    ftol=ftol,
                    xtol=xtol,
                    gtol=gtol,
                    max_nfev=maxiter,
                )
                theta_polish = _project_to_bounds(res_polish.x)
                polish_success = bool(res_polish.success)

                # Keep only if it does not look worse numerically
                cond_before = _jtj_cond(theta_cur)
                cond_after = _jtj_cond(theta_polish)
                sse_before = _sse(theta_cur)
                sse_after = _sse(theta_polish)

                if np.isfinite(sse_after) and (sse_after <= sse_before + 1e-10):
                    # Do not accept obviously much worse conditioning
                    if (not np.isfinite(cond_after)) and np.isfinite(cond_before):
                        pass
                    elif np.isfinite(cond_before) and np.isfinite(cond_after) and cond_after > 1e6 * max(cond_before, 1.0):
                        pass
                    else:
                        theta_cur = theta_polish
                        polish_kept = True

            diag = {
                "path": path_info,
                "polish_attempted": polish_attempted,
                "polish_success": polish_success,
                "polish_kept": polish_kept,
                "sse": _sse(theta_cur),
                "sigma_hat": _sigma_hat(theta_cur),
                "grad_inf_norm": _grad_inf_norm(theta_cur, ridge=0.0),
                "jtj_cond": _jtj_cond(theta_cur),
                "param_norm": float(np.linalg.norm(theta_cur)),
                "max_abs_param": float(np.max(np.abs(theta_cur))),
                "finite_theta": bool(np.all(np.isfinite(theta_cur))),
                "finite_pred": bool(np.all(np.isfinite(_predict(theta_cur, x)))),
            }
            return theta_cur, diag

        except Exception as e:
            diag = {
                "path": path_info,
                "error": str(e),
                "sse": np.inf,
                "sigma_hat": np.inf,
                "grad_inf_norm": np.inf,
                "jtj_cond": np.inf,
                "param_norm": np.inf,
                "max_abs_param": np.inf,
                "finite_theta": False,
                "finite_pred": False,
            }
            return None, diag

    # Build deterministic + random starts
    starts = []

    # 1) warm start if given
    if guess_init is not None:
        gi = np.asarray(guess_init, dtype=float).copy()
        mask = ~np.isfinite(gi)
        if np.any(mask):
            gi[mask] = np.random.normal(loc=init_loc, scale=init_scale, size=mask.sum())
        starts.append(_project_to_bounds(gi))

    # 2) center-like deterministic start
    midpoint = np.where(
        np.isfinite(lb) & np.isfinite(ub),
        0.5 * (lb + ub),
        np.zeros_like(lb)
    )
    midpoint = np.where(np.isfinite(midpoint), midpoint, 0.0)
    starts.append(_project_to_bounds(midpoint))

    # 3) zero-ish deterministic start
    starts.append(_project_to_bounds(np.zeros(p)))

    # 4) random starts
    for _ in range(n_starts):
        z = np.random.normal(loc=init_loc, scale=init_scale, size=p)
        # if finite bounds exist, sample more sensibly inside them when random init is absurd
        if np.all(np.isfinite(lb)) and np.all(np.isfinite(ub)):
            bad = (z <= lb) | (z >= ub)
            if np.any(bad):
                z[bad] = lb[bad] + (ub[bad] - lb[bad]) * np.random.rand(np.sum(bad))
        starts.append(_project_to_bounds(z))

    # Fit all starts and keep best by SSE, then by gradient, then by conditioning
    best_theta = None
    best_diag = None
    best_key = (np.inf, np.inf, np.inf)

    all_diags = []

    for st in starts:
        theta_hat, diag = _fit_one_start(st)
        all_diags.append(diag)

        if theta_hat is None:
            continue
        if not diag["finite_theta"] or not diag["finite_pred"]:
            continue

        key = (
            float(diag["sse"]),
            float(diag["grad_inf_norm"]),
            float(diag["jtj_cond"]) if np.isfinite(diag["jtj_cond"]) else np.inf,
        )
        if key < best_key:
            best_key = key
            best_theta = theta_hat
            best_diag = diag

    if best_theta is None:
        sigma_hat = np.nan
        diagnostics = {
            "success": False,
            "reason": "all_starts_failed",
            "all_diags": all_diags,
        }
        if return_diagnostics:
            return None, sigma_hat, model_func, jac_funcs, diagnostics
        return None, sigma_hat, model_func, jac_funcs

    sigma_hat = _sigma_hat(best_theta)

    diagnostics = {
        "success": True,
        "sse": float(best_diag["sse"]),
        "sigma_hat": float(sigma_hat),
        "grad_inf_norm": float(best_diag["grad_inf_norm"]),
        "jtj_cond": float(best_diag["jtj_cond"]),
        "param_norm": float(best_diag["param_norm"]),
        "max_abs_param": float(best_diag["max_abs_param"]),
        "polish_attempted": bool(best_diag["polish_attempted"]),
        "polish_success": bool(best_diag["polish_success"]),
        "polish_kept": bool(best_diag["polish_kept"]),
        "stable": bool(
            np.isfinite(best_diag["sse"]) and
            np.isfinite(best_diag["grad_inf_norm"]) and
            np.isfinite(best_diag["param_norm"]) and
            np.isfinite(best_diag["max_abs_param"]) and
            best_diag["finite_theta"] and
            best_diag["finite_pred"]
        ),
        "all_diags": all_diags,
    }

    if return_diagnostics:
        return best_theta, sigma_hat, model_func, jac_funcs, diagnostics
    return best_theta, sigma_hat, model_func, jac_funcs

def unidentifiable_parameter_set(
    eigvals, eigvecs,
    lambda_threshold=-10.0,
    lambda_gap_threshold=30.0,
    tie_frac=0.9,
    tie_cos=0.9,
    eps=1e-7,
    resid_tol=1e-10,
):
    """
    Nominate the parameters that fix the numerically unidentifiable directions.

    Step 1 -- flag directions, not parameters.  A direction i is unidentifiable
    when its eigenvalue falls below an absolute floor, or lies more than
    ``lambda_gap_threshold`` nats below the stiffest direction while being itself
    below 1:

        ln(lambda_i) < lambda_threshold
        or  (max_j ln(lambda_j) - ln(lambda_i) > lambda_gap_threshold  and  ln(lambda_i) < 0)

    Step 2 -- nominate representatives.  Let V_null be the p x d matrix whose
    columns span the flagged subspace.  The participation of coordinate k is

        u_k = sum_{i in null} V_ki**2      in [0,1],   sum_k u_k = d

    which is invariant under rotations inside the null subspace (unlike the
    loading on any individual eigenvector, which for a degenerate subspace is an
    arbitrary artefact of ``eigh``).  We then run d steps of greedy subset
    selection: take the coordinate of largest residual participation, record as
    equivalent representatives of the SAME gauge any coordinate whose residual
    participation is within ``tie_frac`` of it AND whose residual row is
    collinear with the winner's to within ``tie_cos`` (equal participation alone
    does not imply the same gauge -- two disjoint gauges also have equal
    participation, but their rows are orthogonal), and deflate the row space
    along the winner before the next step.
    This is the pivoted-QR / Golub-Van Loan subset-selection construction; it
    returns one representative per gauge direction and handles several disjoint
    gauges correctly.

    Returns
    -------
    reps : list[int]     indices of the nominated parameters
    u_null : (p,) float  participation of each coordinate in the null subspace
    gauge_id : (p,) int  gauge a nominated parameter represents (-1 otherwise)
    d : int              dimension of the numerical null subspace
    """
    eigvals = np.asarray(eigvals, dtype=float)
    eigvecs = np.asarray(eigvecs, dtype=float)
    p = eigvecs.shape[0]

    u_null = np.zeros(p, dtype=float)
    gauge_id = np.full(p, -1, dtype=int)

    if eigvals.size == 0:
        return [], u_null, gauge_id, 0

    ln_lam = np.log(np.abs(eigvals) + eps)
    finite = np.isfinite(ln_lam)
    if not np.any(finite):
        return [], u_null, gauge_id, 0
    max_ln = float(np.max(ln_lam[finite]))

    dir_mask = (ln_lam < lambda_threshold) | (
        ((max_ln - ln_lam) > lambda_gap_threshold) & (ln_lam < 0.0)
    )
    d = int(np.count_nonzero(dir_mask))
    if d == 0:
        return [], u_null, gauge_id, 0

    Vn = eigvecs[:, dir_mask]                 # p x d, orthonormal columns
    if not np.all(np.isfinite(Vn)):
        return [], u_null, gauge_id, 0
    u_null = np.sum(Vn * Vn, axis=1)

    R = Vn.copy()
    chosen = []
    for g in range(d):
        w = np.sum(R * R, axis=1)
        if len(chosen):
            w[np.asarray(chosen, dtype=int)] = -np.inf
        kmax = int(np.argmax(w))
        if (not np.isfinite(w[kmax])) or (w[kmax] <= resid_tol):
            break
        # Ties = coordinates that are equivalent representatives of the SAME
        # gauge.  Equal participation is not enough: two disjoint gauges also
        # have equal participation.  A coordinate represents the same gauge as
        # the winner only if its residual row is collinear with the winner's;
        # rows belonging to different gauges are orthogonal.
        thr = tie_frac * w[kmax]
        rmax = R[kmax, :]
        nmax = float(np.linalg.norm(rmax))
        tied = []
        for k in range(p):
            if (k in chosen) or (not np.isfinite(w[k])) or (w[k] < thr):
                continue
            rk = R[k, :]
            nk = float(np.linalg.norm(rk))
            if nk <= resid_tol or nmax <= resid_tol:
                continue
            if abs(float(rk @ rmax)) / (nk * nmax) >= tie_cos:
                tied.append(k)
        if kmax not in tied:
            tied.append(kmax)
        for k in tied:
            gauge_id[k] = g
        chosen.extend(tied)
        # deflate the row space along the winning coordinate: one gauge fixed
        v = R[kmax, :].astype(float).copy()
        nv = float(np.linalg.norm(v))
        if nv <= resid_tol:
            break
        v /= nv
        R = R - np.outer(R @ v, v)

    return chosen, u_null, gauge_id, d


def compute_eigenvecs_eigenvals_and_alignment(
    params, jac_funcs,
    theta_mle, x, y, sigma_noise, model_func,
    eps=1e-7, compute_limits=True, return_ev=False
):
    """
    Computes Fisher eigenspectrum at theta_mle, but returns one row per
    raw parameter rather than one row per eigenvector.

    For each parameter theta_k, we select the eigenvector i* with the
    largest absolute loading on theta_k:

        i*(k) = argmax_i |v_{k,i}|

    The reported ln(lambda), alignment, and whitened coordinate are then
    those of the selected eigenvector.

    This produces a parameter-indexed table, which is more convenient for
    pruning moves such as theta_k -> 0, theta_k -> 1, theta_k -> +/-inf.
    The per-parameter ``ln(lambda)`` column is descriptive (ordering, plots);
    the unidentifiable set used for pruning is nominated on the spectrum itself
    by ``unidentifiable_parameter_set``.

    With ``compute_limits=True`` the table also carries the squared Fisher
    distances (TOTAL Fisher) to the boundaries theta_k = 0, 1, +inf, -inf in
    the columns 'zero', 'one', '+inf', '-inf'.
    """
    p = len(params)
    n = len(y)

    theta_mle = np.asarray(theta_mle, dtype=float)

    # ---------- Build Jacobian J (n x p) ----------
    J = np.zeros((n, p), dtype=float)
    call_args = tuple(theta_mle) + tuple(x)

    for j, jf in enumerate(jac_funcs):
        col = jf(*call_args)
        col = _as_len_n(col, n)
        J[:, j] = col

    # ---------- Empirical Fisher (PER-SAMPLE) ----------
    F_emp = (J.T @ J) / (n * (sigma_noise**2 + eps))

    # Eigendecomposition
    eigvals, eigvecs = np.linalg.eigh(F_emp)

    # Sort by stiffness: descending eigenvalues
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    # Eigenmode alignment with theta
    a = eigvecs.T @ theta_mle

    # Whitened eigenmode coordinate
    tilde_theta_sq = np.maximum(eigvals, 0.0) * (a ** 2)

    rows = []

    for k, par in enumerate(params):
        loadings = eigvecs[k, :]

        abs_loadings = np.abs(loadings)
        max_loading = np.max(abs_loadings)

        tie_idxs = np.where(
            np.isclose(abs_loadings, max_loading, rtol=1e-10, atol=1e-12)
        )[0]

        if len(tie_idxs) == 0:
            # All loadings are NaN (degenerate / all-NaN Jacobian).
            # Fall back to index 0 so the row can still be appended with
            # NaN values rather than crashing.
            tie_idxs = np.array([0], dtype=int)

        best_i = int(tie_idxs[np.argmin(eigvals[tie_idxs])])

        lam_i = float(eigvals[best_i])
        loading_i = float(loadings[best_i])
        alignment_i = float(a[best_i])
        tilde_i = float(tilde_theta_sq[best_i])

        rows.append({
            "main_param": par,
            "main_param_idx": k,
            "best_eig_idx": best_i,
            "ln(lambda)": np.log(np.abs(lam_i) + eps),
            "eigval": lam_i,
            "loading": loading_i,
            "loading_abs": abs(loading_i),
            "loading_sq": loading_i ** 2,
            "param_magnitude": abs(float(theta_mle[k])),
            "alignment": alignment_i,
            "log_tilde_theta_i^2": np.log(tilde_i + eps),
        })

    df = pd.DataFrame(rows)

    if compute_limits:
        # Invert the TOTAL Fisher matrix once for all coordinates.  A failed
        # inversion returns None, which the distance function turns into +inf
        # for every boundary, so no move is nominated.  The event is flagged in
        # df.attrs["fisher_inv_failed"].
        outs = []
        F_tot = n * F_emp
        Finv_tot = safe_fisher_pinv(F_tot)
        for k in range(p):
            outs.append(
                fisher_distances_all_boundaries_for_k(theta_mle, Finv_tot, k))
        outs = pd.DataFrame(outs)

        df['zero'] = outs['zero'].to_list()
        df['one'] = outs['one'].to_list()
        df['+inf'] = outs['+inf'].to_list()
        df['-inf'] = outs['-inf'].to_list()
        df.attrs["fisher_inv_failed"] = bool(Finv_tot is None)

    # Sort by selected eigenvalue: sloppy-associated parameters first
    df = df.sort_values("ln(lambda)", ascending=True).reset_index(drop=True)

    if return_ev:
        return df, eigvecs
    else:
        return df

def fisher_distance_sq_to_coordinate_boundary(
    theta_hat: np.ndarray,
    Finv: np.ndarray,
    k: int,
    target: float,
    eps: float = 1e-12
) -> float:
    """
    Local (quadratic / constant-metric) Fisher distance squared from theta_hat to the
    hyperplane boundary theta_k = target.

    Constraint: c(theta) = theta_k - target = 0,  grad c = e_k.
    Formula:
        d_F^2 = (theta_hat[k] - target)^2 / (F^{-1})_{kk}.

    Notes on scaling:
      - If F is the *per-sample* Fisher (your F_emp = (J^T J)/(n*sigma^2)),
        then this distance is measured in the per-sample Fisher metric.
      - If you want distance in the *total* Fisher metric, use F_tot = n * F_emp,
        which will multiply the distance by n (since (F_tot^{-1}) = (1/n) F_emp^{-1}).

    Returns:
        scalar d_F^2.
    """
    theta_hat = np.asarray(theta_hat, dtype=float)
    denom = float(Finv[k, k])
    denom = max(denom, eps)  # guard against nonpositive/near-zero due to numerics

    num = float(theta_hat[k] - target)
    return (num * num) / denom

def fisher_distance_sq_to_infinity_arctan(
    theta_hat: np.ndarray,
    Finv: np.ndarray,
    k: int,
    sign: int,
    eps: float = 1e-12
) -> float:
    """
    Local Fisher distance squared from theta_hat to the boundary theta_k -> +/- infinity,
    implemented via the compactification phi_k = arctan(theta_k), so that phi_k -> +/- pi/2.

    Reparameterization:
        phi_k = arctan(theta_k)          in (-pi/2, pi/2)
        theta_k = tan(phi_k)
        d theta_k / d phi_k = 1 + theta_k^2

    Metric pullback at theta_hat:
        F_phi = J^T F J, where J is diagonal with J_kk = 1 + theta_k^2 and J_jj = 1 (j != k)

    Boundary in phi-space:
        c(phi) = phi_k - sign*(pi/2) = 0,   sign=+1 for +inf, sign=-1 for -inf.

    Distance formula in phi-space:
        d_F^2 = (phi_hat[k] - sign*pi/2)^2 / (F_phi^{-1})_{kk}.

    IMPORTANT simplification (single-coordinate transform):
        F_phi^{-1} = J^{-1} F^{-1} J^{-1}
        => (F_phi^{-1})_{kk} = (F^{-1})_{kk} / (1+theta_k^2)^2.

    Therefore:
        d_F^2 = (phi_hat - sign*pi/2)^2 * (1+theta_k^2)^2 / (F^{-1})_{kk}.

    Notes on scaling:
      - Same as above: if F is per-sample Fisher, distance is per-sample metric.
        Using F_tot = n*F_emp scales distances by n.

    Args:
        sign: +1 for +infty, -1 for -infty.

    Returns:
        scalar d_F^2.
    """
    if sign not in (+1, -1):
        raise ValueError("sign must be +1 (for +inf) or -1 (for -inf).")

    theta_hat = np.asarray(theta_hat, dtype=float)
    
    denom = float(Finv[k, k])
    denom = max(denom, eps)

    theta_k = float(theta_hat[k])
    phi_k = float(np.arctan(theta_k))  # in (-pi/2, pi/2)

    # Angular gap to boundary in phi-space
    gap = phi_k - sign * (np.pi / 2.0)

    # Jacobian factor d theta / d phi = (1 + theta^2)
    Jkk = 1.0 + theta_k * theta_k

    return (gap * gap) * (Jkk * Jkk) / denom


def safe_fisher_pinv(F, rcond: float = 1e-15):
    """Moore-Penrose pseudo-inverse of the Fisher matrix, or ``None``.

    Returns ``None`` -- never raises -- when the matrix cannot be inverted,
    either because it holds non-finite entries (the Fisher information does not
    exist at this theta on this design: the Jacobian overflowed) or because
    LAPACK's SVD failed to converge.  Callers translate ``None`` into an
    infinite Fisher distance to every boundary: d_F^2 is the evidence that a
    boundary is NEAR, every consumer tests ``d2 < snr_sq``, so +inf means "no
    evidence this boundary is reachable" and nothing is nominated.

    For an invertible matrix the result equals ``np.linalg.pinv(F, rcond)``.
    The inverse is computed once per spectrum by the caller and shared by all
    distance queries.
    """
    F = np.asarray(F, dtype=float)
    if F.size == 0:
        return None
    if not np.all(np.isfinite(F)):
        return None
    try:
        return np.linalg.pinv(F, rcond=rcond)
    except np.linalg.LinAlgError:
        return None


def _all_boundaries_unreachable(boundaries):
    """The d_F^2 dictionary returned when the metric is not invertible."""
    return {b: np.inf for b in boundaries}


def fisher_distances_all_boundaries_for_k(
    theta_hat: np.ndarray,
    Finv: np.ndarray,
    k: int,
    eps: float = 1e-12,
    boundaries: list = ['zero','one','+inf','-inf']
) -> dict:
    """
    Convenience: compute d_F^2 to the canonical scalar boundaries for coordinate k:
      - theta_k = 0
      - theta_k = 1
      - theta_k -> +inf (via arctan)
      - theta_k -> -inf (via arctan)

    Takes the INVERSE Fisher matrix (see ``safe_fisher_pinv``).  ``Finv=None``
    signals that the metric could not be inverted and yields ``+inf`` for every
    boundary, i.e. nothing is nominated.

    Returns dict with keys: "zero", "one", "+inf", "-inf".
    """
    if Finv is None:
        return _all_boundaries_unreachable(boundaries)
    Finv = np.asarray(Finv, dtype=float)

    out = {}
    for b_name in boundaries:
        if b_name=='zero':
            out["zero"]=fisher_distance_sq_to_coordinate_boundary(theta_hat, Finv, k, 0.0, eps=eps)
        elif b_name=='one':
            out["one"]=fisher_distance_sq_to_coordinate_boundary(theta_hat, Finv, k, 1.0, eps=eps )
        elif b_name =='+inf':
            out["+inf"] = fisher_distance_sq_to_infinity_arctan(theta_hat, Finv, k, +1,  eps=eps)
        elif b_name=='-inf':
            out["-inf"] = fisher_distance_sq_to_infinity_arctan(theta_hat, Finv, k, -1,  eps=eps)
    return out

def fisher_distance_sq_to_signed_equality_boundary(
    theta_hat, Finv, i, j, sign=+1, eps=1e-12
):
    """
    Local Fisher distance squared to the signed identification boundary

        theta_i - sign * theta_j = 0

    where
        sign = +1  -> theta_i = theta_j
        sign = -1  -> theta_i = -theta_j

    Use the inverse of the TOTAL Fisher matrix if you want direct chi-square
    comparability.

    Takes the INVERSE Fisher matrix (see ``safe_fisher_pinv``).  ``Finv=None``
    yields ``+inf``, i.e. the identification boundary is treated as unreachable
    and the pair is not nominated.
    """
    if sign not in (+1, -1):
        raise ValueError("sign must be +1 or -1.")
    if Finv is None:
        return np.inf

    theta_hat = np.asarray(theta_hat, dtype=float)
    Finv = np.asarray(Finv, dtype=float)

    u = np.zeros(len(theta_hat))
    u[i] = 1.0
    u[j] = -float(sign)

    denom = float(u @ Finv @ u)
    denom = max(denom, eps)

    num = float(theta_hat[i] - sign * theta_hat[j])
    return (num * num) / denom


def rel_diff(x):
    x=np.array(x)
    num = np.abs(x[:, None] - x[None, :])
    den = (np.abs(x[:, None]) + np.abs(x[None, :])) / 2
    return num / den

def sort_identical_values(x, threshold=0.05):

    diffmatrix=rel_diff(x)
    # Upper triangular part, excluding diagonal
    i, j = np.triu_indices_from(diffmatrix, k=1)

    # Keep only entries below threshold
    mask = diffmatrix[i, j] < threshold
    i_f, j_f = i[mask], j[mask]
    d_f = diffmatrix[i_f, j_f]

    # Sort by increasing relative distance
    order = np.argsort(d_f)
    pairs_sorted = list(zip(i_f[order], j_f[order], d_f[order]))
    return pairs_sorted

def signed_identification_candidates(params, theta_mle, threshold=0.05):
    """
    Find pairs of parameters with similar absolute value.

    Returns a list of dicts with:
      - i, j: parameter indices
      - pi, pj: parameter symbols
      - sign: +1 if same sign, -1 if opposite sign
      - rel_diff: relative difference in |theta|
    """
    theta_mle = np.asarray(theta_mle, dtype=float)
    abs_theta = np.abs(theta_mle)

    pairs = sort_identical_values(abs_theta, threshold=threshold)

    out = []
    for i, j, d in pairs:
        s = +1 if theta_mle[i] * theta_mle[j] >= 0 else -1
        out.append({
            "i": int(i),
            "j": int(j),
            "pi": params[int(i)],
            "pj": params[int(j)],
            "sign": s,
            "rel_diff": float(d),
        })
    return out


# ============================================================================
# Pruning moves and drivers
# ============================================================================

def cancel_removable_singularities(expr, xs):
    """
    Cancel common numerator/denominator factors **term by term**, removing
    *removable* singularities created by a saturation, e.g.

        x**2 / (g*x**2 + p*x)  ->  x / (g*x + p)

    Cancellation is applied to each top-level additive term separately, so the
    additive structure is preserved (we never ``together`` the whole sum into a
    single fraction, which would merge terms and re-parameterise the model).
    Genuine poles (denominator roots that are not shared with the numerator) are
    left intact, and transcendental factors (exp, sin, ...) are untouched.
    Returns the rewritten expression (possibly identical to the input).
    """
    try:
        expr = sp.sympify(expr)
        xset = set(xs)
        new_terms = []
        for t in sp.Add.make_args(expr):
            # A removable 0/0 is a rational phenomenon.  Running
            # sp.cancel on a term containing a transcendental function OF AN
            # INPUT VARIABLE cannot help and is occasionally very expensive
            # (observed: minutes on exponential ratios), so skip those terms.
            try:
                transcendental_in_x = any(
                    (not sub.args[0].free_symbols.isdisjoint(xset))
                    for sub in t.atoms(sp.exp, sp.log, sp.sin, sp.cos,
                                       sp.tan, sp.tanh, sp.sinh, sp.cosh,
                                       sp.besselj, sp.bessely)
                    if sub.args
                )
            except Exception:
                transcendental_in_x = True
            if transcendental_in_x:
                new_terms.append(t)
                continue
            try:
                new_terms.append(sp.cancel(t))
            except Exception:
                new_terms.append(t)
        return sp.Add(*new_terms)
    except Exception:
        return expr


def find_saturation_bound(remove_par, saturation_type_dict, model_sym, model_func, params,
                          theta_mle, xs, x, y, def_bounds, init_loc,
                          init_scale, beta, criterion, sigma_noise=None,
                          criterion_value=None, node_states=None,
                          verbose=False, second_crit='BIC', second_crit_value=None, selection_tol=1e-2,
                          screening_ftol=1e-4, screening_gtol=1e-4, screening_maxiter=500,
                          n_jobs_cv=1,
                          cv_equiv_k=1.0, cancel_removable=False):
    """
    Evaluate every boundary in ``saturation_type_dict`` for ``remove_par``
    (e.g. 0, 1, +oo, -oo, or a signed identification) by refitting the reduced
    model and computing its K-fold CV-MSE and secondary criterion, and return
    the best one according to ``select_best_pruning_result``.

    Only ``criterion='CV'`` is supported; ``second_crit`` is 'BIC' or 'MDL'.

    Screening fits use loose tolerances (screening_ftol/gtol/maxiter) for speed.
    The CV warm start is the saturated model's full-data MLE (theta_k), which is
    usually much closer to each fold's optimum than the original model's theta.

    Parameters
    ----------
    screening_ftol, screening_gtol : float
        Loose function/gradient tolerances used during candidate screening.
        Tighter values produce more accurate rankings at higher cost.
        Defaults: 1e-4 (vs 1e-10 for final fits). For well-specified models
        where convergence is achievable, you may tighten these.
    screening_maxiter : int
        Max function evaluations per least_squares call during screening.
        Default 500 (vs 5000 for final fits).
    n_jobs_cv : int
        n_jobs passed to kfold_cv_mse_for_sympy_model. Set to 1 when the
        candidate loop is already parallelised to avoid nested joblib workers.
    """

    n = len(y)
    # Current model values
    if not criterion:
        print('criterion other than CV not allowed')
    else:
        if criterion == 'CV':
            # criterion_value can be precomputed and passed in
            current_model_values = criterion_value
            current_2crit = second_crit_value
        else:
            print('criterion other than CV not allowed')

    new_params = params.copy()
    new_model_sym = model_sym  # copy not required; SymPy exprs are immutable
    dict_theta = {p: a for p, a in zip(new_params, theta_mle)}

    dict_val_diff = {}
    dict_val_diff_2nd = {}
    dict_model_sym = {}
    dict_params = {}
    dict_model_func = {}
    dict_jac_func = {}
    dict_theta_mle = {}
    dict_sigma_noise = {}
    dict_model_value = {}
    dict_model_value_2nd = {}
    dict_cv_se = {}          # SE of CV estimate for each saturation candidate

    for key, sat_val in saturation_type_dict.items():
        # Saturate remove_par.  No global sp.cancel() here: for transcendental
        # or Abs-containing models it merges additive terms into a single
        # fraction, and for polynomial models it expands factored structures
        # (e.g. C0*(C1+x)**2), losing interpretable structure.  Removable
        # singularities are handled by the opt-in ``cancel_removable`` path.
        sat_model_sym = new_model_sym.subs({remove_par: sat_val})

        # control parameter-only function proliferation
        sat_new_params = add_k_params(list(set(new_params) - set((remove_par,))), k=5, real=True)
        sat_model_sym, contract_init = contract_param_only_subtrees(
            sat_model_sym, sat_new_params, xs, theta_map=dict_theta)
        # adjust params with only used ones; sort for deterministic ordering
        sat_new_params = sorted(sat_model_sym.free_symbols - set(xs), key=lambda s: str(s))

        dict_model_sym[key] = sat_model_sym
        dict_params[key] = sat_new_params

        # Refit with parameter removed/saturated using loose screening tolerances.
        # Two attempts:
        #   Attempt 1 — warm start from the full-model MLE.  Fast convergence
        #               when the landscape is smooth after saturation.
        #   Attempt 2 — pure random initialisation (guess_init=None).  Escapes
        #               bad local minima that the warm start can be trapped in
        #               when a near-degenerate cluster (e.g. C3≈0 entangled with
        #               C13) makes the landscape highly non-convex after the
        #               saturation move.
        # The better result (lower RMSE on the full training set) is kept and
        # used as the warm start for the subsequent k-fold CV.
        try:
            # --- Attempt 1: warm start ---
            guess_init = [contract_init.get(p, dict_theta.get(p, np.nan)) for p in sat_new_params]
            theta_k, sigma_k, f_k, jac_k = compute_lambdify_and_mle_estimates(
                sat_model_sym, sat_new_params, xs, x, y, def_bounds, init_loc, init_scale,
                guess_init=guess_init,
                ftol=screening_ftol,
                gtol=screening_gtol,
                xtol=screening_ftol,
                maxiter=screening_maxiter,
                final_polish=False,
            )

            # --- Attempt 2: random init ---
            theta_k2, sigma_k2, f_k2, jac_k2 = compute_lambdify_and_mle_estimates(
                sat_model_sym, sat_new_params, xs, x, y, def_bounds, init_loc, init_scale,
                guess_init=None,
                ftol=screening_ftol,
                gtol=screening_gtol,
                xtol=screening_ftol,
                maxiter=screening_maxiter,
                final_polish=False,
            )

            # --- Keep the better attempt by RMSE on full training data ---
            _sigma1 = sigma_k  if (isinstance(theta_k,  np.ndarray) and np.isfinite(sigma_k))  else np.inf
            _sigma2 = sigma_k2 if (isinstance(theta_k2, np.ndarray) and np.isfinite(sigma_k2)) else np.inf
            if _sigma2 < _sigma1:
                theta_k, sigma_k, f_k, jac_k = theta_k2, sigma_k2, f_k2, jac_k2

            dict_theta_mle[key] = theta_k
            dict_sigma_noise[key] = sigma_k
            dict_model_func[key] = f_k
            dict_jac_func[key] = jac_k
            dict_val_diff_2nd[key] = None
            dict_model_value_2nd[key] = None

            if isinstance(theta_k, np.ndarray):
                if criterion == 'CV':
                    # Use theta_k (saturated model MLE on full data) as CV warm start.
                    # This is usually closer to each fold's optimum than the original
                    # model's theta, and becomes especially important for misspecified
                    # models where the landscape is irregular.
                    if np.all(np.isfinite(theta_k)):
                        guess_init_cv_use = list(theta_k)
                    else:
                        guess_init_cv_use = None

                    cv_k, cv_k_se = kfold_cv_mse_for_sympy_model(
                        model_sym=sat_model_sym,
                        params=sat_new_params,
                        xs=xs,
                        x=x,
                        y=y,
                        def_bounds=def_bounds,
                        init_loc=init_loc,
                        init_scale=init_scale,
                        guess_init=guess_init_cv_use,
                        return_se=True,
                        verbose=verbose,
                        n_jobs=n_jobs_cv,
                        ftol=screening_ftol,
                        gtol=screening_gtol,
                        xtol=screening_ftol,
                        maxiter=screening_maxiter,
                    )
                    # --- removable-singularity guard (opt-in, cancel_removable=False by default) ---
                    # A saturation can create a removable 0/0 (e.g.
                    # x**2/(g*x**2+p*x)); that makes CV non-finite even though
                    # the cancelled form x/(g*x+p) is well-behaved. Only when the
                    # raw CV is non-finite do we try the per-term-cancelled form
                    # and adopt it if its CV is finite. Genuine poles stay
                    # non-finite and are still rejected.
                    if cancel_removable and not np.isfinite(cv_k):
                        _cand = cancel_removable_singularities(sat_model_sym, xs)
                        if _cand != sat_model_sym:
                            _cpar = sorted(_cand.free_symbols - set(xs), key=lambda s: str(s))
                            _cg = [dict_theta.get(p, np.nan) for p in _cpar]
                            _ct, _cs, _cf, _cj = compute_lambdify_and_mle_estimates(
                                _cand, _cpar, xs, x, y, def_bounds, init_loc, init_scale,
                                guess_init=_cg, ftol=screening_ftol, gtol=screening_gtol,
                                xtol=screening_ftol, maxiter=screening_maxiter, final_polish=False)
                            if isinstance(_ct, np.ndarray) and np.all(np.isfinite(_ct)):
                                _ccv, _ccv_se = kfold_cv_mse_for_sympy_model(
                                    model_sym=_cand, params=_cpar, xs=xs, x=x, y=y,
                                    def_bounds=def_bounds, init_loc=init_loc, init_scale=init_scale,
                                    guess_init=list(_ct), return_se=True, verbose=verbose,
                                    n_jobs=n_jobs_cv, ftol=screening_ftol, gtol=screening_gtol,
                                    xtol=screening_ftol, maxiter=screening_maxiter)
                                if np.isfinite(_ccv):
                                    # adopt the cancelled, finite-CV form for this candidate
                                    sat_model_sym, sat_new_params = _cand, _cpar
                                    theta_k, sigma_k, f_k, jac_k = _ct, _cs, _cf, _cj
                                    cv_k, cv_k_se = _ccv, _ccv_se
                                    dict_model_sym[key] = _cand
                                    dict_params[key] = _cpar
                                    dict_theta_mle[key] = _ct
                                    dict_sigma_noise[key] = _cs
                                    dict_model_func[key] = _cf
                                    dict_jac_func[key] = _cj
                    # ---------------------------------------------------------
                    model_values_k = cv_k
                    dict_cv_se[key] = cv_k_se   # store SE for downstream use
                    delta = model_values_k - current_model_values
                    
                    if second_crit=='BIC':
                        new_2crit = bic_for_sympy_model(sat_new_params,f_k,theta_k,x,y, beta=1)
                        delta_2crit = new_2crit - current_2crit
                    elif second_crit=='MDL':
                        new_2crit = mdl_for_sympy_model(sat_model_sym, xs, sat_new_params, node_states, 
                                        f_k, jac_k, 
                                        theta_k, x, y, return_components=False, beta=1
                                        )
                        delta_2crit = new_2crit - current_2crit
                    
                    
                    dict_val_diff_2nd[key] = delta_2crit
                    dict_model_value_2nd[key] = new_2crit
                else:
                    print('criterion other than CV not allowed')


                dict_val_diff[key] = delta
                dict_model_value[key] = model_values_k
            else:
                if len(dict_params[key])>1:
                    dict_val_diff[key] = np.inf
                    dict_model_value[key] = np.inf
                    if criterion=='CV':
                        dict_val_diff_2nd[key] = np.inf
                        dict_model_value_2nd[key] = np.inf
                        dict_cv_se[key] = np.inf

                else:
                    try:
                        if criterion == 'CV':
                            # No parameters left: just evaluate constant prediction under CV (same constant)
                            const_val = float(sat_model_sym.evalf())
                            # CV-MSE for constant is just MSE on heldout folds; same as overall MSE,
                            # but we keep K-fold for consistency:
                            # compute foldwise MSEs cheaply
                            y_arr = np.asarray(y)
                            n = len(y_arr)
                            rng = np.random.default_rng(0)
                            perm = rng.permutation(n)
                            folds = np.array_split(perm, 5)
                            fold_mses = []
                            for k in range(5):
                                val_idx = folds[k]
                                mse = float(np.nanmean((y_arr[val_idx] - const_val) ** 2))
                                fold_mses.append(mse)
                            model_values_k = float(np.nanmean(fold_mses))
                            fold_arr = np.array(fold_mses, dtype=float)
                            dict_cv_se[key] = float(np.nanstd(fold_arr, ddof=1) / np.sqrt(len(fold_arr)))
                            dict_val_diff[key] = model_values_k - current_model_values
                            dict_model_value[key] = model_values_k

                            if second_crit=='BIC':
                                new_2crit =   n * (1.0 + np.log(2*np.pi*float(np.mean((y - float(sat_model_sym.evalf())) ** 2)))) + np.log(n)
                            elif second_crit=='MDL': 
                                new_2crit = 0.5 * n * (1.0 + np.log(2*np.pi*float(np.mean((y - float(sat_model_sym.evalf())) ** 2)))) + np.log(node_states)
                            
                            delta_2crit = new_2crit - current_2crit
                            dict_val_diff_2nd[key] = delta_2crit
                            dict_model_value_2nd[key] = new_2crit
                        else:
                            print('criterion other than CV not allowed')
                            
                    except TypeError:
                        print('infinities appearing')
                        dict_val_diff[key] = np.inf
                        dict_model_value[key] = np.inf
                        if criterion == 'CV':
                            dict_val_diff_2nd[key] = np.inf
                            dict_model_value_2nd[key] = np.inf

        except NameError:
            if verbose:
                print('weird derivatives are happening. Kill a boundary.')
            dict_theta_mle[key] = np.nan
            dict_sigma_noise[key] = np.nan
            dict_model_func[key] = np.nan
            dict_jac_func[key] = np.nan
            dict_val_diff[key] = np.inf
            dict_model_value[key] = np.inf
            if criterion == 'CV':
                dict_val_diff_2nd[key] = np.inf
                dict_model_value_2nd[key] = np.inf
                dict_cv_se[key] = np.inf
        except ValueError:
            if verbose:
                print('function probably becoming infinity.')
            dict_theta_mle[key] = np.nan
            dict_sigma_noise[key] = np.nan
            dict_model_func[key] = np.nan
            dict_jac_func[key] = np.nan
            dict_val_diff[key] = np.inf
            dict_model_value[key] = np.inf
            if criterion == 'CV':
                dict_val_diff_2nd[key] = np.inf
                dict_model_value_2nd[key] = np.inf
                dict_cv_se[key] = np.inf
        except PrintMethodNotImplementedError:
            if verbose:
                print(f'function without asymptote pushed toward infinity:\n {remove_par} in {sat_model_sym} pushed to {key}')
            dict_theta_mle[key] = np.nan
            dict_sigma_noise[key] = np.nan
            dict_model_func[key] = np.nan
            dict_jac_func[key] = np.nan
            dict_val_diff[key] = np.inf
            dict_model_value[key] = np.inf
            if criterion == 'CV':
                dict_val_diff_2nd[key] = np.inf
                dict_model_value_2nd[key] = np.inf
                dict_cv_se[key] = np.inf
        except KeyError:
            if verbose:
                print(f'Weird complex infinity happening')
            dict_theta_mle[key] = np.nan
            dict_sigma_noise[key] = np.nan
            dict_model_func[key] = np.nan
            dict_jac_func[key] = np.nan
            dict_val_diff[key] = np.inf
            dict_model_value[key] = np.inf
            if criterion == 'CV':
                dict_val_diff_2nd[key] = np.inf
                dict_model_value_2nd[key] = np.inf
                dict_cv_se[key] = np.inf
        except TypeError:
            if verbose:
                print(f'Invalid NaN comparison')
            dict_theta_mle[key] = np.nan
            dict_sigma_noise[key] = np.nan
            dict_model_func[key] = np.nan
            dict_jac_func[key] = np.nan
            dict_val_diff[key] = np.inf
            dict_model_value[key] = np.inf
            if criterion == 'CV':
                dict_val_diff_2nd[key] = np.inf
                dict_model_value_2nd[key] = np.inf
                dict_cv_se[key] = np.inf
        
    if verbose:
        print(remove_par, 'DICT VAL DIFF', dict_val_diff)
        print(remove_par, 'DICT MODEL SYM', dict_model_sym)
        if criterion=='CV':
            print(remove_par, 'DICT 2ND CRIT DIFF', dict_val_diff_2nd)
    # Choose saturation with smallest change


    keys = list(saturation_type_dict.keys())

    boundary_results = []
    for k in keys:
        boundary_results.append({
            'par': remove_par,
            'lim': k,
            'diff': dict_val_diff.get(k, np.inf),
            'sec_diff_value': dict_val_diff_2nd.get(k, np.inf),
            'cv_se': dict_cv_se.get(k, np.inf),
        })


    best_boundary, _bb_accept, _bb_reason = select_best_pruning_result(
        boundary_results,
        criterion=criterion,
        beta=beta,
        sigma_noise=sigma_noise,
        current_cv=criterion_value,
        selection_tol=selection_tol,
        cv_diff_key='diff',
        secondary_key='sec_diff_value',
        allow_infeasible_fallback=False,
        cv_equiv_k=cv_equiv_k,
    )

    # At the per-parameter level we only need a representative move for this
    # parameter; the accept/reject decision is taken later, at the candidate
    # level, by the driver.  When no boundary is feasible
    # (allow_infeasible_fallback=False returns None) fall back to the boundary
    # with the smallest finite Delta-CV; the driver will then reject it.
    if best_boundary is None:
        vals = np.array([dict_val_diff.get(k, np.inf) for k in keys], dtype=float)
        vals = np.where(np.isfinite(vals), vals, np.inf)  # NaN -> inf (infeasible)
        idx = int(np.argmin(vals)) if np.isfinite(vals).any() else 0
        saturation = keys[idx]
    else:
        saturation = best_boundary['lim']

    if verbose:
        print(f'Chosen saturation {remove_par}={saturation}')

    # Recompute eigensystem for the chosen saturation.
    # Skip if the saturated model still contains symbolic infinities (sp.oo,
    # sp.zoo, sp.nan): lambdifying such a model produces all-NaN Jacobians,
    # which makes the eigensystem computation crash on empty tie_idxs.
    _sat_model_has_special = dict_model_sym[saturation].has(
        sp.oo, sp.zoo, sp.nan
    )
    if isinstance(dict_theta_mle[saturation], np.ndarray) and not _sat_model_has_special:
        df = compute_eigenvecs_eigenvals_and_alignment(
                dict_params[saturation],
                dict_jac_func[saturation],
                dict_theta_mle[saturation],
                x, y, dict_sigma_noise[saturation],
                dict_model_func[saturation]
            )
        df = df.sort_values(by='ln(lambda)')
    else:
        df = None

    return (saturation,
            dict_val_diff[saturation],
            dict_theta_mle[saturation],
            dict_sigma_noise[saturation],
            dict_model_sym[saturation],
            dict_params[saturation],
            dict_model_func[saturation],
            dict_jac_func[saturation],
            df,
            dict_model_value[saturation],
            dict_model_value_2nd[saturation],
            dict_val_diff_2nd[saturation],
            dict_cv_se.get(saturation, np.inf))

def remove_and_recalibrate_sloppy_parameters(
    new_model_sym, new_params, model_func, jac_funcs, theta_mle,
    xs, x, y, sigma_noise, df, def_bounds, init_loc, init_scale,
    beta, criterion, criterion_value=None, second_crit=None, second_crit_value=None,
    node_states=None, verbose=False, plot=False, snr_sq=16,
    ident_threshold=0.01,
    lambda_threshold=-10.0,
    lambda_gap_threshold=30.0,  selection_tol=1e-2,
    tie_frac=0.9,
    pruning_mode="unidentifiability",
    history=None,
    stage_name="sloppy",
    screening_ftol=1e-4,
    screening_gtol=1e-4,
    screening_maxiter=500,
    n_jobs_candidates=1,
    cv_equiv_k=1.0,
    cancel_removable=False,
):
    """
    Apply one pruning mode repeatedly until no candidate move is accepted.

    At every iteration the Fisher spectrum is recomputed and the numerically
    unidentifiable directions are flagged on the SPECTRUM (absolute floor
    ``lambda_threshold`` or relative gap ``lambda_gap_threshold`` on ln lambda);
    their coordinate representatives are nominated by
    ``unidentifiable_parameter_set``.  All other parameters ("identifiable")
    form a single pool shared by the three remaining modes, each of which
    screens it with its own Fisher-distance statistic (d_F^2 < ``snr_sq``).

    pruning_mode:
      1) 'unidentifiability'
         candidates: the nominated unidentifiable parameters.
         moves: theta_k -> 0, 1, +inf, -inf, plus signed-identification
         moves theta_i = +/- theta_j that pass the Fisher screen and touch a
         nominated parameter.

      2) 'zero_compatibility'
         candidates: identifiable parameters with d_F^2(theta_k = 0) < snr_sq.
         moves: theta_k -> 0.

      3) 'infinity_compatibility'
         candidates: identifiable parameters whose sign-consistent infinity is
         within reach (d_F^2 < snr_sq).  moves: theta_k -> sign(theta_k)*inf.

      4) 'signed_identifiability'
         candidates: identifiable parameters in a plausible pair
         (|theta_i| ~ |theta_j| within ``ident_threshold`` and
         d_F^2(theta_i = +/- theta_j) < snr_sq).  moves: theta_i = +/- theta_j.

    Each candidate is evaluated by ``find_saturation_bound``; among the
    candidates, ``select_best_pruning_result`` picks the move and decides
    acceptance (CV band, BIC tie-break).

    Three-step screening strategy
    ------------------------------
    Step 1 – Loose-tolerance screening:
        All candidate saturations are evaluated with (screening_ftol,
        screening_gtol, screening_maxiter) instead of the tight defaults.
        For misspecified SR models the gradient tolerance is rarely met
        anyway, so the full maxiter budget is wasted; loose tolerances give
        an equally reliable ranking at a fraction of the cost.

    Step 2 – Optional parallel candidate evaluation:
        If n_jobs_candidates != 1, all candidates are evaluated concurrently
        via joblib.Parallel. Each worker runs its own find_saturation_bound
        independently (no shared state). CV folds are always evaluated
        serially inside find_saturation_bound, so joblib workers are never
        nested.

    Step 3 – High-quality refit of the accepted move:
        Once the best candidate is selected from the screening pass, the
        accepted model is refitted from scratch with full (tight) tolerances
        and the full multi-start budget, and the Fisher eigenspectrum is
        recomputed from this refined solution. This ensures that the model
        handed to the next pruning iteration is as accurate as possible.

    Parameters
    ----------
    screening_ftol, screening_gtol : float
        Loose tolerances for candidate screening. Default 1e-4.
    screening_maxiter : int
        Max evaluations per least_squares call during screening. Default 500.
    n_jobs_candidates : int
        Parallelism over candidates (1 = serial, -1 = all CPUs).
    cancel_removable : bool, default False
        If True, a saturated candidate whose CV is non-finite is retried after
        cancelling removable 0/0 singularities term by term (see
        ``cancel_removable_singularities``).  Off by default.

    Returns
    -------
    (new_model_sym, new_params, model_func, theta_mle, sigma_noise,
     jac_funcs, df, criterion_value, second_crit_value, efficient)
    where ``efficient`` is True if at least one parameter was removed.
    """
    valid_modes = {
        "unidentifiability",
        "zero_compatibility",
        "infinity_compatibility",
        "signed_identifiability",
    }
    if pruning_mode not in valid_modes:
        raise ValueError(
            f"Unknown pruning_mode={pruning_mode}. "
            f"Choose one of {sorted(valid_modes)}."
        )
    
    if verbose:
        print(f'Pruning mode: {pruning_mode}')

    attempt_push_toward_boundary = True
    n = len(y)
    t = np.arange(n)
    in_pars = len(new_params)

    # CV folds run serially inside each candidate evaluation (no nested
    # joblib workers when candidates are parallelised).
    n_jobs_cv = 1

    if not verbose:
        plot = False

    if plot:
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))

    while attempt_push_toward_boundary and len(new_params) > 1:

        df = df.copy()
        
        # ------------------------------------------------------------
        # 0) Fisher geometry and the unidentifiable parameter set
        #
        # lambda is thresholded on the SPECTRUM, not per parameter; the flagged
        # directions then nominate their representatives by greedy subset
        # selection.  Every mode below draws from the same pool -- the
        # complement of the nominated set -- so identifiable parameters are
        # routed by their own boundary statistic alone.
        # ------------------------------------------------------------
        p = len(new_params)
        J = np.zeros((n, p), dtype=float)
        call_args = tuple(theta_mle) + tuple(x)
        for jj, jf in enumerate(jac_funcs):
            J[:, jj] = _as_len_n(jf(*call_args), n)

        F_emp = (J.T @ J) / (n * (sigma_noise**2 + 1e-12))
        F_tot = n * F_emp   # TOTAL Fisher: chi-square / Wald comparability

        try:
            _eigvals_c, _eigvecs_c = np.linalg.eigh(F_emp)
        except np.linalg.LinAlgError:
            _eigvals_c = np.zeros(p)
            _eigvecs_c = np.eye(p)

        _reps, _u_null, _gauge_id, _n_null = unidentifiable_parameter_set(
            _eigvals_c, _eigvecs_c,
            lambda_threshold=lambda_threshold,
            lambda_gap_threshold=lambda_gap_threshold,
            tie_frac=tie_frac,
        )
        _pos = {pp: kk for kk, pp in enumerate(new_params)}
        unident_params = {new_params[k] for k in _reps}

        df['u_null'] = [float(_u_null[_pos[pp]]) if pp in _pos else 0.0
                        for pp in df['main_param']]
        df['gauge_id'] = [int(_gauge_id[_pos[pp]]) if pp in _pos else -1
                          for pp in df['main_param']]
        df['unident'] = [bool(pp in unident_params) for pp in df['main_param']]

        low_lambda_mask = df['unident'].astype(bool)
        high_lambda_mask = ~low_lambda_mask

        if verbose and _n_null > 0:
            print(f'numerical null dim = {_n_null}; '
                  f'nominated {sorted(map(str, unident_params))}')

        theta_dict = {p: float(v) for p, v in zip(new_params, theta_mle)}

        def _sign_consistent_inf_col(par):
            val = theta_dict.get(par, 0.0)
            if val > 0:
                return '+inf'
            elif val < 0:
                return '-inf'
            else:
                # conservative choice: if exactly zero, allow both signs later if needed
                return None

        # ============================================================
        # 1-2) Signed-identification candidate pairs (TOTAL Fisher F_tot)
        # ============================================================
        pair_info = signed_identification_candidates(
            new_params, theta_mle, threshold=ident_threshold
        )

        # One inversion for the whole pair loop.  None => every pair scores
        # +inf and no identification move is proposed.
        Finv_tot = safe_fisher_pinv(F_tot)

        plausible_ident_rows = []
        for pair in pair_info:
            i = pair["i"]
            j = pair["j"]
            s = pair["sign"]

            d2_ident = fisher_distance_sq_to_signed_equality_boundary(
                theta_mle, Finv_tot, i, j, sign=s
            )
            p_ident = chi2.sf(d2_ident, df=1)

            if d2_ident < snr_sq:
                plausible_ident_rows.append({
                    "i": i,
                    "j": j,
                    "pi": pair["pi"],
                    "pj": pair["pj"],
                    "sign": s,
                    "d2_ident": d2_ident,
                    "p_ident": p_ident,
                })

        # ============================================================
        # 3) Build candidate pool according to pruning_mode
        # ============================================================
        if pruning_mode == "unidentifiability":
            candidate_params_curvs = df.loc[
                low_lambda_mask, ['main_param', 'ln(lambda)']
            ].copy()

            # keep only signed-identification pairs touching low-lambda params
            low_lambda_params = set(candidate_params_curvs['main_param'].tolist())
            plausible_ident_rows = [
                row for row in plausible_ident_rows
                if (row["pi"] in low_lambda_params) or (row["pj"] in low_lambda_params)
            ]

            # if a plausible signed-identification move exists, ensure the "pi" parameter is present
            existing_params = set(candidate_params_curvs["main_param"].tolist())
            extra_rows = []
            for row in plausible_ident_rows:
                if row["pi"] not in existing_params:
                    extra_rows.append({
                        "main_param": row["pi"],
                        "ln(lambda)": np.nan
                    })
                    existing_params.add(row["pi"])
            if len(extra_rows) > 0:
                candidate_params_curvs = pd.concat(
                    [candidate_params_curvs, pd.DataFrame(extra_rows)],
                    ignore_index=True
                )

        elif pruning_mode == "zero_compatibility":
            if 'zero' in df.columns:
                mask = high_lambda_mask & (df['zero'] < snr_sq)
                candidate_params_curvs = df.loc[
                    mask, ['main_param', 'ln(lambda)']
                ].copy()
            else:
                candidate_params_curvs = df.iloc[0:0][['main_param', 'ln(lambda)']].copy()
            plausible_ident_rows = []

        elif pruning_mode == "infinity_compatibility":
            rows = []
            for _, row in df.iterrows():
                par = row['main_param']
                if bool(row['unident']):
                    continue   # same pool as the zero and signed modes
                inf_col = _sign_consistent_inf_col(par)
                if inf_col is None:
                    # if MLE is exactly zero, optionally allow both infinities if either is plausible
                    ok = False
                    if '+inf' in df.columns and row.get('+inf', np.inf) < snr_sq:
                        ok = True
                    if '-inf' in df.columns and row.get('-inf', np.inf) < snr_sq:
                        ok = True
                    if ok:
                        rows.append({
                            'main_param': row['main_param'],
                            'ln(lambda)': row['ln(lambda)']
                        })
                else:
                    if inf_col in df.columns and row[inf_col] < snr_sq:
                        rows.append({
                            'main_param': row['main_param'],
                            'ln(lambda)': row['ln(lambda)']
                        })
            candidate_params_curvs = pd.DataFrame(rows)
            if candidate_params_curvs.shape[0] == 0:
                candidate_params_curvs = df.iloc[0:0][['main_param', 'ln(lambda)']].copy()
            plausible_ident_rows = []

        elif pruning_mode == "signed_identifiability":
            high_lambda_params = set(df.loc[high_lambda_mask, 'main_param'].tolist())
            plausible_ident_rows = [
                row for row in plausible_ident_rows
                if row["pi"] in high_lambda_params
            ]

            rows = []
            for row in plausible_ident_rows:
                rows.append({
                    "main_param": row["pi"],
                    "ln(lambda)": np.nan
                })
            candidate_params_curvs = pd.DataFrame(rows)
            if candidate_params_curvs.shape[0] == 0:
                candidate_params_curvs = df.iloc[0:0][['main_param', 'ln(lambda)']].copy()

        # Safety: normalize empty candidate df
        if candidate_params_curvs.shape[0] == 0:
            candidate_params_curvs = df.iloc[0:0][['main_param', 'ln(lambda)']].copy()

        # candidates form a set of parameters
        if candidate_params_curvs.shape[0] > 0:
            candidate_params_curvs = (
                candidate_params_curvs
                .drop_duplicates(subset=["main_param"])
                .reset_index(drop=True)
            )

        if verbose:
            print(f'\nPruning mode: {pruning_mode}')
            print(f'Candidates: {candidate_params_curvs["main_param"].to_list()}')
            print(f'with curvatures: {candidate_params_curvs["ln(lambda)"].to_list()}\n')
            if len(plausible_ident_rows) > 0:
                print("Plausible signed-identification pairs:")
                for row in plausible_ident_rows:
                    sgn = '+' if row["sign"] == 1 else '-'
                    print(f'  {row["pi"]} = {sgn}{row["pj"]}   '
                          f'(d2={row["d2_ident"]:.3g}, p={row["p_ident"]:.3g})')
                print()

        # --------------------------------------------------------
        # 4) Allowed moves for each candidate
        # --------------------------------------------------------
        overall_saturation_boundaries = {}

        for par, curv in candidate_params_curvs[['main_param', 'ln(lambda)']].values:
            saturation_type_dict = {}
            sign_val = np.sign(theta_dict.get(par, 0.0))

            if pruning_mode == "unidentifiability":
                # no Fisher-distance screen in this mode (the metric is
                # degenerate along these directions): test all canonical
                # boundaries, both infinities regardless of the MLE sign
                saturation_type_dict['zero'] = sp.Integer(0)
                saturation_type_dict['one'] = sp.Integer(1)
                saturation_type_dict['+inf'] = +sp.oo
                saturation_type_dict['-inf'] = -sp.oo

            elif pruning_mode == "zero_compatibility":
                saturation_type_dict['zero'] = sp.Integer(0)

            elif pruning_mode == "infinity_compatibility":
                if sign_val > 0:
                    saturation_type_dict['+inf'] = +sp.oo
                elif sign_val < 0:
                    saturation_type_dict['-inf'] = -sp.oo
                else:
                    # exact zero at MLE: allow both if the candidate survived the mask
                    saturation_type_dict['+inf'] = +sp.oo
                    saturation_type_dict['-inf'] = -sp.oo

            elif pruning_mode == "signed_identifiability":
                pass  # only equality moves added below

            overall_saturation_boundaries[par] = saturation_type_dict

        # add signed-identification moves only in the modes that allow them
        if pruning_mode in {"unidentifiability", "signed_identifiability"}:
            for row in plausible_ident_rows:
                pi = row["pi"]
                if pi in overall_saturation_boundaries:
                    move_name = str(row["pi"]) + '=' + str(row['sign']) + str(row["pj"])
                    overall_saturation_boundaries[pi][move_name] = row["sign"] * row["pj"]

        # ============================================================
        # 5) Evaluate all candidate moves  (Step 1 + Step 2)
        #
        # Step 1: loose-tolerance screening — each find_saturation_bound call
        #         uses screening_ftol/gtol/maxiter instead of tight defaults.
        # Step 2: candidates are evaluated in parallel when n_jobs_candidates!=1.
        #         n_jobs_cv is already set to 1 in that case to avoid nesting.
        # ============================================================

        def _eval_one_candidate(par):
            sat_dict = overall_saturation_boundaries.get(par, {})
            if len(sat_dict) == 0:
                return None
            (lim, diff, bound_theta, bound_sigma, bound_model_sym,
             bound_model_params, bound_model_func, bound_jac_funcs,
             bound_df, bound_value, sec_bound_value, sec_diff_value,
             bound_cv_se) = find_saturation_bound(
                par, sat_dict, new_model_sym, model_func, new_params, theta_mle,
                xs, x, y, def_bounds, init_loc, init_scale, beta,
                criterion, sigma_noise=sigma_noise, criterion_value=criterion_value,
                node_states=node_states, verbose=verbose,
                second_crit=second_crit, second_crit_value=second_crit_value,
                selection_tol=selection_tol,
                screening_ftol=screening_ftol,
                screening_gtol=screening_gtol,
                screening_maxiter=screening_maxiter,
                n_jobs_cv=n_jobs_cv,
                cv_equiv_k=cv_equiv_k,
                cancel_removable=cancel_removable,
            )
            return {
                'par': par,
                'lim': lim,
                'diff': diff,
                'cv_se': bound_cv_se,
                'theta': bound_theta,
                'sigma': bound_sigma,
                'model_sym': bound_model_sym,
                'params': bound_model_params,
                'model_func': bound_model_func,
                'jac_funcs': bound_jac_funcs,
                'df': bound_df,
                'model_value': bound_value,
                'sec_crit_value': sec_bound_value,
                'sec_diff_value': sec_diff_value,
            }

        candidate_params_list = candidate_params_curvs["main_param"].tolist()

        if n_jobs_candidates == 1:
            raw_results = [_eval_one_candidate(par) for par in candidate_params_list]
        else:
            raw_results = Parallel(n_jobs=n_jobs_candidates, backend="loky")(
                delayed(_eval_one_candidate)(par) for par in candidate_params_list
            )

        candidate_results = [r for r in raw_results if r is not None]

        if len(candidate_results) > 0:

            if criterion == 'CV':
                # CV-MSE-unit feasibility band (matches select_best_pruning_result).
                if criterion_value is not None and np.isfinite(criterion_value):
                    tol_cv = beta * criterion_value
                else:
                    tol_cv = beta * (sigma_noise ** 2)

                feasible = [
                    r for r in candidate_results
                    if np.isfinite(r['diff']) and r['diff'] <= tol_cv
                ]

                for r in feasible:
                    if verbose:
                        print(f'candidate {r["par"]} : delta {str(second_crit).lower()} {r["sec_diff_value"]}')

                # select_best_pruning_result owns the accept/reject decision.
                # It returns the chosen candidate, an accept flag, and a reason
                # string naming the rule that fired.
                best_result, accept_change, select_reason = select_best_pruning_result(
                    candidate_results,
                    criterion=criterion,
                    beta=beta,
                    sigma_noise=sigma_noise,
                    current_cv=criterion_value,
                    selection_tol=selection_tol,
                    cv_diff_key='diff',
                    secondary_key='sec_diff_value',
                    allow_infeasible_fallback=False,
                    cv_equiv_k=cv_equiv_k,
                )

                if best_result is not None:
                    remove_par = best_result['par']
                    lim        = best_result['lim']
                    diff       = best_result.get('sec_diff_value', np.inf)
                else:
                    remove_par = None
                    lim        = None
                    diff       = np.inf
                    accept_change = False

            else:
                print('criterion other than CV not allowed')
                accept_change = False

            if not accept_change:
                if verbose:
                    if best_result is None:
                        print(f'No acceptable candidate in mode "{pruning_mode}" '
                              f'(reason: {select_reason})')
                    else:
                        print(f'\n Pushing {remove_par} toward {lim} '
                              f'caused major model change: delta = {diff}')
                attempt_push_toward_boundary = False
            else:
                # ----------------------------------------------------------
                # Step 3: high-quality refit of the accepted move.
                # The screening pass used loose tolerances; now refit the
                # winning model with tight tolerances and the full multi-start
                # budget, using the screening solution as a warm start.
                # ----------------------------------------------------------
                accepted_model_sym = best_result['model_sym']
                accepted_params    = best_result['params']
                screen_theta       = best_result['theta']

                guess_refined = (
                    list(screen_theta)
                    if isinstance(screen_theta, np.ndarray) and np.all(np.isfinite(screen_theta))
                    else None
                )

                theta_refined, sigma_refined, model_func_refined, jac_funcs_refined = \
                    compute_lambdify_and_mle_estimates(
                        accepted_model_sym, accepted_params, xs, x, y,
                        def_bounds, init_loc, init_scale,
                        guess_init=guess_refined,
                        # tight defaults (ftol=gtol=1e-10, maxiter=5000)
                    )

                if theta_refined is not None:
                    new_params    = accepted_params
                    new_model_sym = accepted_model_sym
                    theta_mle     = theta_refined
                    sigma_noise   = sigma_refined
                    model_func    = model_func_refined
                    jac_funcs     = jac_funcs_refined
                    # Recompute eigenspectrum from the refined solution
                    df = compute_eigenvecs_eigenvals_and_alignment(
                        new_params, jac_funcs, theta_mle, x, y, sigma_noise, model_func
                    )
                    df = df.sort_values(by='ln(lambda)')
                else:
                    # Fallback: keep the screening solution if the refined fit fails
                    if verbose:
                        print(f'[Step3] Refined refit failed for {remove_par}->{lim}; '
                              f'keeping screening solution.')
                    new_params    = accepted_params
                    new_model_sym = accepted_model_sym
                    theta_mle     = best_result['theta']
                    sigma_noise   = best_result['sigma']
                    model_func    = best_result['model_func']
                    jac_funcs     = best_result['jac_funcs']
                    df            = best_result['df']

                criterion_value    = best_result['model_value']
                second_crit_value  = best_result['sec_crit_value']

                if history is not None:
                    record_snapshot(
                        history,
                        stage=stage_name,
                        action="removed",
                        removed_param=str(remove_par),
                        boundary=str(lim),
                        model_sym=new_model_sym,
                        params=new_params,
                        theta_mle=theta_mle,
                        sigma_noise=sigma_noise,
                        df=df,
                        x=x,
                        model_func=model_func,
                        criterion=criterion,
                        criterion_value=criterion_value,
                        second_crit=second_crit,
                        second_crit_value=second_crit_value,
                        pruning_mode=pruning_mode,
                    )

                if verbose and isinstance(df, pd.DataFrame):
                    print(f'\n Pushing {remove_par} toward {lim}: new model is\n{new_model_sym}\n')
                    print(df.round(2).to_latex(index=False, float_format="%.2f"), '\n')

                if plot:
                    call_args = tuple(theta_mle) + tuple(x)
                    ax.plot(t, model_func(*call_args), label=f'removed {remove_par} ({lim})', linestyle='--')
        else:
            attempt_push_toward_boundary = False

    if plot:
        ax.legend()
        ax.set_ylabel(r'$f(x,\theta)$')
        ax.set_xlabel('sample index')
        plt.show()

    end_pars = len(new_params)
    efficient = (in_pars - end_pars) > 0

    return (
        new_model_sym, new_params, model_func, theta_mle, sigma_noise,
        jac_funcs, df, criterion_value, second_crit_value, efficient
    )


def pruning_move_priority(lim):
    """
    Lower value = preferred pruning move.

    Preference:
      zero > +inf > -inf > one > signed equality > everything else
    """
    lim_str = str(lim)

    if lim_str == 'zero':
        return 0
    if lim_str == '+inf':
        return 1
    if lim_str == '-inf':
        return 2
    if lim_str == 'one':
        return 3

    # signed equality moves are named e.g. "c=-1l"
    if '=' in lim_str:
        return 4

    return 99

def select_best_pruning_result(
    results,
    criterion,
    beta=1.0,
    sigma_noise=None,
    current_cv=None,
    selection_tol=1e-3,
    cv_diff_key='diff',
    secondary_key='sec_diff_value',
    allow_infeasible_fallback=True,
    cv_equiv_k=1.0,
    eps=1e-12,
):
    """
    Select the best pruning move and decide whether to accept it.

    Decision rule: CV primary, BIC secondary. The same
    rule is used at both levels: choosing among boundary moves for a single
    parameter (inside ``find_saturation_bound``) and choosing among candidate
    parameters (inside the driver loop). Only the caller's value of
    ``allow_infeasible_fallback`` differs.

    Each result dict should contain:
      - 'lim'         : move label, e.g. 'zero', '+inf', 'c=-1l'
      - 'par'         : parameter symbol/name (used only for deterministic ties)
      - cv_diff_key   : ΔCV = CV_MSE(reduced) − CV_MSE(current) (negative = better)
      - 'cv_se'       : across-fold SE of the CV estimate
      - secondary_key : ΔBIC = BIC(reduced) − BIC(current), baseline fixed for
                        the whole iteration (negative = better)

    Decision rule (CV decides admissibility; BIC only breaks CV ties)
    -----------------------------------------------------------------
    ① Feasibility : keep finite-ΔCV moves with ΔCV ≤ beta·current_cv
                    (falls back to beta·sigma_noise² if current_cv is missing).
                    Empty → infeasible (least-bad move returned if fallback on).
    ② Select      : among feasible moves take the minimum ΔCV. Moves within
                    cv_equiv_k·cv_se of the best are predictively tied (CV
                    cannot rank them); among those take the minimum ΔBIC
                    (dimension-aware — these moves collapse subtrees of
                    differing dimension), then pruning_move_priority, then
                    (str(par), str(lim)) for deterministic reproducibility.

    BIC enters ONLY at step ② to rank predictively-equivalent, individually-
    regular reduced models; it never decides admissibility (step ①).
    ``cv_se`` is floored to ``eps`` so a degenerate fold spread cannot collapse
    the tie band.

    Returns
    -------
    (best, accept, reason)
        best   : chosen result dict, or None when nothing is selected.
        accept : bool, whether the move should be applied.
        reason : str — one of
                 'empty', 'no_finite', 'criterion_not_cv', 'infeasible', 'cv_min'.

    When ``allow_infeasible_fallback`` is True and the move is rejected, ``best``
    still carries the least-bad ranked move (so per-parameter callers always get
    a representative move) while ``accept`` is False. When it is False, rejected
    cases return (None, False, reason).
    """
    if len(results) == 0:
        return None, False, 'empty'

    if criterion != 'CV':
        print('pruning criterion other than CV not allowed')
        return None, False, 'criterion_not_cv'

    if sigma_noise is None:
        raise ValueError("sigma_noise must be provided when criterion == 'CV'.")

    def _cv(r):
        v = r.get(cv_diff_key, np.inf)
        return np.inf if v is None else float(v)

    def _bic(r):
        v = r.get(secondary_key, np.inf)
        return np.inf if v is None else float(v)

    def _cvse(r):
        v = r.get('cv_se', 0.0)
        try:
            v = float(v)
        except (TypeError, ValueError):
            return eps
        if not np.isfinite(v):
            return eps
        return max(v, eps)

    def _tiebreak(r):
        return (
            pruning_move_priority(r.get('lim')),
            str(r.get('par', '')),
            str(r.get('lim', '')),
        )

    finite = [r for r in results if np.isfinite(_cv(r))]
    if len(finite) == 0:
        return None, False, 'no_finite'

    # ① Feasibility — band in CV-MSE units: tolerate CV worsening up to
    # beta * (current model's CV-MSE). Falls back to beta * sigma_noise**2
    # (training-MSE units) only if current_cv is unavailable/non-finite.
    if current_cv is not None and np.isfinite(current_cv):
        tol_cv = beta * current_cv
    else:
        tol_cv = beta * (sigma_noise ** 2)
    feasible = [r for r in finite if _cv(r) <= tol_cv]
    if len(feasible) == 0:
        if not allow_infeasible_fallback:
            return None, False, 'infeasible'
        best_cv = min(_cv(r) for r in finite)
        near_best = [r for r in finite if _cv(r) <= best_cv + selection_tol]
        return min(near_best, key=_tiebreak), False, 'infeasible'

    # ② Minimum ΔCV, but CV has finite resolution. Competitors whose ΔCV lies
    # within cv_equiv_k * cv_se of the best are statistically indistinguishable
    # in predictive risk — CV cannot rank them. Because these moves collapse
    # whole subtrees, the tied competitors differ in dimension, so we break the
    # tie with a dimension-aware criterion: minimum ΔBIC. Its n*ln(MSE) term
    # penalises destroying real structure, while its k*ln(n) term rewards the
    # larger justified simplification. BIC is used ONLY here — to rank
    # predictively-equivalent, individually-regular reduced models — never to
    # decide admissibility, which remains CV (feasibility) alone.
    best_cv = min(_cv(r) for r in feasible)
    cvse_best = max(
        (_cvse(r) for r in feasible if _cv(r) <= best_cv + eps),
        default=eps,
    )
    band = cv_equiv_k * cvse_best
    tied = [r for r in feasible if _cv(r) - best_cv <= band]
    return (
        min(
            tied,
            key=lambda r: (
                _bic(r),
                pruning_move_priority(r.get('lim')),
                str(r.get('par', '')),
                str(r.get('lim', '')),
            ),
        ),
        True,
        'cv_min',
    )

# ============================================================================
# Model-selection criteria
# ============================================================================
 
def mse_nll_from_data(y, yhat,beta=1.):
    # Gaussian NLL up to constants: n/2 * (1 + log(MSE))
    n = len(y)
    y = np.asarray(y).reshape(-1)
    yhat = _as_len_n(yhat, y.size)
    mse = float(np.mean((y - yhat) ** 2))
    nll = beta * 0.5 * n * (1.0 + np.log(2*np.pi*mse))
    return mse, nll

def sympy_tree_complexity(expr, node_states):
    """Simple structure penalty: (#nodes) * log(#node_states)."""
    def count_nodes(e):
        return 1 + sum(count_nodes(a) for a in getattr(e, "args", ()))
    nodes = count_nodes(expr)

    return nodes * np.log(node_states)

def mdl_for_sympy_model(
    model_sym, xs, params, node_states,
    model_func, jac_funcs,
    theta_mle, x, y,
    return_components=False, beta=1,
    sigma_noise=None, eps=1e-12,
    fisher_scaling="total",   # "total" or "per_sample"
):
    """
    MDL for a SymPy model using:
      - data term: Gaussian NLL from MSE
      - structure term: tree complexity
      - parameter term: diagonal empirical Fisher / Gauss-Newton approximation

    Parameters
    ----------
    model_sym : sympy.Expr
        Symbolic model expression.
    xs : list
        Input symbols.
    params : list
        Parameter symbols.
    node_states : int
        Number of node states for tree complexity.
    model_func : callable
        Lambdified model function with signature model_func(*theta, *x).
    jac_funcs : list
        Lambdified first-derivative functions, one per parameter.
    theta_mle : array-like
        Fitted parameter values.
    x : tuple
        Tuple of input arrays, one per input variable.
    y : array-like
        Target values.
    return_components : bool
        If True, also return component dictionary.
    beta : float
        Passed to mse_nll_from_data.
    sigma_noise : float or None
        If provided, use sigma_noise^2 in Fisher scaling.
        If None, use MSE as the variance scale.
    eps : float
        Numerical floor for stability.
    fisher_scaling : str
        "total"     -> F_jj = sum_i J_ij^2 / sigma^2
        "per_sample"-> F_jj = mean_i J_ij^2 / sigma^2

    Returns
    -------
    MDL : float
        Total MDL score.
    optionally dict of components
    """
    n = len(y)
    p = len(params)

    theta_mle = np.asarray(theta_mle, dtype=float)
    y = np.asarray(y, dtype=float)

    # predictions
    call_args = tuple(theta_mle) + tuple(x)
    yhat = _as_len_n(model_func(*call_args), n).astype(float)

    # data term
    MSE, NLL = mse_nll_from_data(y, yhat, beta=beta)

    # structure term
    tree_cplx = sympy_tree_complexity(model_sym, node_states)

    # variance scale for Fisher
    if sigma_noise is None:
        sigma2 = float(MSE) + eps
    else:
        sigma2 = float(sigma_noise)**2 + eps

    # build Jacobian J (n x p)
    J = np.zeros((n, p), dtype=float)
    for j, jf in enumerate(jac_funcs):
        col = _as_len_n(jf(*call_args), n).astype(float)
        J[:, j] = col

    # diagonal empirical Fisher / Gauss-Newton
    if fisher_scaling == "total":
        FIM_diag = np.sum(J**2, axis=0) / sigma2
    elif fisher_scaling == "per_sample":
        FIM_diag = np.mean(J**2, axis=0) / sigma2
    else:
        raise ValueError("fisher_scaling must be 'total' or 'per_sample'")

    FIM_diag = np.maximum(FIM_diag, eps)

    # parameter resolution
    abs_theta = np.abs(theta_mle)
    Delta = np.minimum(np.sqrt(12.0 / FIM_diag), abs_theta)
    Delta = np.maximum(Delta, eps)

    # parameter code-length
    const_terms = np.where(
        abs_theta > Delta,
        np.log(abs_theta / Delta) + np.log(2.0),
        0.0
    )
    constant_cplx = float(np.sum(const_terms))

    MDL = float(NLL + tree_cplx + constant_cplx)

    if return_components:
        return MDL, {
            "NLL": float(NLL),
            "tree_complexity": float(tree_cplx),
            "constant_complexity": float(constant_cplx),
            "MSE": float(MSE),
            "sigma2_used": float(sigma2),
            "FIM_diag": FIM_diag,
            "Delta": Delta,
            "fisher_scaling": fisher_scaling,
        }

    return MDL

def bic_for_sympy_model(params, 
    model_func,  theta_mle, x, y, beta=1
):
    """
    Gaussian BIC of a fitted model:

        BIC = 2 * NLL + p * ln(n * beta),   NLL = beta * n/2 * (1 + ln(2 pi MSE)).

    With beta = 1 this is n (1 + ln(2 pi MSE)) + p ln n.
    """
    n = len(y)
    p = len(params)

    # predictions
    call_args = tuple(theta_mle) + tuple(x)
    yhat = model_func(*call_args)
    
    # data term
    _, nll = mse_nll_from_data(y, yhat, beta=beta)
    bic = 2 * nll + len(params)*np.log(n*beta)
    return bic


def _slice_inputs(x, idx):
    """x is a list/tuple of arrays; idx is an index array."""
    return [np.asarray(xi)[idx] for xi in x]


def _single_cv_fold_mse(
    k,
    folds,
    model_sym, params, xs, x, y,
    def_bounds, init_loc, init_scale,
    n_starts, maxiter,
    guess_init,
    verbose=False,
    ftol=1e-10,
    gtol=1e-10,
    xtol=1e-10,
):
    val_idx = folds[k]
    train_idx = np.concatenate([folds[j] for j in range(len(folds)) if j != k])

    x_train = _slice_inputs(x, train_idx)
    y_train = y[train_idx]
    x_val   = _slice_inputs(x, val_idx)
    y_val   = y[val_idx]

    try:
        theta_k, sigma_k, f_k, jac_k, diag_k = compute_lambdify_and_mle_estimates(
            model_sym, params, xs, x_train, y_train,
            def_bounds, init_loc, init_scale,
            n_starts=n_starts,
            maxiter=maxiter,
            guess_init=guess_init,
            ftol=ftol,
            gtol=gtol,
            xtol=xtol,
            return_diagnostics=True,
        )

        if (theta_k is None) or (not diag_k["stable"]):
            return np.inf, False

        call_args = tuple(theta_k) + tuple(x_val)
        yhat = f_k(*call_args)
        yhat = np.asarray(yhat, dtype=float).ravel()

        if (len(yhat) != len(y_val)) or (not np.all(np.isfinite(yhat))):
            return np.inf, False

        mse = float(np.nanmean((y_val - yhat) ** 2))
        return mse, True

    except Exception as e:
        if verbose:
            print(f"[CV] Fold {k} failed: {e}")
        return np.inf, False

def kfold_cv_mse_for_sympy_model(
    model_sym, params, xs, x, y,
    def_bounds, init_loc, init_scale,
    K=8, seed=0,
    n_starts=20, maxiter=5000,
    guess_init=None,
    return_se=True,
    verbose=False,
    n_jobs=1,
    backend="loky",
    ftol=1e-10,
    gtol=1e-10,
    xtol=1e-10,
):
    y = np.asarray(y)
    n = len(y)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    folds = np.array_split(perm, K)

    if n_jobs > 1:
        results = Parallel(n_jobs=n_jobs, backend=backend)(
            delayed(_single_cv_fold_mse)(
                k,
                folds,
                model_sym, params, xs, x, y,
                def_bounds, init_loc, init_scale,
                n_starts, maxiter,
                guess_init,
                verbose,
                ftol,
                gtol,
                xtol,
            )
            for k in range(K)
        )
    else: 
        results=[]
        for k in range(K):
            rr=_single_cv_fold_mse(
                k,
                folds,
                model_sym, params, xs, x, y,
                def_bounds, init_loc, init_scale,
                n_starts, maxiter,
                guess_init,
                verbose,
                ftol,
                gtol,
                xtol)
            results.append(rr)


    fold_mses = np.asarray([r[0] for r in results], dtype=float)
    fold_status = [r[1] for r in results]

    cv_mean = float(np.nanmean(fold_mses))

    if not return_se:
        return cv_mean, np.nan

    finite = np.isfinite(fold_mses)
    if finite.sum() <= 1:
        return cv_mean, np.nan

    cv_se = float(np.nanstd(fold_mses[finite], ddof=1) / np.sqrt(finite.sum()))
    return cv_mean, cv_se


# ============================================================================
# Controlled case studies
# ============================================================================

def create_model(model,n,real_sigma_noise):
    """
    Controlled case studies: an over-parametrised symbolic model and noisy data
    generated from a known (simpler) truth.

    Parameters
    ----------
    model : str
        One of 'nested', 'perturbative', 'multidimensional',
        'near_alias_fourier', 'bessel', 'rational_gauge',
        'competing_saturations', 'separable_2d', 'buried_transcendentals',
        'tanh_gauge'.
    n : int
        Number of samples.
    real_sigma_noise : float
        Standard deviation of the additive Gaussian noise.

    Returns
    -------
    x : list[np.ndarray]       one array per input variable
    y : np.ndarray             noisy targets
    model_sym : sympy.Expr     over-parametrised model to be pruned
    xs : list[sympy.Symbol]    input symbols
    params : list[sympy.Symbol]
    f_true : sympy.Expr        data-generating function
    y_true : np.ndarray        noiseless targets
    """
    if model=='nested':
        # truth: 2 sin(1.5 x) + 0.7 x^2, embedded in an 8-parameter model
        x = [np.linspace(1e-4, 5, n)]  # small input range -> x^2 and x^3 may be less informative -> can create sloppiness
        is_positive = [(xi>0).all() for xi in x]

        # 2) Symbolic model and Jacobian with sympy
        params = [sp.Symbol(f'theta{i}', real=True) for i in range(8)]
        # build xs = [x0, x1, ...] with per-dimension assumptions
        xs = []
        for i, pos in enumerate(is_positive):
            if pos:
                xs.append(sp.Symbol(f'x{i+1}', positive=True))  # positive ⇒ real
            else:
                xs.append(sp.Symbol(f'x{i+1}', real=True))      # assert just “real”

        f_true = (2 * sp.sin(1.5 * xs[0]) 
                + 0.7 * xs[0]**2) 
        
        y_true = (2 * np.sin(1.5 * x[0]) 
                + 0.7 * x[0]**2) 
        y = y_true + np.random.normal(0, real_sigma_noise, size=n)


        model_sym = params[0]*sp.sin(params[1]*xs[0]) + params[2]*xs[0]**2 + params[3]*xs[0]**3 + params[4]*1/(1+sp.exp(-params[5]*(xs[0]))) + sp.exp(params[6]*xs[0]+params[7])

    elif model=='perturbative':
        # Generate x and y data
        
        x = [np.linspace(1e-4, 1.5, n),np.linspace(1e-4, 2.5, n)]  # small input range -> x^2 and x^3, x^4... may be less informative -> can create sloppiness
        is_positive = [(xi>0).all() for xi in x]
            
        # 2) Symbolic model and Jacobian with sympy
        params = [sp.Symbol(f'theta{i}', real=True) for i in range(10)]

        # build xs = [x0, x1, ...] with per-dimension assumptions
        xs = []
        for i, pos in enumerate(is_positive):
            if pos:
                xs.append(sp.Symbol(f'x{i+1}', positive=True))  # positive ⇒ real
            else:
                xs.append(sp.Symbol(f'x{i+1}', real=True))      # assert just “real”

        
        # --- 1. True data-generating function ---
        # Example: exponential 
        f_true = sp.exp(0.7 * xs[0])

        # Full parametric model
        model_sym = params[0] + params[1]*xs[0] + params[2]*xs[0]**2 + params[3]*xs[0]**3 + params[4]*xs[0]**4 + params[5]*xs[0]**5 + params[6] * sp.sin( params[7] * xs[0]) + params[8] / (1 + sp.exp(-params[9]*xs[0]))

        
        y_true = np.exp(0.7 * x[0]) 
        y = y_true + np.random.normal(0, real_sigma_noise, size=n)

    elif model=='multidimensional':
        # Generate x and y data
        
        x = [np.random.uniform(0.001, 1.5, n), np.random.uniform(0.001, 2.5, n)]
        is_positive = [(xi>0).all() for xi in x]
            
        # 2) Symbolic model and Jacobian with sympy
        
        params = [sp.Symbol(f'theta{i}', real=True) for i in range(10)]

        # build xs = [x0, x1, ...] with per-dimension assumptions
        xs = []
        for i, pos in enumerate(is_positive):
            if pos:
                xs.append(sp.Symbol(f'x{i+1}', positive=True))  # positive ⇒ real
            else:
                xs.append(sp.Symbol(f'x{i+1}', real=True))      # assert just “real”

        
        # --- 1. True data-generating function ---
        # Example: exponential 
        f_true = sp.exp(0.7 * xs[0])

        # Full parametric model
        model_sym = params[0] + params[1]*xs[0] + params[2]*xs[0]**2 + params[3]*xs[0]**3 + params[4]*xs[1]**4 + params[5]*xs[0]**5 + params[6] * sp.sin( params[7] * xs[0]) + params[8] / (1 + sp.exp(-params[9]*xs[1]))

        
        y_true = np.exp(0.7 * x[0]) 
        y = y_true + np.random.normal(0, real_sigma_noise, size=n)

    elif model=='near_alias_fourier':
        
        x = [np.linspace(0, 30, n)]
        is_positive = [(xi>=0).all() for xi in x]

        xs = []
        for i, pos in enumerate(is_positive):
            if pos:
                xs.append(sp.Symbol(f'x{i+1}', positive=True))  # positive ⇒ real
            else:
                xs.append(sp.Symbol(f'x{i+1}', real=True))      # assert just “real”


        params = [sp.Symbol(f'theta{i}', real=True) for i in range(7)]

        # True
        f_true = 1.0*sp.sin(0.65*xs[0] + 0.5)
        y_true = np.sin(0.65*x[0] + 0.5) 

        # Overparam: add almost-same-frequency sine + bias + tiny trend
        model_sym = params[0]*sp.sin(params[1]*xs[0] + params[2]) + params[3]*sp.sin(params[4]*xs[0]) + params[5] + params[6]*xs[0]

        y = y_true + np.random.normal(0, real_sigma_noise, size=n)

    elif model=='bessel':
        
        x = [np.linspace(1e-2, 8.0, n)]
        is_positive = [(xi>0).all() for xi in x]

        # Symbols
        
        params = [sp.Symbol(f'theta{i}', real=True) for i in range(10)]

        xs = []
        for i, pos in enumerate(is_positive):
            if pos:
                xs.append(sp.Symbol(f'x{i+1}', positive=True))  # positive ⇒ real
            else:
                xs.append(sp.Symbol(f'x{i+1}', real=True))      # assert just “real”


        # True data: J0 with mild frequency
        f_true = 1.3 * sp.besselj(0, 1.6*xs[0])
        y_true = sp.lambdify((xs[0],), f_true, modules='scipy')(x[0],)

        # Overparam model: J0 + J1 + Y0 + low polynomial tail
        model_sym = params[0] * sp.exp(params[1]*xs[0]) * sp.cos(params[2]*xs[0] + params[3]) + params[4] + (params[5] + params[6]*xs[0] + params[7]*xs[0]**2) / (params[8]*xs[0] + params[9]*xs[0]**2)
        y = y_true + np.random.normal(0, real_sigma_noise, size=n)

    elif model=='rational_gauge':
        
        x = [np.linspace(0, 3.0, n)]
        is_positive = [(xi>=0).all() for xi in x]
        xs = []
        for i, pos in enumerate(is_positive):
            if pos:
                xs.append(sp.Symbol(f'x{i+1}', positive=True))  # positive ⇒ real
            else:
                xs.append(sp.Symbol(f'x{i+1}', real=True))      # assert just “real”

        a, b = sp.symbols('a b', real=True)
        # True
        f_true = (a*xs[0])/(1 + b*xs[0])
        y_true = sp.lambdify((a,b,xs[0]), f_true, "numpy")(1.5, 0.8, x[0])

        # Overparam with gauge pair (c,d): only ratio c/d matters
        params = [sp.Symbol(f'theta{i}', real=True) for i in range(9)]
        
        model_sym = (params[0]*xs[0] + params[1] + params[2]*xs[0]**2) / (params[3] + params[4]*xs[0] + params[5]*xs[0]**2) \
                + (params[6]*xs[0])/(params[7] + params[8]*xs[0])

        y = y_true + np.random.normal(0, real_sigma_noise, size=n)

    elif model=='competing_saturations':
        
        x = [np.linspace(-6, 6, n)]
        is_positive = [(xi>0).all() for xi in x]  # false, but we only use real
        xs = []
        for i, pos in enumerate(is_positive):
            if pos:
                xs.append(sp.Symbol(f'x{i+1}', positive=True))  # positive ⇒ real
            else:
                xs.append(sp.Symbol(f'x{i+1}', real=True))      # assert just “real”

        f_true = 1.0 / (1.0 + sp.exp(-1.2*xs[0]+1.5))
        y_true = sp.lambdify((xs[0],), f_true, "numpy")(x[0],)

        params = [sp.Symbol(f'theta{i}', real=True) for i in range(8)]

        # Overparam: logistic + softplus-ish + bias & slope
        model_sym = params[0] + (params[1] +  params[2] * sp.log(params[3] + sp.exp(params[4]*xs[0])))/(params[5] + params[6] * sp.exp(params[7]*xs[0]))

        y = y_true + np.random.normal(0, real_sigma_noise, size=n)

    elif model=='separable_2d':
        
        x = [np.random.uniform(0.001, 3.0, n), np.random.uniform(0.001, 3.0, n)]
        is_positive = [(xi>=0).all() for xi in x]
        xs = []
        for i, pos in enumerate(is_positive):
            if pos:
                xs.append(sp.Symbol(f'x{i+1}', positive=True))  # positive ⇒ real
            else:
                xs.append(sp.Symbol(f'x{i+1}', real=True))      # assert just “real”

        
        f_true =  1.2 * sp.exp(0.6 *xs[0]) / (0.8 + 1.8* sp.exp(0.9* xs[1]))
        y_true = sp.lambdify((xs[0],xs[1]), f_true, "numpy")(x[0], x[1])

        # Overparam: introduce an extra scale s that cancels in optimum
        params = [sp.Symbol(f'theta{i}', real=True) for i in range(10)]

    
        model_sym = params[0] * sp.exp(params[1] *xs[0]) * (params[2] + params[3] * xs[0] + params[4] * xs[0]/xs[1] + sp.exp(params[5] *xs[1] + params[6] *xs[0]+ params[7]))**-1 + params[8] + params[9]*xs[0]
        y = y_true + np.random.normal(0, real_sigma_noise, size=n)
        
    elif model == "buried_transcendentals":
        # ---- targeted move: theta -> +/-inf (remove COEFFICIENT-LESS spurious
        #      transcendental terms) ----
        # The spurious terms exp(theta2 x) and 1/(1+exp(theta3 x)) have NO
        # multiplicative parameter in front, so there is no coefficient to zero.
        # The only way to eliminate their effect is to push the internal
        # parameter to +/-inf:
        #     exp(theta2 x)      -> 0        via theta2 -> -inf   (x>0)
        #     1/(1+exp(theta3 x))-> 0        via theta3 -> +inf   (x>0)
        # A theta->0 move on those internal parameters gives exp(0)=1 and
        # 1/(1+1)=1/2 respectively -- non-zero constant residuals that (with no
        # free constant in the model) worsen the fit and are rejected, so a
        # zero-only pruner cannot remove these terms.
        x = [np.linspace(0.05, 5.0, n)]
        xs = [sp.Symbol("x1", positive=True)]
        params = [sp.Symbol(f"theta{i}", real=True) for i in range(7)]
 
        f_true = 0.5 * xs[0] + 0.7 * xs[0] ** 2
        y_true = 0.5 * x[0] + 0.7 * x[0] ** 2
 
        model_sym = (params[0] * xs[0] + params[1] * xs[0] ** 2       # true part
                     + sp.exp(params[2] * xs[0])                      # bare exp   -> -inf
                     + 1 / (1 + sp.exp(params[3] * xs[0]))            # ratio sig  -> +inf
                     + params[4] * xs[0] ** 3                         # theta->0 distractors
                     + params[5] * xs[0] ** 4
                     + params[6] * sp.sin(xs[0]))
        y = y_true + np.random.normal(0, real_sigma_noise, size=n)
 
    elif model == "tanh_gauge":
        # ---- targeted move: signed identification theta_i = +/- theta_j ----
        # theta0*(exp(theta1 x) - exp(theta2 x)) / (exp(theta3 x) + exp(theta4 x))
        # is invariant under theta_{1..4} -> theta_{1..4} + c (multiply numerator
        # and denominator by exp(c x)): a one-dimensional gauge.  At the truth,
        # 1.2*tanh(2x) = 1.2*(e^{2x} - e^{-2x})/(e^{2x} + e^{-2x}), i.e.
        # theta1 = theta3 ~ 2 and theta2 = theta4 ~ -2, so the term reduces to
        # theta0*tanh(theta1 x) through signed identifications.  The true
        # frequency (2.0) is far enough from 1 that a theta->1 move does not
        # pre-empt them.
        x = [np.linspace(-3.0, 3.0, n)]
        xs = [sp.Symbol("x1", real=True)]
        params = [sp.Symbol(f"theta{i}", real=True) for i in range(9)]
 
        f_true = 1.2 * sp.tanh(2.0 * xs[0])
        y_true = 1.2 * np.tanh(2.0 * x[0])
 
        # The remaining terms (constant, linear, quadratic, oscillation) are
        # theta->0 distractors; a free constant is harmless here since it cannot
        # absorb the frequency gauge.
        num = sp.exp(params[1] * xs[0]) - sp.exp(params[2] * xs[0])
        den = sp.exp(params[3] * xs[0]) + sp.exp(params[4] * xs[0])
        model_sym = (params[0] * num / den
                     + params[5]
                     + params[6] * xs[0]
                     + params[7] * xs[0] ** 2
                     + params[8] * sp.cos(xs[0]))
        y = y_true + np.random.normal(0, real_sigma_noise, size=n)
 
    else:
        raise ValueError(f"unknown model {model!r}")
 
 
    return x,y,model_sym,xs,params,f_true, y_true

# ============================================================================
# Pruning history
# ============================================================================

def init_pruning_history():
    """
    Return an empty pruning history (a list of snapshot dicts).

    Each snapshot written by ``record_snapshot`` stores:
      - step (int), stage (str), action (str: "start", "removed", ...)
      - removed_param (str or None), boundary (str or None: "zero", "one",
        "+inf", "-inf" or a signed identification "a=1b" / "a=-1b")
      - criterion / criterion_value, second_crit / second_crit_value
      - pruning_mode (str or None)
      - model_sym, params, theta_mle, sigma_noise
      - df (Fisher-spectrum table: ln(lambda), main_param, boundary distances)
      - yhat (predictions on the training design, or None)
    """
    return []

def record_snapshot(
    history,
    *,
    stage,
    action,
    model_sym,
    params,
    theta_mle,
    sigma_noise,
    df,
    x,
    model_func,
    removed_param=None,
    boundary=None,
    criterion=None,
    criterion_value=None,
    second_crit=None,
    second_crit_value=None,
    pruning_mode=None,
):
    """
    Append one snapshot to ``history``.  ``yhat`` is computed on the current
    design ``x`` when ``model_func`` and ``theta_mle`` are available.
    """
    step = len(history)

    yhat = None
    if isinstance(theta_mle, np.ndarray) and (model_func is not None):
        try:
            call_args = tuple(theta_mle) + tuple(x)
            yhat = np.asarray(model_func(*call_args), dtype=float).reshape(-1)
        except Exception:
            yhat = None

    history.append({
        "step": step,
        "stage": stage,
        "action": action,
        "removed_param": removed_param,
        "boundary": boundary,
        "criterion": criterion,
        "criterion_value": criterion_value,
        "second_crit": second_crit,
        "second_crit_value": second_crit_value,
        "pruning_mode": pruning_mode,
        "model_sym": model_sym,
        "params": list(params) if params is not None else None,
        "theta_mle": None if theta_mle is None else np.asarray(theta_mle, dtype=float).copy(),
        "sigma_noise": sigma_noise,
        "df": df.copy() if isinstance(df, pd.DataFrame) else None,
        "yhat": yhat,
    })
