"""
emergentSR.ablation_variants
============================

Baseline / ablation pruning drivers.  Each variant changes a single, nameable
design choice with respect to the full method
(``emergent.remove_and_recalibrate_sloppy_parameters`` looped over its four
modes):

    variant       candidate rule                       acceptance / stop rule
    ---------------------------------------------------------------------------
    full method   4 Fisher modes (unident. + zero +    CV feasibility band
                  inf + signed-ident), looped          (+ BIC tie-break)
    sloppy_only   strict MBAM/BR: the SINGLE smallest- CV gate; stop when the
                  eigenvalue direction each step       sloppiest move is rejected
    zero_only     zero move via Wald stat, ALL params  CV gate (unchanged)
    magnitude     smallest |theta| below a relative    threshold ONLY (no test);
                  scale threshold tau, iterative       stop when none below tau
    no_cv_bic     IDENTICAL 4-mode staged procedure    BIC band (dBIC < 3)
                  as the full method (same masks,
                  pools, menus, mode order)

Why these baselines
-------------------
* ``sloppy_only`` is the "sensitivity is the only signal" comparator (MBAM /
  Bayesian-reduction style).  It does NOT restrict itself to formally
  unidentifiable directions -- that would merely be a subset of the full
  method.  Each iteration it recomputes the Fisher spectrum and attempts to
  remove the *single smallest-eigenvalue* direction (a dynamic,
  spectrum-relative choice), saturating it to its best boundary.  The CV gate
  decides admissibility; the loop stops when even the sloppiest direction can
  no longer be removed without leaving the CV band.  It keeps the
  parameter-indexed ordering of ``compute_eigenvecs_eigenvals_and_alignment``
  (dominant eigendirection of each parameter): the baseline is *defined* as
  "follow the smallest-eigenvalue direction".

* ``magnitude`` is the standard naive floor (Han et al.): set a threshold
  ``tau`` relative to the coefficient scale and iteratively zero the
  smallest-|theta| parameter below ``tau``, refitting between removals.  There
  is NO statistical test and NO CV gate.  A coefficient whose zeroing is not
  evaluable (theta=0 is a singular point of the model, e.g. log(theta*x), or
  the refit fails) is skipped rather than aborting the run; the skip list is
  reset after every successful removal.

* ``zero_only`` and ``no_cv_bic`` keep the full method's machinery and change
  exactly one thing (candidate menu scope / acceptance gate).  ``zero_only``
  neutralises the unidentifiable set (``lambda_threshold=-1e18``,
  ``lambda_gap_threshold=1e18``) so every parameter is eligible.

The shared tolerances (snr_sq, beta, cv_equiv_k, ident_threshold,
selection_tol, lambda thresholds) must be passed with the values used by the
full method in the same run, or an ablation changes more than one thing.

Every variant returns the same 10-tuple as
``remove_and_recalibrate_sloppy_parameters`` (last entry = number of removed
parameters):

    (new_model_sym, new_params, model_func, theta_mle, sigma_noise,
     jac_funcs, df, criterion_value, second_crit_value, n_removed)
"""

import numpy as np
import sympy as sp

from .emergent import (
    unidentifiable_parameter_set,
    remove_and_recalibrate_sloppy_parameters,
    find_saturation_bound,
    select_best_pruning_result,
    compute_lambdify_and_mle_estimates,
    compute_eigenvecs_eigenvals_and_alignment,
    bic_for_sympy_model,
    kfold_cv_mse_for_sympy_model,
    signed_identification_candidates,
    record_snapshot,
    fisher_distance_sq_to_signed_equality_boundary,
    safe_fisher_pinv,
    pruning_move_priority,
    _as_len_n,
)


# ============================================================================
# Shared helpers
# ============================================================================
def _recompute_spectrum(new_params, jac_funcs, theta_mle, x, y, sigma_noise, model_func):
    """Fisher eigenspectrum for the current model, sorted ascending by ln(lambda)
    so row 0 is the sloppiest (smallest-eigenvalue) direction."""
    df = compute_eigenvecs_eigenvals_and_alignment(
        new_params, jac_funcs, theta_mle, x, y, sigma_noise, model_func
    )
    return df.sort_values(by="ln(lambda)").reset_index(drop=True)


def _tight_refit(model_sym, params, xs, x, y, def_bounds, init_loc, init_scale,
                 warm_theta):
    """Step-3 style high-quality refit of an accepted reduced model."""
    guess = (
        list(warm_theta)
        if isinstance(warm_theta, np.ndarray) and np.all(np.isfinite(warm_theta))
        else None
    )
    return compute_lambdify_and_mle_estimates(
        model_sym, params, xs, x, y, def_bounds, init_loc, init_scale,
        guess_init=guess,
    )


def _saturation_menu(par, sign_val, ident_rows):
    """Full scalar-saturation menu for a parameter (0, 1, sign-consistent inf),
    plus any signed-identification move whose left member is ``par``."""
    sat = {"zero": sp.Integer(0), "one": sp.Integer(1)}
    if sign_val > 0:
        sat["+inf"] = +sp.oo
    elif sign_val < 0:
        sat["-inf"] = -sp.oo
    else:
        sat["+inf"] = +sp.oo
        sat["-inf"] = -sp.oo
    for row in ident_rows:
        if row["pi"] == par:
            nm = f"{row['pi']}={'+' if row['sign'] > 0 else '-'}{row['pj']}"
            sat[nm] = row["sign"] * row["pj"]
    return sat


def _unpack_fsb(res):
    """Unpack the 13-tuple returned by find_saturation_bound into a dict."""
    (lim, dcv, b_theta, b_sigma, b_model_sym, b_params, b_model_func,
     b_jac, b_df, b_val, b_sec_val, b_sec_diff, b_cv_se) = res
    return {
        "lim": lim,
        "diff": float(dcv) if dcv is not None else np.inf,
        "cv_se": b_cv_se,
        "sec_diff_value": b_sec_diff,
        "model_sym": b_model_sym, "params": b_params, "theta": b_theta,
        "sigma": b_sigma, "model_func": b_model_func, "jac_funcs": b_jac,
        "df": b_df, "model_value": b_val, "sec_crit_value": b_sec_val,
    }


def _apply_accept(best, xs, x, y, def_bounds, init_loc, init_scale):
    """Step-3 tight refit of an accepted reduced model; returns the new state
    dict (falls back to the screening solution if the tight refit fails)."""
    theta_ref, sigma_ref, mfunc_ref, jac_ref = _tight_refit(
        best["model_sym"], best["params"], xs, x, y,
        def_bounds, init_loc, init_scale, best["theta"])
    if theta_ref is not None:
        return dict(model_sym=best["model_sym"], params=best["params"],
                    theta=theta_ref, sigma=sigma_ref,
                    model_func=mfunc_ref, jac_funcs=jac_ref, refit=True)
    return dict(model_sym=best["model_sym"], params=best["params"],
                theta=best["theta"], sigma=best["sigma"],
                model_func=best["model_func"], jac_funcs=best["jac_funcs"],
                refit=False)


# ============================================================================
# 1) SLOPPY-ONLY  (strict MBAM/BR: single smallest-eigenvalue direction, CV gate)
# ============================================================================
def prune_sloppy_only(
    new_model_sym, new_params, model_func, jac_funcs, theta_mle,
    xs, x, y, sigma_noise, df, def_bounds, init_loc, init_scale,
    beta, criterion, criterion_value=None, second_crit="BIC", second_crit_value=None,
    node_states=None, verbose=False, snr_sq=16, ident_threshold=0.01,
    selection_tol=1e-2, cv_equiv_k=0.05, history=None, cancel_removable=False,
    screening_ftol=1e-4, screening_gtol=1e-4, screening_maxiter=500,
    **kwargs
):
    """"Sensitivity is the only signal" baseline (MBAM / Bayesian reduction).

    Each iteration: recompute the Fisher spectrum, take the SINGLE
    smallest-eigenvalue direction, offer it the full saturation menu, and accept
    the move iff it passes the CV feasibility gate. The sloppiness cutoff is
    therefore dynamic (spectrum-relative), not a fixed unidentifiability
    threshold. Stop as soon as the sloppiest direction can no longer be removed
    within the CV band.

    Note: ``lambda_threshold`` / ``lambda_gap_threshold`` are intentionally NOT
    parameters here -- the whole point is that the cutoff is the running minimum
    of the spectrum, so no absolute threshold is used.
    """
    n_removed = 0

    while len(new_params) > 1:
        # dynamic cutoff: the sloppiest direction of the CURRENT spectrum
        df = _recompute_spectrum(new_params, jac_funcs, theta_mle,
                                 x, y, sigma_noise, model_func)
        par = df.iloc[0]["main_param"]                 # smallest ln(lambda)
        theta_dict = {p: float(v) for p, v in zip(new_params, theta_mle)}
        sign_val = np.sign(theta_dict.get(par, 0.0))
        ident_rows = signed_identification_candidates(
            new_params, theta_mle, threshold=ident_threshold)
        sat_dict = _saturation_menu(par, sign_val, ident_rows)

        res = find_saturation_bound(
            par, sat_dict, new_model_sym, model_func, new_params, theta_mle,
            xs, x, y, def_bounds, init_loc, init_scale, beta,
            criterion, sigma_noise=sigma_noise, criterion_value=criterion_value,
            node_states=node_states, verbose=False,
            second_crit=second_crit, second_crit_value=second_crit_value,
            selection_tol=selection_tol,
            screening_ftol=screening_ftol, screening_gtol=screening_gtol,
            screening_maxiter=screening_maxiter, n_jobs_cv=1,
            cv_equiv_k=cv_equiv_k, cancel_removable=cancel_removable,
        )
        cand = _unpack_fsb(res)
        cand["par"] = par

        # CV feasibility gate for this single sloppiest move.
        best, accept, reason = select_best_pruning_result(
            [cand], criterion=criterion, beta=beta, sigma_noise=sigma_noise,
            current_cv=criterion_value, selection_tol=selection_tol,
            cv_diff_key="diff", secondary_key="sec_diff_value",
            allow_infeasible_fallback=False, cv_equiv_k=cv_equiv_k,
        )
        if not accept or best is None:
            if verbose:
                print(f"[sloppy_only] stop: sloppiest {par} inadmissible "
                      f"(reason={reason}, dCV={cand['diff']:.3g})")
            break

        state = _apply_accept(best, xs, x, y, def_bounds, init_loc, init_scale)
        new_model_sym, new_params = state["model_sym"], state["params"]
        theta_mle, sigma_noise = state["theta"], state["sigma"]
        model_func, jac_funcs = state["model_func"], state["jac_funcs"]
        criterion_value = best["model_value"]
        second_crit_value = best["sec_crit_value"]
        n_removed += 1
        df = _recompute_spectrum(new_params, jac_funcs, theta_mle,
                                 x, y, sigma_noise, model_func)

        if history is not None:
            record_snapshot(
                history, stage="sloppy_only", action="removed",
                removed_param=str(par), boundary=str(best["lim"]),
                model_sym=new_model_sym, params=new_params, theta_mle=theta_mle,
                sigma_noise=sigma_noise, df=df, x=x, model_func=model_func,
                criterion=criterion, criterion_value=criterion_value,
                second_crit=second_crit, second_crit_value=second_crit_value,
                pruning_mode="sloppy_only")
        if verbose:
            print(f"[sloppy_only] {par} -> {best['lim']} "
                  f"(dCV={best['diff']:.3g})")

    return (new_model_sym, new_params, model_func, theta_mle, sigma_noise,
            jac_funcs, df, criterion_value, second_crit_value, n_removed)


# ============================================================================
# 2) ZERO-ONLY  (wrapper: Wald-to-zero over ALL parameters, CV gate)
# ============================================================================
def prune_zero_only(
    new_model_sym, new_params, model_func, jac_funcs, theta_mle,
    xs, x, y, sigma_noise, df, def_bounds, init_loc, init_scale,
    beta, criterion, criterion_value=None, second_crit=None, second_crit_value=None,
    node_states=None, verbose=False, snr_sq=16,
    selection_tol=1e-2, history=None, cv_equiv_k=0.05, cancel_removable=False,
    **kwargs
):
    """Classical zero-pruning: drop any parameter whose Wald statistic to the
    zero boundary (df['zero'] ~ chi^2_1) is below snr_sq, one at a time, under
    the same CV gate. The eigenvalue mask is neutralised so ALL parameters are
    eligible (not only the stiff ones), making this a faithful standalone
    baseline. Answers: "do the non-zero moves (saturation-at-inf, signed
    identification) earn their keep beyond plain zero-testing?"
    """
    in_pars = len(new_params)
    (new_model_sym, new_params, model_func, theta_mle, sigma_noise,
     jac_funcs, df, criterion_value, second_crit_value, _eff) = \
        remove_and_recalibrate_sloppy_parameters(
            new_model_sym, new_params, model_func, jac_funcs, theta_mle,
            xs, x, y, sigma_noise, df, def_bounds, init_loc, init_scale,
            beta, criterion, criterion_value, second_crit, second_crit_value,
            node_states, verbose=verbose, snr_sq=snr_sq,
            lambda_threshold=-1e18,      # neutralise the eigenvalue mask ...
            lambda_gap_threshold=1e18,   # ... => every parameter is eligible
            selection_tol=selection_tol,
            pruning_mode="zero_compatibility",
            history=history, stage_name="zero_only",
            cv_equiv_k=cv_equiv_k, cancel_removable=cancel_removable,
        )
    n_removed = in_pars - len(new_params)
    return (new_model_sym, new_params, model_func, theta_mle, sigma_noise,
            jac_funcs, df, criterion_value, second_crit_value, n_removed)


# ============================================================================
# 3) MAGNITUDE  (naive floor: relative-scale threshold, test-free)
# ============================================================================


def prune_magnitude(
    new_model_sym, new_params, model_func, jac_funcs, theta_mle,
    xs, x, y, sigma_noise, df, def_bounds, init_loc, init_scale,
    beta=None, criterion=None, criterion_value=None, second_crit="BIC",
    second_crit_value=None, node_states=None, verbose=False,
    magnitude_frac=0.05, scale="max", history=None, report_cv=True,
    **kwargs
):
    """Well-established magnitude pruning (Han et al. style), test-free.

    Threshold ``tau = magnitude_frac * scale(|theta|)`` where ``scale`` is the
    max (default) or std of the current |theta|. Each iteration: zero the
    single smallest-|theta| parameter whose |theta| < tau, then refit the
    remaining parameters. NO acceptance test and NO CV gate -- tau is the only
    criterion. Stop when no |theta| is below tau (or one parameter remains).

    ``criterion``/``beta``/CV play no role in any decision. If ``report_cv`` is
    True, the final model's CV-MSE is computed ONCE at the end purely so the
    returned tuple carries a comparable CV for the ablation table (never used to
    decide a removal).

    A coefficient whose zeroing is not evaluable is SKIPPED, not fatal: if
    theta_par = 0 is a singular point of the model (sympy zoo/nan/oo) or the
    refit fails or raises, ``par`` is added to ``blocked`` and the next-smallest
    coefficient is tried.  ``blocked`` is cleared after every successful
    removal, since the expression and the MLE have changed.  Termination is
    guaranteed: each iteration removes a parameter or grows a finite set.
    This is not a statistical test -- it only checks that the move exists.
    """
    n_removed = 0
    blocked = set()          # coefficients ruled out for the CURRENT expression

    while len(new_params) > 1:
        theta_dict = {p: float(v) for p, v in zip(new_params, theta_mle)}
        absvals = {p: abs(theta_dict[p]) for p in new_params}
        arr = np.array(list(absvals.values()), dtype=float)
        scale_val = np.std(arr) if scale == "std" else np.max(arr)
        tau = magnitude_frac * scale_val

        below = [p for p in new_params if absvals[p] < tau and p not in blocked]
        if not below:
            # either nothing is below tau, or everything below tau is blocked;
            # in both cases the current model is the answer.
            break
        par = min(below, key=lambda p: (absvals[p], str(p)))

        reduced_params = [p for p in new_params if p != par]
        if len(reduced_params) < 1:
            break

        # --- is theta_par = 0 a point at which the model is defined? ---------
        reduced_sym = new_model_sym.subs({par: sp.Integer(0)})
        if reduced_sym.has(sp.zoo, sp.nan, sp.oo):
            # e.g. par inside log(par*x) or in a denominator: the substituted
            # expression is not a model.  Not a statistical judgement.
            blocked.add(par)
            if verbose:
                print(f"[magnitude] {par}=0 is singular; skipping it this cycle.")
            continue

        try:
            (theta_ref, sigma_ref, mfunc_ref,
             jac_ref) = compute_lambdify_and_mle_estimates(
                reduced_sym, reduced_params, xs, x, y,
                def_bounds, init_loc, init_scale)
        except Exception as exc:
            # lambdify/fit raised rather than returned None: skip and go on.
            blocked.add(par)
            if verbose:
                print(f"[magnitude] refit raised on {par}: {exc}; skipping it.")
            continue

        if theta_ref is None:            # cannot fit the reduced model
            blocked.add(par)
            if verbose:
                print(f"[magnitude] refit failed after zeroing {par}; skipping it.")
            continue

        new_model_sym, new_params = reduced_sym, reduced_params
        blocked.clear()   # new expression, new MLE: re-open every coefficient
        theta_mle, sigma_noise = theta_ref, sigma_ref
        model_func, jac_funcs = mfunc_ref, jac_ref
        df = _recompute_spectrum(new_params, jac_funcs, theta_mle,
                                 x, y, sigma_noise, model_func)
        second_crit_value = bic_for_sympy_model(new_params, model_func,
                                                theta_mle, x, y)
        n_removed += 1

        if history is not None:
            record_snapshot(
                history, stage="magnitude", action="removed",
                removed_param=str(par), boundary="zero",
                model_sym=new_model_sym, params=new_params, theta_mle=theta_mle,
                sigma_noise=sigma_noise, df=df, x=x, model_func=model_func,
                criterion=criterion, criterion_value=criterion_value,
                second_crit=second_crit, second_crit_value=second_crit_value,
                pruning_mode="magnitude")
        if verbose:
            print(f"[magnitude] zeroed {par} "
                  f"(|theta|={absvals[par]:.3g} < tau={tau:.3g})")

    # Reporting-only CV of the final model (never used for a pruning decision).
    if report_cv and n_removed > 0:
        try:
            criterion_value, _ = kfold_cv_mse_for_sympy_model(
                new_model_sym, new_params, xs, x, y,
                def_bounds, init_loc, init_scale, return_se=True, n_jobs=1)
        except Exception:
            pass

    return (new_model_sym, new_params, model_func, theta_mle, sigma_noise,
            jac_funcs, df, criterion_value, second_crit_value, n_removed)


# ============================================================================
# 4) NO-CV / BIC-ONLY  (full physical menu, BIC band, dBIC < bic_accept_band)
# ============================================================================
def _mode_candidate_moves(mode, df, new_params, theta_mle, jac_funcs,
                          sigma_noise, x, y, snr_sq, ident_threshold,
                          lambda_threshold, lambda_gap_threshold):
    """Replicate, move for move, the candidate pool + per-parameter menus that
    ``remove_and_recalibrate_sloppy_parameters`` builds for ``mode`` (masks,
    Wald screens, signed-identification plausibility). Returns a flat list of
    single-boundary moves [(par, name, value), ...] so the caller can rank them
    with its own criterion. Any change here breaks the ablation's single-
    variable guarantee -- keep in lockstep with the full driver."""
    d = df
    n = len(y)

    # ---- pool: threshold the SPECTRUM, nominate representatives, and use the
    # SAME set for every mode (as in the full driver).
    p_ = len(new_params)
    Jm = np.zeros((n, p_), dtype=float)
    call_args0 = tuple(theta_mle) + tuple(x)
    for jj, jf in enumerate(jac_funcs):
        Jm[:, jj] = _as_len_n(jf(*call_args0), n)
    F_emp0 = (Jm.T @ Jm) / (n * (sigma_noise ** 2 + 1e-12))
    try:
        _ev, _V = np.linalg.eigh(F_emp0)
    except np.linalg.LinAlgError:
        _ev, _V = np.zeros(p_), np.eye(p_)
    _reps, _u, _g, _dnull = unidentifiable_parameter_set(
        _ev, _V,
        lambda_threshold=lambda_threshold,
        lambda_gap_threshold=lambda_gap_threshold,
    )
    unident_set = {new_params[k] for k in _reps}
    low_mask = d['main_param'].isin(unident_set)
    high_mask = ~low_mask
    theta_dict = {p: float(v) for p, v in zip(new_params, theta_mle)}

    # signed-identification pairs surviving the Fisher-Wald screen (d2 < snr_sq),
    # exactly as in the full method (TOTAL Fisher = J^T J / sigma^2)
    ident_rows = []
    if mode in ("unidentifiability", "signed_identifiability"):
        p = len(new_params)
        J = np.zeros((n, p), dtype=float)
        call_args = tuple(theta_mle) + tuple(x)
        for jj, jf in enumerate(jac_funcs):
            J[:, jj] = _as_len_n(jf(*call_args), n)
        F_tot = (J.T @ J) / (sigma_noise ** 2 + 1e-12)
        # Invert once, outside the pair loop.  ``None`` (a Fisher matrix that is
        # non-finite or that LAPACK cannot invert) scores every pair at +inf, so
        # no identification move is nominated.
        Finv_tot = safe_fisher_pinv(F_tot)
        for pair in signed_identification_candidates(
                new_params, theta_mle, threshold=ident_threshold):
            d2 = fisher_distance_sq_to_signed_equality_boundary(
                theta_mle, Finv_tot, pair["i"], pair["j"], sign=pair["sign"])
            if d2 < snr_sq:
                ident_rows.append(pair)

    moves = []  # flat list of (par, boundary_name, boundary_value)

    if mode == "unidentifiability":
        pool = list(d.loc[low_mask, 'main_param'])
        low_set = set(pool)
        ident_rows = [r for r in ident_rows
                      if (r["pi"] in low_set) or (r["pj"] in low_set)]
        for r in ident_rows:                      # ensure pi is in the pool
            if r["pi"] not in low_set:
                pool.append(r["pi"])
                low_set.add(r["pi"])
        for par in pool:
            moves.append((par, 'zero', sp.Integer(0)))
            moves.append((par, 'one', sp.Integer(1)))
            moves.append((par, '+inf', +sp.oo))
            moves.append((par, '-inf', -sp.oo))
        for r in ident_rows:
            name = str(r["pi"]) + '=' + str(r["sign"]) + str(r["pj"])
            moves.append((r["pi"], name, r["sign"] * r["pj"]))

    elif mode == "zero_compatibility":
        if 'zero' in d.columns:
            for par in d.loc[high_mask & (d['zero'] < snr_sq), 'main_param']:
                moves.append((par, 'zero', sp.Integer(0)))

    elif mode == "infinity_compatibility":
        for _, row in d.iterrows():
            if row['main_param'] in unident_set:
                continue          # same pool as zero / signed
            par = row['main_param']
            val = theta_dict.get(par, 0.0)
            if val > 0:
                if '+inf' in d.columns and row.get('+inf', np.inf) < snr_sq:
                    moves.append((par, '+inf', +sp.oo))
            elif val < 0:
                if '-inf' in d.columns and row.get('-inf', np.inf) < snr_sq:
                    moves.append((par, '-inf', -sp.oo))
            else:  # exact zero at MLE: both signs if either is plausible
                ok = (('+inf' in d.columns and row.get('+inf', np.inf) < snr_sq) or
                      ('-inf' in d.columns and row.get('-inf', np.inf) < snr_sq))
                if ok:
                    moves.append((par, '+inf', +sp.oo))
                    moves.append((par, '-inf', -sp.oo))

    elif mode == "signed_identifiability":
        high_set = set(d.loc[high_mask, 'main_param'])
        for r in ident_rows:
            if r["pi"] in high_set:
                name = str(r["pi"]) + '=' + str(r["sign"]) + str(r["pj"])
                moves.append((r["pi"], name, r["sign"] * r["pj"]))

    return moves


def prune_no_cv_bic(
    new_model_sym, new_params, model_func, jac_funcs, theta_mle,
    xs, x, y, sigma_noise, df, def_bounds, init_loc, init_scale,
    beta, criterion="CV", criterion_value=None, second_crit="BIC",
    second_crit_value=None, node_states=None, verbose=False,
    snr_sq=16, ident_threshold=0.01, selection_tol=1e-2, bic_accept_band=3.0,
    lambda_threshold=-10.0, lambda_gap_threshold=30.0,
    history=None, cancel_removable=False, report_cv=True,
    screening_ftol=1e-4, screening_gtol=1e-4, screening_maxiter=500,
    **kwargs
):
    """CV-free ablation that MIMICS the full method's staged procedure exactly.

    It runs the SAME outer sweep as the full driver -- the four Fisher modes
    (unidentifiability -> zero_compatibility -> infinity_compatibility ->
    signed_identifiability), each iterated to exhaustion, the sweep repeated
    until a whole pass removes nothing -- with the SAME candidate pools,
    eigenvalue masks (lambda_threshold / lambda_gap_threshold), Wald screens
    (snr_sq), per-mode saturation menus, and Step-3 tight refit of every
    accepted move. The ONLY change is the selection/acceptance rule: where the
    full method applies the CV feasibility band (+ BIC tiebreak), this variant
    uses BIC alone -- at both levels (boundary choice within a parameter AND
    parameter choice within a mode): take the move with the smallest dBIC,
    accept while dBIC < bic_accept_band (3). Ties break deterministically via
    (pruning_move_priority, str(par), str(lim)), mirroring the full method's
    secondary ordering.

    CV is never consulted for a decision. find_saturation_bound is still called
    with criterion='CV' (it needs a current-CV value internally), but each call
    passes a SINGLE boundary, so its internal CV selection is inert and we read
    only the returned dBIC (sec_diff_value). ``criterion_value`` is NOT used for
    any pruning decision; if ``report_cv`` is True it is overwritten ONCE at the
    very end with the pruned model's actual CV-MSE so the returned tuple carries
    a comparable CV for the table. ``second_crit_value`` (the BIC baseline) is
    threaded and advanced after every accepted move, exactly as in the full
    method.

    IMPORTANT: ``lambda_threshold`` / ``lambda_gap_threshold`` / ``snr_sq`` must
    be passed with the SAME values the full method uses in the run, or the
    candidate pools diverge and the ablation is no longer single-variable.

    Answers: "does predictive (CV) validation prevent harmful over-pruning that
    an information criterion alone would wave through?"
    """
    n_removed = 0
    cv_placeholder = criterion_value if (criterion_value is not None
                                         and np.isfinite(criterion_value)) \
        else sigma_noise ** 2

    MODES = ("unidentifiability", "zero_compatibility",
             "infinity_compatibility", "signed_identifiability")

    effectiveness = True
    while effectiveness and len(new_params) > 1:
        removed_this_sweep = 0

        for mode in MODES:
            # iterate this mode to exhaustion (mirrors the inner while-loop of
            # remove_and_recalibrate_sloppy_parameters)
            while len(new_params) > 1:
                moves = _mode_candidate_moves(
                    mode, df, new_params, theta_mle, jac_funcs, sigma_noise,
                    x, y, snr_sq, ident_threshold,
                    lambda_threshold, lambda_gap_threshold)
                if not moves:
                    break

                evaluated = []
                for par, name, val in moves:
                    res = find_saturation_bound(
                        par, {name: val}, new_model_sym, model_func, new_params,
                        theta_mle, xs, x, y, def_bounds, init_loc, init_scale,
                        beta, criterion, sigma_noise=sigma_noise,
                        criterion_value=cv_placeholder,
                        node_states=node_states, verbose=False,
                        second_crit=second_crit, second_crit_value=second_crit_value,
                        selection_tol=selection_tol,
                        screening_ftol=screening_ftol, screening_gtol=screening_gtol,
                        screening_maxiter=screening_maxiter, n_jobs_cv=1,
                        cancel_removable=cancel_removable,
                    )
                    r = _unpack_fsb(res)
                    r["par"] = par
                    r["dbic"] = (float(r["sec_diff_value"])
                                 if r["sec_diff_value"] is not None else np.inf)
                    evaluated.append(r)

                finite = [r for r in evaluated if np.isfinite(r["dbic"])]
                if not finite:
                    break
                best = min(finite, key=lambda r: (
                    r["dbic"], pruning_move_priority(r.get("lim")),
                    str(r["par"]), str(r["lim"])))
                if not (best["dbic"] < bic_accept_band):
                    if verbose:
                        print(f"[no_cv_bic:{mode}] stop: best dBIC "
                              f"{best['dbic']:.3g} >= {bic_accept_band}")
                    break

                # Step-3 tight refit of the accepted move (as in the full method)
                state = _apply_accept(best, xs, x, y, def_bounds, init_loc, init_scale)
                new_model_sym, new_params = state["model_sym"], state["params"]
                theta_mle, sigma_noise = state["theta"], state["sigma"]
                model_func, jac_funcs = state["model_func"], state["jac_funcs"]
                df = _recompute_spectrum(new_params, jac_funcs, theta_mle,
                                         x, y, sigma_noise, model_func)
                # NOTE: CV baseline (criterion_value) intentionally NOT updated/used.
                second_crit_value = best["sec_crit_value"]  # advance BIC baseline
                n_removed += 1
                removed_this_sweep += 1

                if history is not None:
                    record_snapshot(
                        history, stage="no_cv_bic", action="removed",
                        removed_param=str(best["par"]), boundary=str(best["lim"]),
                        model_sym=new_model_sym, params=new_params,
                        theta_mle=theta_mle, sigma_noise=sigma_noise, df=df, x=x,
                        model_func=model_func, criterion=criterion,
                        criterion_value=criterion_value,
                        second_crit=second_crit, second_crit_value=second_crit_value,
                        pruning_mode=mode)
                if verbose:
                    print(f"[no_cv_bic:{mode}] {best['par']} -> {best['lim']} "
                          f"(dBIC={best['dbic']:.3g})")

        effectiveness = bool(removed_this_sweep)

    # Reporting-only: overwrite the (unused) CV baseline with the pruned model's
    # actual CV so the table is comparable. Never influences a pruning decision.
    if report_cv and n_removed > 0:
        try:
            criterion_value, _ = kfold_cv_mse_for_sympy_model(
                new_model_sym, new_params, xs, x, y,
                def_bounds, init_loc, init_scale, return_se=True, n_jobs=1)
        except Exception:
            pass

    return (new_model_sym, new_params, model_func, theta_mle, sigma_noise,
            jac_funcs, df, criterion_value, second_crit_value, n_removed)


# Convenient registry for the ablation driver / plotting loop.
ABLATION_VARIANTS = {
    "sloppy_only": prune_sloppy_only,
    "zero_only":   prune_zero_only,
    "magnitude":   prune_magnitude,
    "no_cv_bic":   prune_no_cv_bic,
}
