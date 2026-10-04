"""
emergentSR.operon_formatting
============================

Import / preprocessing pipeline that turns Operon symbolic-regression
expressions into pruning-ready SymPy models, plus the Friedman benchmark
helpers used by ``operon_ablation.py``.

Pre-processing choices
----------------------
1. Variable assumption ``"nonnegative"`` for the Friedman wrapper.  All
   Friedman inputs live in [0, 1], so SymPy can prove nonnegativity and
   collapses trivial absolute values (``Abs(x3) -> x3``,
   ``sqrt(Abs(x1*x2)) -> sqrt(x1*x2)``) while keeping ``Abs(C*x3 + C)``
   (interior of unknown sign).

2. Targeted exponential normalization (``_normalize_exponentials``):
   ``exp(const + var) -> exp(const) * exp(var)``.  In ``C0 * exp(C1 + f(x))``
   only ``C0 * exp(C1)`` is identifiable, so the additive constant is folded
   into the multiplicative coefficient.  Implemented as a targeted recursion
   (not ``sp.expand``) to avoid blow-up on large nested Operon trees.

3. ``_rewrite_numeric_powers_to_exp`` only rewrites positive numeric bases.
   A negative numeric base raised to a symbolic exponent would give a complex
   ``log``; such terms are left untouched (they are complex / NaN on real
   data, as Operon's ``pow``) and a warning is emitted.

4. Occurrence-wise constant extraction
   (``replace_numeric_constants_occurrencewise``): each numeric leaf becomes
   its own parameter.  Merging by value would silently couple unrelated
   coefficients (``C0*x1 + C0*x2``) and corrupt the identifiability analysis.
   Pure-constant arithmetic is folded beforehand, so this does not inflate the
   parameter count.

5. Structural numbers are protected by value (``PROTECTED_VALUES``): after
   ``evalf`` every number is a Float, so small integers {-3..3} (powers, sign
   flips) and simple rational exponents {+-1/2, +-1/3, +-2/3} are recognised
   within ``PROTECT_TOL`` and kept out of the parameter list.
"""

import sympy as sp
import pandas as pd
import numpy as np
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
)


LAMBDA_MODULES = [{"Abs": np.abs, "abs": np.abs}, "numpy"]


# ============================================================
# 0) Allowed functions + parsing transforms
# ============================================================

ALLOWED_FUNCS = {
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "asin": sp.asin,
    "acos": sp.acos,
    "atan": sp.atan,
    "sinh": sp.sinh,
    "cosh": sp.cosh,
    "tanh": sp.tanh,
    "exp": sp.exp,
    "log": sp.log,
    "sqrt": sp.sqrt,
    "Abs": sp.Abs,
    "abs": sp.Abs,  # Operon often writes abs(...)
    "sign": sp.sign,
    "pow": sp.Pow,
}

TRANSFORMS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)


# ============================================================
# 0a) Protected structural values (protect by value, not type)
# ============================================================

# Small integers (structural powers, the +-1 in (...-1)**2, sign flips) plus
# simple rational root exponents that sqrt/cbrt decay into under evalf.
PROTECTED_VALUES = (
    -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0,
    0.5, -0.5,
    1.0 / 3.0, -1.0 / 3.0,
    2.0 / 3.0, -2.0 / 3.0,
)

PROTECT_TOL = 1e-9


def _is_protected_value(num, protected_values=PROTECTED_VALUES, tol=PROTECT_TOL):
    """True if ``num`` is within ``tol`` of any protected structural value."""
    try:
        v = float(num)
    except (TypeError, ValueError):
        return False
    return any(abs(v - pv) < tol for pv in protected_values)


# ============================================================
# 0b) Targeted exponential normalization
# ============================================================

def _normalize_exponentials(expr: sp.Expr, xs) -> sp.Expr:
    """
    Pull constant additive terms out of every ``exp`` argument:

        exp(const + var)  ->  exp(const) * exp(var)

    ``exp(const)`` with a Float constant auto-evaluates to a number, so the
    constant folds into the surrounding multiplicative coefficient.  This
    removes the redundant additive-inside-exp parameter (only the product of
    the leading coefficient and ``exp(const)`` is identifiable).

    Done as a targeted recursion rather than ``sp.expand`` / ``expand_power_exp``
    so it never distributes ``(...)**2`` style structure and cannot blow up on
    the large nested Operon expressions.

    Only constants additive *at the top level of an exp argument* are pulled;
    arguments that fully contain a variable (e.g. ``C*sin(C*x + x)``) are left
    untouched.
    """
    xset = set(xs)

    def is_var(term):
        return bool(term.free_symbols & xset)

    def rec(e):
        if e.is_Atom:
            return e
        e = e.func(*[rec(a) for a in e.args])
        if e.func is sp.exp:
            arg = e.args[0]
            if arg.is_Add:
                const_part = sp.Add(*[t for t in arg.args if not is_var(t)])
                if const_part != sp.S.Zero:
                    var_part = sp.Add(*[t for t in arg.args if is_var(t)])
                    # exp(const_part) auto-evaluates when const_part is a Float.
                    return sp.exp(const_part) * sp.exp(var_part)
        return e

    return rec(expr)


# ============================================================
# 0c) Numeric-base power rewriting (positive bases only)
# ============================================================

def _rewrite_numeric_powers_to_exp(expr: sp.Expr, warnings=None) -> sp.Expr:
    """
    Recursively convert ``positive_numeric_base ** symbolic_expr`` to
    ``exp(log(base) * symbolic_expr)``.

    This collapses near-singular parametrizations such as
    ``abs(-654.6) ** ((3693.6 - X1) / 7671.4)`` (two entangled numeric
    constants) into a single log-scale coefficient inside ``exp``, keeping the
    Fisher information matrix well-conditioned.

    The rewrite fires only when ``base.is_positive``.  A *negative* numeric
    base raised to a symbolic exponent would produce a complex ``log`` (e.g.
    ``log(-654.5) = 6.48 + i*pi``).  Such a term is already complex / NaN on
    real data (Operon's ``pow`` returns NaN there), so it is left untouched and
    a warning is recorded instead of injecting an imaginary part.
    """
    if expr.is_Atom:
        return expr

    new_args = [_rewrite_numeric_powers_to_exp(a, warnings) for a in expr.args]
    expr = expr.func(*new_args)

    if isinstance(expr, sp.Pow):
        base, exponent = expr.args
        if base.is_Number and not exponent.is_Number:
            if base.is_positive:
                inner = sp.simplify(sp.log(base) * exponent).evalf()
                return sp.exp(inner)
            elif base.is_negative:
                msg = (
                    f"negative numeric base {base} raised to a symbolic "
                    f"exponent ({exponent}); left untouched "
                    f"(complex / NaN on real data)."
                )
                if warnings is not None:
                    warnings.append(msg)
                else:
                    print(msg)

    return expr


# ============================================================
# 1) Parse Operon-style expression -> SymPy
# ============================================================

def parse_operon_expression(expr_str: str, x_symbols: dict) -> sp.Expr:
    """
    Parse an Operon-style expression into a SymPy expression.

    ``evaluate=True`` so SymPy immediately simplifies purely-numeric
    sub-expressions such as ``abs(-654.58997) -> 654.58997`` before any further
    processing.

    x_symbols maps names appearing in the expression to SymPy symbols, e.g.
        {"X1": x1, "X2": x2, ...}
    """
    local_dict = {**ALLOWED_FUNCS, **x_symbols}
    return parse_expr(
        expr_str,
        local_dict=local_dict,
        transformations=TRANSFORMS,
        evaluate=True,
    )


# ============================================================
# 2) Dataset -> (x_tuple, y)
# ============================================================

def extract_xy_from_dataset(dataset: pd.DataFrame, x_cols, y_col="target"):
    """Extract x_tuple (one array per input) and target vector y."""
    missing_x = [c for c in x_cols if c not in dataset.columns]
    if missing_x:
        raise ValueError(f"Missing input columns in dataset: {missing_x}")

    if y_col not in dataset.columns:
        raise ValueError(
            f"Target column {y_col!r} not found. "
            f"Available columns are: {list(dataset.columns)}"
        )

    X = dataset[x_cols].to_numpy(dtype=float)
    y = dataset[y_col].to_numpy(dtype=float)

    x_tuple = tuple(X[:, i] for i in range(X.shape[1]))
    return x_tuple, y


# ============================================================
# 3) Occurrence-wise constant -> parameter replacement
# ============================================================

def replace_numeric_constants_occurrencewise(
    expr: sp.Expr,
    prefix="C",
    protected_values=PROTECTED_VALUES,
    protect_tol=PROTECT_TOL,
):
    """
    Replace every (non-protected) numeric leaf with its OWN free parameter.

    Identical numeric values appearing in different structural slots become
    *different* parameters.  This avoids
    silently coupling unrelated coefficients (the ``C0*x1 + C0*x2`` artifact)
    and is the correct behaviour for an identifiability / pruning analysis.

    Protection
    ----------
    A numeric leaf is left untouched (not parametrized) when its value is within
    ``protect_tol`` of a protected structural value (small integers and simple
    rational root exponents).  After ``evalf`` these are all Floats, so the test
    is by value, not by SymPy type.

    Parameters are numbered ``C0, C1, ...`` in deterministic traversal order.

    Returns
    -------
    new_expr, params, init_map
    """
    counter = {"i": 0}
    params = []
    init_vals = []

    def rec(e):
        # Numeric leaves: Float / Integer / Rational only (not pi, E, I, ...).
        if isinstance(e, (sp.Float, sp.Integer, sp.Rational)):
            if _is_protected_value(e, protected_values, protect_tol):
                return e
            p = sp.Symbol(f"{prefix}{counter['i']}", real=True)
            counter["i"] += 1
            params.append(p)
            init_vals.append(float(e))
            return p
        if e.is_Atom:
            return e
        return e.func(*[rec(a) for a in e.args])

    new_expr = rec(expr)
    init_map = {p: v for p, v in zip(params, init_vals)}
    return new_expr, params, init_map


# ============================================================
# 4) Main create_model_from_string
# ============================================================

def create_model_from_string(
    dataset: pd.DataFrame,
    expr_str: str,
    x_cols=None,
    y_col="target",
    expr_var_names=None,
    reparametrize_constants=True,
    const_prefix="C",
    variable_assumption="nonnegative",
    true_model_kind=None,
    normalize_exponentials=True,
    simplify_before_extraction=True,
    rewrite_numeric_powers=True,
    return_warnings=False,
):
    """
    Build a pruning-compatible SymPy model from an Operon expression string.

    Pre-processing pipeline (in order)
    ----------------------------------
    1. Parse (``evaluate=True``) with the requested variable assumption.
       ``nonnegative`` collapses trivial ``Abs`` / ``sqrt(Abs(...))`` on
       nonnegative-variable interiors.
    2. ``_normalize_exponentials`` (if enabled): pull additive constants out of
       exp arguments so they fold into multiplicative coefficients.
    3. ``simplify().evalf()`` (if enabled): fold all pure-constant arithmetic to
       single floats and tidy algebraic structure.
    4. ``_rewrite_numeric_powers_to_exp`` (if enabled): collapse
       positive-numeric-base symbolic powers into ``exp(log(base)*...)``;
       warn (and skip) on negative bases.
    5. Occurrence-wise constant extraction with value-based protection.

    Returns
    -------
    x_tuple, y, model_sym, xs, params, f_true, y_true, model_func, init_map
        (plus ``warnings`` appended at the end if ``return_warnings=True``)
    """
    if x_cols is None:
        x_cols = [c for c in dataset.columns if c != y_col]
    x_cols = list(x_cols)

    if expr_var_names is None:
        expr_var_names = [f"X{i + 1}" for i in range(len(x_cols))]
    else:
        expr_var_names = list(expr_var_names)

    if len(x_cols) != len(expr_var_names):
        raise ValueError(
            "x_cols and expr_var_names must have the same length. "
            f"Got len(x_cols)={len(x_cols)} and "
            f"len(expr_var_names)={len(expr_var_names)}."
        )

    if variable_assumption == "positive":
        xs = [sp.Symbol(c, positive=True) for c in x_cols]
    elif variable_assumption == "nonnegative":
        xs = [sp.Symbol(c, nonnegative=True) for c in x_cols]
    elif variable_assumption == "real":
        xs = [sp.Symbol(c, real=True) for c in x_cols]
    else:
        raise ValueError(
            "variable_assumption must be one of: "
            "'real', 'positive', 'nonnegative'."
        )

    x_symbols = dict(zip(expr_var_names, xs))
    warnings = []

    # Step 1: parse.
    model_sym = parse_operon_expression(expr_str, x_symbols=x_symbols)

    # Step 2: pull additive constants out of exp arguments.
    if normalize_exponentials:
        model_sym = _normalize_exponentials(model_sym, xs)

    # Step 3: algebraic simplification + numeric folding.
    if simplify_before_extraction:
        model_sym = sp.simplify(model_sym).evalf()

    # Step 4: collapse positive-numeric-base symbolic powers (warn on negative).
    if rewrite_numeric_powers:
        model_sym = _rewrite_numeric_powers_to_exp(model_sym, warnings=warnings)

    params = []
    init_map = {}

    # Step 5: occurrence-wise constant extraction.
    if reparametrize_constants:
        model_sym, params, init_map = replace_numeric_constants_occurrencewise(
            model_sym,
            prefix=const_prefix,
        )

    x_tuple, y = extract_xy_from_dataset(dataset, x_cols=x_cols, y_col=y_col)

    if params:
        model_func = sp.lambdify(
            tuple(params) + tuple(xs), model_sym, modules=LAMBDA_MODULES,
        )
    else:
        model_func = sp.lambdify(tuple(xs), model_sym, modules=LAMBDA_MODULES)

    if true_model_kind is None:
        f_true, y_true = None, None
    else:
        f_true = friedman_true_model(true_model_kind, xs)
        f_true_func = sp.lambdify(tuple(xs), f_true, modules=LAMBDA_MODULES)
        y_true = np.asarray(f_true_func(*x_tuple), dtype=float)

    if warnings:
        for w in warnings:
            print(w)

    result = (x_tuple, y, model_sym, xs, params, f_true, y_true, model_func, init_map)
    if return_warnings:
        return result + (warnings,)
    return result


# ============================================================
# 4b) Convenience wrapper for Friedman / Operon files
# ============================================================

def create_friedman_operon_model_from_string(
    dataset: pd.DataFrame,
    expr_str: str,
    n_features=10,
    y_col="target",
    reparametrize_constants=True,
    const_prefix="C",
    true_model_kind=None,
    variable_assumption="nonnegative",
    normalize_exponentials=True,
    simplify_before_extraction=True,
    rewrite_numeric_powers=True,
    return_warnings=False,
):
    """
    Convenience wrapper for the Friedman files.

    Assumes:
        dataframe columns: x1, x2, ..., x10, target
        expression vars:   X1, X2, ..., X10

    Defaults to ``variable_assumption="nonnegative"`` because the Friedman
    inputs live in [0, 1].
    """
    x_cols = [f"x{i}" for i in range(1, n_features + 1)]
    expr_var_names = [f"X{i}" for i in range(1, n_features + 1)]

    return create_model_from_string(
        dataset=dataset,
        expr_str=expr_str,
        x_cols=x_cols,
        y_col=y_col,
        expr_var_names=expr_var_names,
        reparametrize_constants=reparametrize_constants,
        const_prefix=const_prefix,
        variable_assumption=variable_assumption,
        true_model_kind=true_model_kind,
        normalize_exponentials=normalize_exponentials,
        simplify_before_extraction=simplify_before_extraction,
        rewrite_numeric_powers=rewrite_numeric_powers,
        return_warnings=return_warnings,
    )


# ============================================================
# 5) Warm starts, population file, misc helpers
# ============================================================

def init_from_map(params, init_map, fallback=np.nan):
    """Return a parameter vector ordered like ``params`` using ``init_map``."""
    return np.array([init_map.get(p, fallback) for p in params], dtype=float)


def read_sr_population_file(path, **read_csv_kwargs):
    """Read the population/results file (filename, MSE_train, MSE_test, expression)."""
    return pd.read_csv(path, **read_csv_kwargs)


def count_sympy_nodes(expr):
    """Total number of SymPy nodes (expression complexity)."""
    return 1 + sum(count_sympy_nodes(arg) for arg in getattr(expr, "args", ()))


def friedman_true_model(kind, xs):
    """Symbolic true model for the Friedman case studies."""
    if kind in {"additive", "nonlin_additive", "friedman_additive"}:
        return (
            0.1 * sp.exp(4 * xs[0])
            + 4 / (1 + sp.exp(-20 * (xs[1] - 0.5)))
            + 3 * xs[2]
            + 2 * xs[3]
            + xs[4]
        )

    if kind in {"standard", "friedman", "mars_4_3"}:
        return (
            10 * sp.sin(sp.pi * xs[0] * xs[1])
            + 20 * (xs[2] - 0.5) ** 2
            + 10 * xs[3]
            + 5 * xs[4]
        )

    raise ValueError(
        "Unknown Friedman kind. Use one of: "
        "'additive', 'nonlin_additive', 'friedman_additive', "
        "'standard', 'friedman', 'mars_4_3'."
    )


def generate_friedman_nonlin_additive_test(n, noiselvl,suffix='test',seed=24):
    """Friedman (MARS 4.2) additive test set; also written to
    ``friedman_additive_n-<n>_sigma-<noiselvl>_<suffix>.csv.gz`` in the cwd."""
    def inner(X, sigma, name_suffix):
        n = X.shape[0]
        # J.H. Friedman, Multivariate adaptive regression splines
        # 4.2 MARS Modeling on Additive Data
        y_clean = (
            0.1 * np.exp(4 * X[:, 0])
            + 4 / (1 + np.exp(-20 * (X[:, 1] - 0.5)))
            + 3 * X[:, 2]
            + 2 * X[:, 3]
            + 1 * X[:, 4]
        )
        y = y_clean + np.random.randn(n) * sigma
       
        df = pd.DataFrame(
            np.concatenate((X, y.reshape(n, -1)), axis=1)
        )
        df.rename(
            columns={
                0: "x1",
                1: "x2",
                2: "x3",
                3: "x4",
                4: "x5",
                5: "x6",
                6: "x7",
                7: "x8",
                8: "x9",
                9: "x10",
                10: "target",
            },
            inplace=True,
        )
   
        df.to_csv(f"friedman_additive_n-{n}_sigma-{sigma}_{name_suffix}.csv.gz", header=True, index=False, compression="gzip")
        return df
    np.random.seed(seed)
    X_test = np.random.rand(n, 10)
    df=inner(X_test, noiselvl, suffix)
    return df

def generate_friedman_test(n,noiselvl,suffix='test',seed=24):
    """Friedman (MARS 4.3) test set; also written to
    ``friedman_sigma-<noiselvl>_<suffix>.csv.gz`` in the cwd."""
    def inner(X, sigma, name_suffix):
        n = X.shape[0]
        # J.H. Friedman, Multivariate adaptive regression splines
        # 4.3 A simple function in ten variables
        y_clean = (
            10 * np.sin(np.pi * X[:, 0] * X[:, 1])
            + 20 * np.square(X[:, 2] - 0.5)
            + 10 * X[:, 3]
            + 5 * X[:, 4]
        )
        y = y_clean + np.random.randn(n) * sigma
       
        df = pd.DataFrame(
            np.concatenate((X, y.reshape(n, -1)), axis=1)
        )
        df.rename(
            columns={
                0: "x1",
                1: "x2",
                2: "x3",
                3: "x4",
                4: "x5",
                5: "x6",
                6: "x7",
                7: "x8",
                8: "x9",
                9: "x10",
                10: "target",
            },
            inplace=True,
        )
   
        df.to_csv(f"friedman_sigma-{sigma}_{name_suffix}.csv.gz", header=True, index=False, compression="gzip")
        return df
    
    np.random.seed(seed)
    X_test = np.random.rand(n, 10)
    df=inner(X_test, noiselvl, suffix)
    return df
