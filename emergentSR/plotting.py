"""
emergentSR.plotting
===================

Figures for the pruning tutorial notebook.  These functions only read a
pruning history (see ``emergentSR.emergent.init_pruning_history`` and
``record_snapshot``) or arrays produced by the notebook; the experiment
scripts do not use them.

plot_spectrum_waterfall    ln(lambda) of the Fisher spectrum at every step
plot_stepwise_cv_bic       CV-MSE and BIC along the accepted moves
plot_model_fit             data, starting model and pruned model
plot_pruning_summary       the three panels above in one row
latex_formula, latex_param_label, add_text_below_legend
                           labelling helpers
"""

import re

import numpy as np
import pandas as pd
import sympy as sp
import matplotlib.pyplot as plt


def latex_formula(expr, params, xs, lhs=r"f"):
    """
    LaTeX string ``$f(x_1, ...; \theta) = ...$`` for a symbolic model.

    Parameters are rendered as theta_j with j their index in ``params`` (pass
    the ORIGINAL parameter list to keep the original numbering after pruning);
    inputs are rendered as x_1, x_2, ... following ``xs``.
    """
    symbol_names = {}

    for j, p in enumerate(params):
        symbol_names[p] = rf"\theta_{{{j}}}"

    for i, x_sym in enumerate(xs, start=1):
        symbol_names[x_sym] = rf"x_{{{i}}}"

    rhs = sp.latex(
        expr,
        symbol_names=symbol_names,
        fold_short_frac=False,
        mul_symbol=" "
    )

    x_args = ", ".join(rf"x_{{{i}}}" for i in range(1, len(xs) + 1))
    return rf"${lhs}({x_args};\theta) = {rhs}$"


def latex_param_label(s, wrap=True):
    """
    Convert labels like:
        theta1       -> $\\theta_{1}$
        theta_12     -> $\\theta_{12}$
        x1           -> $x_{1}$
        zero         -> $0$
        one          -> $1$
        theta1->zero -> $\\theta_{1} \\rightarrow 0$
        inf          -> $\\infty$

    If wrap=False, returns the LaTeX body without outer $...$.
    """
    if s is None:
        return ""

    s = str(s).strip()

    # Remove existing math delimiters if present
    if s.startswith("$") and s.endswith("$"):
        s = s[1:-1]

    # Arrows first
    s = s.replace("-->", r"\rightarrow")
    s = s.replace("->", r"\rightarrow")
    s = s.replace("→", r"\rightarrow")

    # Normalize common boundary names.
    # Important: use lambda in re.sub because replacements may contain backslashes.
    replacements = {
        "minus_one": "-1",
        "plus_one": "1",
        "-one": "-1",
        "+one": "1",
        "zero": "0",
        "one": "1",
        "-pi/2": r"-\frac{\pi}{2}",
        "pi/2": r"\frac{\pi}{2}",
    }

    for old, new in replacements.items():
        pattern = rf"(?<![A-Za-z0-9_]){re.escape(old)}(?![A-Za-z0-9_])"
        s = re.sub(pattern, lambda m, new=new: new, s, flags=re.IGNORECASE)

    # Infinity spellings
    s = s.replace("np.inf", r"\infty")
    s = s.replace("+inf", r"\infty")
    s = s.replace("-inf", r"-\infty")
    s = re.sub(r"(?<![A-Za-z0-9_])inf(?![A-Za-z0-9_])", r"\\infty", s)

    # theta1 or theta_1 -> \theta_{1}
    s = re.sub(r"\btheta_?(\d+)\b", r"\\theta_{\1}", s)

    # x1 or x_1 -> x_{1}
    s = re.sub(r"\bx_?(\d+)\b", r"x_{\1}", s)

    if wrap:
        return rf"${s}$"

    return s


def add_text_below_legend(
    ax,
    text,
    legend=None,
    pad_points=4,
    fontsize=9,
    bbox_alpha=0.5,
):
    """
    Place text immediately below the legend.

    pad_points controls the vertical gap between legend and text.
    """

    fig = ax.figure

    if legend is None:
        legend = ax.get_legend()

    if legend is None:
        raise ValueError("No legend found. Create the legend before calling this.")

    # Needed so the legend has a computed bounding box
    fig.canvas.draw()

    renderer = fig.canvas.get_renderer()
    leg_bbox = legend.get_window_extent(renderer=renderer)

    # Convert a vertical padding from points to display pixels
    pad_pixels = pad_points * fig.dpi / 72.0

    # Position: left edge of legend, slightly below bottom edge
    x_disp = leg_bbox.x0 + 5
    y_disp = leg_bbox.y0 - pad_pixels

    # Convert display coordinates to axes coordinates
    x_ax, y_ax = ax.transAxes.inverted().transform((x_disp, y_disp))

    txt = ax.text(
        x_ax,
        y_ax,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=fontsize,
        bbox=dict(boxstyle="round,pad=0.3", alpha=bbox_alpha, color='#ff7f0e'),
    )

    return txt


def plot_spectrum_waterfall(
    history,
    *,
    ax=None,
    alpha=0.35,
    linewidth=1.2,
    label_points=True,
    label_every=1,
    max_labels_per_curve=30,
    fontsize=7,
    save=False,
    namefile="file",
    figsize=(7, 3),
    show=True,
):
    """
    Waterfall of ln(lambda) over pruning steps.

    Requirements per snapshot:
      - snap["df"] is a DataFrame with columns: "ln(lambda)", "main_param"

    Works both standalone and inside a subplot grid.
    If ax is provided, the plot is drawn into that axis.
    """

    created_fig = ax is None

    if created_fig:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure

    lines = []
    labels = []

    for t, snap in enumerate(history):
        df = snap.get("df", None)

        if not isinstance(df, pd.DataFrame):
            continue

        if "ln(lambda)" not in df.columns or "main_param" not in df.columns:
            continue

        vals = np.asarray(df["ln(lambda)"], dtype=float)
        mode_idx = np.arange(len(vals))

        action = snap.get("action", "")
        removed = snap.get("removed_param", None)
        boundary = snap.get("boundary", None)

        # Compact legend label
        leg = f"{t}"
        if action:
            leg += ":"

        if removed is not None:
            removed_ltx = latex_param_label(removed, wrap=False)

            if boundary is None:
                leg += rf" ${removed_ltx}$"
            else:
                boundary_ltx = latex_param_label(boundary, wrap=False)
                leg += rf" ${removed_ltx} \rightarrow {boundary_ltx}$"

        (ln,) = ax.plot(
            mode_idx,
            vals,
            alpha=alpha,
            linewidth=linewidth,
        )

        lines.append(ln)
        labels.append(leg)

        # Point labels: main parameter per eigenvector
        if label_points:
            main_params = [
                latex_param_label(p)
                for p in df["main_param"].astype(str).to_list()
            ]

            idxs = list(range(0, len(vals), max(1, int(label_every))))

            if len(idxs) > max_labels_per_curve:
                keep = np.linspace(
                    0,
                    len(idxs) - 1,
                    max_labels_per_curve,
                ).astype(int)
                idxs = [idxs[k] for k in keep]

            for i in idxs:
                ax.text(
                    mode_idx[i],
                    vals[i],
                    main_params[i],
                    fontsize=fontsize,
                    alpha=min(1.0, alpha + 0.2),
                    ha="left",
                    va="bottom",
                )

    ax.set_xlabel(r"mode index $i$")
    ax.set_ylabel(r"$\ln(\lambda_i)$")
    ax.set_title("Fisher spectrum")
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)

    if lines:
        ax.legend(
            lines,
            labels,
            fontsize=7,
            frameon=False,
            loc="best",
        )

    if created_fig:
        plt.tight_layout()

    if save:
        fig.savefig(namefile + ".pdf", bbox_inches="tight")

    if show:
        plt.show()

    return ax


def plot_stepwise_cv_bic(
    history,
    *,
    ax=None,
    mode="absolute",
    annotate=True,
    save=False,
    namefile="file",
    figsize=(7, 3),
    show=True,
):
    """
    Pruning trajectory in the quantities that drive the decision:
    cross-validated MSE and BIC.

    mode = "absolute"
        Plot CV MSE and BIC - BIC_0 against pruning step.

    mode = "delta"
        Plot per-move Delta CV and Delta BIC.

    Works both standalone and inside a subplot grid.
    If ax is provided, the plot is drawn into that axis.
    """

    created_fig = ax is None

    if created_fig:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure

    steps = []

    for s in history:
        cv = s.get("criterion_value")
        bic = s.get("second_crit_value")

        try:
            cv = float(cv)
            bic = float(bic)
        except (TypeError, ValueError):
            continue

        if not (np.isfinite(cv) and np.isfinite(bic)):
            continue

        rp = s.get("removed_param")
        bd = s.get("boundary")

        if rp is None:
            label = ""
        else:
            rp_ltx = latex_param_label(rp, wrap=False)

            if bd is None:
                label = rf"${rp_ltx}$"
            else:
                bd_ltx = latex_param_label(bd, wrap=False)
                label = rf"${rp_ltx} \rightarrow {bd_ltx}$"

        steps.append((cv, bic, label))

    if len(steps) < 2:
        print("Need >=2 snapshots carrying criterion_value and second_crit_value.")
        return ax

    cvs = np.array([t[0] for t in steps])
    bics = np.array([t[1] for t in steps])
    labels = [t[2] for t in steps]

    ax2 = ax.twinx()

    c_cv = "tab:blue"
    c_bic = "tab:red"

    if mode == "absolute":
        xs = np.arange(len(cvs))
        bic_rel = bics - bics[0]

        ax.plot(
            xs,
            cvs,
            marker="o",
            color=c_cv,
            lw=1.5,
            label="CV MSE",
        )

        ax2.plot(
            xs,
            bic_rel,
            marker="s",
            color=c_bic,
            lw=1.5,
            label=r"BIC $-$ BIC$_0$",
        )

        ax.set_ylabel("CV MSE", color=c_cv)
        ax2.set_ylabel(r"BIC $-$ BIC$_0$", color=c_bic)
        tick_labels = labels

    elif mode == "delta":
        xs = np.arange(1, len(cvs))

        ax.bar(
            xs,
            np.diff(cvs),
            width=0.6,
            color=c_cv,
            alpha=0.55,
            label=r"$\Delta$CV",
        )

        ax.axhline(0.0, color="0.4", lw=0.8)

        ax2.plot(
            xs,
            np.diff(bics),
            marker="s",
            color=c_bic,
            lw=1.5,
            label=r"$\Delta$BIC",
        )

        ax.set_ylabel(r"$\Delta$CV MSE", color=c_cv)
        ax2.set_ylabel(r"$\Delta$BIC", color=c_bic)
        tick_labels = labels[1:]

    else:
        raise ValueError("mode must be either 'absolute' or 'delta'.")

    ax.tick_params(axis="y", labelcolor=c_cv)
    ax2.tick_params(axis="y", labelcolor=c_bic)

    ax.set_xticks(xs)
    ax.set_xticklabels(
        tick_labels if annotate else [""] * len(xs),
        rotation=45,
        ha="right",
        fontsize=9,
    )

    ax.set_xlabel("pruning move")
    ax.set_title("Selection trajectory")
    ax.grid(True, axis="y", linestyle=":", linewidth=0.5, alpha=0.5)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()

    ax.legend(
        h1 + h2,
        l1 + l2,
        loc="upper center",
        fontsize=7,
        frameon=False,
    )

    if created_fig:
        plt.tight_layout()

    if save:
        fig.savefig(namefile + ".pdf", bbox_inches="tight")

    if show:
        plt.show()

    return ax



def plot_model_fit(x, y, start_pred, pruned_pred, *, ax=None, formula=None,
                   figsize=(5, 4), show=True):
    """
    Data, starting (over-parametrised) model and pruned model against the
    first input variable.  For multi-input models the predictions are drawn as
    markers, since they are not functions of x_1 alone.

    ``formula`` (optional) is a LaTeX string, e.g. from ``latex_formula``,
    printed in a box below the legend.
    """
    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    x1 = np.asarray(x[0], dtype=float)
    order = np.argsort(x1)
    x_sorted = x1[order]
    start_sorted = np.broadcast_to(np.asarray(start_pred, dtype=float), x1.shape)[order]
    pruned_sorted = np.broadcast_to(np.asarray(pruned_pred, dtype=float), x1.shape)[order]
    one_input = len(x) == 1

    ax.scatter(x_sorted, np.asarray(y)[order], label="Observed data",
               color="k", s=7, alpha=0.7)
    if one_input:
        ax.plot(x_sorted, start_sorted, "--", linewidth=1.8,
                label="Starting model")
        ax.plot(x_sorted, pruned_sorted, "-.", linewidth=1.8,
                label="Pruned model")
    else:
        ax.scatter(x_sorted, start_sorted, s=10, marker="x",
                   label="Starting model")
        ax.scatter(x_sorted, pruned_sorted, s=10, marker="+",
                   label="Pruned model")

    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$f(x;\theta)$")
    ax.set_title("Model fit")
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)
    leg = ax.legend(frameon=False, fontsize=8, loc="upper left")

    if formula is not None:
        add_text_below_legend(ax, formula, legend=leg, pad_points=10,
                              fontsize=9, bbox_alpha=0.5)

    if created_fig:
        plt.tight_layout()
    if show:
        plt.show()
    return ax


def plot_pruning_summary(x, y, start_pred, pruned_pred, history, *,
                         formula=None, savepath=None, show=True):
    """
    One-row summary of a pruning run: model fit, Fisher-spectrum waterfall and
    CV/BIC trajectory.  Saves the figure to ``savepath`` (e.g. a .pdf) if given.
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4),
                                        constrained_layout=True)
    plot_model_fit(x, y, start_pred, pruned_pred, ax=ax1, formula=formula,
                   show=False)
    plot_spectrum_waterfall(history, ax=ax2, label_points=True, label_every=1,
                            fontsize=9, max_labels_per_curve=20, show=False)
    plot_stepwise_cv_bic(history, ax=ax3, mode="absolute", annotate=True,
                         show=False)
    if savepath is not None:
        fig.savefig(savepath, bbox_inches="tight")
    if show:
        plt.show()
    return fig
