# emergentSR — Fisher-geometry pruning for symbolic regression

Symbolic-regression (SR) models are often over-parametrised: several constants
compensate each other, some terms are not supported by the data, and the fitted
expression is larger than the function that generated the data. `emergentSR`
prunes a fitted symbolic model by applying, one at a time, the parameter
reductions that the data cannot distinguish from the current fit.

## Method in brief

At the current fit, the Fisher information `F = JᵀJ / σ²` is used to nominate
candidate moves:

| move | description |
|---|---|
| `θ_k → 0`, `θ_k → 1` | remove a term / a coefficient |
| `θ_k → ±∞` | saturation (distance computed through `φ_k = arctan θ_k`) |
| `θ_i = ±θ_j` | signed identification (gauge fixing) |

Candidates come from four modes — numerically **unidentifiable** directions
(flagged on the Fisher spectrum, representatives chosen by pivoted-QR subset
selection), and identifiable parameters whose **zero**, **infinity** or
**signed-identification** boundary is within reach in Fisher distance. Every
candidate is refitted; a move is accepted only if the K-fold CV-MSE stays
within `β · CV_current`, and BIC only breaks ties between moves that CV cannot
distinguish. The accepted model is refitted with tight tolerances and the
procedure is repeated until nothing more can be removed.

## Repository layout

```
emergentSR/
    emergent.py            fitting, Fisher geometry, pruning drivers, case studies
    ablation_variants.py   baselines: sloppy_only, zero_only, magnitude, no_cv_bic
    operon_formatting.py   Operon expression import + Friedman benchmark helpers
    plotting.py            figures used by the tutorial notebook
Case_studies_notebook.ipynb   step-by-step tutorial on one controlled case study
ablation_case_studies.py      full method vs baselines on all controlled case studies
operon_ablation.py            full method vs baselines on Operon SR populations
SR_population_models/         Operon populations (expressions.txt) and Friedman training sets
```

## Installation

```bash
conda env create -f environment.yml && conda activate emergentsr
# or
pip install -r requirements.txt
```

Run everything from the repository root so that `emergentSR` is importable.

## Usage

**Tutorial.** Open `Case_studies_notebook.ipynb`; all settings are in a single
cell (case study, noise level, Fisher thresholds, acceptance band).

**Controlled case studies** (10 models × 2 noise levels × 30 seeds, five methods):

```bash
python ablation_case_studies.py                 # all case studies
python ablation_case_studies.py nested bessel   # a subset
```

Outputs go to `results_ablation_cases/`.

**Operon populations** (paired Wilcoxon comparison of the five methods):

```bash
python operon_ablation.py friedman_sigma-1.0 [max_rows]   # one population
python operon_ablation.py <shard_id> <n_shards>            # shard all populations
```

Outputs go to `results_ablation/`. The number of workers is read from
`SLURM_CPUS_PER_TASK` (default: all CPUs).

## Minimal example

```python
import numpy as np
from emergentSR.emergent import (create_model, compute_lambdify_and_mle_estimates,
    kfold_cv_mse_for_sympy_model, bic_for_sympy_model,
    compute_eigenvecs_eigenvals_and_alignment, remove_and_recalibrate_sloppy_parameters)

np.random.seed(0)
x, y, model_sym, xs, params, f_true, y_true = create_model("nested", 100, 0.5)
bounds = [(-100, 100)]
theta, sigma, f, jac = compute_lambdify_and_mle_estimates(model_sym, params, xs, x, y, bounds, 0.0, 0.1)
cv, _ = kfold_cv_mse_for_sympy_model(model_sym, params, xs, x, y, bounds, 0.0, 0.1, guess_init=list(theta))
bic = bic_for_sympy_model(params, f, theta, x, y)
df = compute_eigenvecs_eigenvals_and_alignment(params, jac, theta, x, y, sigma, f)

out = remove_and_recalibrate_sloppy_parameters(
    model_sym, params, f, jac, theta, xs, x, y, sigma, df, bounds, 0.0, 0.1,
    beta=0.1, criterion="CV", criterion_value=cv, second_crit="BIC", second_crit_value=bic,
    pruning_mode="unidentifiability", cv_equiv_k=2, lambda_gap_threshold=25.0)
print(out[0])   # pruned expression after one mode; see the notebook for the full loop
```

## Citation

If you use this code, please cite the accompanying paper: Guidetti V., La Rocca L., Olivetti de França F., Kronberger G., "From Overparameterised Expressions to Parsimonious Models: Fisher-Geometric Pruning for Symbolic Regression", (under review). 

## License

MIT — see `LICENSE`.
