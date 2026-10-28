# Bayes Code by Maria

This project estimates per-voxel tissue mixture weights (`W`) and shared tissue-compartment spectra (`C`) from diffusion-relaxation MRI signals, both simulated and real (ex vivo macaque).

The default signal model uses diffusion coefficients (`D`) and T2 relaxation times, `S = exp(-b D) · exp(-TE / T2)`. It also uses a Dirichlet prior on the voxel mixture weights and SVD compression of the forward model before optimisation.

## Optimisation modes

Two optimisation paths are available.

- `run_optimisation` optimises the reduced coefficients `C_small` directly. It has only `R × K` spectral parameters and is therefore faster, but the inferred full-grid spectra require inverse-SVD mapping and post-processing to enforce nonnegativity and normalization.
- `run_constrained_optimisation` optimises full-grid canonical spectra `C` through softmax logits. Every component remains nonnegative and sums to one throughout optimization. It still evaluates the SVD-reduced forward model `U @ (S @ V.T @ C) @ W`, but has more free spectral parameters and benefits from a meaningful prior or regularizer on `C`.

Use the constrained mode when physically valid canonical spectra are the priority. Use the reduced mode for rapid experiments or as a comparison baseline.

## Signal models

The signal model lives in `src/signal_models.py`. Every model is a class built on `SignalModel`, and provides:

- a grid of parameter values, made with `make_grid(...)` (any number of axes)
- `A(acq)`: the forward matrix for an acquisition, e.g. `acq = {"b": b_list, "TE": TE_list}`
- `initial_C(...)` and `initial_M0(...)`: starting values for the fit

`DiffusionT2Model` is the D–T2 model. To try another model, write a new class in `src/signal_models.py` and change the `model = ...` line in the notebook. The optimiser only uses `A`, so it doesn't need to change.

## Getting started

Install the Python dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Run the test suite from the project directory:

```bash
python3 -m pytest new_tests/ -q
```

The code is imported as a package, e.g. `from src.signal_models import DiffusionT2Model`. In a notebook, add the repository root to the path first:

```python
from pathlib import Path
import sys
root = Path.cwd().resolve(); root = root if (root / "src").exists() else root.parent
sys.path.insert(0, str(root))
```

## Notebooks

- `real_data_walkthrough.ipynb` is the best place to start. It fits one region of the real macaque data step by step, with an explanation before each step.
- `maria-tests-simulation_for_poster.ipynb` fits simulated data with known ground truth, and makes the simulation poster figures.
- `maria-tests-real-data_for_poster_2.ipynb` builds the real-data signal matrix `Y` (noise-floor correction, powder averaging), fits several regions, and makes the real-data poster figures.
- `amy-tests.ipynb` holds Amy's experiments.

Run the notebooks in VS Code or Jupyter after installing the dependencies. The real-data notebooks read the macaque dataset from a local OneDrive folder, so the path at the top needs changing on another machine.

## Project structure

```text
src/
  signal_models.py          Signal models (grid + forward matrix A + starting values)
  simulation.py             Ground truth for simulations (compartments, weights, M0, noise)
  optimiser_alpha_gamma.py  Bayesian optimiser, initialisation methods and SVD of A
  plotting.py               Plots (take `model` for grid shape, axes and labels)
  metrics.py                Error measures and component matching
  postprocessing.py         Spectrum thresholding and peak finding
notebooks/                  Analyses and the walkthrough
new_tests/                  Automated tests (use the model and simulation code)
```
