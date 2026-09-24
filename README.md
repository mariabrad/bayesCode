# Bayes Code by Maria

This project estimates per-voxel tissue mixture weights (`W`) and shared tissue-compartment spectra (`C`) from simulated diffusion-relaxation MRI signals.

The signal model uses diffusion coefficients (`D`), T2 relaxation times, a Dirichlet prior on voxel mixture weights and SVD compression of the forward model before optimisation.

## Getting started

Install the Python dependencies:

```bash
python3 -m pip install numpy scipy scikit-learn matplotlib dirichlet pytest jupyter
```

Run the test suite from the project directory:

```bash
python3 -m pytest tests/ -q
```

## Notebooks

- `demo_pipeline.ipynb` is the main walkthrough. It builds synthetic data, runs the optimiser, compares ground-truth, least-squares and NNLS initialisation + explores sensitivity to `alpha`, `M0` and `sigma2`.
- `testing_nnls_madco.ipynb` compares NNLS and MADCO spectrum initialisation. MADCO is more computationally expensive because it solves a constrained fit for each selected voxel.
- `testing_w_convergence.ipynb` plots how the voxel weights `W` change over optimisation iterations.

Run the notebooks in VS Code or Jupyter after installing the dependencies.

## Runtime note

The demo is configured for `n_voxels = 1000` and may take a while, especially during initialisation comparisons and parameter sweeps. For a quick run, change `n_voxels = 1000` in the acquisition/grid setup cell of `demo_pipeline.ipynb` to a smaller value such as `100` or `20`.

The NNLS initialisation in the demo uses a representative subset of voxels. The subset size is controlled by `init_voxel_idx` in Section 5.

## Project structure

```text
create_grid.py                 Forward model and synthetic-data generation
optimiser_alpha_gamma.py      Bayesian optimiser and initialisation methods
plotting.py                    Visualisation and matched-error utilities
demo_pipeline.ipynb            Main walkthrough
testing_nnls_madco.ipynb        NNLS/MADCO initialisation experiments
testing_w_convergence.ipynb     Weight-convergence diagnostic
tests/                          Automated tests
```
