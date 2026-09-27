# PINN-Based Shock Detection for a Quasi-1D Nozzle

This repository contains a PyTorch implementation of a physics-informed neural network (PINN) for a quasi-1D converging-diverging nozzle with a moving shock. The main workflow is implemented in the notebook `unsteady_fb_add_pinn.ipynb` and includes:

- analytical reference solution generation
- single-network shock detection training
- dual-network split PINN training around the shock location
- loading saved checkpoints and evaluating final predictions
- plotting loss and solution comparisons

## Project structure

- `unsteady_fb_add_pinn.ipynb` — main training, evaluation, and visualization notebook
- `saved_models/` — trained model checkpoints (`.pt` files)
- `saved_histories/` — training histories and field snapshots (`.csv`, `.npz`)
- `saved_figures/` — generated figures and plots

## Requirements

- Python 3.10+ recommended
- PyTorch with CUDA support optional but not required
- NumPy, Pandas, Matplotlib
- scikit-learn is used in some plotting/model-metric sections

## Setup

### 1) Create and activate a virtual environment

```bash
cd /home/prantik/projects/napc_2027
python -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install torch numpy pandas matplotlib scikit-learn jupyter
```

If you want to use GPU acceleration, install a CUDA-enabled PyTorch build that matches your system. The notebook automatically selects CUDA when available and otherwise falls back to CPU.

## Running the notebook

Open the workspace in VS Code and launch the notebook:

```bash
jupyter notebook
```

Then open `unsteady_fb_add_pinn.ipynb` and run the cells in order.

The notebook is organized as a training pipeline:

1. define constants, domain, and analytical solution
2. train the single shock-detection network
3. detect the shock location and initialize the split-domain setup
4. train the left/right dual networks with interface conditions
5. load saved checkpoints from `saved_models/`
6. evaluate final predictions against the analytical solution
7. generate loss and comparison plots in `saved_figures/`

## Reproducing saved outputs

The repository already includes trained artifacts in `saved_models/` and `saved_histories/`.

You can skip the long training process by loading the saved checkpoints already present in the project. The notebook contains cells like:

```python
model_paths = (
    "saved_models/dual_net_pre_model.2026-09-26_16-22-40.pt",
    "saved_models/dual_net_post_model.2026-09-26_16-22-40.pt",
    "saved_models/interface_model.2026-09-26_16-22-40.pt",
)
```

and loading code for the history and snapshot files.

## Notes

- The notebook uses a deterministic random seed (`SEED = 16`) for reproducibility.
- Training can take a while on CPU; it is much faster on CUDA-enabled hardware.
- Model checkpoints and histories are written automatically under `saved_models/` and `saved_histories/` when training runs.
- The main forward model outputs `(rho, u, P, T)` for each `(x, t)` sample.

## Typical workflow

```bash
cd /home/prantik/projects/napc_2027
source .venv/bin/activate
jupyter notebook
```

Then in the notebook:

- run all cells from top to bottom, or
- load pretrained checkpoints and evaluate only the final results cells.

## Troubleshooting

- If a package is missing, run `python -m pip install <package-name>`.
- If the notebook cannot find saved files, verify that the working directory is the repository root.
- If you use a local font path and a plotting cell fails, update the Open Sans font path in the notebook to a valid font file on your system.
