# Learning grid cells by predictive coding

This repository contains the code and analysis notebooks for the manuscript
**Learning grid cells by predictive coding**. The project studies whether
predictive coding networks can learn path integration and develop grid-cell-like
representations with local Hebbian updates. It also includes analyses of grid
cell development, grid-scale modularity, other mEC-like cell types, and transfer
learning in trapezoid environments.

## Code Structure

- `place_cell_tpc.py`: main entry point for training and evaluating temporal
  predictive coding networks (tPCN).
- `place_cell_pcn.py`: training script for the static predictive coding network
  (PCN).
- `src/data/`: place-cell code, trajectory generation, and preloaded data tools.
- `src/model.py`: model definitions for PCN, tPCN, and related networks.
- `src/trainer.py`: training and inference loops.
- `src/visualize.py`: rate-map generation, grid-score computation, shuffled-null
  classification, plotting utilities, and cell-type analyses.
- `src/evaluation/`: grid score, border score, grid scale, modularity, and other
  evaluation utilities.
- `notebooks/`: analysis notebooks used to reproduce manuscript and
  supplementary figures.
- `results/`: trained runs, cached metrics, intermediate analysis outputs, and
  generated figure panels.

## Training

Train a standard tPCN run:

```bash
python place_cell_tpc.py
```

Outputs are written to a timestamped folder under `results/tpc/`. The folder
contains model checkpoints, configs, rate maps, SACs, grid scores, and related
evaluation files.

Train a static PCN run:

```bash
python place_cell_pcn.py
```

Outputs are written to `results/pcn/`.

Most manuscript figures use existing saved runs in `results/`; the notebooks
load these runs and regenerate the corresponding panels.

## Manuscript Figure Notebooks

Use the following notebooks to regenerate the figure components:

| Manuscript figure | Content | Main notebook(s) |
| --- | --- | --- |
| Figure 2: Path integration | trajectory decoding and tPCN/RNN learning curves | `notebooks/task_completion.ipynb` |
| Figure 3: Cell-type emergence | grid, border/band, and head-direction cell analyses | `notebooks/cell_type_emergence.ipynb` |
| Figure 4: Grid-scale modularity | module-specific grid scales and module controls | `notebooks/module_metrics.ipynb` |
| Figure 5: Grid-cell development | grid development, stability, scaffold similarity, and Gini | `notebooks/tpcn_grid_development.ipynb` |
| Figure 6: Trapezoid transfer learning | transfer to trapezoid and square-size controls | `notebooks/trapezoid_extension.ipynb` |
