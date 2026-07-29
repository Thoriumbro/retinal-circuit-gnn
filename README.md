# Retinal Circuit GNN

A biologically structured Graph Neural Network that models the full retinal circuit — photoreceptors → horizontal cells → bipolar cells → amacrine cells → ganglion cells — to predict real ganglion cell firing rates from natural scene stimuli.

## Overview

Retinal ganglion cells (RGCs) are the output neurons of the eye, and predicting their firing rates from raw visual stimuli is a classic benchmark in computational neuroscience (see the Stanford/Baccus lab CNN work this project builds on). Standard CNN approaches treat the retina as a black-box image-to-firing-rate function. This project instead represents the retina as a **heterogeneous graph** whose node types and connectivity mirror the actual anatomy of the circuit, then trains a GNN to pass messages through that structure the way real synaptic signals would.

As an additional experiment, a small spatial-smoothness regularization term is added on top of the base GNN (see Results), encouraging neighboring ganglion cells to learn similar internal representations — but the core contribution of this project is the graph-structured model itself.

## Architecture

- **Nodes**: five cell-type populations — photoreceptor, horizontal, bipolar, amacrine, ganglion — sized proportionally to known retinal cell-count ratios, laid out as spatial mosaics.
- **Edges**: distance-weighted, directed connections per synapse type (e.g. photoreceptor→bipolar, bipolar→amacrine, amacrine→ganglion), built with radius queries and k-NN convergence, weighted by a Gaussian distance-decay kernel normalized per destination node.
- **Node features**: each stimulus-carrying node type gets a real per-frame time series (pooled from the raw stimulus), passed through a shared temporal Conv1d front-end. Ganglion cells additionally receive a per-cell receptive-field patch processed by a small 3D spatiotemporal CNN branch.
- **Message passing**: multiple rounds of heterogeneous graph convolution (`HeteroConv` + `GraphConv`) with residual connections and LayerNorm.
- **Readout**: per-cell learned linear readout (embedding-based weight/bias per ganglion cell) followed by a softplus nonlinearity to produce a non-negative firing rate.
- **Optional regularization**: a small spatial-smoothness (diffusion) loss term over a k-NN graph of ganglion cells' true receptive-field centers, tested as an addition to the main Poisson loss (see Results).

## Data

This project uses the [Baccus Lab `torch-deep-retina`](https://github.com/baccuslab/torch-deep-retina) dataset loader and the associated natural-scene ganglion cell recordings. Ganglion cell receptive-field centers are taken directly from the dataset's published `CENTERS_DICT` rather than estimated in-notebook, for anatomical accuracy.

Data is not included in this repository. To run the code, download the relevant `.h5` recording file and point `RETINA_DATA_PATH` (see Setup) at it.

## Repository Structure

```
retinal-circuit-gnn/
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
├── src/
│   ├── pinn_in_gnn.py                 # main model: heterogeneous GNN (with optional diffusion regularization)
│   ├── uniform_grid_retina_gnn.py     # variant: uniform grid cell-position layout
│   └── random_mosaic_retina_gnn.py    # variant: random jittered mosaic layout
├── docs/
│   ├── cnn_notes.md
│   ├── gnn_notes.md
│   └── pinn_background.md
└── results/
    ├── Results.docx                       # prediction-vs-ground-truth plots
```

## Setup

```bash
git clone https://github.com/<Thoriumbro>/retinal-circuit-gnn.git
cd retinal-circuit-gnn
pip install -r requirements.txt
```

Set the data path as an environment variable before running:

```bash
export RETINA_DATA_PATH=/path/to/naturalscene.h5
python src/pinn_in_gnn.py
```

## Results

Mean Pearson correlation between predicted and true firing rates, evaluated on the held-out test split across 17 ganglion cells:

| Model variant | Mean test Pearson r |
|---|---|
| **GNN — Poisson loss only** | **0.656** |
| GNN + diffusion regularization, λ=0.01 | 0.655 |
| GNN + diffusion regularization, λ=0.1 | 0.662 |
| GNN + diffusion regularization, λ=1 | 0.672 |
| GNN + diffusion regularization, λ=10 | 0.667 |

The base GNN reaches a mean test Pearson r of 0.656 predicting real ganglion cell firing rates from natural scene stimuli. Adding a small spatial-smoothness regularization term gave a modest, consistent improvement across a range of weights, with λ=1 performing best (0.672).

*Note: the numbers above are from short (2-epoch) training runs used for rapid iteration during development. A longer final training run is recommended before treating these as final reported results.*

Example prediction trace vs. ground truth for the best-correlating cell:

![prediction vs ground truth](results/figures/example_cell_prediction.png)

## What This Project Demonstrates

- Encoding real anatomical structure (cell-type ratios, synaptic connectivity, receptive-field geometry) directly into a graph learning problem, rather than relying on a generic CNN.
- Working with heterogeneous graphs in PyTorch Geometric (`HeteroData`, `HeteroConv`) across multiple node and edge types.
- Combining a spatiotemporal CNN branch with graph message passing in a single model.
- Testing a lightweight spatial-smoothness regularizer as a secondary experiment on top of the base GNN.

## Acknowledgments

- Built on the data loading utilities from [Baccus Lab's `torch-deep-retina`](https://github.com/baccuslab/torch-deep-retina).
- Developed with [collaborator name] — see commit history for contributions.

## License

MIT — see [LICENSE](LICENSE).
