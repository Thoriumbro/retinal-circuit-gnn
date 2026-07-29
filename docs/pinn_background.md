# Physics-Informed Component: Background

This project incorporates a physics-informed-style regularization term, adapted for a
biological (rather than a strict PDE-governed) system.

## Motivation

Classical PINNs constrain a neural network's output to satisfy a known governing
differential equation (e.g. FitzHugh-Nagumo, Navier-Stokes) as a soft loss penalty
alongside data-fitting loss. The retina doesn't have a single clean governing PDE for
ganglion cell firing rates, but neighboring ganglion cells (by real receptive-field
position) are known to exhibit correlated activity due to shared upstream circuitry
(shared bipolar/amacrine inputs, gap junctions, common noise sources).

## Implementation

A k-NN graph is built over the true receptive-field centers of the ganglion cells used
in this project. During training, a diffusion loss term is added:

    L_diffusion = mean( (h_i - h_j)^2 )   for each edge (i, j) in the ganglion k-NN graph

where `h` is the shared hidden representation produced by the model's readout layer,
before the per-cell linear decoding step. This encourages spatially nearby cells to
learn similar internal representations, acting as a smoothness prior consistent with
known correlated RGC activity.

Total training loss:

    L = L_poisson(pred, target) + lambda * L_diffusion

## Notes for future work

- Consider logging `L_poisson` and `L_diffusion` separately per epoch to check that the
  diffusion term isn't dominating training.
- Consider an ablation sweep over `lambda` to quantify its effect on held-out test
  Pearson correlation.
