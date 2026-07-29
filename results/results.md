# Results Log — All Model Variants

Full experimental results across all layout and loss-function variants tested during development. This is the detailed working log; see the main [README](../README.md) for the curated headline results.

All results are mean Pearson correlation between predicted and true firing rates on the held-out test split, per ganglion cell, using `CELLS_USED = [0, 1, 3, 5, 8, 9, 13, 14, 16, 17, 18, 20, 21, 22, 23, 24, 25]` (17 cells). Training runs are 2 epochs unless noted otherwise.

> Note: a few individual per-cell values below were transcribed from screenshots and have minor OCR uncertainty in the last decimal digit; the **Mean Pearson** and **FINAL TEST corr** values are taken directly from the printed summary line and are exact.

---

## Random Mosaic Layout

Notebook: https://colab.research.google.com/drive/11eZAXwvyKUEznOptSyqd90wwArZE8Xj2

### Both losses (Poisson + regularization)

| Cell | Pearson r |
|---|---|
| 0 | 0.760 |
| 1 | 0.687 |
| 3 | 0.532 |
| 5 | 0.536 |
| 8 | 0.378 |
| 9 | 0.737 |
| 13 | 0.690 |
| 14 | 0.718 |
| 16 | 0.726 |
| 17 | 0.582 |
| 18 | 0.716 |
| 20 | 0.771 |
| 21 | 0.788 |
| 22 | 0.643 |
| 23 | 0.737 |
| 24 | 0.497 |
| 25 | 0.642 |

**Mean Pearson = 0.655**

### Only Poisson loss

| Cell | Pearson r |
|---|---|
| 0 | 0.770 |
| 1 | 0.687 |
| 3 | 0.518 |
| 5 | 0.535 |
| 8 | 0.362 |
| 9 | 0.730 |
| 13 | 0.689 |
| 14 | 0.725 |
| 16 | 0.725 |
| 17 | 0.589 |
| 18 | 0.722 |
| 20 | 0.774 |
| 21 | 0.792 |
| 22 | 0.645 |
| 23 | 0.730 |
| 24 | 0.503 |
| 25 | 0.671 |

**Mean Pearson = 0.656**

---

## Uniform Grid Layout

Notebook: https://colab.research.google.com/drive/1ksr5ym6UcomuQ7g4ojIgL9X7ix2bsv3J

Poisson loss only.

| Cell | Pearson r |
|---|---|
| 0 | 0.763 |
| 1 | 0.689 |
| 3 | 0.520 |
| 5 | 0.534 |
| 8 | 0.374 |
| 9 | 0.735 |
| 13 | 0.709 |
| 14 | 0.722 |
| 16 | 0.737 |
| 17 | 0.612 |
| 18 | 0.710 |
| 20 | 0.754 |
| 21 | 0.795 |
| 22 | 0.625 |
| 23 | 0.742 |
| 24 | 0.509 |
| 25 | 0.657 |

**Mean Pearson = 0.658**

---

## New Algo (Bridson blue-noise mosaic layout)

Notebook: https://colab.research.google.com/drive/1OwhMXLmGXnirquLhxtZoj6n-0CUvaFqN

Poisson loss only.

| Cell | Pearson r |
|---|---|
| 0 | 0.801 |
| 1 | 0.717 |
| 3 | 0.549 |
| 5 | 0.538 |
| 8 | 0.376 |
| 9 | 0.769 |
| 13 | 0.767 |
| 14 | 0.756 |
| 16 | 0.745 |
| 17 | 0.629 |
| 18 | 0.733 |
| 20 | 0.786 |
| 21 | 0.808 |
| 22 | 0.618 |
| 23 | 0.794 |
| 24 | 0.539 |
| 25 | 0.724 |

**FINAL TEST corr (Poisson-only) = 0.685**

---

## PINN-Based Model (Random Mosaic Layout + Diffusion/Center-Surround Regularization)

Notebook: https://colab.research.google.com/drive/1-oZkrYEHpjXIQfAYHeq5HztvpUx3ZLdz

Poisson loss + a spatial-smoothness regularization term, swept across regularization weight λ.

### λ = 0.01

| Cell | Pearson r |
|---|---|
| 0 | 0.761 |
| 1 | 0.682 |
| 3 | 0.524 |
| 5 | 0.520 |
| 8 | 0.347 |
| 9 | 0.736 |
| 13 | 0.703 |
| 14 | 0.731 |
| 16 | 0.721 |
| 17 | 0.596 |
| 18 | 0.715 |
| 20 | 0.771 |
| 21 | 0.793 |
| 22 | 0.634 |
| 23 | 0.734 |
| 24 | 0.507 |
| 25 | 0.662 |

**FINAL TEST corr = 0.655**

### λ = 0.1

| Cell | Pearson r |
|---|---|
| 0 | 0.765 |
| 1 | 0.679 |
| 3 | 0.550 |
| 5 | 0.557 |
| 8 | 0.384 |
| 9 | 0.747 |
| 13 | 0.689 |
| 14 | 0.731 |
| 16 | 0.718 |
| 17 | 0.593 |
| 18 | 0.685 |
| 20 | 0.786 |
| 21 | 0.800 |
| 22 | 0.640 |
| 23 | 0.755 |
| 24 | 0.511 |
| 25 | 0.660 |

**FINAL TEST corr = 0.662**

### λ = 1

| Cell | Pearson r |
|---|---|
| 0 | 0.771 |
| 1 | 0.692 |
| 3 | 0.563 |
| 5 | 0.573 |
| 8 | 0.371 |
| 9 | 0.761 |
| 13 | 0.713 |
| 14 | 0.745 |
| 16 | 0.738 |
| 17 | 0.606 |
| 18 | 0.707 |
| 20 | 0.801 |
| 21 | 0.808 |
| 22 | 0.612 |
| 23 | 0.757 |
| 24 | 0.527 |
| 25 | 0.678 |

**FINAL TEST corr = 0.672** ← best λ

### λ = 10

| Cell | Pearson r |
|---|---|
| 0 | 0.773 |
| 1 | 0.705 |
| 3 | 0.555 |
| 5 | 0.554 |
| 8 | 0.312 |
| 9 | 0.757 |
| 13 | 0.706 |
| 14 | 0.750 |
| 16 | 0.737 |
| 17 | 0.600 |
| 18 | 0.709 |
| 20 | 0.806 |
| 21 | 0.819 |
| 22 | 0.603 |
| 23 | 0.749 |
| 24 | 0.517 |
| 25 | 0.677 |

**FINAL TEST corr = 0.667**

---

## Summary Table

| Variant | Mean / Final Test Pearson r |
|---|---|
| Random Mosaic — Both losses | 0.655 |
| Random Mosaic — Poisson only | 0.656 |
| Uniform Grid — Poisson only | 0.658 |
| New Algo (Bridson) — Poisson only | 0.685 |
| PINN — λ=0.01 | 0.655 |
| PINN — λ=0.1 | 0.662 |
| PINN — λ=1 | 0.672 |
| PINN — λ=10 | 0.667 |

**Takeaways:**
- The spatial-smoothness regularizer gives a consistent, modest improvement over the Poisson-only baseline on the same layout, peaking at λ=1 (0.656 → 0.672).
- The Bridson (blue-noise) cell layout outperforms both the random mosaic and uniform grid layouts on its own, and is the strongest single result recorded here.
- All numbers above are from short (2-epoch) runs used for fast iteration; a longer final training run is recommended before treating any of these as final reported numbers.