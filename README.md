# OmniVelo

**Mechanistic inference of epigenomic and transcriptional dynamics from partially overlapping single-cell multi-omic data**

OmniVelo is a mechanistic deep generative model that infers how different epigenomic marks and gene expression change
together during cell differentiation. It is designed for the common setting in which each epigenomic layer is measured
together with RNA, but in a different set of cells, for example:

| dataset | measured in each cell |
|---|---|
| 10x Multiome | RNA + chromatin accessibility (ATAC) |
| Paired-Tag | RNA + H3K27ac |
| Paired-Tag | RNA + H3K9me3 |

Shared transcriptomic measurements connect these datasets. OmniVelo places all cells on one latent state space and one
latent time axis, learns the kinetics of every mark, and predicts every mark in every cell, including the marks a cell
was not measured with.

## Model

For each gene, a chromatin state $c$ gates transcription of unspliced RNA $u$, which is spliced into $s$:

$$\frac{dc}{dt} = \alpha_c (k_c - c), \qquad \frac{du}{dt} = \rho\,\alpha\,c - \beta u, \qquad \frac{ds}{dt} = \beta u - \gamma s .$$

Activating marks (accessibility, H3K27ac, H3K4me3) promote transcription. Repressive marks (H3K9me3, H3K27me3,
DNA methylation) silence it, so for them transcription is driven by the absence of the mark:

$$\frac{du}{dt} = \rho\,\alpha\,(1 - c) - \beta u .$$

The two forms are related by the change of variables $\tilde c = 1 - c$. OmniVelo uses this to put all marks into one
model: repressive marks are flipped gene by gene before training, so that every mark reads a single permissive chromatin
state, and the readout of a repressive mark is flipped back to its measured scale.

The parameters are learned with autoencoding variational Bayes:

- **Encoder.** Maps each cell's RNA and its one measured mark to a latent cell state $z$ and a latent time $t$.
- **Decoder.** Conditioned on a mark label, it outputs the chromatin target $k_c$ and the transcriptional activity $\rho$
  of every gene. The analytical solution of the ODEs then gives $c$, $u$ and $s$ at time $t$.
- **Kinetic rates.** Each mark has its own rates $(\alpha_c, \alpha, \beta, \gamma)$, so different marks can change at
  different speeds.

## What OmniVelo provides

- **Shared latent time and RNA velocity** for cells from all datasets.
- **Prediction of unmeasured marks.** A cell is decoded with the label of every mark, including the marks of datasets
  it does not belong to, which predicts its level of each mark.
- **Coupling and decoupling of chromatin and transcription.** The factors $\kappa = k_c + \rho - 1$ and
  $\delta = k_c - \rho$ classify each gene in each cell as coupled on, coupled off, primed, or transcribed while its
  chromatin closes.
- **Order of epigenomic events.** Comparing the predicted marks within the same cells shows, for example, whether
  H3K27ac is gained or H3K9me3 is lost before a gene's chromatin becomes accessible.

## Repository

| path | content |
|---|---|
| `OmniVelo/` | the model used for the human embryoid body data in the paper |
| `OmniVelo/model/` | encoder and decoder, ODE solution, training, velocity and differential dynamics |
| `OmniVelo/analysis/` | evaluation metrics (latent time, cross-boundary direction correctness) |
| `OmniVelo/plotting*.py` | plotting utilities |

See [`OmniVelo/README.md`](OmniVelo/README.md) for usage and the exact settings of the embryoid body run.

## Requirements

Python 3 with PyTorch, NumPy, SciPy, pandas, scikit-learn, scanpy, scVelo, anndata, matplotlib, seaborn,
python-igraph, pynndescent, hnswlib, joblib and tqdm. A GPU is recommended for training.

## Citation

Mingjia Yao, Cuining (Choo) Liu, Cynthia DeLong, Chongyuan Luo and Joshua D. Welch.
*Mechanistic Inference of Epigenomic and Transcriptional Dynamics from Partially Overlapping Single-Cell Multi-Omic Data.*

OmniVelo builds on [MultiVelo](https://doi.org/10.1038/s41587-022-01476-y) and
[MultiVeloVAE](https://doi.org/10.1038/s41467-025-66287-6).

## Contact

Please open an issue on this repository for questions or bug reports.
