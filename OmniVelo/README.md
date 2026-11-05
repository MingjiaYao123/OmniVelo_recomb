# OmniVelo

OmniVelo jointly infers activating and repressive epigenomic and transcriptional dynamics from single-cell
multi-omic datasets in which each epigenomic mark is measured with RNA in a different set of cells.
This folder contains the model used for the human embryoid body data in the paper
(H3K27ac + RNA, H3K9me3 + RNA and ATAC + RNA, measured in separate cells).

## Usage

Place the `OmniVelo` folder on your Python path. `adata` holds the RNA of all cells (unspliced and spliced layers,
a `batch` column naming the dataset of each cell); `adata_atac` holds the gene-level mark of each cell, in the same
cell order. The embryoid body model was fit with:

```python
import OmniVelo as ov

model = ov.VAEChrom(adata, adata_atac,
                    batch_key='batch', ref_batch=2,            # datasets: H3K27ac, H3K9me3, ATAC (reference)
                    batch_hvg_key='highly_variable',
                    var_to_regress=['total_unspliced3'],
                    device='cuda:0', cluster_key='leiden',
                    repressive_idx=1,                          # H3K9me3 enters as a repressive mark
                    repressive_flip='own_scale',               # flip of Eq. 7 in the paper
                    encoder_own_batch=False, reg_param_target='ref', refine_guard=True,
                    reg_mmd=5, reg_mmd_t=5,
                    time_prior='geo', time_prior_root=['Pluripotent'],
                    time_prior_start=0.5, time_prior_std=0.03,
                    velocity_from='data', velocity_blend=0.75,
                    train_ton=True, t_network=False, tmax=20, run_2nd_stage=False)
model.train(config={'n_refine': 1})
model.save_anndata('out', file_name='omnivelo_EB.h5ad')
```

Requirements: Python 3, PyTorch, NumPy, SciPy, pandas, scikit-learn, scanpy, scVelo, anndata, matplotlib, seaborn,
python-igraph, pynndescent, hnswlib, joblib and tqdm.
