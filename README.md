# NP-HighPerm

**NP-HighPerm** is an assay-aware multimodal learning framework for predicting the membrane permeability of nonpeptidic macrocycles (NPMCs). It integrates molecular fingerprints, SMILES-based sequence representations, molecular graphs, and assay identity to support continuous permeability estimation and auxiliary high-/low-permeability classification.

## Overview

Nonpeptidic macrocycles occupy a chemically diverse, beyond-rule-of-five space and offer opportunities for targeting challenging intracellular proteins. Their membrane permeability is difficult to predict because experimental data combine different assay contexts and permeability endpoints, and the continuous measurements are unevenly distributed.

NP-HighPerm combines:

- **Complementary molecular representations:** Morgan and MACCS fingerprints, a Transformer-based SMILES encoder, and a graph convolutional network (GCN).
- **Assay-conditioned fusion:** a Condition-Aware Fusion (CAF) module uses PAMPA or Caco-2 assay identity to assign weights to the three molecular modalities.
- **Joint training objectives:** continuous permeability regression (MSE) with auxiliary threshold-based classification (BCE-with-logits; auxiliary loss weight 0.1).
- **Validation-controlled evaluation:** five-fold cross-validation, internal validation-based checkpoint selection, held-out test evaluation, and additional scaffold and subgroup analyses.

Assay-conditioned fusion does not imply that PAMPA, Caco-2, or their permeability endpoints are physically equivalent or experimentally harmonized.

## Repository Structure

```text
NP-HighPerm/
├── all_data_split/
│   └── folds/
├── split/                    # If included in this release
│   ├── scaffold_split.py
│   └── folds/
├── visualization/            # If included in this release
│   ├── attention_visualization.py
│   ├── umap_visualization.py
│   └── scatter_plot.py
├── new_result/
│   └── best_result/          # Existing archived results, if present
├── feature_engineering.py
├── main.py
├── metrics.py
├── model.py
├── train.py
├── test.py
├── requirements.txt
├── .gitignore
└── LICENSE
```

The exact directory contents depend on the repository revision. Check the local paths referenced by `main.py` and `test.py` before running them.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/xiaohou-88/NP-HighPerm.git
cd NP-HighPerm
```

### 2. Create a Python environment

```bash
conda create -n np-highperm python=3.10 -y
conda activate np-highperm
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

If needed, install RDKit using conda:

```bash
conda install -c conda-forge rdkit
```

## Dataset and Preprocessing

The study used the July 2025 version of the Non-Peptidic Macrocycle Membrane Permeability Database (NPMMPD). After preprocessing, the curated dataset contains **4,148 unique macrocycles**, comprising **3,611 PAMPA** and **537 Caco-2** records.

The documented preprocessing workflow includes:

1. Retaining PAMPA and Caco-2 permeability measurements and excluding invalid measurements.
2. Selecting one record per standardized SMILES using the following **endpoint selection priority** (not a numerical inequality): `Log Papp` → `Log Papp AB` → `Log Papp AB+` → `Log Peff`.
3. Representing the selected measurements as a common **logarithmic permeability target** (`y_reg`). This representation does **not** establish the physical equivalence of the endpoints.
4. Clipping the continuous target to the predefined range **[-8, -4]**.
5. Deriving binary labels at **`y_reg = -6`**: label **1** if `y_reg >= -6`, and label **0** otherwise.
6. Creating five train/test folds using eight quantile-based bins of the continuous target, within-bin shuffling (seed **42**), and round-robin assignment.

The supplied five-fold train/test files are located in:

```text
all_data_split/folds/
├── fold_1_train.csv
├── fold_1_test.csv
├── fold_2_train.csv
├── fold_2_test.csv
├── fold_3_train.csv
├── fold_3_test.csv
├── fold_4_train.csv
├── fold_4_test.csv
├── fold_5_train.csv
└── fold_5_test.csv
```

These files define the *outer* train/test folds. The internal validation subsets are generated separately during validation-controlled training.

## Model

### Fingerprint branch

Morgan fingerprints (radius 2, 2,048 bits) and MACCS keys (167 bits) provide complementary molecular substructure features.

### Sequence branch

A Transformer-based sequence encoder learns representations from tokenized SMILES strings; its sequence output uses masked mean pooling over non-padding tokens.

### Graph branch

A GCN-based encoder learns molecular graph representations from atom features and chemical connectivity.

### Condition-Aware Fusion (CAF)

An assay embedding representing PAMPA or Caco-2 is used to generate three softmax-normalized modality weights. These weights are conditioned on **assay identity**, rather than individually generated from each compound's structure.

### Joint regression-classification learning

The network predicts a continuous logarithmic permeability target and an auxiliary classification logit. The training objective is:

`Loss = MSE(regression) + 0.1 × BCEWithLogits(classification)`

The auxiliary classification task uses labels derived from the same experimental permeability measurements; it does not provide an independent experimental endpoint.

## Training and Evaluation

### Validation-controlled five-fold evaluation

The current manuscript reports results using the following protocol:

1. Hold out one predefined fold for final testing.
2. Split **10% of the remaining training portion** into an internal validation subset (quantile-based stratification when feasible).
3. Train for at most **100 epochs**, using Adam (`lr=0.001`) and batch size **128**.
4. Use validation MSE for checkpoint selection and early stopping (patience **20** epochs). The learning-rate scheduler responds to validation loss.
5. Restore the checkpoint with the **lowest validation MSE** and evaluate it on the held-out test fold.
6. Report the **mean ± standard deviation across five test folds**.

Run the training entry point:

```bash
python main.py
```

**Path check:** The validation-controlled `main.py` provided with the manuscript currently points to `all_data_split2/folds/`, whereas the public repository layout lists `all_data_split/folds/`. Set `folds_dir` in `main.py` to the location containing your five fold CSV pairs before running. Do not assume that an older repository revision implements the validation-controlled protocol without verifying the code.

### Testing an existing checkpoint

```bash
python test.py
```

Before running `test.py`, verify the test CSV path, checkpoint path, and model configuration in the script. Use a checkpoint generated for the corresponding fold and architecture; the path can differ across repository revisions.

## Manuscript-Reported Results

The following values are from the **validation-controlled five-fold test evaluation** reported in the current manuscript, *not* from a test-selected best-checkpoint protocol.

| Metric     | Mean ± SD           |
| ---------- | ------------------- |
| MSE ↓      | **0.2459 ± 0.0084** |
| RMSE ↓     | **0.4958 ± 0.0085** |
| R² ↑       | **0.6522 ± 0.0150** |
| CI ↑       | **0.8113 ± 0.0045** |
| Accuracy ↑ | **0.8382 ± 0.0141** |
| F1 ↑       | **0.8420 ± 0.0131** |
| AUROC ↑    | **0.9165 ± 0.0051** |

For this evaluation, **Accuracy** and **F1** are calculated by thresholding the *regression output* at `y_reg = -6`, and **AUROC** uses the continuous regression prediction as its score. They are **not** metrics calculated directly from the auxiliary classification head. Fold-wise ROC and precision–recall analyses are also reported separately in the study's supplementary material and should not be conflated with the main-table evaluation.

**Version note:** Earlier experiments used a different model-selection/evaluation procedure and yielded an MSE of approximately **0.2387**. Those results should not be described as validation-controlled held-out test results. Archived files under `new_result/best_result/` may correspond to the earlier experiment; verify run metadata and checkpoint-selection rules before citing or reusing them as the manuscript results.

## Results and Output Files

The validation-controlled training script creates timestamped output directories under `new_result/`, typically named:

```text
new_result/<timestamp>_trainval_crossval/
├── average_results.csv
├── fold_summary.csv
├── fold_1/
│   ├── best_model.pth
│   ├── train_results.csv
│   ├── validation_results.csv
│   ├── test_results.csv
│   ├── detailed_predictions.csv
│   └── ...
├── fold_2/
├── fold_3/
├── fold_4/
└── fold_5/
```

The repository may also contain older results in `new_result/best_result/`. **Do not assume that the archived `average_results.csv` or model weights match the current manuscript without checking their provenance.**

The detailed prediction export includes regression-based `true_cls` and `pred_cls` labels. Its `pred_cls_prob` field is a sigmoid-transformed regression score, **not** the auxiliary classification head probability or a calibrated probability estimate.

## Evaluation Metrics

- **Regression:** MSE, RMSE, coefficient of determination (R²), and concordance index (CI).
- **Threshold-based screening evaluation:** Accuracy, F1, and AUROC based on regression outputs, as specified above.
- **Additional analyses:** AUPRC/AP for separately reported precision–recall evaluations, Murcko scaffold-based assessment, and physicochemical/assay subgroup analyses.

## Reproducibility Notes

1. Install the dependencies in `requirements.txt` and verify compatibility with your hardware.
2. Confirm that the five predefined train/test CSV pairs are available, and update `folds_dir` if necessary.
3. Confirm that the `main.py` version implements internal validation-based checkpoint selection before claiming to reproduce the manuscript results.
4. Run `python main.py`; inspect each fold's `split_info.csv`, `validation_results.csv`, `test_results.csv`, and `detailed_predictions.csv` if generated.
5. Inspect the newly generated `average_results.csv`; compare its metrics and run settings with the manuscript-reported values above.
6. When using `test.py`, supply the matching checkpoint, test fold, and model definition.

## Citation

If you use NP-HighPerm or its associated results, please cite the manuscript once its publication details are available:

*NP-HighPerm: An Assay-Aware Multimodal Learning Framework for Predicting Nonpeptidic Macrocycle Membrane Permeability.*

The journal, DOI, and final bibliographic metadata will be added after publication.

## License

See [`LICENSE`](LICENSE) for the repository's license terms.
