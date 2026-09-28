# A3IDPS: Adversarial GAN-Driven Adaptive Deep Learning Framework for Intrusion Detection with Iterative Feedback Training

Code, results, and figures accompanying the paper under revision at the *International Journal of Computers and Applications* (Taylor & Francis).

## What this repo contains

This is the corrected experimental pipeline produced during the major-revision cycle. Everything here — preprocessing, model training, attack evaluation, and the analysis scripts used for the paper's tables and figures — is runnable end to end from the master notebook, with the same logic also split out into individual scripts for readability.

- **`notebook/a3idps_revision_colab.ipynb`** — the master Colab notebook. This is the canonical, ground-truth source: every number reported in the paper's revised tables was produced by running these cells (including the in-notebook fixes applied along the way, kept visible rather than silently squashed, for transparency). Requires a Google Drive mount at `/content/drive/MyDrive/A3IDPS_COLAB` with the expected folder layout (see below) and a GPU runtime.
- **`scripts/`** — the same cells, split into individually named files per pipeline stage, for reading outside Colab:
  - `scripts/nsl_kdd/` — NSL-KDD pipeline: preprocessing, undefended/FGSM-AT/PGD-AT baselines, A3IDPS (Stage 3), the 5-seed robustness study (Stage 4a), the AdvGAN-as-attack sweep (Stage 4b), and cross-checks against the `adversarial-robustness-toolbox` (ART) library's APGD attacks.
  - `scripts/cicids2017/` — the equivalent CICIDS2017 pipeline, built to the same protocol (dedup, log1p + MinMax scaling, SMOTE class balancing, the same 5-seed / attack-sweep / ART cross-check structure).
  - `scripts/shared/` — dataset-agnostic analysis: confusion matrices (clean and under worst-case PGD at eps=0.05) and the corrected SHAP feature-importance analysis (fixes an axis-averaging bug present in an earlier pass).
- **`results/`** — the result JSONs that back the paper's tables:
  - `results/cicids_v2/stage4_final_corrected.json` — the headline 5-seed CICIDS result table: `corrected_worst_case = min(white-box worst-case, best AdvGAN-sweep accuracy)` per model/eps, plus the paired-difference statistics (kd_feedback vs. undefended / PGD-AT / fixed-generator / no-KD ablations) cited in the paper.
  - `results/{nsl_v2,cicids_v2}/art_crosscheck_results.json` — the independent-library cross-check: our own hand-written multi-attack worst-case protocol vs. ART's APGD-CE / APGD-DLR (components of AutoAttack; full AutoAttack fails outright on tabular input since it includes an image-only attack).
  - `results/{nsl_v2,cicids_v2}/confusion_matrices.json` — raw confusion-matrix counts (clean and adversarial, seed 0) behind `figures/confusion_matrices/`.
- **`figures/`** — the confusion-matrix heatmaps and SHAP top-15 feature bar charts referenced above, for the four representative models (undefended, FGSM-AT, PGD-AT, A3IDPS with 5 feedback rounds) on both datasets.

## What is *not* checked in, and why

- **Raw NSL-KDD / CICIDS2017 data** — both are standard public benchmark datasets, redistributed elsewhere under their own terms; the preprocessing scripts here regenerate the processed arrays from the originals rather than shipping a copy.
- **Trained model weights** (`.keras` files) and the full per-seed result JSONs — these are large (dozens of models across two datasets × multiple training regimes × 5 seeds) and are regenerable from the scripts in a few hours on a single GPU. Only the aggregated results actually cited in the paper are checked in here; happy to share the raw per-seed artifacts on request.

## Reproducing a result

Each stage script/cell reads its inputs from, and writes its outputs to, a fixed folder layout under a base directory (`A3IDPS_COLAB/` in the notebook):

```
data_processed/{nslkdd_v2,cicids_v2}/   preprocessed arrays + feature_names.json
models/{nsl_v2,cicids_v2}/               trained .keras models, one per (stage, config, seed)
results/{nsl_v2,cicids_v2}/              per-stage JSON results
results/{confusion_matrices,shap}/       cross-dataset analysis outputs
```

Run the stage scripts in numeric order (Stage 1 → Stage 4b) within each dataset folder; later stages assume the models and result JSONs from earlier stages already exist and will skip retraining a model that is already saved on disk.

## Methodology notes

- **Worst-case robustness** is evaluated as the *minimum* accuracy across an ensemble of attacks (FGSM, PGD-20 with 3 restarts under both cross-entropy and margin loss, plus a fresh AdvGAN generator swept over 8 configurations and trained specifically to attack each final model), not any single attack in isolation.
- ε (perturbation budget) selection for adversarial training is validation-based and frozen before the test-set evaluation.
- All headline numbers are reported as mean ± std over 5 random seeds (same data split; only model/generator initialization and batch order vary by seed).
- We do not claim A3IDPS outperforms standard PGD adversarial training on raw worst-case accuracy — it does not, on either dataset, at the harder ε values. The contribution is (1) a corrected, leakage-free, honestly-evaluated protocol for this class of problem, and (2) the A3IDPS iterative-feedback framework and its ablations (feedback vs. fixed generator, with vs. without distillation), evaluated under that same protocol.

## Citation

If you use this code, please cite the paper (details to be added on acceptance).
