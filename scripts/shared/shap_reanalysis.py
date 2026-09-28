# SHAP re-analysis on the CORRECTED models, for both datasets. Fixes the axis-averaging bug diagnosed early on
# (shap_vals came back shaped (n_samples, n_features, n_classes); the original code averaged over the wrong
# axis, producing meaningless "top features"). Uses KernelExplainer's model-agnostic wrapper around the Keras
# model's predicted-class probabilities via a small background sample (standard for tabular NNs; a
# DeepExplainer/GradientExplainer would need a TF1-style graph, not worth fighting for a logits-output tf.keras
# model here). Reports per-class mean(|SHAP value|) top-15 features for undefended vs kd_feedback_r5, so the
# paper can show whether/how A3IDPS's training changes which features the model actually relies on.

# %% [0] Setup
from google.colab import drive
drive.mount('/content/drive')
import os, json
import numpy as np, tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

try:
    import shap
except ImportError:
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "pip", "install", "shap", "--quiet"])
    import shap

BASE = "/content/drive/MyDrive/A3IDPS_COLAB"
OUT = f"{BASE}/results/shap"
os.makedirs(OUT, exist_ok=True)

N_BACKGROUND = 100     # background sample for KernelExplainer (kept small - KernelExplainer is O(features x background))
N_EXPLAIN = 200         # test rows to explain per model (kept small for runtime; increase if Colab time allows)
TOP_K = 15

DATASETS = {
    "nsl": {
        "data": f"{BASE}/data_processed/nslkdd_v2/nsl_v2_arrays.npz",
        "mod": f"{BASE}/models/nsl_v2", "res": f"{BASE}/results/nsl_v2",
        "class_names": ["Normal", "DoS", "Probe", "R2L", "U2R"],
        "feature_names": json.load(open(f"{BASE}/data_processed/nslkdd_v2/feature_names.json")),
    },
    "cicids": {
        "data": f"{BASE}/data_processed/cicids_v2/cicids_v2_primary.npz",
        "mod": f"{BASE}/models/cicids_v2", "res": f"{BASE}/results/cicids_v2",
        "class_names": ["BENIGN", "Brute Force", "DoS/DDoS", "PortScan", "Other"],
        "feature_names": json.load(open(f"{BASE}/data_processed/cicids_v2/feature_names.json")),
    },
}

def get_model_paths(ds_name, cfg):
    if ds_name == "cicids":
        eps_fgsm, eps_pgd = json.load(open(f"{cfg['res']}/stage2b_selected_seed0.json")).values()
        eps_max = json.load(open(f"{cfg['res']}/stage3_selection_seed0.json"))["selected_eps"]
        return {
            "undefended": f"{cfg['mod']}/undefended_smote_seed0.keras",
            "kd_feedback_r5": f"{cfg['mod']}/s3_kd_feedback_eps{eps_max}_r5_seed0.keras",
        }
    else:
        # NSL Stage 2 saves undefended as f"{MOD}/s2_none_seed0.keras" (confirmed against nsl_stage2.py).
        # kd_feedback_r5 is saved by nsl_stage4a_seeds.py under its own naming ("s4_" prefix, no eps in the
        # name), not Stage 3's "s3_..._eps{e}_..." pattern - confirmed via directory listing.
        return {
            "undefended": f"{cfg['mod']}/s2_none_seed0.keras",
            "kd_feedback_r5": f"{cfg['mod']}/s4_kd_feedback_r5_seed0.keras",
        }

# %% [1] Per-dataset, per-model SHAP (KernelExplainer over predicted class probabilities, correct axis handling)
all_top_features = {}
for ds_name, cfg in DATASETS.items():
    print(f"\n========== {ds_name.upper()} ==========")
    d = np.load(cfg["data"])
    Xte = d["Xte"].astype("float32")
    D = Xte.shape[1]
    ncls = len(cfg["class_names"])
    feat_names = cfg["feature_names"] if cfg["feature_names"] else [f"f{i}" for i in range(D)]
    assert len(feat_names) == D, f"{ds_name}: feature name count ({len(feat_names)}) != D ({D})"

    rng = np.random.RandomState(0)
    bg_idx = rng.choice(len(Xte), N_BACKGROUND, replace=False)
    ex_idx = rng.choice(len(Xte), N_EXPLAIN, replace=False)
    background, X_explain = Xte[bg_idx], Xte[ex_idx]

    model_paths = get_model_paths(ds_name, cfg)
    ds_top = {}
    for tag, path in model_paths.items():
        if not os.path.exists(path):
            print(f"  SKIP {tag}: model not found at {path}"); continue
        model = keras.models.load_model(path, compile=False)
        def f(x):  # KernelExplainer needs a plain numpy-in numpy-out probability function
            return tf.nn.softmax(model.predict(x, batch_size=4096, verbose=0), axis=1).numpy()

        explainer = shap.KernelExplainer(f, background)
        shap_vals = explainer.shap_values(X_explain, nsamples="auto")
        # shap_vals can come back as either a list of (n_explain, D) arrays (one per class) or a single
        # (n_explain, D, n_classes) array depending on the shap version - handle both, and NEVER average over
        # the wrong axis (the bug that broke the original analysis): we want, per class c, mean over samples of
        # |shap value for feature j, class c| - i.e. mean over axis 0 only, keeping features and classes separate.
        if isinstance(shap_vals, list):
            sv = np.stack(shap_vals, axis=-1)   # (n_explain, D, n_classes)
        else:
            sv = shap_vals
        assert sv.shape == (len(X_explain), D, ncls), f"unexpected SHAP output shape {sv.shape}, expected {(len(X_explain), D, ncls)}"

        mean_abs = np.mean(np.abs(sv), axis=0)   # (D, n_classes) - correct axis: average over SAMPLES only
        ds_top[tag] = {}
        for c, cname in enumerate(cfg["class_names"]):
            order = np.argsort(-mean_abs[:, c])[:TOP_K]
            ds_top[tag][cname] = [{"feature": feat_names[j], "mean_abs_shap": float(mean_abs[j, c])} for j in order]

        # one summary plot per model: overall top-15 features by mean(|SHAP|) averaged across classes too
        overall = mean_abs.mean(axis=1)
        order = np.argsort(-overall)[:TOP_K]
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.barh([feat_names[j] for j in order][::-1], overall[order][::-1])
        ax.set_xlabel("mean(|SHAP value|), averaged across classes")
        ax.set_title(f"{ds_name.upper()} - {tag} - top {TOP_K} features")
        fig.tight_layout(); fig.savefig(f"{OUT}/{ds_name}_{tag}_shap_top{TOP_K}.png", dpi=150); plt.close(fig)
        print(f"  {tag}: top-3 overall features = {[feat_names[j] for j in order[:3]]}")

    all_top_features[ds_name] = ds_top
    json.dump(all_top_features[ds_name], open(f"{OUT}/{ds_name}_shap_top_features.json", "w"), indent=1)

print(f"\nSaved plots and JSON to {OUT}/")
