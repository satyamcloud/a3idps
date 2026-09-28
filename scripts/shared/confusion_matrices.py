# Confusion matrices for both datasets, clean AND under worst-case attack at eps=0.05 (the eps both papers'
# headline tables report). Covers 4 models per dataset (undefended, fgsm_at, pgd_at, kd_feedback_r5, seed 0 -
# the same seed used for any single-seed illustration in the manuscript; the 5-seed tables already give the
# statistical claim, this gives the per-class qualitative picture Reviewer 2 asked for). Saves both a PNG heatmap
# per (dataset, model, condition) and the raw counts as JSON (for building a LaTeX table instead, if preferred).
# Self-contained, read-only over existing saved models - no training.

# %% [0] Setup
from google.colab import drive
drive.mount('/content/drive')
import os, json
import numpy as np, tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

BASE = "/content/drive/MyDrive/A3IDPS_COLAB"
OUT = f"{BASE}/results/confusion_matrices"
os.makedirs(OUT, exist_ok=True)

DATASETS = {
    "nsl": {
        "data": f"{BASE}/data_processed/nslkdd_v2/nsl_v2_arrays.npz",
        "mod": f"{BASE}/models/nsl_v2", "res": f"{BASE}/results/nsl_v2",
        "class_names": ["Normal", "DoS", "Probe", "R2L", "U2R"],
        "mask_categorical": True,  # NSL has one-hot categorical columns to freeze during the attack
    },
    "cicids": {
        "data": f"{BASE}/data_processed/cicids_v2/cicids_v2_primary.npz",
        "mod": f"{BASE}/models/cicids_v2", "res": f"{BASE}/results/cicids_v2",
        "class_names": ["BENIGN", "Brute Force", "DoS/DDoS", "PortScan", "Other"],
        "mask_categorical": False,
    },
}
EPS = 0.05
SEED = 0

def make_attacks(model, mask, ncls):
    M = tf.constant(mask, tf.float32)
    def loss_fn(z, y):
        return tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=z)
    @tf.function
    def pgd(x0, y, eps, steps=20, restarts=3):
        E = eps * M
        best_x, best_l = x0, tf.fill(tf.shape(y), -1e30)
        for r in range(restarts):
            x = x0 if r == 0 else tf.clip_by_value(x0 + tf.random.uniform(tf.shape(x0), -1., 1.) * E, 0., 1.)
            for s_ in range(steps):
                alpha = (1.0 if s_ < steps // 2 else 0.25) * E / 4.0
                with tf.GradientTape() as t:
                    t.watch(x); s = tf.reduce_sum(loss_fn(model(x, training=False), y))
                x = tf.clip_by_value(tf.clip_by_value(x + alpha * tf.sign(t.gradient(s, x)), x0 - E, x0 + E), 0., 1.)
                l_new = loss_fn(model(x, training=False), y)
                upd = l_new > best_l
                best_x, best_l = tf.where(upd[:, None], x, best_x), tf.where(upd, l_new, best_l)
        return best_x
    return pgd

def predict(model, X): return np.argmax(model.predict(X, batch_size=8192, verbose=0), axis=1)

def plot_cm(cm, class_names, title, path):
    cm_norm = cm.astype("float64") / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(class_names))); ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names))); ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            txt = f"{cm[i,j]}\n({cm_norm[i,j]*100:.1f}%)"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7,
                    color="white" if cm_norm[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)

# %% [1] Run for both datasets
all_results = {}
for ds_name, cfg in DATASETS.items():
    print(f"\n========== {ds_name.upper()} ==========")
    d = np.load(cfg["data"])
    Xte, yte = d["Xte"].astype("float32"), d["yte"].astype("int32")
    D, ncls = Xte.shape[1], len(cfg["class_names"])
    if cfg["mask_categorical"]:
        names = json.load(open(f"{BASE}/data_processed/nslkdd_v2/feature_names.json"))
        mask = np.array([0. if n.startswith(("protocol_type_", "service_", "flag_")) else 1. for n in names], "float32")
    else:
        mask = np.ones(D, "float32")

    model_paths = None
    if ds_name == "cicids":
        eps_fgsm, eps_pgd = json.load(open(f"{cfg['res']}/stage2b_selected_seed0.json")).values()
        eps_max = json.load(open(f"{cfg['res']}/stage3_selection_seed0.json"))["selected_eps"]
        model_paths = {
            "undefended": f"{cfg['mod']}/undefended_smote_seed0.keras",
            "fgsm_at": f"{cfg['mod']}/fgsm_eps{eps_fgsm}_seed0.keras",
            "pgd_at": f"{cfg['mod']}/pgd_eps{eps_pgd}_seed0.keras",
            "kd_feedback_r5": f"{cfg['mod']}/s3_kd_feedback_eps{eps_max}_r5_seed0.keras",
        }
    else:
        s2 = json.load(open(f"{cfg['res']}/stage2_test_seed0.json"))
        eps_fgsm, eps_pgd = s2["selected"]["fgsm"], s2["selected"]["pgd"]
        eps_max = json.load(open(f"{cfg['res']}/stage3_selection_seed0.json"))["selected_eps"]
        # NSL Stage 2 saves as f"{MOD}/s2_{tag}_seed{seed}.keras" where tag is "none" for undefended, or
        # "{kind}_eps{eps_train}" for fgsm/pgd - confirmed against nsl_stage2.py's train_model().
        # kd_feedback_r5 is saved by nsl_stage4a_seeds.py under its OWN naming ("s4_" prefix, no eps in the
        # name) rather than reusing Stage 3's "s3_..._eps{e}_..." pattern - confirmed via directory listing.
        model_paths = {
            "undefended": f"{cfg['mod']}/s2_none_seed0.keras",
            "fgsm_at": f"{cfg['mod']}/s2_fgsm_eps{eps_fgsm}_seed0.keras",
            "pgd_at": f"{cfg['mod']}/s2_pgd_eps{eps_pgd}_seed0.keras",
            "kd_feedback_r5": f"{cfg['mod']}/s4_kd_feedback_r5_seed0.keras",
        }

    ds_results = {}
    for tag, path in model_paths.items():
        if not os.path.exists(path):
            print(f"  SKIP {tag}: model file not found at {path} - check the path convention for this dataset")
            continue
        model = keras.models.load_model(path, compile=False)

        p_clean = predict(model, Xte)
        cm_clean = confusion_matrix(yte, p_clean, labels=range(ncls))
        plot_cm(cm_clean, cfg["class_names"], f"{ds_name.upper()} - {tag} - clean",
               f"{OUT}/{ds_name}_{tag}_clean.png")

        pgd = make_attacks(model, mask, ncls)
        Xa = np.vstack([pgd(tf.constant(Xte[i:i+4096]), tf.constant(yte[i:i+4096]), eps=EPS).numpy()
                        for i in range(0, len(Xte), 4096)])
        p_adv = predict(model, Xa)
        cm_adv = confusion_matrix(yte, p_adv, labels=range(ncls))
        plot_cm(cm_adv, cfg["class_names"], f"{ds_name.upper()} - {tag} - PGD eps={EPS}",
               f"{OUT}/{ds_name}_{tag}_adv_eps{EPS}.png")

        ds_results[tag] = {"clean_cm": cm_clean.tolist(), "adv_cm": cm_adv.tolist(),
                           "clean_acc": float((p_clean == yte).mean()), "adv_acc": float((p_adv == yte).mean())}
        print(f"  {tag:<16} clean_acc={ds_results[tag]['clean_acc']:.4f}  acc@eps={EPS}={ds_results[tag]['adv_acc']:.4f}")

    all_results[ds_name] = ds_results
    json.dump(all_results[ds_name], open(f"{OUT}/{ds_name}_confusion_matrices.json", "w"), indent=1)

print(f"\nSaved PNGs and JSON summaries to {OUT}/")
