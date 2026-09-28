# C&W (L2) generalization check - the attack-ensemble-generalization angle proposed for the revision.
#
# Everything reported elsewhere (Stage 2/3/4, the ART APGD cross-check) evaluates worst-case accuracy against
# an ensemble that is entirely Linf / sign-gradient based: FGSM, PGD-CE, PGD-margin, the AdvGAN sweep, and
# ART's APGD-CE/APGD-DLR (both also Linf, sign-step attacks). A3IDPS was tuned and validated against exactly
# that family. This script checks whether its (or PGD-AT's) accuracy holds up against an attack that is
# algorithmically unrelated - not just a new epsilon of the same attack:
#
#   Carlini-Wagner L2 (ART's CarliniL2Method) - optimization-based, L2-norm, minimizes ||delta||_2 subject to
#   misclassification via a binary search over a soft constraint, no sign-gradient steps, no fixed epsilon.
#
# Because C&W has no epsilon, "worst-case at eps=X" doesn't apply here - instead we report (a) accuracy under
# the attack, and (b) the median/mean L2 perturbation norm C&W needed to achieve it, so a reader can judge
# whether the two models are being fooled with comparably small perturbations or not. This is a DIFFERENT
# threat model from the rest of the paper, not a drop-in replacement worst-case number - keep that framing
# whatever the numbers turn out to be.
#
# Same 4 models per dataset as confusion_matrices.py (undefended, fgsm_at, pgd_at, kd_feedback_r5), seed 0,
# same 300-row subset convention (RandomState(0)) as the other ART cross-checks. Self-contained, read-only
# over existing saved models - no training.

# %% [0] Install and load
import subprocess
subprocess.run(["pip", "install", "-q", "adversarial-robustness-toolbox"], check=True)

from google.colab import drive
drive.mount('/content/drive')
import json, os, time
import numpy as np, tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

BASE = "/content/drive/MyDrive/A3IDPS_COLAB"
N_SUBSET = 300

DATASETS = {
    "nsl": {
        "data": f"{BASE}/data_processed/nslkdd_v2/nsl_v2_arrays.npz",
        "mod": f"{BASE}/models/nsl_v2", "res": f"{BASE}/results/nsl_v2",
        "mask_categorical": True,
    },
    "cicids": {
        "data": f"{BASE}/data_processed/cicids_v2/cicids_v2_primary.npz",
        "mod": f"{BASE}/models/cicids_v2", "res": f"{BASE}/results/cicids_v2",
        "mask_categorical": False,
    },
}

import art
from art.estimators.classification import TensorFlowV2Classifier
from art.attacks.evasion import CarliniL2Method
print("ART version:", art.__version__)


def snapped(model, mask_num, D):
    # Re-round one-hot columns after every forward pass so an L2 attack can't spend its budget there -
    # same wrapper used in art_apgd_crosscheck_numeric_only.py, since CarliniL2Method has no native mask kwarg.
    M_T = tf.constant(mask_num)
    inp = keras.Input(shape=(D,))
    x = layers.Lambda(lambda t: M_T * t + (1. - M_T) * tf.round(t))(inp)
    return keras.Model(inp, model(x))


# %% [1] Run for both datasets
for ds_name, cfg in DATASETS.items():
    print(f"\n========== {ds_name.upper()} ==========")
    RES, MOD = cfg["res"], cfg["mod"]
    d = np.load(cfg["data"])
    Xte, yte = d["Xte"].astype("float32"), d["yte"].astype("int32")
    D, NCLS = Xte.shape[1], int(yte.max()) + 1

    if cfg["mask_categorical"]:
        names = json.load(open(f"{BASE}/data_processed/nslkdd_v2/feature_names.json"))
        mask_num = np.array([0. if n.startswith(("protocol_type_", "service_", "flag_")) else 1. for n in names], "float32")
    else:
        mask_num = np.ones(D, "float32")

    if ds_name == "cicids":
        eps_fgsm, eps_pgd = json.load(open(f"{RES}/stage2b_selected_seed0.json")).values()
        eps_max = json.load(open(f"{RES}/stage3_selection_seed0.json"))["selected_eps"]
        model_paths = {
            "undefended": f"{MOD}/undefended_smote_seed0.keras",
            "fgsm_at": f"{MOD}/fgsm_eps{eps_fgsm}_seed0.keras",
            "pgd_at": f"{MOD}/pgd_eps{eps_pgd}_seed0.keras",
            "kd_feedback_r5": f"{MOD}/s3_kd_feedback_eps{eps_max}_r5_seed0.keras",
        }
    else:
        s2 = json.load(open(f"{RES}/stage2_test_seed0.json"))
        eps_fgsm, eps_pgd = s2["selected"]["fgsm"], s2["selected"]["pgd"]
        eps_max = json.load(open(f"{RES}/stage3_selection_seed0.json"))["selected_eps"]
        model_paths = {
            "undefended": f"{MOD}/s2_none_seed0.keras",
            "fgsm_at": f"{MOD}/s2_fgsm_eps{eps_fgsm}_seed0.keras",
            "pgd_at": f"{MOD}/s2_pgd_eps{eps_pgd}_seed0.keras",
            "kd_feedback_r5": f"{MOD}/s4_kd_feedback_r5_seed0.keras",
        }

    sub = np.random.RandomState(0).choice(len(Xte), N_SUBSET, replace=False)
    X, y = Xte[sub], yte[sub]
    Y1 = keras.utils.to_categorical(y, NCLS)

    ds_results = {}
    for tag, path in model_paths.items():
        model = keras.models.load_model(path, compile=False)
        wm = snapped(model, mask_num, D) if cfg["mask_categorical"] else model
        clean_acc = float((np.argmax(wm.predict(X, batch_size=512, verbose=0), 1) == y).mean())

        clf = TensorFlowV2Classifier(model=wm, nb_classes=NCLS, input_shape=(D,),
                                      loss_object=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                                      clip_values=(0.0, 1.0))
        atk = CarliniL2Method(classifier=clf, confidence=0.0, targeted=False, learning_rate=0.01,
                               binary_search_steps=9, max_iter=100, initial_const=0.01,
                               max_halving=5, max_doubling=5, batch_size=100, verbose=False)
        t0 = time.time()
        Xa = atk.generate(x=X, y=Y1)
        adv_acc = float((np.argmax(wm.predict(Xa, batch_size=512, verbose=0), 1) == y).mean())
        l2 = np.linalg.norm((Xa - X).reshape(len(X), -1), axis=1)
        row = {
            "clean_acc": clean_acc,
            "cw_l2_acc": adv_acc,
            "cw_l2_median_perturbation_norm": float(np.median(l2)),
            "cw_l2_mean_perturbation_norm": float(np.mean(l2)),
            "seconds": round(time.time() - t0, 1),
        }
        ds_results[tag] = row
        print(f"  {tag:<16} clean={clean_acc:.4f}  under C&W-L2={adv_acc:.4f}  "
              f"median||delta||2={row['cw_l2_median_perturbation_norm']:.4f}  ({row['seconds']}s)")

    out_path = f"{RES}/cw_generalization_check.json"
    json.dump(ds_results, open(out_path, "w"), indent=1)
    print(f"saved {out_path}")

print("\nDone. Note: C&W-L2 has no fixed epsilon, so these accuracy numbers are NOT directly comparable to the "
      "eps=0.02/0.05/0.1 worst-case tables elsewhere - compare accuracy alongside the perturbation-norm columns.")
