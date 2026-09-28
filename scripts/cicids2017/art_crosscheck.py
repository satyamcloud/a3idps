# CICIDS2017 v2 - ART cross-check, mirroring nsl_art_crosscheck.py. Purpose: an independent, well-established
# attack library (Adversarial Robustness Toolbox) is run against the SAME models on the SAME fixed subset of
# test rows, to confirm our own hand-written PGD is not secretly weaker than it should be (the thing we'd want
# a reviewer to be able to check independently). CICIDS has no categorical columns, so - unlike the NSL-KDD
# version - there is no numeric-only/all-features distinction and no snapping-wrapper needed: every one of the
# 78 features is a real numeric flow statistic, so ART's mask (which NSL-KDD's cross-check discovered ART
# silently ignores) is not needed here since there is nothing to freeze in the first place.
#
# This runs on seed-0 models only (it is a sanity/validity check on the attack implementation, not part of the
# 5-seed statistical claims): undefended, fgsm_at, pgd_at, kd_feedback_r5. Uses ART's AutoProjectedGradientDescent
# (APGD-CE and APGD-DLR) - full AutoAttack failed on tabular data in the NSL-KDD check (SquareAttack requires
# image inputs), so APGD components are used and described in the paper as "components of AutoAttack", not
# "AutoAttack" itself, exactly as agreed for NSL-KDD.

# %% [0] Setup
from google.colab import drive
drive.mount('/content/drive')
import os, json
import numpy as np, tensorflow as tf
from tensorflow import keras

BASE = "/content/drive/MyDrive/A3IDPS_COLAB"
DATA = f"{BASE}/data_processed/cicids_v2"; MOD = f"{BASE}/models/cicids_v2"; RES = f"{BASE}/results/cicids_v2"

d = np.load(f"{DATA}/cicids_v2_primary.npz")
Xte, yte = d["Xte"].astype("float32"), d["yte"].astype("int32")
D, NCLS = Xte.shape[1], 5
CLASS_NAMES = ["BENIGN", "Brute Force", "DoS/DDoS", "PortScan", "Other"]

sel_at = json.load(open(f"{RES}/stage2b_selected_seed0.json"))
sel_gen = json.load(open(f"{RES}/stage3_selection_seed0.json"))
EPS_FGSM, EPS_PGD, EPS_MAX = sel_at["fgsm"], sel_at["pgd"], sel_gen["selected_eps"]

MODEL_PATHS = {
    "undefended":     f"{MOD}/undefended_smote_seed0.keras",
    "fgsm_at":         f"{MOD}/fgsm_eps{EPS_FGSM}_seed0.keras",
    "pgd_at":          f"{MOD}/pgd_eps{EPS_PGD}_seed0.keras",
    "kd_feedback_r5":  f"{MOD}/s3_kd_feedback_eps{EPS_MAX}_r5_seed0.keras",
}
EPS_LIST = [0.02, 0.05, 0.10]
N_SUBSET = 300

def jsonable(o):
    if isinstance(o, dict): return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, np.floating): return float(o)
    return o

# %% [1] Fixed subset (same one used for every model, saved so it is reproducible / inspectable)
subset_path = f"{RES}/art_crosscheck_cicids_subset_idx.json"
if os.path.exists(subset_path):
    idx = np.array(json.load(open(subset_path)))
else:
    idx = np.random.RandomState(0).choice(len(Xte), size=N_SUBSET, replace=False)
    json.dump(idx.tolist(), open(subset_path, "w"))
Xs, ys = Xte[idx], yte[idx]
print(f"Fixed cross-check subset: {len(idx)} rows (seed 0, saved to {subset_path})")

# %% [2] Our own attack (verbatim logic from Stage 2b/3, re-run on this exact subset for a fair comparison)
def make_attacks(model):
    def loss_fn(z, y, margin):
        if margin:
            oh = tf.one_hot(y, NCLS)
            return tf.reduce_max(z - 1e9 * oh, axis=1) - tf.reduce_sum(z * oh, axis=1)
        return tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=z)
    @tf.function
    def fgsm(x0, y, eps):
        with tf.GradientTape() as t:
            t.watch(x0); s = tf.reduce_sum(loss_fn(model(x0, training=False), y, False))
        return tf.clip_by_value(x0 + eps * tf.sign(t.gradient(s, x0)), 0., 1.)
    @tf.function
    def pgd(x0, y, eps, steps, restarts, margin):
        E = eps
        best_x, best_l = x0, tf.fill(tf.shape(y), -1e30)
        for r in range(restarts):
            x = x0 if r == 0 else tf.clip_by_value(x0 + tf.random.uniform(tf.shape(x0), -1., 1.) * E, 0., 1.)
            for s_ in range(steps):
                alpha = (1.0 if s_ < steps // 2 else 0.25) * E / 4.0
                with tf.GradientTape() as t:
                    t.watch(x); s = tf.reduce_sum(loss_fn(model(x, training=False), y, margin))
                x = tf.clip_by_value(tf.clip_by_value(x + alpha * tf.sign(t.gradient(s, x)), x0 - E, x0 + E), 0., 1.)
                l_new = loss_fn(model(x, training=False), y, margin)
                upd = l_new > best_l
                best_x, best_l = tf.where(upd[:, None], x, best_x), tf.where(upd, l_new, best_l)
        return best_x
    return {"fgsm": fgsm, "pgd": pgd}

def predict(model, X): return np.argmax(model.predict(X, batch_size=8192, verbose=0), axis=1)

def our_worst_case(model, X, y, eps):
    A = make_attacks(model)
    et = tf.constant(eps, tf.float32)
    c_f = predict(model, A["fgsm"](tf.constant(X), tf.constant(y), eps=et).numpy()) == y
    c_c = predict(model, A["pgd"](tf.constant(X), tf.constant(y), eps=et, steps=20, restarts=3, margin=False).numpy()) == y
    c_m = predict(model, A["pgd"](tf.constant(X), tf.constant(y), eps=et, steps=20, restarts=3, margin=True).numpy()) == y
    return float((c_f & c_c & c_m).mean())

# %% [3] ART cross-check (APGD-CE, APGD-DLR - components of AutoAttack; full AutoAttack fails on tabular data)
import art
from art.estimators.classification import TensorFlowV2Classifier
from art.attacks.evasion import AutoProjectedGradientDescent

print("ART version:", art.__version__)

def art_classifier(model):
    return TensorFlowV2Classifier(
        model=model, nb_classes=NCLS, input_shape=(D,),
        loss_object=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        clip_values=(0.0, 1.0))

results = {}
for tag, path in MODEL_PATHS.items():
    model = keras.models.load_model(path, compile=False)
    clf = art_classifier(model)
    results[tag] = {}
    for eps in EPS_LIST:
        ours = our_worst_case(model, Xs, ys, eps)
        row = {"our_worst_case": ours}
        for loss_type, label in [("cross_entropy", "apgd_ce"), ("difference_logits_ratio", "apgd_dlr")]:
            atk = AutoProjectedGradientDescent(estimator=clf, norm="inf", eps=eps, eps_step=eps / 4,
                                               max_iter=20, nb_random_init=3, loss_type=loss_type,
                                               batch_size=len(Xs), verbose=False)
            Xa = atk.generate(x=Xs, y=ys)
            max_delta = float(np.max(np.abs(Xa - Xs)))
            if max_delta > eps + 1e-3:
                print(f"  WARNING [{tag} eps={eps} {label}]: ART perturbation {max_delta:.4f} exceeds eps bound")
            art_acc = float((predict(model, Xa) == ys).mean())
            row[label] = art_acc
            flag = " <== ART STRONGER (>4pp)" if (ours - art_acc) > 0.04 else ""
            print(f"{tag:<16} eps={eps:<5} {label:<9} ART acc={art_acc:.4f}  ours worst-case={ours:.4f}{flag}")
        results[tag][str(eps)] = row
        json.dump(jsonable(results), open(f"{RES}/art_crosscheck_cicids_results.json", "w"), indent=1)

# %% [4] Summary
print("\n=== Summary: cases where ART's APGD beat our worst-case by more than 4pp ===")
any_flag = False
for tag, by_eps in results.items():
    for eps, row in by_eps.items():
        for label in ("apgd_ce", "apgd_dlr"):
            gap = row["our_worst_case"] - row[label]
            if gap > 0.04:
                any_flag = True
                print(f"  {tag} eps={eps} {label}: ART acc={row[label]:.4f} vs ours={row['our_worst_case']:.4f} (gap {gap*100:.1f}pp)")
if not any_flag:
    print("  none - our attack implementation is at least as strong as ART's APGD components on every model/eps checked.")
print(f"\nSaved: {RES}/art_crosscheck_cicids_results.json")
