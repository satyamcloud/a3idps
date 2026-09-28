# ART APGD cross-check (Option A): independent published implementation vs our worst-case attacks,
# on a FIXED 300-row test subset, for any list of models. Resumable (results cached in Drive).
# Reused later for the A3IDPS models: just add them to MODELS below and re-run.
# In the paper this is described as "APGD-CE and APGD-DLR (components of AutoAttack)", not "AutoAttack".

# %% [0] Install, load, configuration
import subprocess
subprocess.run(["pip", "install", "-q", "adversarial-robustness-toolbox"], check=True)

from google.colab import drive
drive.mount('/content/drive')
import json, os, time, numpy as np, tensorflow as tf
from tensorflow import keras

BASE = "/content/drive/MyDrive/A3IDPS_COLAB"
RES = f"{BASE}/results/nsl_v2"; MOD = f"{BASE}/models/nsl_v2"
d = np.load(f"{BASE}/data_processed/nslkdd_v2/nsl_v2_arrays.npz")
Xte, yte = d["Xte"].astype("float32"), d["yte"].astype("int32")
names = json.load(open(f"{BASE}/data_processed/nslkdd_v2/feature_names.json"))
MASK_NUM = np.array([0. if n.startswith(("protocol_type_", "service_", "flag_")) else 1. for n in names], "float32")
MASK_ALL = np.ones_like(MASK_NUM)
D, NCLS = Xte.shape[1], 5

sel = json.load(open(f"{RES}/stage2_selected_seed0.json"))
MODELS = {
    "undefended": f"{MOD}/s2_none_seed0.keras",
    f"fgsm_eps{sel['fgsm']}": f"{MOD}/s2_fgsm_eps{sel['fgsm']}_seed0.keras",
    f"pgd_eps{sel['pgd']}": f"{MOD}/s2_pgd_eps{sel['pgd']}_seed0.keras",
}
# Later, after Stage 3, add e.g.:
# MODELS["a3idps_kd_feedback_r2"] = f"{MOD}/s3_kd_feedback_eps0.05_r2_seed0.keras"   (use the eps_max Stage 3 selected)
EPS_LIST = (0.05, 0.10)
PROTOCOLS = ("all_features", "numeric_only")
N = 300
sub = np.random.RandomState(0).choice(len(Xte), N, replace=False)     # same rows as the earlier ART check
json.dump(sub.tolist(), open(f"{RES}/art_crosscheck_subset_idx.json", "w"))
X, y = Xte[sub], yte[sub]
Y1 = keras.utils.to_categorical(y, NCLS)
OUT = f"{RES}/art_crosscheck_results.json"
results = json.load(open(OUT)) if os.path.exists(OUT) else {}

# %% [1] Our attack (same logic as Stage 2: worst case over PGD-CE and PGD-margin, 20 steps x 3 restarts)
def loss_fn(z, yy, margin):
    if margin:
        oh = tf.one_hot(yy, NCLS)
        return tf.reduce_max(z - 1e9 * oh, axis=1) - tf.reduce_sum(z * oh, axis=1)
    return tf.nn.sparse_softmax_cross_entropy_with_logits(labels=yy, logits=z)

def our_pgd(model, X, y, eps, mask, margin, steps=20, restarts=3):
    E = tf.constant(eps * mask); x0 = tf.constant(X); yb = tf.constant(y)
    best_x, best_l = x0, tf.fill(tf.shape(yb), -1e30)
    for r in range(restarts):
        x = x0 if r == 0 else tf.clip_by_value(x0 + tf.random.uniform(tf.shape(x0), -1., 1.) * E, 0., 1.)
        for s_ in range(steps):
            alpha = (1.0 if s_ < steps // 2 else 0.25) * E / 4.0
            with tf.GradientTape() as t:
                t.watch(x); s = tf.reduce_sum(loss_fn(model(x, training=False), yb, margin))
            x = tf.clip_by_value(tf.clip_by_value(x + alpha * tf.sign(t.gradient(s, x)), x0 - E, x0 + E), 0., 1.)
            l_new = loss_fn(model(x, training=False), yb, margin)
            upd = l_new > best_l
            best_x = tf.where(upd[:, None], x, best_x); best_l = tf.where(upd, l_new, best_l)
    return best_x.numpy()

def correct(model, Xa):
    return np.argmax(model.predict(Xa, verbose=0), 1) == y

# %% [2] ART wrapper; find out which mask argument (if any) ART's APGD accepts
import art
from art.estimators.classification import TensorFlowV2Classifier
from art.attacks.evasion import AutoProjectedGradientDescent
print("ART version:", art.__version__)

def make_art(model):
    return TensorFlowV2Classifier(model=model, nb_classes=NCLS, input_shape=(D,),
                                  loss_object=tf.keras.losses.CategoricalCrossentropy(from_logits=True), clip_values=(0.0, 1.0))

def make_apgd(clf, eps, loss_type, n_init=2, iters=100):
    return AutoProjectedGradientDescent(estimator=clf, norm=np.inf, eps=eps, eps_step=eps / 4, max_iter=iters, targeted=False,
                                        nb_random_init=n_init, batch_size=150, loss_type=loss_type, verbose=False)

def find_art_mask(model):
    clf = make_art(model); last = None
    for cand, label in ((MASK_NUM, "1-D (122,)"), (MASK_NUM[None, :], "(1,122)")):
        try:
            Xa = make_apgd(clf, 0.05, "cross_entropy", n_init=1, iters=20).generate(x=X[:60], y=Y1[:60], mask=cand)
            frozen = float(np.abs((Xa - X[:60])[:, MASK_NUM == 0]).max())
            print(f"mask form {label}: accepted, max change on frozen columns = {frozen}")
            if frozen == 0.0: return cand
        except Exception as ex:
            last = ex; print(f"mask form {label}: rejected ({type(ex).__name__}: {ex})")
    return None

first_model = keras.models.load_model(next(iter(MODELS.values())), compile=False)
ART_MASK = find_art_mask(first_model)
print("numeric-only cross-check available:", ART_MASK is not None)

# %% [3] Run the comparison
for name, path in MODELS.items():
    model = keras.models.load_model(path, compile=False); clf = make_art(model)
    print(f"\n##### {name}  (clean acc on subset: {correct(model, X).mean():.4f})")
    for eps in EPS_LIST:
        for prot in PROTOCOLS:
            key = f"{name}|{eps}|{prot}"
            if key in results: continue
            if prot == "numeric_only" and ART_MASK is None:
                print(f"skip {key}: ART mask unavailable"); continue
            mask = MASK_NUM if prot == "numeric_only" else MASK_ALL
            ce = correct(model, our_pgd(model, X, y, eps, mask, False)); mg = correct(model, our_pgd(model, X, y, eps, mask, True))
            row = {"ours_pgd_ce": float(ce.mean()), "ours_pgd_margin": float(mg.mean()), "ours_worst": float((ce & mg).mean())}
            art_c = {}
            for lt in ("cross_entropy", "difference_logits_ratio"):
                t0 = time.time()
                kw = {"mask": ART_MASK} if prot == "numeric_only" else {}
                Xa = make_apgd(clf, eps, lt).generate(x=X, y=Y1, **kw)
                art_c[lt] = correct(model, Xa)
                row[f"art_{lt}"] = float(art_c[lt].mean())
                row[f"art_{lt}_max_delta"] = float(np.abs(Xa - X).max())
                if prot == "numeric_only": row[f"art_{lt}_frozen_change"] = float(np.abs((Xa - X)[:, MASK_NUM == 0]).max())
                print(f"  {key} ART {lt}: acc={art_c[lt].mean():.4f} ({time.time()-t0:.0f}s)")
            row["art_worst"] = float((art_c["cross_entropy"] & art_c["difference_logits_ratio"]).mean())
            row["combined_worst"] = float((ce & mg & art_c["cross_entropy"] & art_c["difference_logits_ratio"]).mean())
            row["art_stronger_by"] = row["ours_worst"] - row["art_worst"]
            results[key] = row
            json.dump(results, open(OUT, "w"), indent=1)

# %% [4] Summary (lower accuracy = stronger attack; N=300 so differences under ~3-4 pp are noise)
print(f"\n{'model|eps|protocol':<34}{'ours worst':>11}{'ART worst':>11}{'combined':>10}  note")
for key, r in results.items():
    note = "ART clearly stronger - our attack may be weak here" if r["art_stronger_by"] > 0.04 else ""
    print(f"{key:<34}{r['ours_worst']:>11.4f}{r['art_worst']:>11.4f}{r['combined_worst']:>10.4f}  {note}")
print("saved:", OUT)
