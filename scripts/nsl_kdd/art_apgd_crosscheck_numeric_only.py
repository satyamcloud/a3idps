# Numeric-only cross-check for ART's APGD. ART accepted the `mask` argument but did not apply it (frozen columns changed by eps).
# Workaround: wrap each model so its one-hot columns are re-snapped to {0,1} before the network. For eps < 0.5 this erases any
# perturbation ART puts on those columns, so the attack is effectively numeric-only. Results merge into art_crosscheck_results.json.

# %% [0] Setup (same subset and models as nsl_art_crosscheck.py)
import subprocess
subprocess.run(["pip", "install", "-q", "adversarial-robustness-toolbox"], check=True)

from google.colab import drive
drive.mount('/content/drive')
import json, os, time, numpy as np, tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

BASE = "/content/drive/MyDrive/A3IDPS_COLAB"
RES = f"{BASE}/results/nsl_v2"; MOD = f"{BASE}/models/nsl_v2"
d = np.load(f"{BASE}/data_processed/nslkdd_v2/nsl_v2_arrays.npz")
Xte, yte = d["Xte"].astype("float32"), d["yte"].astype("int32")
names = json.load(open(f"{BASE}/data_processed/nslkdd_v2/feature_names.json"))
MASK_NUM = np.array([0. if n.startswith(("protocol_type_", "service_", "flag_")) else 1. for n in names], "float32")
D, NCLS = Xte.shape[1], 5
FROZEN = MASK_NUM == 0

sel = json.load(open(f"{RES}/stage2_selected_seed0.json"))
MODELS = {"undefended": f"{MOD}/s2_none_seed0.keras",
          f"fgsm_eps{sel['fgsm']}": f"{MOD}/s2_fgsm_eps{sel['fgsm']}_seed0.keras",
          f"pgd_eps{sel['pgd']}": f"{MOD}/s2_pgd_eps{sel['pgd']}_seed0.keras"}
# Later add the A3IDPS models here as well.
EPS_LIST = (0.05, 0.10)
sub = np.array(json.load(open(f"{RES}/art_crosscheck_subset_idx.json")))
X, y = Xte[sub], yte[sub]
Y1 = keras.utils.to_categorical(y, NCLS)
assert np.abs(X[:, FROZEN] - np.round(X[:, FROZEN])).max() == 0, "one-hot columns are not exactly 0/1 - snapping would be invalid"
OUT = f"{RES}/art_crosscheck_results.json"
results = json.load(open(OUT))

# %% [1] Snapping wrapper, our attack, ART wrapper
M_T = tf.constant(MASK_NUM)
def snapped(model):
    inp = keras.Input(shape=(D,))
    x = layers.Lambda(lambda t: M_T * t + (1. - M_T) * tf.round(t))(inp)
    return keras.Model(inp, model(x))

def loss_fn(z, yy, margin):
    if margin:
        oh = tf.one_hot(yy, NCLS)
        return tf.reduce_max(z - 1e9 * oh, axis=1) - tf.reduce_sum(z * oh, axis=1)
    return tf.nn.sparse_softmax_cross_entropy_with_logits(labels=yy, logits=z)

def our_pgd(model, X, y, eps, margin, steps=20, restarts=3):     # numeric-only via the mask, as in Stage 2
    E = tf.constant(eps * MASK_NUM); x0 = tf.constant(X); yb = tf.constant(y)
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
            best_x, best_l = tf.where(upd[:, None], x, best_x), tf.where(upd, l_new, best_l)
    return best_x.numpy()

import art
from art.estimators.classification import TensorFlowV2Classifier
from art.attacks.evasion import AutoProjectedGradientDescent
def make_apgd(clf, eps, lt):
    return AutoProjectedGradientDescent(estimator=clf, norm=np.inf, eps=eps, eps_step=eps / 4, max_iter=100, targeted=False,
                                        nb_random_init=2, batch_size=150, loss_type=lt, verbose=False)
correct = lambda m, Xa: np.argmax(m.predict(Xa, verbose=0), 1) == y

# %% [2] Run
for name, path in MODELS.items():
    model = keras.models.load_model(path, compile=False); wm = snapped(model)
    assert (np.argmax(wm.predict(X, verbose=0), 1) == np.argmax(model.predict(X, verbose=0), 1)).all()
    clf = TensorFlowV2Classifier(model=wm, nb_classes=NCLS, input_shape=(D,),
                                 loss_object=tf.keras.losses.CategoricalCrossentropy(from_logits=True), clip_values=(0.0, 1.0))
    for eps in EPS_LIST:
        key = f"{name}|{eps}|numeric_only"
        if key in results: continue
        ce = correct(model, our_pgd(model, X, y, eps, False)); mg = correct(model, our_pgd(model, X, y, eps, True))
        row = {"ours_pgd_ce": float(ce.mean()), "ours_pgd_margin": float(mg.mean()), "ours_worst": float((ce & mg).mean()),
               "method": "ART on snapped wrapper (one-hot columns re-rounded)"}
        art_c = {}
        for lt in ("cross_entropy", "difference_logits_ratio"):
            t0 = time.time(); Xa = make_apgd(clf, eps, lt).generate(x=X, y=Y1)
            art_c[lt] = correct(wm, Xa); row[f"art_{lt}"] = float(art_c[lt].mean())
            snap = MASK_NUM * Xa + (1 - MASK_NUM) * np.round(Xa)
            row[f"art_{lt}_effective_frozen_change"] = float(np.abs((snap - X)[:, FROZEN]).max())
            row[f"art_{lt}_effective_numeric_delta"] = float(np.abs((snap - X)[:, ~FROZEN]).max())
            print(f"  {key} ART {lt}: acc={art_c[lt].mean():.4f} | effective change on frozen cols={row[f'art_{lt}_effective_frozen_change']} ({time.time()-t0:.0f}s)")
        row["art_worst"] = float((art_c["cross_entropy"] & art_c["difference_logits_ratio"]).mean())
        row["combined_worst"] = float((ce & mg & art_c["cross_entropy"] & art_c["difference_logits_ratio"]).mean())
        row["art_stronger_by"] = row["ours_worst"] - row["art_worst"]
        results[key] = row; json.dump(results, open(OUT, "w"), indent=1)

print(f"\n{'model|eps|protocol':<34}{'ours worst':>11}{'ART worst':>11}{'combined':>10}  note")
for key, r in results.items():
    note = "ART clearly stronger - our attack may be weak here" if r["art_stronger_by"] > 0.04 else ""
    print(f"{key:<34}{r['ours_worst']:>11.4f}{r['art_worst']:>11.4f}{r['combined_worst']:>10.4f}  {note}")
