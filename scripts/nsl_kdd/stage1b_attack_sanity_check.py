# Stage 1b - stronger attack sanity check (no retraining: loads the saved Stage-1 baseline from Drive)
# Fixes in the attack code: logits-based losses, best-iterate tracking, restarts, worst-case over attacks,
# optional mask that keeps one-hot columns fixed. Paste into Colab.

# %% [0] Load
from google.colab import drive
drive.mount('/content/drive')
import json, numpy as np, tensorflow as tf
from tensorflow import keras
from sklearn.metrics import accuracy_score

BASE = "/content/drive/MyDrive/A3IDPS_COLAB"
d = np.load(f"{BASE}/data_processed/nslkdd_v2/nsl_v2_arrays.npz")
Xte, yte = d["Xte"].astype("float32"), d["yte"].astype("int32")
names = json.load(open(f"{BASE}/data_processed/nslkdd_v2/feature_names.json"))
NUMERIC_MASK = np.array([0. if n.startswith(("protocol_type_", "service_", "flag_")) else 1. for n in names], "float32")
print("perturbable numeric features:", int(NUMERIC_MASK.sum()), "of", len(names))
model = keras.models.load_model(f"{BASE}/models/nsl_v2/undefended_seed0.keras")

# %% [1] Logits function (avoids softmax saturation / zero gradients) and losses
def make_logits_fn(m):
    W, b = [tf.constant(w) for w in m.layers[-1].get_weights()]
    body = keras.Model(m.input, m.layers[-2].output)
    return lambda x: tf.matmul(body(x, training=False), W) + b

logits_fn = make_logits_fn(model)
assert np.mean(np.argmax(logits_fn(tf.constant(Xte)).numpy(), 1) == np.argmax(model.predict(Xte, verbose=0), 1)) == 1.0

def ce_loss(z, y):
    return tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=z)

def margin_loss(z, y):                     # >0 means misclassified
    oh = tf.one_hot(y, z.shape[-1])
    return tf.reduce_max(z - 1e9 * oh, 1) - tf.reduce_sum(z * oh, 1)

# %% [2] Attacks (all respect the [0,1] box and the per-feature budget eps * mask)
def fgsm(X, y, eps, mask, bs=4096):
    out = []
    for i in range(0, len(X), bs):
        x0 = tf.constant(X[i:i+bs]); yb = tf.constant(y[i:i+bs])
        with tf.GradientTape() as t:
            t.watch(x0); s = tf.reduce_sum(ce_loss(logits_fn(x0), yb))
        g = t.gradient(s, x0)
        out.append(tf.clip_by_value(x0 + eps * mask * tf.sign(g), 0., 1.).numpy())
    return np.vstack(out)

def pgd(X, y, eps, mask, loss_fn, steps=20, restarts=3, bs=4096):
    E = tf.constant(eps * mask)
    out = []
    for i in range(0, len(X), bs):
        x0 = tf.constant(X[i:i+bs]); yb = tf.constant(y[i:i+bs])
        best_x, best_l = x0, tf.fill([x0.shape[0]], -1e30)
        for r in range(restarts):
            x = x0 if r == 0 else tf.clip_by_value(x0 + tf.random.uniform(x0.shape, -1., 1.) * E, 0., 1.)
            for t_ in range(steps):
                alpha = (1.0 if t_ < steps // 2 else 0.25) * E / 4.0
                with tf.GradientTape() as t:
                    t.watch(x); s = tf.reduce_sum(loss_fn(logits_fn(x), yb))
                g = t.gradient(s, x)
                x = tf.clip_by_value(tf.clip_by_value(x + alpha * tf.sign(g), x0 - E, x0 + E), 0., 1.)
                l_new = loss_fn(logits_fn(x), yb)
                upd = l_new > best_l
                best_x = tf.where(upd[:, None], x, best_x); best_l = tf.where(upd, l_new, best_l)
        out.append(best_x.numpy())
    return np.vstack(out)

def correct_mask(Xa, y):
    return np.argmax(logits_fn(tf.constant(Xa)).numpy(), 1) == y

# %% [3] Sanity table, with and without the categorical mask
EPS = [0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.0]
results = {}
for use_mask in (False, True):
    mask = NUMERIC_MASK if use_mask else np.ones_like(NUMERIC_MASK)
    tag = "numeric_only" if use_mask else "all_features"
    rows, flags = [], []
    print(f"\n=== perturbation on: {tag} ===")
    for e in EPS:
        if e == 0:
            a = float(accuracy_score(yte, np.argmax(logits_fn(tf.constant(Xte)).numpy(), 1)))
            rows.append({"eps": e, "fgsm": a, "pgd_ce": a, "pgd_margin": a, "worst_case": a}); continue
        c_f = correct_mask(fgsm(Xte, yte, e, mask), yte)
        c_c = correct_mask(pgd(Xte, yte, e, mask, ce_loss), yte)
        c_m = correct_mask(pgd(Xte, yte, e, mask, margin_loss), yte)
        worst = c_f & c_c & c_m
        rows.append({"eps": e, "fgsm": float(c_f.mean()), "pgd_ce": float(c_c.mean()),
                     "pgd_margin": float(c_m.mean()), "worst_case": float(worst.mean())})
        r = rows[-1]
        print(f"eps={e:<5} FGSM={r['fgsm']:.4f}  PGD-CE={r['pgd_ce']:.4f}  PGD-margin={r['pgd_margin']:.4f}  worst-case={r['worst_case']:.4f}")
        if r["pgd_ce"] > r["fgsm"] + 0.02:
            flags.append(f"PGD-CE weaker than FGSM at eps={e}")
    for i in range(len(rows) - 1):
        if rows[i + 1]["worst_case"] > rows[i]["worst_case"] + 0.01:
            flags.append(f"worst-case accuracy rises from eps={rows[i]['eps']} to {rows[i+1]['eps']}")
    if use_mask is False and rows[-1]["worst_case"] > 0.05:
        flags.append("worst-case at eps=1.0 (all features) is not near 0")
    print("RED FLAGS:", flags if flags else "none")
    results[tag] = {"rows": rows, "flags": flags}

json.dump(results, open(f"{BASE}/results/nsl_v2/stage1b_attack_sanity_seed0.json", "w"), indent=1)
print("saved stage1b_attack_sanity_seed0.json")
