# CICIDS2017 v2 - Stage 2: shared architecture, undefended baseline, attack sanity checks.
# Same structure and same frozen conventions as the NSL-KDD stages: logits-output model, worst-case attacks
# (FGSM / PGD-CE / PGD-margin), class-weighted training, and Carlini et al. sanity checks across an eps sweep.
# Primary protocol = the stratified row-level split (cicids_v2_primary.npz). The blocked split is evaluated
# separately at the end, purely as a robustness check, never used for training or selection.

# %% [0] Setup
from google.colab import drive
drive.mount('/content/drive')
import os, json, random, time
import numpy as np, tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight

BASE = "/content/drive/MyDrive/A3IDPS_COLAB"
DATA = f"{BASE}/data_processed/cicids_v2"
MOD  = f"{BASE}/models/cicids_v2"
RES  = f"{BASE}/results/cicids_v2"
os.makedirs(MOD, exist_ok=True); os.makedirs(RES, exist_ok=True)

d = np.load(f"{DATA}/cicids_v2_primary.npz")
Xtr, Xva, Xte = [d[k].astype("float32") for k in ("Xtr", "Xva", "Xte")]
ytr, yva, yte = [d[k].astype("int32") for k in ("ytr", "yva", "yte")]
names = json.load(open(f"{DATA}/feature_names.json"))
D, NCLS = Xtr.shape[1], 5
CLASS_NAMES = ["BENIGN", "Brute Force", "DoS/DDoS", "PortScan", "Other"]
print("D =", D, "| train/val/test:", Xtr.shape, Xva.shape, Xte.shape)
print("class counts (train):", np.bincount(ytr), "| (test):", np.bincount(yte))

# No categorical one-hot columns here (unlike NSL-KDD), so the "numeric-only" vs "all-features" distinction
# does not apply - every one of the 78 CICIDS features is a genuine numeric flow statistic. All eps budgets
# below perturb every feature.
MASK_ALL = np.ones(D, "float32")

SEED = 0
EPS_EVAL = [0.01, 0.02, 0.05, 0.10, 0.15]
CLASS_WEIGHT_CAP, BATCH, MAX_EPOCHS = 20.0, 2048, 60          # larger batch: CICIDS train set is ~50x NSL-KDD's
cw = np.minimum(compute_class_weight("balanced", classes=np.arange(NCLS), y=ytr), CLASS_WEIGHT_CAP).astype("float32")
print("class weights:", cw.tolist())

def set_seed(s): random.seed(s); np.random.seed(s); tf.keras.utils.set_random_seed(s)
def jsonable(o):
    if isinstance(o, dict): return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, np.floating): return float(o)
    return o

# %% [1] Model (LOGITS output) and attacks - identical structure to the NSL-KDD stages
def build_model(d, widths=(256, 128, 64)):
    inp = keras.Input(shape=(d,))
    x = layers.Dense(widths[0])(inp); x = layers.BatchNormalization()(x); x = layers.Activation("relu")(x)
    x = layers.Dense(widths[1])(x);   x = layers.Dropout(0.3)(x);          x = layers.Activation("relu")(x)
    x = layers.Dense(widths[2])(x);   x = layers.Dropout(0.3)(x);          x = layers.Activation("relu")(x)
    return keras.Model(inp, layers.Dense(NCLS)(x))

def make_attacks(model, mask):
    M = tf.constant(mask, tf.float32)
    def loss_fn(z, y, margin):
        if margin:
            oh = tf.one_hot(y, NCLS)
            return tf.reduce_max(z - 1e9 * oh, axis=1) - tf.reduce_sum(z * oh, axis=1)
        return tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=z)
    @tf.function
    def fgsm(x0, y, eps):
        with tf.GradientTape() as t:
            t.watch(x0); s = tf.reduce_sum(loss_fn(model(x0, training=False), y, False))
        return tf.clip_by_value(x0 + eps * M * tf.sign(t.gradient(s, x0)), 0., 1.)
    @tf.function
    def pgd(x0, y, eps, steps, restarts, margin):
        E = eps * M
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

def run_attack(fn, X, y, bs=4096, **kw):
    return np.vstack([fn(tf.constant(X[i:i+bs]), tf.constant(y[i:i+bs]), **kw).numpy() for i in range(0, len(X), bs)])

def predict(model, X): return np.argmax(model.predict(X, batch_size=8192, verbose=0), axis=1)

def clean_eval(model, X, y):
    p = predict(model, X)
    present = sorted(np.unique(y).tolist())
    return {"acc": float(accuracy_score(y, p)),
            "macro_f1": float(f1_score(y, p, average="macro", labels=present, zero_division=0)),
            "per_class_f1": {CLASS_NAMES[c]: float(f1_score(y == c, p == c, zero_division=0)) for c in present},
            "per_class_support": {CLASS_NAMES[c]: int((y == c).sum()) for c in present}}

def robust_eval(model, X, y, mask, eps_list, steps=20, restarts=3):
    A = make_attacks(model, mask); out = {}
    present = sorted(np.unique(y).tolist())
    for e in eps_list:
        et = tf.constant(e, tf.float32)
        c_f = predict(model, run_attack(A["fgsm"], X, y, eps=et)) == y
        c_c = predict(model, run_attack(A["pgd"], X, y, eps=et, steps=steps, restarts=restarts, margin=False)) == y
        c_m = predict(model, run_attack(A["pgd"], X, y, eps=et, steps=steps, restarts=restarts, margin=True)) == y
        w = c_f & c_c & c_m
        out[str(e)] = {"fgsm": float(c_f.mean()), "pgd_ce": float(c_c.mean()), "pgd_margin": float(c_m.mean()),
                       "worst": float(w.mean()),
                       "worst_recall": {CLASS_NAMES[c]: float(w[y == c].mean()) for c in present}}
    return out

def train(model, X, y, Xv, yv, epochs=MAX_EPOCHS):
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    cw_dict = {i: float(w) for i, w in enumerate(cw)}
    model.fit(X, y, validation_data=(Xv, yv), epochs=epochs, batch_size=BATCH,
              class_weight=cw_dict, verbose=2,
              callbacks=[keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True),
                         keras.callbacks.ReduceLROnPlateau(patience=4, factor=0.5)])
    return model

# %% [2] Train the undefended baseline
set_seed(SEED)
model = build_model(D)
print("parameters (total):", model.count_params())
train(model, Xtr, ytr, Xva, yva)
model.save(f"{MOD}/undefended_seed{SEED}.keras")

clean = clean_eval(model, Xte, yte)
print("CLEAN test:", {k: clean[k] for k in ("acc", "macro_f1")})
print("per-class F1:", clean["per_class_f1"])
print("per-class support:", clean["per_class_support"])

# %% [3] Attack sanity checks on the PRIMARY test set (Carlini et al. red flags), same as NSL-KDD Stage 1
rows = []
for e in [0.0] + EPS_EVAL:
    if e == 0:
        rows.append({"eps": 0.0, "fgsm": clean["acc"], "pgd_ce": clean["acc"], "pgd_margin": clean["acc"], "worst": clean["acc"]}); continue
    A = make_attacks(model, MASK_ALL)
    c_f = predict(model, run_attack(A["fgsm"], Xte, yte, eps=tf.constant(e, tf.float32))) == yte
    c_c = predict(model, run_attack(A["pgd"], Xte, yte, eps=tf.constant(e, tf.float32), steps=20, restarts=3, margin=False)) == yte
    c_m = predict(model, run_attack(A["pgd"], Xte, yte, eps=tf.constant(e, tf.float32), steps=20, restarts=3, margin=True)) == yte
    w = c_f & c_c & c_m
    rows.append({"eps": e, "fgsm": float(c_f.mean()), "pgd_ce": float(c_c.mean()), "pgd_margin": float(c_m.mean()), "worst": float(w.mean())})
    print(f"eps={e:<5} FGSM={rows[-1]['fgsm']:.4f}  PGD-CE={rows[-1]['pgd_ce']:.4f}  PGD-margin={rows[-1]['pgd_margin']:.4f}  worst={rows[-1]['worst']:.4f}")

flags = []
for i in range(len(rows) - 1):
    if rows[i + 1]["worst"] > rows[i]["worst"] + 0.01:
        flags.append(f"worst-case accuracy rises from eps={rows[i]['eps']} to {rows[i+1]['eps']}")
    if rows[i + 1]["pgd_ce"] > rows[i]["fgsm"] + 0.02:
        flags.append(f"PGD-CE weaker than FGSM going from eps={rows[i]['eps']} to {rows[i+1]['eps']}")
if rows[-1]["worst"] > 0.20:
    flags.append(f"worst-case at eps={rows[-1]['eps']} (largest tested) still above 20% - may need a larger eps for full saturation")
print("RED FLAGS:", flags if flags else "none")

# %% [4] Robustness eval at the standard eps grid (primary test set)
primary_robust = robust_eval(model, Xte, yte, MASK_ALL, EPS_EVAL)
for e in EPS_EVAL:
    r = primary_robust[str(e)]
    print(f"eps={e}: worst={r['worst']:.4f} | per-class recall under worst-case attack: {r['worst_recall']}")

# %% [5] Blocked-split robustness check: evaluate the SAME model (trained only on the primary split's train data)
# on the blocked split's test set, to see whether performance differs meaningfully from the primary test set.
# This is NOT retraining - it's checking whether the primary split's row-level randomisation is inflating results
# relative to a split that respects (approximate) temporal/session locality.
db = np.load(f"{DATA}/cicids_v2_blocked.npz")
Xte_b, yte_b = db["Xte"].astype("float32"), db["yte"].astype("int32")
print(f"\nblocked test set: {Xte_b.shape}, classes present: {sorted(np.unique(yte_b).tolist())}")
clean_b = clean_eval(model, Xte_b, yte_b)
print("CLEAN on blocked test:", {k: clean_b[k] for k in ("acc", "macro_f1")}, "| per-class F1:", clean_b["per_class_f1"])
blocked_robust = robust_eval(model, Xte_b, yte_b, MASK_ALL, [0.05, 0.10])
for e in (0.05, 0.10):
    print(f"blocked test, eps={e}: worst={blocked_robust[str(e)]['worst']:.4f}  (primary test at same eps: {primary_robust[str(e)]['worst']:.4f})")
gap = abs(clean_b["acc"] - clean["acc"])
if gap > 0.05:
    print(f"NOTE: clean accuracy differs by {gap:.4f} between primary and blocked test sets - "
          f"this is evidence worth discussing in Limitations (possible temporal / near-duplicate leakage in the primary split).")
else:
    print(f"Clean accuracy differs by only {gap:.4f} between primary and blocked test sets - no strong evidence of leakage inflating the primary split's numbers.")

# %% [6] Save everything
out = {
    "params": model.count_params(),
    "clean_primary": clean,
    "attack_sanity_primary": rows,
    "flags": flags,
    "robust_primary": primary_robust,
    "clean_blocked": clean_b,
    "robust_blocked": blocked_robust,
    "clean_accuracy_gap_primary_vs_blocked": gap,
}
json.dump(jsonable(out), open(f"{RES}/stage2_undefended_seed{SEED}.json", "w"), indent=1)
print("\nsaved:", f"{RES}/stage2_undefended_seed{SEED}.json")
