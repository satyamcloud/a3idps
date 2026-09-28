# CICIDS2017 v2 - Stage 2b: FGSM-AT / PGD-AT with an eps_train sweep, validation-based selection (same rule as
# NSL-KDD), and worst-case test evaluation. Self-contained (re-defines everything needed in case of a fresh session).
# Carries forward two fixes learned from the undefended baseline: SMOTE on the training pool, and lr=3e-4.
# A collapse-guard checks per-class F1 after every training run before accepting it.

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
DATA = f"{BASE}/data_processed/cicids_v2"; MOD = f"{BASE}/models/cicids_v2"; RES = f"{BASE}/results/cicids_v2"
os.makedirs(MOD, exist_ok=True); os.makedirs(RES, exist_ok=True)

d = np.load(f"{DATA}/cicids_v2_primary.npz")
Xtr, Xva, Xte = [d[k].astype("float32") for k in ("Xtr", "Xva", "Xte")]
ytr, yva, yte = [d[k].astype("int32") for k in ("ytr", "yva", "yte")]
D, NCLS = Xtr.shape[1], 5
CLASS_NAMES = ["BENIGN", "Brute Force", "DoS/DDoS", "PortScan", "Other"]
MASK_ALL = np.ones(D, "float32")

SEED = 0
EPS_EVAL, EPS_SELECT = [0.01, 0.02, 0.05, 0.10, 0.15], [0.01, 0.02, 0.05, 0.10]
EPS_TRAIN = [0.02, 0.05, 0.10]
CLASS_WEIGHT_CAP, VAL_CLEAN_TOL, BATCH, MAX_EPOCHS, LR = 20.0, 0.05, 2048, 60, 3e-4
FAST = True
def maybe_tf(fn): return tf.function(fn, reduce_retracing=True) if FAST else fn
def set_seed(s): random.seed(s); np.random.seed(s); tf.keras.utils.set_random_seed(s)
def jsonable(o):
    if isinstance(o, dict): return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, np.floating): return float(o)
    return o

# %% [1] SMOTE the training pool ONCE (same target as the corrected undefended baseline)
from imblearn.over_sampling import SMOTE
SMOTE_TARGET = 50_000
counts = np.bincount(ytr)
strategy = {c: SMOTE_TARGET for c in range(len(counts)) if counts[c] < SMOTE_TARGET}
print("SMOTE strategy:", strategy)
Xtr_sm, ytr_sm = SMOTE(random_state=0, k_neighbors=5, sampling_strategy=strategy).fit_resample(Xtr, ytr)
Xtr_sm = np.clip(Xtr_sm, 0.0, 1.0).astype("float32"); ytr_sm = ytr_sm.astype("int32")
print("post-SMOTE train class counts:", np.bincount(ytr_sm))
cw = np.minimum(compute_class_weight("balanced", classes=np.arange(NCLS), y=ytr_sm), CLASS_WEIGHT_CAP).astype("float32")
print("class weights:", cw.tolist())

# %% [2] Model, attacks, eval (same structure as NSL-KDD / CICIDS Stage 2)
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
    p = predict(model, X); present = sorted(np.unique(y).tolist())
    return {"acc": float(accuracy_score(y, p)),
            "macro_f1": float(f1_score(y, p, average="macro", labels=present, zero_division=0)),
            "per_class_f1": {CLASS_NAMES[c]: float(f1_score(y == c, p == c, zero_division=0)) for c in present}}

def robust_eval(model, X, y, mask, eps_list, steps=20, restarts=3):
    A = make_attacks(model, mask); out = {}; present = sorted(np.unique(y).tolist())
    for e in eps_list:
        et = tf.constant(e, tf.float32)
        c_f = predict(model, run_attack(A["fgsm"], X, y, eps=et)) == y
        c_c = predict(model, run_attack(A["pgd"], X, y, eps=et, steps=steps, restarts=restarts, margin=False)) == y
        c_m = predict(model, run_attack(A["pgd"], X, y, eps=et, steps=steps, restarts=restarts, margin=True)) == y
        w = c_f & c_c & c_m
        out[str(e)] = {"fgsm": float(c_f.mean()), "pgd_ce": float(c_c.mean()), "pgd_margin": float(c_m.mean()),
                       "worst": float(w.mean()), "worst_recall": {CLASS_NAMES[c]: float(w[y == c].mean()) for c in present}}
    return out

def collapse_check(clean, tag):
    bad = [c for c in ("Brute Force", "DoS/DDoS", "PortScan") if clean["per_class_f1"].get(c, 0.0) == 0.0]
    if bad:
        raise RuntimeError(f"[{tag}] COLLAPSED: F1=0 for {bad} - stop and inspect before continuing")
    print(f"[{tag}] collapse check passed: {clean['per_class_f1']}")

# %% [3] Training loop: undefended reload, or FGSM-AT / PGD-AT with adversarial examples generated on the fly
def fit_classifier(model, X, y, Xv, yv, at=None, lr=LR, max_epochs=MAX_EPOCHS, patience=8, seed=SEED, tag=""):
    set_seed(seed)
    opt = keras.optimizers.Adam(lr); opt.build(model.trainable_variables)
    A = make_attacks(model, MASK_ALL) if at is not None else None
    et = tf.constant(at[1] if at else 0., tf.float32)
    cwt = tf.constant(cw)
    def wce(z, yy): return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=yy, logits=z) * tf.gather(cwt, yy))

    @maybe_tf
    def step(xb, yb):
        with tf.GradientTape() as t:
            loss = wce(model(xb, training=True), yb)
        opt.apply_gradients(zip(t.gradient(loss, model.trainable_variables), model.trainable_variables))
        return loss

    rng, n = np.random.RandomState(seed), len(X)
    best, best_w, wait, cur_lr = 1e9, None, 0, lr
    for ep in range(max_epochs):
        t0 = time.time(); perm = rng.permutation(n)
        for i in range(0, n - BATCH + 1, BATCH):
            idx = perm[i:i+BATCH]; xt, yt = tf.constant(X[idx]), tf.constant(y[idx])
            if at is not None:
                xa = A["fgsm"](xt, yt, eps=et) if at[0] == "fgsm" else A["pgd"](xt, yt, eps=et, steps=10, restarts=1, margin=False)
                xt, yt = tf.concat([xt, xa], 0), tf.concat([yt, yt], 0)
            step(xt, yt)
        vl = float(wce(model(tf.constant(Xv), training=False), tf.constant(yv)))
        if at is not None:
            xv = run_attack(A["fgsm"], Xv, yv, eps=et)
            vl += float(wce(model(tf.constant(xv), training=False), tf.constant(yv)))
        if vl < best - 1e-4: best, best_w, wait = vl, model.get_weights(), 0
        else:
            wait += 1
            if wait % 4 == 0: cur_lr *= 0.5; opt.learning_rate = cur_lr
            if wait >= patience: break
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"[{tag}] epoch {ep+1} val_loss={vl:.4f} best={best:.4f} ({time.time()-t0:.1f}s)")
    model.set_weights(best_w)
    return model

# %% [4] Train / load undefended (already done in Stage 2) + FGSM-AT / PGD-AT sweep, score on VALIDATION
und = keras.models.load_model(f"{MOD}/undefended_smote_seed{SEED}.keras", compile=False)
und_val = clean_eval(und, Xva, yva); collapse_check(und_val, "undefended(val)")
print("undefended val clean acc:", und_val["acc"])

SPEC = [("fgsm", e) for e in EPS_TRAIN] + [("pgd", e) for e in EPS_TRAIN]
models, val_res = {}, {}
for kind, e in SPEC:
    tag = f"{kind}_eps{e}"
    path = f"{MOD}/{tag}_seed{SEED}.keras"
    if os.path.exists(path):
        m = keras.models.load_model(path, compile=False)
    else:
        m = build_model(D)
        fit_classifier(m, Xtr_sm, ytr_sm, Xva, yva, at=(kind, e), tag=tag)
        m.save(path)
    models[tag] = m
    vc = clean_eval(m, Xva, yva); collapse_check(vc, f"{tag}(val)")
    vr = robust_eval(m, Xva, yva, MASK_ALL, EPS_SELECT, steps=10, restarts=1)
    val_res[tag] = {"val_clean": vc, "val_rob": vr, "val_rob_mean": float(np.mean([vr[str(x)]["worst"] for x in EPS_SELECT]))}
    print(f"VAL {tag:<12} clean acc={vc['acc']:.4f}  mean worst-case robust acc={val_res[tag]['val_rob_mean']:.4f}")
    json.dump(jsonable(val_res), open(f"{RES}/stage2b_val_seed{SEED}.json", "w"), indent=1)

# %% [5] Select eps_train per method (same rule as NSL-KDD, validation only)
selected = {}
for method in ("fgsm", "pgd"):
    cands = [(e, val_res[f"{method}_eps{e}"]) for e in EPS_TRAIN]
    elig = [(e, v) for e, v in cands if und_val["acc"] - v["val_clean"]["acc"] <= VAL_CLEAN_TOL]
    pool = elig if elig else cands
    if not elig: print(f"WARNING: no {method} candidate within {VAL_CLEAN_TOL:.0%} clean-accuracy tolerance")
    e_best = max(pool, key=lambda ev: (round(ev[1]["val_rob_mean"], 6), -ev[0]))[0]
    selected[method] = e_best
    for e, v in cands:
        print(f"{method:<5} eps_train={e:<5} val clean={v['val_clean']['acc']:.4f} mean robust={v['val_rob_mean']:.4f} "
              f"eligible={(e, v) in elig} {'<== selected' if e == e_best else ''}")
json.dump(selected, open(f"{RES}/stage2b_selected_seed{SEED}.json", "w"))

# %% [6] TEST evaluation (undefended + the two selected AT models)
test_res = {"undefended": {"clean": clean_eval(und, Xte, yte), "robust": robust_eval(und, Xte, yte, MASK_ALL, EPS_EVAL)}}
for method in ("fgsm", "pgd"):
    tag = f"{method}_eps{selected[method]}"; m = models[tag]
    tc = clean_eval(m, Xte, yte); collapse_check(tc, f"{tag}(test)")
    test_res[tag] = {"clean": tc, "robust": robust_eval(m, Xte, yte, MASK_ALL, EPS_EVAL)}
json.dump(jsonable({"selected": selected, "test": test_res}), open(f"{RES}/stage2b_test_seed{SEED}.json", "w"), indent=1)

print(f"\n=== TEST worst-case robust accuracy (selected models) ===")
print(f"{'model':<16}{'clean':>8}{'macroF1':>9}" + "".join(f"{('e='+str(e)):>9}" for e in EPS_EVAL))
for name, key in [("undefended", "undefended"), (f"fgsm eps={selected['fgsm']}", f"fgsm_eps{selected['fgsm']}"),
                  (f"pgd eps={selected['pgd']}", f"pgd_eps{selected['pgd']}")]:
    r, c = test_res[key]["robust"], test_res[key]["clean"]
    print(f"{name:<16}{c['acc']:>8.4f}{c['macro_f1']:>9.4f}" + "".join(f"{r[str(e)]['worst']:>9.4f}" for e in EPS_EVAL))
print("\nsaved:", f"{RES}/stage2b_test_seed{SEED}.json")
