# Stage 2 - NSL-KDD: shared logits architecture, undefended / FGSM-AT / PGD-AT with an epsilon_train sweep,
# validation-based selection (rule fixed in advance), and full test evaluation with worst-case attacks.
# Paste into Colab cell by cell. Resumable: finished models are loaded from Drive, not retrained.
# Protocol (frozen):
#   - main protocol perturbs the 38 numeric features only; all 122 features is a stress test
#   - epsilon_train in {0.02, 0.05, 0.10}, chosen per method on VALIDATION data only:
#       highest mean validation worst-case accuracy over eps in {0.01, 0.02, 0.05, 0.10},
#       among candidates whose clean validation accuracy is within 5 pp of the undefended model (ties -> smaller eps)
#   - evaluation eps {0.01, 0.02, 0.05, 0.10, 0.15}; 0.15 lies outside the training grid

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
DATA = f"{BASE}/data_processed/nslkdd_v2"
MOD  = f"{BASE}/models/nsl_v2"
RES  = f"{BASE}/results/nsl_v2"
os.makedirs(MOD, exist_ok=True); os.makedirs(RES, exist_ok=True)

d = np.load(f"{DATA}/nsl_v2_arrays.npz")
Xtr, Xva, Xte = [d[k].astype("float32") for k in ("Xtr", "Xva", "Xte")]
ytr, yva, yte = [d[k].astype("int32") for k in ("ytr", "yva", "yte")]
names = json.load(open(f"{DATA}/feature_names.json"))
D, NCLS = Xtr.shape[1], 5
CLASS_NAMES = ["Normal", "DoS", "Probe", "R2L", "U2R"]
MASK_NUM = np.array([0. if n.startswith(("protocol_type_", "service_", "flag_")) else 1. for n in names], "float32")
MASK_ALL = np.ones(D, "float32")
assert int(MASK_NUM.sum()) == 38

SEED = 0
EPS_EVAL   = [0.01, 0.02, 0.05, 0.10, 0.15]
EPS_SELECT = EPS_EVAL[:4]
EPS_TRAIN  = [0.02, 0.05, 0.10]
CLASS_WEIGHT_CAP, VAL_CLEAN_TOL, BATCH, MAX_EPOCHS = 20.0, 0.05, 512, 40
cw = np.minimum(compute_class_weight("balanced", classes=np.arange(NCLS), y=ytr), CLASS_WEIGHT_CAP).astype("float32")

def set_seed(s):
    random.seed(s); np.random.seed(s); tf.keras.utils.set_random_seed(s)

def jsonable(o):
    if isinstance(o, dict): return {k: jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, np.floating): return float(o)
    return o

# %% [1] Model (outputs LOGITS; softmax is folded into the loss) and attacks
def build_model(d):
    inp = keras.Input(shape=(d,))
    x = layers.Dense(256)(inp); x = layers.BatchNormalization()(x); x = layers.Activation("relu")(x)
    x = layers.Dense(128)(x);   x = layers.Dropout(0.3)(x);          x = layers.Activation("relu")(x)
    x = layers.Dense(64)(x);    x = layers.Dropout(0.3)(x);          x = layers.Activation("relu")(x)
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
            t.watch(x0)
            s = tf.reduce_sum(loss_fn(model(x0, training=False), y, False))
        g = t.gradient(s, x0)
        return tf.clip_by_value(x0 + eps * M * tf.sign(g), 0., 1.)

    @tf.function
    def pgd(x0, y, eps, steps, restarts, margin):
        E = eps * M
        best_x = x0
        best_l = tf.fill(tf.shape(y), -1e30)
        for r in range(restarts):
            if r == 0:
                x = x0
            else:
                x = tf.clip_by_value(x0 + tf.random.uniform(tf.shape(x0), -1., 1.) * E, 0., 1.)
            for s_ in range(steps):
                alpha = (1.0 if s_ < steps // 2 else 0.25) * E / 4.0
                with tf.GradientTape() as t:
                    t.watch(x)
                    s = tf.reduce_sum(loss_fn(model(x, training=False), y, margin))
                g = t.gradient(s, x)
                x = tf.clip_by_value(tf.clip_by_value(x + alpha * tf.sign(g), x0 - E, x0 + E), 0., 1.)
                l_new = loss_fn(model(x, training=False), y, margin)
                upd = l_new > best_l
                best_x = tf.where(upd[:, None], x, best_x)
                best_l = tf.where(upd, l_new, best_l)
        return best_x
    return {"fgsm": fgsm, "pgd": pgd}

def run_attack(fn, X, y, bs=4096, **kw):
    return np.vstack([fn(tf.constant(X[i:i+bs]), tf.constant(y[i:i+bs]), **kw).numpy()
                      for i in range(0, len(X), bs)])

def predict(model, X):
    return np.argmax(model.predict(X, batch_size=4096, verbose=0), axis=1)

def clean_eval(model, X, y):
    p = predict(model, X)
    return {"acc": float(accuracy_score(y, p)),
            "macro_f1": float(f1_score(y, p, average="macro", labels=range(NCLS), zero_division=0)),
            "per_class_f1": dict(zip(CLASS_NAMES, f1_score(y, p, average=None, labels=range(NCLS), zero_division=0).tolist()))}

def robust_eval(model, X, y, mask, eps_list, steps, restarts):
    A = make_attacks(model, mask); out = {}
    for e in eps_list:
        et = tf.constant(e, tf.float32)
        c_f = predict(model, run_attack(A["fgsm"], X, y, eps=et)) == y
        c_c = predict(model, run_attack(A["pgd"], X, y, eps=et, steps=steps, restarts=restarts, margin=False)) == y
        c_m = predict(model, run_attack(A["pgd"], X, y, eps=et, steps=steps, restarts=restarts, margin=True)) == y
        w = c_f & c_c & c_m
        out[str(e)] = {"fgsm": float(c_f.mean()), "pgd_ce": float(c_c.mean()), "pgd_margin": float(c_m.mean()),
                       "worst": float(w.mean()),
                       "worst_recall": {CLASS_NAMES[k]: float(w[y == k].mean()) for k in range(NCLS)}}
    return out

# %% [2] Training: identical recipe for every model (undefended, FGSM-AT, PGD-AT)
def train_model(kind, eps_train, seed, mask=MASK_NUM):
    tag = kind if kind == "none" else f"{kind}_eps{eps_train}"
    path = f"{MOD}/s2_{tag}_seed{seed}.keras"
    if os.path.exists(path):
        print("loading", path); return keras.models.load_model(path, compile=False), tag
    set_seed(seed)
    model = build_model(D)
    opt = keras.optimizers.Adam(1e-3)
    A = make_attacks(model, mask) if kind != "none" else None
    cwt = tf.constant(cw)
    wce = lambda z, y: tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=z) * tf.gather(cwt, y))
    et = tf.constant(eps_train or 0., tf.float32)
    best, best_w, wait, lr = 1e9, None, 0, 1e-3
    rng = np.random.RandomState(seed)
    for ep in range(MAX_EPOCHS):
        t0 = time.time(); perm = rng.permutation(len(Xtr))
        for i in range(0, len(perm), BATCH):
            idx = perm[i:i+BATCH]
            xb, yb = tf.constant(Xtr[idx]), tf.constant(ytr[idx])
            if kind == "fgsm":
                xa = A["fgsm"](xb, yb, eps=et)
            elif kind == "pgd":
                xa = A["pgd"](xb, yb, eps=et, steps=10, restarts=1, margin=False)
            if kind != "none":                      # clean + adversarial version of the same batch
                xb, yb = tf.concat([xb, xa], 0), tf.concat([yb, yb], 0)
            with tf.GradientTape() as t:
                loss = wce(model(xb, training=True), yb)
            opt.apply_gradients(zip(t.gradient(loss, model.trainable_variables), model.trainable_variables))
        vl = float(wce(model(tf.constant(Xva), training=False), tf.constant(yva)))
        if kind != "none":
            xv = run_attack(A["fgsm"], Xva, yva, eps=et)
            vl += float(wce(model(tf.constant(xv), training=False), tf.constant(yva)))
        if vl < best - 1e-4:
            best, best_w, wait = vl, model.get_weights(), 0
        else:
            wait += 1
            if wait % 4 == 0:
                lr *= 0.5; opt.learning_rate = lr
            if wait >= 8:
                break
        print(f"[{tag}] epoch {ep+1} val_loss={vl:.4f} best={best:.4f} ({time.time()-t0:.1f}s)")
    model.set_weights(best_w); model.save(path)
    return model, tag

# %% [3] Train everything (seed 0) and score on VALIDATION data
SPEC = [("none", None)] + [("fgsm", e) for e in EPS_TRAIN] + [("pgd", e) for e in EPS_TRAIN]
models, val_res = {}, {}
for kind, e in SPEC:
    model, tag = train_model(kind, e, SEED)
    models[tag] = model
    vc = clean_eval(model, Xva, yva)
    vr = robust_eval(model, Xva, yva, MASK_NUM, EPS_SELECT, steps=10, restarts=1)
    val_res[tag] = {"val_clean": vc, "val_rob": vr,
                    "val_rob_mean": float(np.mean([vr[str(x)]["worst"] for x in EPS_SELECT]))}
    print(f"VAL {tag:<12} clean acc={vc['acc']:.4f}  mean worst-case robust acc={val_res[tag]['val_rob_mean']:.4f}")
    json.dump(jsonable(val_res), open(f"{RES}/stage2_val_seed{SEED}.json", "w"), indent=1)

# %% [4] Select epsilon_train per method with the pre-stated rule (validation data only)
und = val_res["none"]["val_clean"]["acc"]
selected = {}
for method in ("fgsm", "pgd"):
    cands = [(e, val_res[f"{method}_eps{e}"]) for e in EPS_TRAIN]
    elig = [(e, v) for e, v in cands if und - v["val_clean"]["acc"] <= VAL_CLEAN_TOL]
    pool = elig if elig else cands
    if not elig: print(f"WARNING: no {method} candidate within {VAL_CLEAN_TOL:.0%} clean-accuracy tolerance")
    e_best = max(pool, key=lambda ev: (round(ev[1]["val_rob_mean"], 6), -ev[0]))[0]
    selected[method] = e_best
    for e, v in cands:
        print(f"{method:<5} eps_train={e:<5} val clean={v['val_clean']['acc']:.4f} mean robust={v['val_rob_mean']:.4f} "
              f"eligible={(e, v) in elig} {'<== selected' if e == e_best else ''}")
json.dump(selected, open(f"{RES}/stage2_selected_seed{SEED}.json", "w"))

# %% [5] TEST evaluation (worst-case over FGSM, PGD-CE, PGD-margin; 20 steps x 3 restarts)
test_res = {}
for tag, model in models.items():
    test_res[tag] = {"clean": clean_eval(model, Xte, yte),
                     "numeric_only": robust_eval(model, Xte, yte, MASK_NUM, EPS_EVAL, steps=20, restarts=3),
                     "all_features": robust_eval(model, Xte, yte, MASK_ALL, EPS_EVAL, steps=20, restarts=3)}
    json.dump(jsonable({"selected": selected, "test": test_res}), open(f"{RES}/stage2_test_seed{SEED}.json", "w"), indent=1)

show = ["none", f"fgsm_eps{selected['fgsm']}", f"pgd_eps{selected['pgd']}"]
for prot in ("numeric_only", "all_features"):
    print(f"\n=== TEST, worst-case robust accuracy, {prot} (selected models) ===")
    print(f"{'model':<14}{'clean':>8}{'macroF1':>9}" + "".join(f"{('e='+str(e)):>9}" for e in EPS_EVAL))
    for tag in show:
        r, c = test_res[tag][prot], test_res[tag]["clean"]
        print(f"{tag:<14}{c['acc']:>8.4f}{c['macro_f1']:>9.4f}" + "".join(f"{r[str(e)]['worst']:>9.4f}" for e in EPS_EVAL))
print("\nAppendix grid (all eps_train) is in", f"{RES}/stage2_test_seed{SEED}.json")
