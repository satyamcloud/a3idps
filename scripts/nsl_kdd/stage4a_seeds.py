# Stage 4a - NSL-KDD, 5 seeds: undefended, FGSM-AT, PGD-AT, teacher, and the A3IDPS pipelines up to round 5.
# Protocol frozen from Stages 2-3: same data split (seed 42), numeric-only main protocol, attack suite = worst case over
# FGSM / PGD-CE / PGD-margin (20 steps x 3 restarts). Hyper-parameters chosen on seed-0 VALIDATION data are reused for all seeds:
#   eps_train for FGSM-AT / PGD-AT  <- stage2_selected_seed0.json
#   generator eps_max               <- stage3_selection_seed0.json
# Only model / generator initialisation and batch order change with the seed. The test set is used only for final reporting.
# Resumable: models and per-seed results are cached in Drive. Run cell [0]-[5] once; then run cell [6] as often as you like.

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

SEEDS = [0, 1, 2, 3, 4]
ROUNDS = 5
sel2 = json.load(open(f"{RES}/stage2_selected_seed0.json"))
sel3 = json.load(open(f"{RES}/stage3_selection_seed0.json"))
EPS_AT  = {"fgsm": sel2["fgsm"], "pgd": sel2["pgd"]}
EPS_GEN = sel3["selected_eps"]
print("fixed from seed-0 validation:", EPS_AT, "| generator eps_max:", EPS_GEN)

EPS_NUM, EPS_ALL = [0.01, 0.02, 0.05, 0.10, 0.15], [0.05, 0.10]
KEY_ALL = {"undefended", "fgsm_at", "pgd_at", "kd_feedback_r3", "kd_feedback_r5"}   # tags that also get the all-features stress test
CLASS_WEIGHT_CAP, BATCH = 20.0, 512
KD_ALPHA, KD_T = 0.7, 3.0
GEN_STEPS, LAM_CLS, LAM_L1, LAM_GAN, KAPPA = 1200, 20.0, 0.5, 1.0, 1.0
FAST = True                                   # set False to fall back to eager steps if tf.function tracing errors occur
cw = np.minimum(compute_class_weight("balanced", classes=np.arange(NCLS), y=ytr), CLASS_WEIGHT_CAP).astype("float32")

def maybe_tf(fn): return tf.function(fn, reduce_retracing=True) if FAST else fn
def set_seed(s): random.seed(s); np.random.seed(s); tf.keras.utils.set_random_seed(s)
def jsonable(o):
    if isinstance(o, dict): return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, np.floating): return float(o)
    return o

# %% [1] Models and attacks (identical to Stages 2-3)
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

def predict(model, X): return np.argmax(model.predict(X, batch_size=4096, verbose=0), axis=1)

def clean_eval(model, X, y):
    p = predict(model, X)
    return {"acc": float(accuracy_score(y, p)),
            "macro_f1": float(f1_score(y, p, average="macro", labels=range(NCLS), zero_division=0)),
            "per_class_f1": dict(zip(CLASS_NAMES, f1_score(y, p, average=None, labels=range(NCLS), zero_division=0).tolist()))}

def robust_eval(model, X, y, mask, eps_list, steps=20, restarts=3):
    A = make_attacks(model, mask); out = {}
    for e in eps_list:
        et = tf.constant(e, tf.float32)
        c_f = predict(model, run_attack(A["fgsm"], X, y, eps=et)) == y
        c_c = predict(model, run_attack(A["pgd"], X, y, eps=et, steps=steps, restarts=restarts, margin=False)) == y
        c_m = predict(model, run_attack(A["pgd"], X, y, eps=et, steps=steps, restarts=restarts, margin=True)) == y
        w = c_f & c_c & c_m
        out[str(e)] = {"fgsm": float(c_f.mean()), "pgd_ce": float(c_c.mean()), "pgd_margin": float(c_m.mean()),
                       "worst": float(w.mean()), "worst_recall": {CLASS_NAMES[k]: float(w[y == k].mean()) for k in range(NCLS)}}
    return out

# %% [2] Classifier training: plain / distilled / adversarial training on the fly (at) / feedback (X_extra)
def fit_classifier(model, X, y, Xv, yv, X_extra=None, Xv_extra=None, zt=None, at=None,
                   lr=1e-3, max_epochs=40, patience=8, seed=0, tag=""):
    set_seed(seed)
    opt = keras.optimizers.Adam(lr); opt.build(model.trainable_variables)
    use_kd, cwt = zt is not None, tf.constant(cw)
    A = make_attacks(model, MASK_NUM) if at is not None else None
    et = tf.constant(at[1] if at else 0., tf.float32)
    def wce(z, yy):
        return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=yy, logits=z) * tf.gather(cwt, yy))

    @maybe_tf
    def step(xb, yb, ztb):
        with tf.GradientTape() as t:
            z = model(xb, training=True)
            hard = wce(z, yb)
            if use_kd:
                pt = tf.nn.softmax(ztb / KD_T); log_ps = tf.nn.log_softmax(z / KD_T)
                kl = tf.reduce_mean(tf.reduce_sum(pt * (tf.math.log(pt + 1e-8) - log_ps), axis=1))
                loss = KD_ALPHA * hard + (1. - KD_ALPHA) * KD_T * KD_T * kl
            else:
                loss = hard
        opt.apply_gradients(zip(t.gradient(loss, model.trainable_variables), model.trainable_variables))
        return loss

    rng, n = np.random.RandomState(seed), len(X)
    best, best_w, wait, cur_lr = 1e9, None, 0, lr
    for ep in range(max_epochs):
        t0 = time.time(); perm = rng.permutation(n)
        for i in range(0, n - BATCH + 1, BATCH):
            idx = perm[i:i+BATCH]; xb, yb = X[idx], y[idx]
            if X_extra is not None:
                xb, yb = np.concatenate([xb, X_extra[idx]]), np.concatenate([yb, y[idx]])
            ztb = zt[idx] if use_kd else np.zeros((len(idx), NCLS), "float32")
            xt, yt = tf.constant(xb), tf.constant(yb)
            if at is not None:
                xa = A["fgsm"](xt, yt, eps=et) if at[0] == "fgsm" else A["pgd"](xt, yt, eps=et, steps=10, restarts=1, margin=False)
                xt, yt = tf.concat([xt, xa], 0), tf.concat([yt, yt], 0)
            step(xt, yt, tf.constant(ztb))
        vl = float(wce(model(tf.constant(Xv), training=False), tf.constant(yv)))
        if Xv_extra is not None:
            vl += float(wce(model(tf.constant(Xv_extra), training=False), tf.constant(yv)))
        if at is not None:
            xv = run_attack(A["fgsm"], Xv, yv, eps=et)
            vl += float(wce(model(tf.constant(xv), training=False), tf.constant(yv)))
        if vl < best - 1e-4:
            best, best_w, wait = vl, model.get_weights(), 0
        else:
            wait += 1
            if wait % 4 == 0: cur_lr *= 0.5; opt.learning_rate = cur_lr
            if wait >= patience: break
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"[{tag}] epoch {ep+1} val_loss={vl:.4f} best={best:.4f} ({time.time()-t0:.1f}s)")
    model.set_weights(best_w)
    return model

# %% [3] AdvGAN generator (corrected objective, as in Stage 3)
def build_generator(d):
    inp = keras.Input(shape=(d,))
    x = layers.Dense(256, activation="relu")(inp); x = layers.Dense(256, activation="relu")(x)
    return keras.Model(inp, layers.Dense(d)(x))

def build_disc(d):
    inp = keras.Input(shape=(d,))
    x = layers.Dense(256)(inp); x = layers.LeakyReLU(0.2)(x)
    x = layers.Dense(256)(x);   x = layers.LeakyReLU(0.2)(x)
    return keras.Model(inp, layers.Dense(1)(x))

def apply_G(G, X, eps, bs=8192):
    E = eps * MASK_NUM
    return np.vstack([np.clip(X[i:i+bs] + E * np.tanh(G(tf.constant(X[i:i+bs]), training=False).numpy()), 0., 1.)
                      for i in range(0, len(X), bs)]).astype("float32")

def train_generator(clf, eps, seed, tag="", diag_path=None):
    set_seed(seed)
    G, Dn = build_generator(D), build_disc(D)
    g_opt = keras.optimizers.Adam(1e-3, beta_1=0.5); g_opt.build(G.trainable_variables)
    d_opt = keras.optimizers.Adam(1e-4, beta_1=0.5); d_opt.build(Dn.trainable_variables)
    E = tf.constant(eps * MASK_NUM)
    bce = lambda labels, logits: tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits))
    def gen(x): return tf.clip_by_value(x + E * tf.tanh(G(x, training=True)), 0., 1.)

    @maybe_tf
    def step(x, y):
        with tf.GradientTape() as td:
            xa = gen(x)
            ld = bce(tf.ones([tf.shape(x)[0], 1]), Dn(x, training=True)) + bce(tf.zeros([tf.shape(x)[0], 1]), Dn(xa, training=True))
        d_opt.apply_gradients(zip(td.gradient(ld, Dn.trainable_variables), Dn.trainable_variables))
        with tf.GradientTape() as tg:
            xa = gen(x); z = clf(xa, training=False); oh = tf.one_hot(y, NCLS)
            margin = tf.reduce_sum(z * oh, 1) - tf.reduce_max(z - 1e9 * oh, 1)
            lg = (LAM_CLS * tf.reduce_mean(tf.nn.relu(margin + KAPPA))
                  + LAM_GAN * bce(tf.ones([tf.shape(x)[0], 1]), Dn(xa, training=False))
                  + LAM_L1 * tf.reduce_mean(tf.abs(xa - x)))
        g_opt.apply_gradients(zip(tg.gradient(lg, G.trainable_variables), G.trainable_variables))
        return lg

    rng = np.random.RandomState(seed)
    for s in range(GEN_STEPS):
        idx = rng.randint(0, len(Xtr), BATCH)
        step(tf.constant(Xtr[idx]), tf.constant(ytr[idx]))
    Xa = apply_G(G, Xva, eps)
    a0, a1 = float((predict(clf, Xva) == yva).mean()), float((predict(clf, Xa) == yva).mean())
    print(f"[{tag}] generator sanity (validation): acc clean={a0:.4f} -> under G={a1:.4f}")
    if diag_path: json.dump({"acc_clean": a0, "acc_under_G": a1, "mean_abs_delta": float(np.abs(Xa - Xva).mean())}, open(diag_path, "w"))
    return G

# %% [4] Teacher, baselines, pipelines (per seed)
def get_teacher(seed):
    path = f"{MOD}/s4_teacher_seed{seed}.keras"
    if os.path.exists(path): return keras.models.load_model(path, compile=False)
    t = build_model(D, widths=(512, 256, 128))
    fit_classifier(t, Xtr, ytr, Xva, yva, seed=seed, tag=f"teacher s{seed}"); t.save(path); return t

def get_baseline(distil, seed):
    path = f"{MOD}/s4_base_{'kd' if distil else 'nokd'}_seed{seed}.keras"
    if os.path.exists(path): return keras.models.load_model(path, compile=False)
    zt = get_teacher(seed).predict(Xtr, batch_size=4096, verbose=0).astype("float32") if distil else None
    m = build_model(D)
    fit_classifier(m, Xtr, ytr, Xva, yva, zt=zt, seed=seed, tag=f"base_{'kd' if distil else 'nokd'} s{seed}")
    m.save(path); return m

def run_pipeline(distil, mode, eps, seed, rounds=ROUNDS):
    ptag = f"{'kd' if distil else 'nokd'}_{mode}"
    prev = get_baseline(distil, seed); clfs, G_fixed = {0: prev}, None
    for r in range(1, rounds + 1):
        cpath = f"{MOD}/s4_{ptag}_r{r}_seed{seed}.keras"; gpath = f"{MOD}/s4_gen_{ptag}_r{r}_seed{seed}.keras"
        if os.path.exists(cpath):
            clfs[r] = prev = keras.models.load_model(cpath, compile=False)
            if mode == "fixed_gen" and r == 1: G_fixed = keras.models.load_model(gpath, compile=False)
            continue
        if mode == "feedback" or r == 1:
            G = train_generator(prev, eps, seed * 100 + r, tag=f"{ptag} s{seed} r{r}", diag_path=gpath.replace(".keras", ".json"))
            G.save(gpath)
            if r == 1: G_fixed = G
        else:
            G = G_fixed
        Xa_tr, Xa_va = apply_G(G, Xtr, eps), apply_G(G, Xva, eps)
        new = build_model(D); new.set_weights(prev.get_weights())
        fit_classifier(new, Xtr, ytr, Xva, yva, X_extra=Xa_tr, Xv_extra=Xa_va, lr=5e-4, max_epochs=30,
                       seed=seed * 100 + r, tag=f"{ptag} s{seed} r{r}")
        new.save(cpath); clfs[r] = prev = new
    return clfs

# %% [5] Run every seed (resumable) and evaluate on TEST
def evaluate_tag(res, tag, m, path):
    have = res.get(tag, {})
    if "numeric_only" in have and (tag not in KEY_ALL or "all_features" in have): return
    r = {"clean": clean_eval(m, Xte, yte), "numeric_only": robust_eval(m, Xte, yte, MASK_NUM, EPS_NUM)}
    if tag in KEY_ALL: r["all_features"] = robust_eval(m, Xte, yte, MASK_ALL, EPS_ALL)
    res[tag] = r; json.dump(jsonable(res), open(path, "w"), indent=1)
    print(f"  evaluated {tag}: clean={r['clean']['acc']:.4f}, worst@0.05={r['numeric_only']['0.05']['worst']:.4f}, worst@0.10={r['numeric_only']['0.1']['worst']:.4f}")

def run_seed(seed):
    out = f"{RES}/stage4a_seed{seed}.json"
    res = json.load(open(out)) if os.path.exists(out) else {}
    print(f"\n########## seed {seed} ##########")
    for tag, at in [("undefended", None), ("fgsm_at", ("fgsm", EPS_AT["fgsm"])), ("pgd_at", ("pgd", EPS_AT["pgd"]))]:
        path = f"{MOD}/s4_{tag}_seed{seed}.keras"
        if os.path.exists(path): m = keras.models.load_model(path, compile=False)
        else:
            m = build_model(D); fit_classifier(m, Xtr, ytr, Xva, yva, at=at, seed=seed, tag=f"{tag} s{seed}"); m.save(path)
        evaluate_tag(res, tag, m, out)
    for distil, mode in [(True, "feedback"), (True, "fixed_gen"), (False, "feedback")]:
        clfs = run_pipeline(distil, mode, EPS_GEN, seed)
        base_tag = "kd_base" if distil else "nokd_base"
        if mode == "feedback": evaluate_tag(res, base_tag, clfs[0], out)
        for r in range(1, ROUNDS + 1):
            evaluate_tag(res, f"{'kd' if distil else 'nokd'}_{mode}_r{r}", clfs[r], out)
    keras.backend.clear_session()

for s in SEEDS:
    run_seed(s)

# %% [6] Aggregate (mean +- std over the seeds finished so far) and paired differences
def collect():
    return {s: json.load(open(f"{RES}/stage4a_seed{s}.json")) for s in SEEDS if os.path.exists(f"{RES}/stage4a_seed{s}.json")}
ORDER = (["undefended", "fgsm_at", "pgd_at", "kd_base"] + [f"kd_feedback_r{r}" for r in range(1, ROUNDS + 1)]
         + [f"kd_fixed_gen_r{r}" for r in range(1, ROUNDS + 1)] + ["nokd_base"] + [f"nokd_feedback_r{r}" for r in range(1, ROUNDS + 1)])
def ms(v):
    a = np.array(v); return f"{100*a.mean():5.1f}+-{100*(a.std(ddof=1) if len(a) > 1 else 0):4.1f}"

allres = collect(); n_seeds = len(allres)
print(f"seeds available: {sorted(allres)}")
for prot, eps_list in (("numeric_only", EPS_NUM), ("all_features", EPS_ALL)):
    print(f"\n=== TEST worst-case accuracy (%), {prot}, mean+-std over {n_seeds} seeds ===")
    print(f"{'model':<20}{'clean':>12}{'macroF1':>12}" + "".join(f"{('e='+str(e)):>12}" for e in eps_list))
    for tag in ORDER:
        rows = [allres[s][tag] for s in allres if tag in allres[s] and prot in allres[s][tag]]
        if not rows: continue
        print(f"{tag:<20}{ms([r['clean']['acc'] for r in rows]):>12}{ms([r['clean']['macro_f1'] for r in rows]):>12}"
              + "".join(f"{ms([r[prot][str(e)]['worst'] for r in rows]):>12}" for e in eps_list))

print("\n=== Paired differences in worst-case accuracy, numeric_only (percentage points, mean+-std over seeds) ===")
PAIRS = [("kd_feedback_r3", "kd_fixed_gen_r3"), ("kd_feedback_r5", "kd_fixed_gen_r5"),
         ("kd_feedback_r3", "nokd_feedback_r3"), ("kd_feedback_r5", "nokd_feedback_r5"),
         ("kd_feedback_r5", "undefended"), ("kd_feedback_r5", "pgd_at")]
for a, b in PAIRS:
    for e in (0.05, 0.10):
        diffs = [100 * (allres[s][a]["numeric_only"][str(e)]["worst"] - allres[s][b]["numeric_only"][str(e)]["worst"])
                 for s in allres if a in allres[s] and b in allres[s]]
        if diffs:
            print(f"{a} - {b} @ eps={e}: {np.mean(diffs):+6.1f} +- {(np.std(diffs, ddof=1) if len(diffs) > 1 else 0):4.1f}  (positive in {sum(x > 0 for x in diffs)}/{len(diffs)} seeds)")
