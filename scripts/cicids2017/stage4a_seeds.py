# CICIDS2017 v2 - Stage 4a: 5-seed robustness study (seeds 0-4), mirroring nsl_stage4a_seeds.py exactly in
# protocol: the SAME train/val/test split and the SAME SMOTE'd training pool (random_state=0, fixed) are used
# for every seed - only model/generator weight initialization and minibatch order vary by seed. Reuses the
# eps_train (fgsm/pgd) selected on seed-0 validation in Stage 2b and the eps_max selected on seed-0 validation
# in Stage 3 - selection is NOT repeated per seed (same rule as NSL-KDD). Trains, per seed: undefended, fgsm_at,
# pgd_at, teacher, kd_feedback (r0-5), kd_fixed_gen (r0-5), nokd_feedback (r0-5). Collapse-guard after every
# training stage. Fully resumable (every artifact checks os.path.exists before retraining/regenerating).
#
# fit_classifier_at and fit_classifier_kd below are copied verbatim (only SEED->seed made an explicit arg) from
# cicids_stage2b_at_sweep.py and cicids_stage3_a3idps.py respectively, and are NOT merged into one function -
# merging AT and KD training paths would be new, untested logic, and neither Stage 2b nor Stage 3 ever combined
# them, so keeping two faithful copies is the safer choice given we cannot afford to introduce a new bug here.

# %% [0] Setup
from google.colab import drive
drive.mount('/content/drive')
import os, json, random, time
import numpy as np, tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE

BASE = "/content/drive/MyDrive/A3IDPS_COLAB"
DATA = f"{BASE}/data_processed/cicids_v2"; MOD = f"{BASE}/models/cicids_v2"; RES = f"{BASE}/results/cicids_v2"
os.makedirs(MOD, exist_ok=True); os.makedirs(RES, exist_ok=True)

d = np.load(f"{DATA}/cicids_v2_primary.npz")
Xtr, Xva, Xte = [d[k].astype("float32") for k in ("Xtr", "Xva", "Xte")]
ytr, yva, yte = [d[k].astype("int32") for k in ("ytr", "yva", "yte")]
D, NCLS = Xtr.shape[1], 5
CLASS_NAMES = ["BENIGN", "Brute Force", "DoS/DDoS", "PortScan", "Other"]
MASK = np.ones(D, "float32")   # all 78 CICIDS features are numeric - no categorical columns to freeze

SEEDS = [0, 1, 2, 3, 4]
ROUNDS = 5
EPS_EVAL, EPS_SELECT = [0.01, 0.02, 0.05, 0.10, 0.15], [0.01, 0.02, 0.05, 0.10]
CLASS_WEIGHT_CAP, VAL_CLEAN_TOL, BATCH = 20.0, 0.05, 2048
BASE_LR, FEEDBACK_LR = 3e-4, 1.5e-4
KD_ALPHA, KD_T = 0.7, 3.0
GEN_STEPS, LAM_CLS, LAM_L1, LAM_GAN, KAPPA = 1200, 20.0, 0.5, 1.0, 1.0
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

# %% [1] SMOTE the training pool ONCE - identical across all seeds (random_state=0, fixed), so every seed trains
# on exactly the same data; only weight init / batch order differ by seed. This matches the NSL-KDD 5-seed
# protocol, where the data split was fixed at seed 42 for every seed and only model randomness varied.
SMOTE_TARGET = 50_000
counts = np.bincount(ytr)
strategy = {c: SMOTE_TARGET for c in range(len(counts)) if counts[c] < SMOTE_TARGET}
Xtr_sm, ytr_sm = SMOTE(random_state=0, k_neighbors=5, sampling_strategy=strategy).fit_resample(Xtr, ytr)
Xtr_sm = np.clip(Xtr_sm, 0.0, 1.0).astype("float32"); ytr_sm = ytr_sm.astype("int32")
print("post-SMOTE train class counts (shared across all 5 seeds):", np.bincount(ytr_sm))
cw = np.minimum(compute_class_weight("balanced", classes=np.arange(NCLS), y=ytr_sm), CLASS_WEIGHT_CAP).astype("float32")

# %% [2] Reuse seed-0-selected hyperparameters (frozen; NOT re-selected per seed)
sel_at = json.load(open(f"{RES}/stage2b_selected_seed0.json"))          # {"fgsm": e, "pgd": e}
sel_gen = json.load(open(f"{RES}/stage3_selection_seed0.json"))        # {"selected_eps": e, ...}
EPS_FGSM, EPS_PGD, EPS_MAX = sel_at["fgsm"], sel_at["pgd"], sel_gen["selected_eps"]
print(f"Reusing seed-0 selections: fgsm eps_train={EPS_FGSM}  pgd eps_train={EPS_PGD}  generator eps_max={EPS_MAX}")

# %% [3] Model / attack / eval / collapse-guard - identical to Stage 2b and Stage 3
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
    if bad: raise RuntimeError(f"[{tag}] COLLAPSED: F1=0 for {bad} - stop and inspect before continuing")
    print(f"[{tag}] collapse check passed")

# %% [4a] Classifier training - undefended / FGSM-AT / PGD-AT (verbatim from Stage 2b, seed made an explicit arg)
def fit_classifier_at(model, X, y, Xv, yv, seed, at=None, lr=BASE_LR, max_epochs=60, patience=8, tag=""):
    set_seed(seed)
    opt = keras.optimizers.Adam(lr); opt.build(model.trainable_variables)
    A = make_attacks(model, MASK) if at is not None else None
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
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"[{tag}] epoch {ep+1} val_loss={vl:.4f} best={best:.4f} ({time.time()-t0:.1f}s)")
    model.set_weights(best_w)
    return model

# %% [4b] Classifier training - plain / distilled / +feedback samples (verbatim from Stage 3, seed made an explicit arg)
def fit_classifier_kd(model, X, y, Xv, yv, seed, X_extra=None, Xv_extra=None, zt=None,
                      lr=BASE_LR, max_epochs=60, patience=8, tag=""):
    set_seed(seed)
    opt = keras.optimizers.Adam(lr); opt.build(model.trainable_variables)
    use_kd, cwt = zt is not None, tf.constant(cw)
    def wce(z, yy): return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=yy, logits=z) * tf.gather(cwt, yy))

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
            ztb = zt[idx] if use_kd else np.zeros((len(idx), NCLS), "float32")
            if X_extra is not None:
                xb, yb = np.concatenate([xb, X_extra[idx]]), np.concatenate([yb, y[idx]])
                # BUGFIX (found when seed 1 crashed with "2048 vs 4096"): xb/yb double to cover the clean +
                # generator-perturbed copies, so ztb (the teacher's soft labels used by the KD loss) must double
                # the same way, giving the perturbed copy the same teacher target as its clean counterpart.
                if use_kd:
                    ztb = np.concatenate([ztb, ztb])
            step(tf.constant(xb), tf.constant(yb), tf.constant(ztb))
        vl = float(wce(model(tf.constant(Xv), training=False), tf.constant(yv)))
        if Xv_extra is not None:
            vl += float(wce(model(tf.constant(Xv_extra), training=False), tf.constant(yv)))
        if vl < best - 1e-4: best, best_w, wait = vl, model.get_weights(), 0
        else:
            wait += 1
            if wait % 4 == 0: cur_lr *= 0.5; opt.learning_rate = cur_lr
            if wait >= patience: break
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"[{tag}] epoch {ep+1} val_loss={vl:.4f} best={best:.4f} ({time.time()-t0:.1f}s)")
    model.set_weights(best_w)
    return model

# %% [5] AdvGAN generator (verbatim from Stage 3, seed made an explicit arg)
def build_generator(d):
    inp = keras.Input(shape=(d,))
    x = layers.Dense(256, activation="relu")(inp); x = layers.Dense(256, activation="relu")(x)
    return keras.Model(inp, layers.Dense(d)(x))
def build_disc(d):
    inp = keras.Input(shape=(d,))
    x = layers.Dense(256)(inp); x = layers.LeakyReLU(0.2)(x); x = layers.Dense(256)(x); x = layers.LeakyReLU(0.2)(x)
    return keras.Model(inp, layers.Dense(1)(x))

def apply_G(G, X, eps, bs=8192):
    E = eps * MASK
    return np.vstack([np.clip(X[i:i+bs] + E * np.tanh(G(tf.constant(X[i:i+bs]), training=False).numpy()), 0., 1.)
                      for i in range(0, len(X), bs)]).astype("float32")

def train_generator(clf, eps, seed, tag="", diag_path=None):
    set_seed(seed)
    G, Dn = build_generator(D), build_disc(D)
    g_opt = keras.optimizers.Adam(1e-3, beta_1=0.5); g_opt.build(G.trainable_variables)
    d_opt = keras.optimizers.Adam(1e-4, beta_1=0.5); d_opt.build(Dn.trainable_variables)
    E = tf.constant(eps * MASK)
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
        idx = rng.randint(0, len(Xtr_sm), BATCH)
        step(tf.constant(Xtr_sm[idx]), tf.constant(ytr_sm[idx]))
    Xa = apply_G(G, Xva, eps)
    a0, a1 = float((predict(clf, Xva) == yva).mean()), float((predict(clf, Xa) == yva).mean())
    print(f"[{tag}] generator sanity (validation): acc clean={a0:.4f} -> under G={a1:.4f}")
    if diag_path: json.dump({"acc_clean": a0, "acc_under_G": a1}, open(diag_path, "w"))
    return G

# %% [6] Per-seed pipelines - same model paths/naming as Stage 2b / Stage 3 so seed 0 loads the existing files
def get_undefended(seed):
    path = f"{MOD}/undefended_smote_seed{seed}.keras"
    if os.path.exists(path): return keras.models.load_model(path, compile=False)
    m = build_model(D)
    fit_classifier_at(m, Xtr_sm, ytr_sm, Xva, yva, seed=seed, at=None, tag=f"undefended seed{seed}")
    collapse_check(clean_eval(m, Xva, yva), f"undefended seed{seed}(val)"); m.save(path); return m

def get_at(method, seed):
    eps = EPS_FGSM if method == "fgsm" else EPS_PGD
    path = f"{MOD}/{method}_eps{eps}_seed{seed}.keras"
    if os.path.exists(path): return keras.models.load_model(path, compile=False)
    m = build_model(D)
    fit_classifier_at(m, Xtr_sm, ytr_sm, Xva, yva, seed=seed, at=(method, eps), tag=f"{method}_at seed{seed}")
    collapse_check(clean_eval(m, Xva, yva), f"{method}_at seed{seed}(val)"); m.save(path); return m

def get_teacher(seed):
    path = f"{MOD}/s3_teacher_seed{seed}.keras"
    if os.path.exists(path): return keras.models.load_model(path, compile=False)
    t = build_model(D, widths=(512, 256, 128))
    fit_classifier_kd(t, Xtr_sm, ytr_sm, Xva, yva, seed=seed, tag=f"teacher seed{seed}")
    collapse_check(clean_eval(t, Xva, yva), f"teacher seed{seed}(val)"); t.save(path); return t

def get_baseline(distil, seed, zt_tr):
    path = f"{MOD}/s3_base_{'kd' if distil else 'nokd'}_seed{seed}.keras"
    if os.path.exists(path): return keras.models.load_model(path, compile=False)
    m = build_model(D)
    fit_classifier_kd(m, Xtr_sm, ytr_sm, Xva, yva, seed=seed, zt=zt_tr if distil else None,
                      tag=f"base_{'kd' if distil else 'nokd'} seed{seed}")
    collapse_check(clean_eval(m, Xva, yva), f"base_{'kd' if distil else 'nokd'} seed{seed}(val)"); m.save(path); return m

def run_pipeline(distil, mode, eps, seed, zt_tr, rounds=ROUNDS):
    ptag = f"{'kd' if distil else 'nokd'}_{mode}"
    prev = get_baseline(distil, seed, zt_tr); clfs, G_fixed = {0: prev}, None
    for r in range(1, rounds + 1):
        cpath = f"{MOD}/s3_{ptag}_eps{eps}_r{r}_seed{seed}.keras"; gpath = f"{MOD}/s3_gen_{ptag}_eps{eps}_r{r}_seed{seed}.keras"
        if os.path.exists(cpath):
            clfs[r] = prev = keras.models.load_model(cpath, compile=False)
            if mode == "fixed_gen" and r == 1: G_fixed = keras.models.load_model(gpath, compile=False)
            continue
        if mode == "feedback" or r == 1:
            G = train_generator(prev, eps, tag=f"{ptag} eps={eps} r{r} seed{seed}", seed=seed * 100 + r,
                                diag_path=gpath.replace(".keras", ".json")); G.save(gpath)
            if r == 1: G_fixed = G
        else:
            G = G_fixed
        Xa_tr, Xa_va = apply_G(G, Xtr_sm, eps), apply_G(G, Xva, eps)
        new = build_model(D); new.set_weights(prev.get_weights())
        # NOTE: zt is deliberately NOT passed here, matching cicids_stage3_a3idps.py's run_pipeline (and
        # nsl_stage3.py's) exactly - in the original design, distillation only shapes the round-0 baseline's
        # warm-start weights; per-round feedback retraining is plain hard-label + class-weighted training on
        # clean+generated samples for every pipeline (kd_feedback included). An earlier draft of this script
        # added zt here to make KD apply continuously through the feedback rounds - that is a real, separate
        # design question worth deciding deliberately, but it must not be smuggled in as a silent deviation from
        # what already produced the existing seed-0 numbers, so it is reverted here to keep seed 0 comparable.
        fit_classifier_kd(new, Xtr_sm, ytr_sm, Xva, yva, seed=seed * 100 + r, X_extra=Xa_tr, Xv_extra=Xa_va,
                          lr=FEEDBACK_LR, max_epochs=30, tag=f"{ptag} eps={eps} r{r} seed{seed}")
        collapse_check(clean_eval(new, Xva, yva), f"{ptag} r{r} seed{seed}(val)")
        new.save(cpath); clfs[r] = prev = new
    return clfs

# %% [7] Run all 5 seeds, evaluate every model on TEST, save per-seed JSON as we go (resumable)
ALL = {}   # ALL[seed][tag] = {"clean":..., "robust":...}
for seed in SEEDS:
    seed_out_path = f"{RES}/stage4a_seed{seed}.json"
    if os.path.exists(seed_out_path):
        ALL[seed] = json.load(open(seed_out_path)); print(f"seed {seed}: loaded existing results"); continue

    print(f"\n========== SEED {seed} ==========")
    und = get_undefended(seed)
    fgsm_at = get_at("fgsm", seed)
    pgd_at = get_at("pgd", seed)
    teacher = get_teacher(seed)
    zt_tr = teacher.predict(Xtr_sm, batch_size=8192, verbose=0).astype("float32")

    kd_feedback = run_pipeline(True, "feedback", EPS_MAX, seed, zt_tr)
    kd_fixed_gen = run_pipeline(True, "fixed_gen", EPS_MAX, seed, zt_tr)
    nokd_feedback = run_pipeline(False, "feedback", EPS_MAX, seed, zt_tr)

    seed_res = {}
    for tag, m in [("undefended", und), (f"fgsm_at_eps{EPS_FGSM}", fgsm_at), (f"pgd_at_eps{EPS_PGD}", pgd_at),
                   ("teacher", teacher)]:
        tc = clean_eval(m, Xte, yte); collapse_check(tc, f"{tag} seed{seed}(test)")
        seed_res[tag] = {"clean": tc, "robust": robust_eval(m, Xte, yte, MASK, EPS_EVAL)}
    for ptag, clfs in [("kd_feedback", kd_feedback), ("kd_fixed_gen", kd_fixed_gen), ("nokd_feedback", nokd_feedback)]:
        for r, m in clfs.items():
            tc = clean_eval(m, Xte, yte)
            if r > 0: collapse_check(tc, f"{ptag}_r{r} seed{seed}(test)")
            seed_res[f"{ptag}_r{r}"] = {"clean": tc, "robust": robust_eval(m, Xte, yte, MASK, EPS_EVAL)}

    json.dump(jsonable(seed_res), open(seed_out_path, "w"), indent=1)
    ALL[seed] = seed_res
    print(f"seed {seed}: done, saved to {seed_out_path}")

# %% [8] Aggregate mean +- std across seeds, and paired differences (per-seed, then averaged)
TAGS = ["undefended", f"fgsm_at_eps{EPS_FGSM}", f"pgd_at_eps{EPS_PGD}"] + \
       [f"kd_feedback_r{r}" for r in range(ROUNDS + 1)] + \
       [f"kd_fixed_gen_r{r}" for r in range(ROUNDS + 1)] + \
       [f"nokd_feedback_r{r}" for r in range(ROUNDS + 1)]

def agg(tag, eps=None):
    vals = []
    for s in SEEDS:
        r = ALL[s][tag]
        vals.append(r["clean"]["acc"] if eps is None else r["robust"][str(eps)]["worst"])
    return float(np.mean(vals)), float(np.std(vals)), vals

print(f"\n=== 5-SEED TEST RESULTS (mean +- std over seeds {SEEDS}), worst-case robust accuracy ===")
print(f"{'model':<24}{'clean':>12}" + "".join(f"{('e='+str(e)):>13}" for e in EPS_EVAL))
agg_table = {}
for tag in TAGS:
    if tag not in ALL[SEEDS[0]]: continue
    cm, cs, _ = agg(tag)
    row = {"clean": (cm, cs)}
    line = f"{tag:<24}{cm*100:>8.1f}+-{cs*100:<3.1f}"
    for e in EPS_EVAL:
        m_, s_, _ = agg(tag, e); row[str(e)] = (m_, s_)
        line += f"  {m_*100:>6.1f}+-{s_*100:<3.1f}"
    print(line)
    agg_table[tag] = row

def paired(tag_a, tag_b, eps):
    diffs = [ALL[s][tag_a]["robust"][str(eps)]["worst"] - ALL[s][tag_b]["robust"][str(eps)]["worst"] for s in SEEDS]
    return float(np.mean(diffs)), float(np.std(diffs)), sum(1 for x in diffs if x > 0)

print(f"\n=== Paired differences at eps=0.05 and eps=0.10 (worst-case robust acc, kd_feedback_r{ROUNDS} vs others) ===")
comparisons = [(f"kd_feedback_r{ROUNDS}", "undefended"), (f"kd_feedback_r{ROUNDS}", f"pgd_at_eps{EPS_PGD}"),
               (f"kd_feedback_r{ROUNDS}", f"kd_fixed_gen_r{ROUNDS}"), (f"kd_feedback_r{ROUNDS}", f"nokd_feedback_r{ROUNDS}")]
paired_table = {}
for a, b in comparisons:
    entry = {}
    for e in (0.05, 0.10):
        m_, s_, npos = paired(a, b, e)
        entry[str(e)] = {"mean": m_, "std": s_, "n_positive": npos}
        print(f"{a} - {b}  @eps={e}: {m_*100:+.1f}+-{s_*100:.1f} pp  ({npos}/{len(SEEDS)} seeds positive)")
    paired_table[f"{a}__minus__{b}"] = entry

json.dump(jsonable({"eps_fgsm": EPS_FGSM, "eps_pgd": EPS_PGD, "eps_max": EPS_MAX, "seeds": SEEDS,
                    "aggregate": agg_table, "paired_differences": paired_table}),
          open(f"{RES}/stage4a_summary.json", "w"), indent=1)
print(f"\nSaved: {RES}/stage4a_summary.json")
