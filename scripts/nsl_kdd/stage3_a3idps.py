# Stage 3 - A3IDPS on NSL-KDD under the frozen protocol (same data, attacks, selection rule as Stage 2).
# Implements what the paper describes:
#   (i)  wide teacher -> distilled student (alpha=0.7, T=3; KL taken teacher||student, the standard direction)
#   (ii) AdvGAN generator: delta = eps_max * mask * tanh(net(x)), untargeted margin objective, connected discriminator,
#        lambda_cls=20, lambda_l1=0.5, 1200 steps, trained on the TRAINING split only
#   (iii) feedback rounds: generator retrained against the current classifier, classifier retrained on clean + generated
# Ablations: with/without distillation x feedback vs fixed generator (control), rounds 0..3.
# The test set is used only for final reporting. Paste into Colab cell by cell. Resumable via files on Drive.

# %% [0] Setup (same conventions as Stage 2)
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
EPS_EVAL, EPS_SELECT = [0.01, 0.02, 0.05, 0.10, 0.15], [0.01, 0.02, 0.05, 0.10]
EPS_TRAIN = [0.02, 0.05, 0.10]           # generator eps_max grid (same grid as the AT baselines)
CLASS_WEIGHT_CAP, VAL_CLEAN_TOL, BATCH = 20.0, 0.05, 512
KD_ALPHA, KD_T = 0.7, 3.0
GEN_STEPS, LAM_CLS, LAM_L1, LAM_GAN, KAPPA = 1200, 20.0, 0.5, 1.0, 1.0
ROUNDS = 3
FAST = True                              # tf.function training steps; set False to fall back to eager if tracing errors occur
cw = np.minimum(compute_class_weight("balanced", classes=np.arange(NCLS), y=ytr), CLASS_WEIGHT_CAP).astype("float32")

def maybe_tf(fn):
    return tf.function(fn, reduce_retracing=True) if FAST else fn

def set_seed(s):
    random.seed(s); np.random.seed(s); tf.keras.utils.set_random_seed(s)

def jsonable(o):
    if isinstance(o, dict): return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, np.floating): return float(o)
    return o

# %% [1] Models and attacks (identical to Stage 2; models output LOGITS)
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
                best_x = tf.where(upd[:, None], x, best_x); best_l = tf.where(upd, l_new, best_l)
        return best_x
    return {"fgsm": fgsm, "pgd": pgd}

def run_attack(fn, X, y, bs=4096, **kw):
    return np.vstack([fn(tf.constant(X[i:i+bs]), tf.constant(y[i:i+bs]), **kw).numpy() for i in range(0, len(X), bs)])

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
                       "worst": float(w.mean()), "worst_recall": {CLASS_NAMES[k]: float(w[y == k].mean()) for k in range(NCLS)}}
    return out

# %% [2] Generic classifier training: plain, distilled (zt given), or feedback (X_extra = generated samples, Eq. 4)
def fit_classifier(model, X, y, Xv, yv, X_extra=None, Xv_extra=None, zt=None,
                   lr=1e-3, max_epochs=40, patience=8, seed=0, tag=""):
    set_seed(seed)
    opt = keras.optimizers.Adam(lr); opt.build(model.trainable_variables)
    use_kd, cwt = zt is not None, tf.constant(cw)
    def wce(z, yy):
        return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=yy, logits=z) * tf.gather(cwt, yy))

    @maybe_tf
    def step(xb, yb, ztb):
        with tf.GradientTape() as t:
            z = model(xb, training=True)
            hard = wce(z, yb)
            if use_kd:
                pt = tf.nn.softmax(ztb / KD_T)
                log_ps = tf.nn.log_softmax(z / KD_T)
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
            step(tf.constant(xb), tf.constant(yb), tf.constant(ztb))
        vl = float(wce(model(tf.constant(Xv), training=False), tf.constant(yv)))
        if Xv_extra is not None:
            vl += float(wce(model(tf.constant(Xv_extra), training=False), tf.constant(yv)))
        if vl < best - 1e-4:
            best, best_w, wait = vl, model.get_weights(), 0
        else:
            wait += 1
            if wait % 4 == 0:
                cur_lr *= 0.5; opt.learning_rate = cur_lr
            if wait >= patience: break
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"[{tag}] epoch {ep+1} val_loss={vl:.4f} best={best:.4f} ({time.time()-t0:.1f}s)")
    model.set_weights(best_w)
    return model

# %% [3] AdvGAN generator (corrected): bounded delta on numeric features, untargeted margin loss, connected discriminator
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

def train_generator(clf, eps, seed=SEED, tag=""):
    set_seed(seed)
    G, Dn = build_generator(D), build_disc(D)
    g_opt = keras.optimizers.Adam(1e-3, beta_1=0.5); g_opt.build(G.trainable_variables)
    d_opt = keras.optimizers.Adam(1e-4, beta_1=0.5); d_opt.build(Dn.trainable_variables)
    E = tf.constant(eps * MASK_NUM)
    bce = lambda labels, logits: tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits))

    def gen(x):
        return tf.clip_by_value(x + E * tf.tanh(G(x, training=True)), 0., 1.)

    @maybe_tf
    def step(x, y):
        with tf.GradientTape() as td:
            xa = gen(x)
            ld = bce(tf.ones([tf.shape(x)[0], 1]), Dn(x, training=True)) + bce(tf.zeros([tf.shape(x)[0], 1]), Dn(xa, training=True))
        d_opt.apply_gradients(zip(td.gradient(ld, Dn.trainable_variables), Dn.trainable_variables))
        with tf.GradientTape() as tg:
            xa = gen(x)
            z = clf(xa, training=False)
            oh = tf.one_hot(y, NCLS)
            margin = tf.reduce_sum(z * oh, 1) - tf.reduce_max(z - 1e9 * oh, 1)       # >0: still correctly classified
            l_adv = tf.reduce_mean(tf.nn.relu(margin + KAPPA))                        # untargeted, bounded below
            l_gan = bce(tf.ones([tf.shape(x)[0], 1]), Dn(xa, training=False))
            l_l1 = tf.reduce_mean(tf.abs(xa - x))
            lg = LAM_CLS * l_adv + LAM_GAN * l_gan + LAM_L1 * l_l1
        g_opt.apply_gradients(zip(tg.gradient(lg, G.trainable_variables), G.trainable_variables))
        return lg

    rng = np.random.RandomState(seed)
    for s in range(GEN_STEPS):
        idx = rng.randint(0, len(Xtr), BATCH)
        lg = step(tf.constant(Xtr[idx]), tf.constant(ytr[idx]))
        if (s + 1) % 400 == 0: print(f"[{tag}] generator step {s+1}/{GEN_STEPS} loss={float(lg):.4f}")
    Xa = apply_G(G, Xva, eps)
    a0, a1 = float((predict(clf, Xva) == yva).mean()), float((predict(clf, Xa) == yva).mean())
    print(f"[{tag}] generator sanity (validation): acc clean={a0:.4f} -> under G={a1:.4f}; mean|delta|={np.abs(Xa - Xva).mean():.5f}")
    if a0 - a1 < 0.05:
        print(f"[{tag}] WARNING: generator lowers accuracy by < 5 pp - it is a weak attack; check objective/lr before trusting rounds")
    return G

# %% [4] Teacher, baselines (with / without distillation)
def get_teacher():
    path = f"{MOD}/s3_teacher_seed{SEED}.keras"
    if os.path.exists(path): return keras.models.load_model(path, compile=False)
    t = build_model(D, widths=(512, 256, 128))
    fit_classifier(t, Xtr, ytr, Xva, yva, seed=SEED, tag="teacher"); t.save(path); return t

teacher = get_teacher()
zt_tr = teacher.predict(Xtr, batch_size=4096, verbose=0).astype("float32")
teacher_report = {"val": clean_eval(teacher, Xva, yva), "test": clean_eval(teacher, Xte, yte),
                  "params": int(teacher.count_params())}
print("TEACHER clean: val acc", round(teacher_report["val"]["acc"], 4), "| test acc", round(teacher_report["test"]["acc"], 4),
      "| test macro-F1", round(teacher_report["test"]["macro_f1"], 4), "| params", teacher_report["params"])

def get_baseline(distil):
    path = f"{MOD}/s3_base_{'kd' if distil else 'nokd'}_seed{SEED}.keras"
    if os.path.exists(path): return keras.models.load_model(path, compile=False)
    m = build_model(D)
    fit_classifier(m, Xtr, ytr, Xva, yva, zt=zt_tr if distil else None, seed=SEED, tag=f"base_{'kd' if distil else 'nokd'}")
    m.save(path); return m

# %% [5] Feedback pipeline: mode "feedback" retrains the generator each round; "fixed_gen" is the control
def run_pipeline(distil, mode, eps, rounds=ROUNDS):
    ptag = f"{'kd' if distil else 'nokd'}_{mode}"
    prev = get_baseline(distil); clfs, G_fixed = {0: prev}, None
    for r in range(1, rounds + 1):
        cpath = f"{MOD}/s3_{ptag}_eps{eps}_r{r}_seed{SEED}.keras"
        gpath = f"{MOD}/s3_gen_{ptag}_eps{eps}_r{r}_seed{SEED}.keras"
        if os.path.exists(cpath):
            clfs[r] = prev = keras.models.load_model(cpath, compile=False)
            if mode == "fixed_gen" and r == 1: G_fixed = keras.models.load_model(gpath, compile=False)
            continue
        if mode == "feedback" or r == 1:
            G = train_generator(prev, eps, tag=f"{ptag} eps={eps} r{r}"); G.save(gpath)
            if r == 1: G_fixed = G
        else:
            G = G_fixed
        Xa_tr, Xa_va = apply_G(G, Xtr, eps), apply_G(G, Xva, eps)
        new = build_model(D); new.set_weights(prev.get_weights())
        fit_classifier(new, Xtr, ytr, Xva, yva, X_extra=Xa_tr, Xv_extra=Xa_va, lr=5e-4, max_epochs=30,
                       seed=SEED + r, tag=f"{ptag} eps={eps} r{r}")
        new.save(cpath); clfs[r] = prev = new
    return clfs

def val_score(m):
    vr = robust_eval(m, Xva, yva, MASK_NUM, EPS_SELECT, steps=10, restarts=1)
    return {"clean": clean_eval(m, Xva, yva)["acc"], "robust_mean": float(np.mean([vr[str(e)]["worst"] for e in EPS_SELECT]))}

# %% [6] Select the generator eps_max on VALIDATION (main pipeline: distilled + feedback, paper's R=2), same rule as Stage 2
und_val_clean = json.load(open(f"{RES}/stage2_val_seed{SEED}.json"))["none"]["val_clean"]["acc"]
grid = {}
for eps in EPS_TRAIN:
    clfs = run_pipeline(True, "feedback", eps)
    grid[eps] = {"models": clfs, "val": {r: val_score(clfs[r]) for r in clfs}}
    print(f"VAL eps_max={eps}: " + ", ".join(f"r{r}: clean={v['clean']:.4f} robust={v['robust_mean']:.4f}" for r, v in grid[eps]["val"].items()))
elig = [e for e in EPS_TRAIN if und_val_clean - grid[e]["val"][2]["clean"] <= VAL_CLEAN_TOL]
pool = elig if elig else EPS_TRAIN
if not elig: print("WARNING: no eps_max within the clean-accuracy tolerance at round 2")
eps_sel = max(pool, key=lambda e: (round(grid[e]["val"][2]["robust_mean"], 6), -e))
print("SELECTED generator eps_max:", eps_sel)
json.dump(jsonable({"selected_eps": eps_sel, "teacher": teacher_report, "val_grid": {e: grid[e]["val"] for e in EPS_TRAIN}}),
          open(f"{RES}/stage3_selection_seed{SEED}.json", "w"), indent=1)

# %% [7] Remaining pipelines at the selected eps_max, then TEST evaluation of every (pipeline, round)
all_models = {("kd_feedback", r): m for r, m in grid[eps_sel]["models"].items()}
for distil, mode in [(False, "feedback"), (True, "fixed_gen"), (False, "fixed_gen")]:
    for r, m in run_pipeline(distil, mode, eps_sel).items():
        all_models[(f"{'kd' if distil else 'nokd'}_{mode}", r)] = m

test_res = {}
for (ptag, r), m in sorted(all_models.items()):
    test_res[f"{ptag}_r{r}"] = {"clean": clean_eval(m, Xte, yte),
                                "numeric_only": robust_eval(m, Xte, yte, MASK_NUM, EPS_EVAL, steps=20, restarts=3),
                                "all_features": robust_eval(m, Xte, yte, MASK_ALL, EPS_EVAL, steps=20, restarts=3)}
    json.dump(jsonable({"eps_max": eps_sel, "test": test_res}), open(f"{RES}/stage3_test_seed{SEED}.json", "w"), indent=1)

s2 = json.load(open(f"{RES}/stage2_test_seed{SEED}.json"))
ref = {"undefended (S2)": "none", f"FGSM-AT eps={s2['selected']['fgsm']} (S2)": f"fgsm_eps{s2['selected']['fgsm']}",
       f"PGD-AT eps={s2['selected']['pgd']} (S2)": f"pgd_eps{s2['selected']['pgd']}"}
for prot in ("numeric_only", "all_features"):
    print(f"\n=== TEST worst-case robust accuracy, {prot} ===")
    print(f"{'model':<26}{'clean':>8}{'mF1':>8}" + "".join(f"{('e='+str(e)):>8}" for e in EPS_EVAL))
    for name, key in ref.items():
        r_, c_ = s2["test"][key][prot], s2["test"][key]["clean"]
        print(f"{name:<26}{c_['acc']:>8.4f}{c_['macro_f1']:>8.4f}" + "".join(f"{r_[str(e)]['worst']:>8.4f}" for e in EPS_EVAL))
    for key in sorted(test_res):
        r_, c_ = test_res[key][prot], test_res[key]["clean"]
        print(f"{key:<26}{c_['acc']:>8.4f}{c_['macro_f1']:>8.4f}" + "".join(f"{r_[str(e)]['worst']:>8.4f}" for e in EPS_EVAL))
print("\nSaved:", f"{RES}/stage3_test_seed{SEED}.json")
