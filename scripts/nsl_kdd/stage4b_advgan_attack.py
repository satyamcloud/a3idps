# Stage 4b - AdvGAN used as an ATTACK: fresh generators trained against each final model, sweeping generator
# capacity, lambda_cls and number of steps (Reviewer 2, point 6). Reports accuracy under each generator, the worst
# generator configuration, and the white-box worst case from Stage 4a for comparison.
# Integrity checks are built in: target models must reproduce Stage 4a's clean accuracy, and every generated perturbation
# must stay within eps and leave the one-hot columns untouched. Resumable. Start with SEEDS = [0], then extend.

# %% [0] Setup
from google.colab import drive
drive.mount('/content/drive')
import os, json, random, time
import numpy as np, tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

BASE = "/content/drive/MyDrive/A3IDPS_COLAB"
DATA = f"{BASE}/data_processed/nslkdd_v2"; MOD = f"{BASE}/models/nsl_v2"; RES = f"{BASE}/results/nsl_v2"
d = np.load(f"{DATA}/nsl_v2_arrays.npz")
Xtr, Xte = d["Xtr"].astype("float32"), d["Xte"].astype("float32")
ytr, yte = d["ytr"].astype("int32"), d["yte"].astype("int32")
names = json.load(open(f"{DATA}/feature_names.json"))
D, NCLS, BATCH = Xtr.shape[1], 5, 512
MASK_NUM = np.array([0. if n.startswith(("protocol_type_", "service_", "flag_")) else 1. for n in names], "float32")
FROZEN = MASK_NUM == 0

SEEDS = [0, 1, 2, 3, 4]                       # first run: seed 0 only; extend to [0, 1, 2, 3, 4] once the output looks right
EPS_ATTACK = [0.05, 0.10]
TARGETS = {"undefended": "s4_undefended", "fgsm_at": "s4_fgsm_at", "pgd_at": "s4_pgd_at", "kd_feedback_r5": "s4_kd_feedback_r5"}
CONFIGS = {
    "default":   dict(width=256,  lam_cls=20., steps=1200),      # the paper's setting
    "width64":   dict(width=64,   lam_cls=20., steps=1200),
    "width1024": dict(width=1024, lam_cls=20., steps=1200),
    "lam5":      dict(width=256,  lam_cls=5.,  steps=1200),
    "lam80":     dict(width=256,  lam_cls=80., steps=1200),
    "steps400":  dict(width=256,  lam_cls=20., steps=400),
    "steps4800": dict(width=256,  lam_cls=20., steps=4800),
    "strong":    dict(width=1024, lam_cls=80., steps=4800),
}
LAM_L1, LAM_GAN, KAPPA = 0.5, 1.0, 1.0
FAST = True                        # set False to fall back to eager steps if tf.function tracing errors occur
def maybe_tf(fn): return tf.function(fn, reduce_retracing=True) if FAST else fn
def set_seed(s): random.seed(s); np.random.seed(s); tf.keras.utils.set_random_seed(s)
def predict(model, X): return np.argmax(model.predict(X, batch_size=4096, verbose=0), axis=1)

# %% [1] Generator (same objective as the defence generator) with configurable capacity / lambda_cls / steps
def build_generator(width):
    inp = keras.Input(shape=(D,))
    x = layers.Dense(width, activation="relu")(inp); x = layers.Dense(width, activation="relu")(x)
    return keras.Model(inp, layers.Dense(D)(x))

def build_disc():
    inp = keras.Input(shape=(D,))
    x = layers.Dense(256)(inp); x = layers.LeakyReLU(0.2)(x)
    x = layers.Dense(256)(x);   x = layers.LeakyReLU(0.2)(x)
    return keras.Model(inp, layers.Dense(1)(x))

def apply_G(G, X, eps, bs=8192):
    E = eps * MASK_NUM
    return np.vstack([np.clip(X[i:i+bs] + E * np.tanh(G(tf.constant(X[i:i+bs]), training=False).numpy()), 0., 1.)
                      for i in range(0, len(X), bs)]).astype("float32")

def train_generator(clf, eps, seed, width, lam_cls, steps):
    set_seed(seed)
    G, Dn = build_generator(width), build_disc()
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
            lg = (lam_cls * tf.reduce_mean(tf.nn.relu(margin + KAPPA))
                  + LAM_GAN * bce(tf.ones([tf.shape(x)[0], 1]), Dn(xa, training=False))
                  + LAM_L1 * tf.reduce_mean(tf.abs(xa - x)))
        g_opt.apply_gradients(zip(tg.gradient(lg, G.trainable_variables), G.trainable_variables))
        return lg

    rng = np.random.RandomState(seed)
    for _ in range(steps):
        idx = rng.randint(0, len(Xtr), BATCH)
        step(tf.constant(Xtr[idx]), tf.constant(ytr[idx]))
    return G

# %% [2] Run: fresh generator per (model, seed, eps, configuration); integrity checks on every one
OUT = f"{RES}/stage4b_advgan_attack.json"
res = json.load(open(OUT)) if os.path.exists(OUT) else {}
s4a = {s: json.load(open(f"{RES}/stage4a_seed{s}.json")) for s in SEEDS}

for seed in SEEDS:
    for tname, prefix in TARGETS.items():
        model = keras.models.load_model(f"{MOD}/{prefix}_seed{seed}.keras", compile=False)
        clean = float((predict(model, Xte) == yte).mean())
        ref_clean = s4a[seed][tname]["clean"]["acc"]
        assert abs(clean - ref_clean) < 1e-6, f"{tname} seed {seed}: clean acc {clean} != Stage 4a {ref_clean} (wrong model file?)"
        print(f"\n##### seed {seed} | {tname} | clean acc {clean:.4f} (matches Stage 4a)")
        for eps in EPS_ATTACK:
            for cname, cfg in CONFIGS.items():
                key = f"{tname}|seed{seed}|eps{eps}|{cname}"
                if key in res: continue
                t0 = time.time()
                G = train_generator(model, eps, 1000 + seed, **cfg)
                Xa = apply_G(G, Xte, eps)
                delta = Xa - Xte
                assert np.abs(delta).max() <= eps + 1e-6, "perturbation exceeds eps"
                assert np.abs(delta[:, FROZEN]).max() == 0.0, "one-hot columns were modified"
                acc = float((predict(model, Xa) == yte).mean())
                res[key] = {"acc_under_generator": acc, "mean_abs_delta": float(np.abs(delta).mean()), "config": cfg}
                json.dump(res, open(OUT, "w"), indent=1)
                print(f"  eps={eps} {cname:<10} acc under generator={acc:.4f} | mean|delta|={np.abs(delta).mean():.5f} ({time.time()-t0:.0f}s)")
        keras.backend.clear_session()

# %% [3] Summary (run again after more seeds finish)
res = json.load(open(OUT)); seeds_done = sorted({int(k.split("|")[1][4:]) for k in res})
def ms(v): a = np.array(v); return f"{100*a.mean():5.1f}+-{100*(a.std(ddof=1) if len(a) > 1 else 0):4.1f}"
print(f"\nseeds with results: {seeds_done}   (accuracy in %, lower = stronger attack; white-box = Stage 4a worst case)")
for eps in EPS_ATTACK:
    print(f"\n=== eps = {eps}: AdvGAN attack accuracy per model ===")
    print(f"{'model':<18}{'default cfg':>14}{'worst cfg':>14}{'white-box':>14}   worst config (per seed)")
    for tname in TARGETS:
        dflt, worst, wb, who = [], [], [], []
        for s in seeds_done:
            accs = {c: res[f"{tname}|seed{s}|eps{eps}|{c}"]["acc_under_generator"] for c in CONFIGS if f"{tname}|seed{s}|eps{eps}|{c}" in res}
            if len(accs) < len(CONFIGS): continue
            dflt.append(accs["default"]); c_min = min(accs, key=accs.get); worst.append(accs[c_min]); who.append(c_min)
            s4 = json.load(open(f"{RES}/stage4a_seed{s}.json"))[tname]["numeric_only"][str(eps)]["worst"]; wb.append(s4)
        if dflt:
            flag = "  <-- generator STRONGER than white-box: check for a bug" if np.mean(worst) < np.mean(wb) - 0.02 else ""
            print(f"{tname:<18}{ms(dflt):>14}{ms(worst):>14}{ms(wb):>14}   {who}{flag}")
    print(f"\n--- per-configuration mean accuracy at eps={eps} ---")
    print(f"{'config':<12}" + "".join(f"{t:>16}" for t in TARGETS))
    for c in CONFIGS:
        row = []
        for tname in TARGETS:
            v = [res[f"{tname}|seed{s}|eps{eps}|{c}"]["acc_under_generator"] for s in seeds_done if f"{tname}|seed{s}|eps{eps}|{c}" in res]
            row.append(ms(v) if v else "-")
        print(f"{c:<12}" + "".join(f"{r:>16}" for r in row))
