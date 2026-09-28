# CICIDS2017 v2 - Stage 4b: AdvGAN-as-attack sweep, mirroring nsl_stage4b_advgan_attack.py. Purpose: answer
# Reviewer 2's "sign-error / F1-contradiction" concern with a worst-case check - train a FRESH generator, with
# several architectures/hyperparameters, whose only job is to attack each already-finished classifier (it is not
# part of that classifier's own training loop), and confirm it never beats the gradient-based white-box attacks
# (FGSM/PGD) already reported in Stage 4a. If a generator config ever attacked a model MORE effectively than the
# white-box PGD attack, that would mean the white-box numbers understate real-world risk - this script checks
# for that directly, with integrity assertions, rather than asserting it never happens.
#
# NOTE ON THE 8 CONFIGS: these mirror the *names and intent* of the 8 configs used in nsl_stage4b_advgan_attack.py
# (default / width64 / width1024 / lam5 / lam80 / steps400 / steps4800 / strong), but the exact hyperparameter
# values below are freshly chosen for CICIDS (78-dim, no categorical mask) rather than copied from the NSL file,
# which was not available to reconstruct byte-for-byte. What matters for the paper is the SWEEP METHODOLOGY and
# the integrity checks, not that the two datasets share identical generator hyperparameters.
#
# Requires cicids_stage4a_seeds.py to have produced, per seed: undefended_smote_seed{s}.keras,
# {fgsm,pgd}_eps{e}_seed{s}.keras, and s3_kd_feedback_eps{EPS_MAX}_r5_seed{s}.keras. Fully resumable.

# %% [0] Setup
from google.colab import drive
drive.mount('/content/drive')
import os, json, time
import numpy as np, tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import accuracy_score

BASE = "/content/drive/MyDrive/A3IDPS_COLAB"
DATA = f"{BASE}/data_processed/cicids_v2"; MOD = f"{BASE}/models/cicids_v2"; RES = f"{BASE}/results/cicids_v2"
os.makedirs(RES, exist_ok=True)

d = np.load(f"{DATA}/cicids_v2_primary.npz")
Xte, yte = d["Xte"].astype("float32"), d["yte"].astype("int32")
D, NCLS = Xte.shape[1], 5
MASK = np.ones(D, "float32")   # no categorical columns in CICIDS - nothing to freeze

SEEDS = [0, 1, 2, 3, 4]
EPS_LIST = [0.05, 0.10]
BATCH = 2048

sel_at = json.load(open(f"{RES}/stage2b_selected_seed0.json"))
sel_gen = json.load(open(f"{RES}/stage3_selection_seed0.json"))
EPS_FGSM, EPS_PGD, EPS_MAX = sel_at["fgsm"], sel_at["pgd"], sel_gen["selected_eps"]

# model tag -> path template (must match Stage 4a's saved filenames exactly)
MODEL_PATHS = {
    "undefended":  lambda s: f"{MOD}/undefended_smote_seed{s}.keras",
    "fgsm_at":     lambda s: f"{MOD}/fgsm_eps{EPS_FGSM}_seed{s}.keras",
    "pgd_at":      lambda s: f"{MOD}/pgd_eps{EPS_PGD}_seed{s}.keras",
    "kd_feedback_r5": lambda s: f"{MOD}/s3_kd_feedback_eps{EPS_MAX}_r5_seed{s}.keras",
}

def jsonable(o):
    if isinstance(o, dict): return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, np.floating): return float(o)
    return o

def predict(model, X): return np.argmax(model.predict(X, batch_size=8192, verbose=0), axis=1)
def acc(model, X, y): return float(accuracy_score(y, predict(model, X)))

# %% [1] Generator/discriminator builders, parameterized by width - and the 8 config grid
def build_generator(d, width):
    inp = keras.Input(shape=(d,))
    x = layers.Dense(width, activation="relu")(inp); x = layers.Dense(width, activation="relu")(x)
    return keras.Model(inp, layers.Dense(d)(x))
def build_disc(d, width=256):
    inp = keras.Input(shape=(d,))
    x = layers.Dense(width)(inp); x = layers.LeakyReLU(0.2)(x); x = layers.Dense(width)(x); x = layers.LeakyReLU(0.2)(x)
    return keras.Model(inp, layers.Dense(1)(x))

# name -> dict(width, lam_cls, lam_gan, lam_l1, steps)
CONFIGS = {
    "default":   dict(width=256,  lam_cls=20., lam_gan=1.0, lam_l1=0.5, steps=1200),
    "width64":   dict(width=64,   lam_cls=20., lam_gan=1.0, lam_l1=0.5, steps=1200),
    "width1024": dict(width=1024, lam_cls=20., lam_gan=1.0, lam_l1=0.5, steps=1200),
    "lam5":      dict(width=256,  lam_cls=5.,  lam_gan=1.0, lam_l1=0.5, steps=1200),
    "lam80":     dict(width=256,  lam_cls=80., lam_gan=1.0, lam_l1=0.5, steps=1200),
    "steps400":  dict(width=256,  lam_cls=20., lam_gan=1.0, lam_l1=0.5, steps=400),
    "steps4800": dict(width=256,  lam_cls=20., lam_gan=1.0, lam_l1=0.5, steps=4800),
    "strong":    dict(width=1024, lam_cls=80., lam_gan=0.5, lam_l1=0.25, steps=4800),   # most aggressive combination
}
KAPPA = 1.0

def apply_G(G, X, eps, bs=8192):
    E = eps * MASK
    return np.vstack([np.clip(X[i:i+bs] + E * np.tanh(G(tf.constant(X[i:i+bs]), training=False).numpy()), 0., 1.)
                      for i in range(0, len(X), bs)]).astype("float32")

def train_attack_generator(clf, Xtr_for_gen, ytr_for_gen, eps, cfg, seed):
    tf.keras.utils.set_random_seed(seed)
    G = build_generator(D, cfg["width"]); Dn = build_disc(D)
    g_opt = keras.optimizers.Adam(1e-3, beta_1=0.5); g_opt.build(G.trainable_variables)
    d_opt = keras.optimizers.Adam(1e-4, beta_1=0.5); d_opt.build(Dn.trainable_variables)
    E = tf.constant(eps * MASK)
    bce = lambda labels, logits: tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits))
    def gen(x): return tf.clip_by_value(x + E * tf.tanh(G(x, training=True)), 0., 1.)

    @tf.function(reduce_retracing=True)
    def step(x, y):
        with tf.GradientTape() as td:
            xa = gen(x)
            ld = bce(tf.ones([tf.shape(x)[0], 1]), Dn(x, training=True)) + bce(tf.zeros([tf.shape(x)[0], 1]), Dn(xa, training=True))
        d_opt.apply_gradients(zip(td.gradient(ld, Dn.trainable_variables), Dn.trainable_variables))
        with tf.GradientTape() as tg:
            xa = gen(x); z = clf(xa, training=False); oh = tf.one_hot(y, NCLS)
            margin = tf.reduce_sum(z * oh, 1) - tf.reduce_max(z - 1e9 * oh, 1)
            lg = (cfg["lam_cls"] * tf.reduce_mean(tf.nn.relu(margin + KAPPA))
                  + cfg["lam_gan"] * bce(tf.ones([tf.shape(x)[0], 1]), Dn(xa, training=False))
                  + cfg["lam_l1"] * tf.reduce_mean(tf.abs(xa - x)))
        g_opt.apply_gradients(zip(tg.gradient(lg, G.trainable_variables), G.trainable_variables))
        return lg

    rng = np.random.RandomState(seed); n = len(Xtr_for_gen)
    for s in range(cfg["steps"]):
        idx = rng.randint(0, n, BATCH)
        step(tf.constant(Xtr_for_gen[idx]), tf.constant(ytr_for_gen[idx]))
    return G

# %% [2] Data the generator trains against - use a fixed subsample of TEST inputs' distribution is wrong (would
# leak test into training the attack); use the held-out VALIDATION split instead, exactly as Stage 3's
# train_generator() did, then attack the TEST set with the trained generator. Re-load val split from primary npz.
Xva, yva = d["Xva"].astype("float32"), d["yva"].astype("int32")

# %% [3] Run the sweep: per seed, per model tag, per eps, per config - fully resumable via a flat results dict
OUT_PATH = f"{RES}/stage4b_results.json"
results = json.load(open(OUT_PATH)) if os.path.exists(OUT_PATH) else {}

# integrity: baseline clean test accuracy per (seed, model tag), to catch any load mismatch immediately
def get_model(seed, tag):
    return keras.models.load_model(MODEL_PATHS[tag](seed), compile=False)

for seed in SEEDS:
    seed_key = str(seed)
    results.setdefault(seed_key, {})
    for tag in MODEL_PATHS:
        model = get_model(seed, tag)
        clean_acc = acc(model, Xte, yte)
        results[seed_key].setdefault(tag, {"clean_acc": clean_acc, "configs": {}})
        # integrity check 1: clean accuracy must be stable/deterministic for this saved model (no retraining happened)
        prev = results[seed_key][tag].get("clean_acc")
        if prev is not None and abs(prev - clean_acc) > 1e-6:
            raise RuntimeError(f"seed{seed} {tag}: clean acc changed between runs ({prev} -> {clean_acc}) - model file changed underfoot")
        results[seed_key][tag]["clean_acc"] = clean_acc

        for eps in EPS_LIST:
            eps_key = str(eps)
            for cfg_name, cfg in CONFIGS.items():
                done = results[seed_key][tag]["configs"].get(cfg_name, {}).get(eps_key)
                if done is not None:
                    continue
                t0 = time.time()
                G = train_attack_generator(model, Xva, yva, eps, cfg, seed=seed * 1000 + hash(cfg_name) % 1000)
                Xa = apply_G(G, Xte, eps)

                # integrity check 2: perturbation must not exceed eps anywhere (tanh bounds it by construction,
                # but verify numerically in case of a future code change)
                max_delta = float(np.max(np.abs(Xa - Xte)))
                if max_delta > eps + 1e-4:
                    raise RuntimeError(f"seed{seed} {tag} {cfg_name} eps={eps}: perturbation {max_delta:.5f} exceeds eps bound")

                acc_under_g = acc(model, Xa, yte)
                mean_abs_delta = float(np.mean(np.abs(Xa - Xte)))
                results[seed_key].setdefault(tag, {}).setdefault("configs", {}).setdefault(cfg_name, {})[eps_key] = {
                    "acc_under_generator": acc_under_g, "mean_abs_delta": mean_abs_delta, "max_abs_delta": max_delta,
                    "seconds": round(time.time() - t0, 1)}
                print(f"seed{seed} {tag:<14} eps={eps} {cfg_name:<10} acc_under_G={acc_under_g:.4f}  "
                      f"mean|d|={mean_abs_delta:.5f} max|d|={max_delta:.5f} ({time.time()-t0:.0f}s)")
                json.dump(jsonable(results), open(OUT_PATH, "w"), indent=1)

# %% [4] Compare against the white-box worst-case from Stage 4a - flag ANY case where the generator attack is
# stronger (lower accuracy) than the white-box PGD/FGSM worst-case, which would mean Stage 4a understated risk.
s4a = {s: json.load(open(f"{RES}/stage4a_seed{s}.json")) for s in SEEDS}
S4A_TAG = {"undefended": "undefended", "fgsm_at": f"fgsm_at_eps{EPS_FGSM}", "pgd_at": f"pgd_at_eps{EPS_PGD}",
           "kd_feedback_r5": "kd_feedback_r5"}

flags = []
print(f"\n=== AdvGAN-attack accuracy vs white-box worst-case (accuracy %, lower = stronger attack) ===")
for eps in EPS_LIST:
    print(f"\n--- eps={eps} ---")
    print(f"{'model':<16}{'default cfg':>12}{'worst cfg':>12}{'white-box':>12}   worst config per seed")
    for tag in MODEL_PATHS:
        default_accs, worst_accs, wb_accs, worst_cfgs = [], [], [], []
        for seed in SEEDS:
            cfgs = results[str(seed)][tag]["configs"]
            per_cfg = {c: cfgs[c][str(eps)]["acc_under_generator"] for c in CONFIGS if str(eps) in cfgs.get(c, {})}
            default_accs.append(per_cfg["default"])
            worst_name = min(per_cfg, key=per_cfg.get); worst_accs.append(per_cfg[worst_name]); worst_cfgs.append(worst_name)
            wb = s4a[seed][S4A_TAG[tag]]["robust"][str(eps)]["worst"]; wb_accs.append(wb)
            if per_cfg[worst_name] < wb - 1e-6:
                flags.append(f"seed{seed} {tag} eps={eps}: generator ({worst_name})={per_cfg[worst_name]:.4f} "
                             f"< white-box worst-case={wb:.4f}")
        print(f"{tag:<16}{np.mean(default_accs)*100:>9.1f}+-{np.std(default_accs)*100:<3.1f}"
              f"{np.mean(worst_accs)*100:>9.1f}+-{np.std(worst_accs)*100:<3.1f}"
              f"{np.mean(wb_accs)*100:>9.1f}+-{np.std(wb_accs)*100:<3.1f}   {worst_cfgs}")

if flags:
    print(f"\n*** {len(flags)} FLAG(S): generator attack beat the white-box worst-case - inspect before trusting Stage 4a numbers ***")
    for f in flags: print("  " + f)
else:
    print(f"\nNo flags: across {len(SEEDS)} seeds x {len(MODEL_PATHS)} models x {len(EPS_LIST)} eps x {len(CONFIGS)} configs "
          f"= {len(SEEDS)*len(MODEL_PATHS)*len(EPS_LIST)*len(CONFIGS)} generator-attack runs, the white-box "
          f"FGSM/PGD worst-case was never beaten by the AdvGAN attack sweep.")

json.dump({"flags": flags}, open(f"{RES}/stage4b_flags.json", "w"), indent=1)
print(f"\nSaved: {OUT_PATH} and {RES}/stage4b_flags.json")
