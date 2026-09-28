# Stage 2c - ART APGD-only cross-check (replaces the default AutoAttack suite, which fails on tabular data).
# Runs ART's APGD-CE and APGD-DLR on a 300-row test subset and compares them with our own worst-case PGD attacks
# on the SAME rows. Also tests whether ART's APGD accepts a per-feature mask. Run after restarting the runtime.

# %% [0] Install and load
import subprocess
subprocess.run(["pip", "install", "-q", "adversarial-robustness-toolbox"], check=True)

from google.colab import drive
drive.mount('/content/drive')
import json, time, numpy as np, tensorflow as tf
from tensorflow import keras

BASE = "/content/drive/MyDrive/A3IDPS_COLAB"
d = np.load(f"{BASE}/data_processed/nslkdd_v2/nsl_v2_arrays.npz")
Xte, yte = d["Xte"].astype("float32"), d["yte"].astype("int32")
names = json.load(open(f"{BASE}/data_processed/nslkdd_v2/feature_names.json"))
MASK_NUM = np.array([0. if n.startswith(("protocol_type_", "service_", "flag_")) else 1. for n in names], "float32")
MASK_ALL = np.ones_like(MASK_NUM)
model = keras.models.load_model(f"{BASE}/models/nsl_v2/s2_none_seed0.keras", compile=False)   # logits output
D, NCLS = Xte.shape[1], 5

N = 300
sub = np.random.RandomState(0).choice(len(Xte), N, replace=False)
X, y = Xte[sub], yte[sub]
Y1 = keras.utils.to_categorical(y, NCLS)
print("clean acc on subset:", float((np.argmax(model.predict(X, verbose=0), 1) == y).mean()))

# %% [1] Our own attacks (same logic as Stage 2) on the same subset
def loss_fn(z, yy, margin):
    if margin:
        oh = tf.one_hot(yy, NCLS)
        return tf.reduce_max(z - 1e9 * oh, axis=1) - tf.reduce_sum(z * oh, axis=1)
    return tf.nn.sparse_softmax_cross_entropy_with_logits(labels=yy, logits=z)

def our_pgd(X, y, eps, mask, margin, steps=20, restarts=3):
    E = tf.constant(eps * mask); x0 = tf.constant(X); yb = tf.constant(y)
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
            best_x = tf.where(upd[:, None], x, best_x); best_l = tf.where(upd, l_new, best_l)
    return best_x.numpy()

def acc_correct(Xa):
    return np.argmax(model.predict(Xa, verbose=0), 1) == y

# %% [2] ART APGD-CE and APGD-DLR (all features), compared with our worst-case
import art
from art.estimators.classification import TensorFlowV2Classifier
from art.attacks.evasion import AutoProjectedGradientDescent
print("ART version:", art.__version__)

clf = TensorFlowV2Classifier(model=model, nb_classes=NCLS, input_shape=(D,),
                             loss_object=tf.keras.losses.CategoricalCrossentropy(from_logits=True), clip_values=(0.0, 1.0))
results = {}
for eps in (0.02, 0.05):
    row = {}
    ours_ce = acc_correct(our_pgd(X, y, eps, MASK_ALL, False)); ours_m = acc_correct(our_pgd(X, y, eps, MASK_ALL, True))
    row["ours_pgd_ce"], row["ours_pgd_margin"], row["ours_worst"] = float(ours_ce.mean()), float(ours_m.mean()), float((ours_ce & ours_m).mean())
    art_correct = {}
    for loss_type in ("cross_entropy", "difference_logits_ratio"):
        t0 = time.time()
        try:
            atk = AutoProjectedGradientDescent(estimator=clf, norm=np.inf, eps=eps, eps_step=eps / 4, max_iter=100,
                                               targeted=False, nb_random_init=2, batch_size=150, loss_type=loss_type, verbose=False)
            Xa = atk.generate(x=X, y=Y1)
            c = acc_correct(Xa); art_correct[loss_type] = c
            linf = float(np.abs(Xa - X).max())
            row[f"art_apgd_{loss_type}"] = float(c.mean())
            print(f"eps={eps} ART APGD-{loss_type}: acc={c.mean():.4f} | max|delta|={linf:.4f} (<=eps: {linf <= eps + 1e-6}) | {time.time()-t0:.0f}s")
        except Exception as ex:
            print(f"eps={eps} ART APGD-{loss_type} FAILED: {type(ex).__name__}: {ex}")
    if len(art_correct) == 2:
        row["art_worst"] = float((art_correct["cross_entropy"] & art_correct["difference_logits_ratio"]).mean())
    results[str(eps)] = row
    print(f"eps={eps} SUMMARY:", {k: round(v, 4) for k, v in row.items()})

# %% [3] Does ART's APGD accept a perturbation mask (numeric features only)?
try:
    atk = AutoProjectedGradientDescent(estimator=clf, norm=np.inf, eps=0.05, eps_step=0.0125, max_iter=50, targeted=False,
                                       nb_random_init=1, batch_size=100, loss_type="cross_entropy", verbose=False)
    n_ = 100
    Xa = atk.generate(x=X[:n_], y=Y1[:n_], mask=np.tile(MASK_NUM, (n_, 1)))
    print("mask accepted; max change on frozen (one-hot) columns:", float(np.abs((Xa - X[:n_])[:, MASK_NUM == 0]).max()))
except Exception as ex:
    print("mask NOT accepted by APGD.generate:", type(ex).__name__, ex)

json.dump(results, open(f"{BASE}/results/nsl_v2/stage2c_art_apgd_check.json", "w"), indent=1)
print("saved stage2c_art_apgd_check.json")
