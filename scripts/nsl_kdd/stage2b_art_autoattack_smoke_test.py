# Stage 2b - ART AutoAttack smoke test. Run AFTER nsl_stage2.py has trained the undefended model.
# Purpose: does ART's AutoAttack run cleanly here, and is it at least as strong as our PGD worst-case?
# If anything below errors, paste the traceback: the fallback is our own APGD-CE + APGD-DLR in TensorFlow.

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
model = keras.models.load_model(f"{BASE}/models/nsl_v2/s2_none_seed0.keras", compile=False)   # outputs logits
D = Xte.shape[1]

rng = np.random.RandomState(0)
sub = rng.choice(len(Xte), 1000, replace=False)
X, y = Xte[sub], yte[sub]

# %% [1] Wrap the model for ART
import art
from art.estimators.classification import TensorFlowV2Classifier
from art.attacks.evasion import AutoAttack
print("ART version:", art.__version__)

loss_obj = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
clf = TensorFlowV2Classifier(model=model, nb_classes=5, input_shape=(D,), loss_object=loss_obj,
                             clip_values=(0.0, 1.0))
p_art = np.argmax(clf.predict(X), 1)
p_our = np.argmax(model.predict(X, verbose=0), 1)
print("ART vs our predictions agree:", bool((p_art == p_our).all()), "| clean acc on subset:", float((p_our == y).mean()))
Y1 = keras.utils.to_categorical(y, 5)

# %% [2] AutoAttack, all features (ART does not know about our mask)
ref = json.load(open(f"{BASE}/results/nsl_v2/stage1b_attack_sanity_seed0.json"))["all_features"]["rows"]
for eps in (0.02, 0.05):
    t0 = time.time()
    try:
        aa = AutoAttack(estimator=clf, norm=np.inf, eps=eps, eps_step=eps / 4, batch_size=256)
        Xa = aa.generate(x=X, y=Y1)
        acc = float((np.argmax(clf.predict(Xa), 1) == y).mean())
        linf = float(np.abs(Xa - X).max())
        ok_box = bool((Xa >= 0).all() and (Xa <= 1).all())
        ref_w = [r["worst_case"] for r in ref if abs(r["eps"] - eps) < 1e-9][0]
        print(f"eps={eps}: AutoAttack acc={acc:.4f} | max|delta|={linf:.4f} (<= eps: {linf <= eps + 1e-6}) | in [0,1]: {ok_box} "
              f"| our Stage-1b worst-case on full test (undefended v1 model)={ref_w:.4f} | {time.time()-t0:.0f}s")
    except Exception as ex:
        print(f"eps={eps}: AutoAttack FAILED: {type(ex).__name__}: {ex}")

# %% [3] Does AutoAttack accept a perturbation mask (numeric features only)?
try:
    aa = AutoAttack(estimator=clf, norm=np.inf, eps=0.05, eps_step=0.0125, batch_size=256)
    Xa = aa.generate(x=X[:200], y=Y1[:200], mask=np.tile(MASK_NUM, (200, 1)))
    frozen = float(np.abs((Xa - X[:200])[:, MASK_NUM == 0]).max())
    print("mask accepted; max change on frozen (one-hot) columns:", frozen)
except Exception as ex:
    print("mask NOT supported by AutoAttack.generate:", type(ex).__name__, ex)
print("Done.")

