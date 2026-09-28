# Stage 1 - NSL-KDD corrected pipeline (preprocessing + undefended baseline + attack sanity checks)
# Paste into Colab cell by cell (cells are separated by "# %%"). One seed. Writes only to *_v2 folders.

# %% [0] Setup
from google.colab import drive
drive.mount('/content/drive')

import os, json, random, hashlib
import numpy as np, pandas as pd, joblib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

BASE = "/content/drive/MyDrive/A3IDPS_COLAB"
RAW  = f"{BASE}/data_raw/NSL-KDD"
OUT  = f"{BASE}/data_processed/nslkdd_v2"
MOD  = f"{BASE}/models/nsl_v2"
RES  = f"{BASE}/results/nsl_v2"
for p in (OUT, MOD, RES):
    os.makedirs(p, exist_ok=True)

SPLIT_SEED = 42          # fixed data split (same for every model and seed)
SEED = 0                 # model seed (later: 0..4)
CLASS_NAMES = ["Normal", "DoS", "Probe", "R2L", "U2R"]
CLASS_WEIGHT_CAP = 20.0  # same cap for every model
BATCH = 512

def set_seed(s):
    random.seed(s); np.random.seed(s); tf.keras.utils.set_random_seed(s)

def jsonable(o):
    if isinstance(o, dict):  return {k: jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    return o

# %% [1] Load official KDDTrain+ / KDDTest+ and map labels (fails loudly on unknown labels)
COLS = ["duration","protocol_type","service","flag","src_bytes","dst_bytes","land","wrong_fragment",
        "urgent","hot","num_failed_logins","logged_in","num_compromised","root_shell","su_attempted",
        "num_root","num_file_creations","num_shells","num_access_files","num_outbound_cmds",
        "is_host_login","is_guest_login","count","srv_count","serror_rate","srv_serror_rate",
        "rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate","srv_diff_host_rate",
        "dst_host_count","dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate",
        "dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate",
        "dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate",
        "label","difficulty"]

GROUPS = {
    0: {"normal"},
    1: {"back","land","neptune","pod","smurf","teardrop","apache2","udpstorm","processtable","worm","mailbomb"},
    2: {"satan","ipsweep","nmap","portsweep","mscan","saint"},
    3: {"guess_passwd","ftp_write","imap","phf","multihop","warezmaster","warezclient","spy",
        "xlock","xsnoop","snmpguess","snmpgetattack","httptunnel","sendmail","named"},
    4: {"buffer_overflow","loadmodule","rootkit","perl","sqlattack","xterm","ps"},
}
LOOKUP = {lbl: c for c, s in GROUPS.items() for lbl in s}

def to_5class(l):
    if l not in LOOKUP:
        raise ValueError(f"unmapped attack label: {l}")
    return LOOKUP[l]

df_tr = pd.read_csv(f"{RAW}/KDDTrain+.txt", header=None, names=COLS)
df_te = pd.read_csv(f"{RAW}/KDDTest+.txt",  header=None, names=COLS)
y_all = df_tr["label"].map(to_5class).values.astype("int32")
y_te  = df_te["label"].map(to_5class).values.astype("int32")

print("train counts:", np.bincount(y_all), "| test counts:", np.bincount(y_te))
assert list(np.bincount(y_all)) == [67343, 45927, 11656, 995, 52],  "train class counts differ from expected"
assert list(np.bincount(y_te))  == [9711, 7460, 2421, 2885, 67],    "test class counts differ from expected"

# %% [2] Features: labels/difficulty are NEVER inputs; official split; min-max fit on TRAIN only
CAT = ["protocol_type", "service", "flag"]
feat_tr = df_tr.drop(columns=["label", "difficulty"])
feat_te = df_te.drop(columns=["label", "difficulty"])
assert not any(c.lower().startswith("label") or c == "difficulty" for c in feat_tr.columns)

cats = {c: sorted(feat_tr[c].unique()) for c in CAT}          # vocabulary from KDDTrain+ only
print({c: len(v) for c, v in cats.items()})

def encode(df):
    d = df.copy()
    for c in CAT:
        d[c] = pd.Categorical(d[c], categories=cats[c])
    return pd.get_dummies(d, columns=CAT, dtype=np.float32)

X_all_df, X_te_df = encode(feat_tr), encode(feat_te)
assert list(X_all_df.columns) == list(X_te_df.columns)
print("input dim:", X_all_df.shape[1])
assert X_all_df.shape[1] == 122, "expected 38 numeric + 3 + 70 + 11 = 122"

unseen = int((feat_te[CAT].apply(lambda s: ~s.isin(cats[s.name]))).any(axis=1).sum())
print("test rows with a category unseen in train:", unseen)

idx_tr, idx_va = train_test_split(np.arange(len(y_all)), test_size=0.10,
                                  stratify=y_all, random_state=SPLIT_SEED)
Xtr_raw = X_all_df.values[idx_tr]; Xva_raw = X_all_df.values[idx_va]; Xte_raw = X_te_df.values
ytr, yva = y_all[idx_tr], y_all[idx_va]

scaler = MinMaxScaler().fit(Xtr_raw)
Xtr = scaler.transform(Xtr_raw).astype("float32")
Xva = np.clip(scaler.transform(Xva_raw), 0, 1).astype("float32")
te_scaled = scaler.transform(Xte_raw)
print("share of test values outside train range (clipped to [0,1]):",
      float(((te_scaled < 0) | (te_scaled > 1)).mean()))
Xte = np.clip(te_scaled, 0, 1).astype("float32")

def hashes(X): return {hashlib.md5(r.tobytes()).hexdigest() for r in np.round(X, 6)}
overlap = len(hashes(Xte) & (hashes(Xtr) | hashes(Xva)))
print("exact test rows also present in train/val (informational):", overlap, "of", len(Xte))

np.savez_compressed(f"{OUT}/nsl_v2_arrays.npz", Xtr=Xtr, ytr=ytr, Xva=Xva, yva=yva, Xte=Xte, yte=y_te)
joblib.dump(scaler, f"{OUT}/minmax_scaler.joblib")
json.dump(list(X_all_df.columns), open(f"{OUT}/feature_names.json", "w"), indent=1)
print("splits:", Xtr.shape, Xva.shape, Xte.shape)

# %% [3] Shared architecture and training recipe (every model in the paper uses these)
def build_model(d, k=5):
    inp = keras.Input(shape=(d,))
    x = layers.Dense(256)(inp); x = layers.BatchNormalization()(x); x = layers.Activation("relu")(x)
    x = layers.Dense(128)(x);   x = layers.Dropout(0.3)(x);          x = layers.Activation("relu")(x)
    x = layers.Dense(64)(x);    x = layers.Dropout(0.3)(x);          x = layers.Activation("relu")(x)
    return keras.Model(inp, layers.Dense(k, activation="softmax")(x))

def class_weights(y):
    w = compute_class_weight("balanced", classes=np.arange(5), y=y)
    return {i: float(min(v, CLASS_WEIGHT_CAP)) for i, v in enumerate(w)}

def train(model, X, y, Xv, yv, epochs=60):
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(X, y, validation_data=(Xv, yv), epochs=epochs, batch_size=BATCH,
              class_weight=class_weights(y), verbose=2,
              callbacks=[keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True),
                         keras.callbacks.ReduceLROnPlateau(patience=4, factor=0.5)])
    return model

def evaluate(model, X, y):
    p = np.argmax(model.predict(X, batch_size=4096, verbose=0), axis=1)
    return {"acc": float(accuracy_score(y, p)),
            "macro_f1": float(f1_score(y, p, average="macro", labels=range(5), zero_division=0)),
            "per_class_f1": dict(zip(CLASS_NAMES, f1_score(y, p, average=None, labels=range(5), zero_division=0).tolist())),
            "confusion": confusion_matrix(y, p, labels=range(5)).tolist()}

# %% [4] Attacks with the box constraint [0,1] (sanity versions; full suite comes in Stage 2)
def _loss_and_grad(model, x, y):
    with tf.GradientTape() as t:
        t.watch(x)
        l = keras.losses.sparse_categorical_crossentropy(y, model(x, training=False))
        s = tf.reduce_sum(l)
    return l, t.gradient(s, x)

def fgsm(model, X, y, eps, bs=4096):
    out = []
    for i in range(0, len(X), bs):
        xb = tf.constant(X[i:i+bs]); yb = tf.constant(y[i:i+bs])
        _, g = _loss_and_grad(model, xb, yb)
        out.append(tf.clip_by_value(xb + eps * tf.sign(g), 0., 1.).numpy())
    return np.vstack(out)

def pgd(model, X, y, eps, steps=10, restarts=1, bs=4096):
    alpha = 2.5 * eps / steps
    out = []
    for i in range(0, len(X), bs):
        x0 = tf.constant(X[i:i+bs]); yb = tf.constant(y[i:i+bs])
        best, best_loss = x0, -np.inf * tf.ones(len(x0))
        for _ in range(restarts):
            x = tf.clip_by_value(x0 + tf.random.uniform(x0.shape, -eps, eps), 0., 1.)
            for _ in range(steps):
                _, g = _loss_and_grad(model, x, yb)
                x = tf.clip_by_value(tf.clip_by_value(x + alpha * tf.sign(g), x0 - eps, x0 + eps), 0., 1.)
            l = keras.losses.sparse_categorical_crossentropy(yb, model(x, training=False))
            better = l > best_loss
            best = tf.where(better[:, None], x, best); best_loss = tf.where(better, l, best_loss)
        out.append(best.numpy())
    return np.vstack(out)

# %% [5] Train the undefended baseline (test set is never touched during training)
set_seed(SEED)
model = build_model(Xtr.shape[1])
print("parameters (total):", model.count_params())
train(model, Xtr, ytr, Xva, yva)
model.save(f"{MOD}/undefended_seed{SEED}.keras")

clean = evaluate(model, Xte, y_te)
print("CLEAN test:", {k: clean[k] for k in ("acc", "macro_f1")}, clean["per_class_f1"])

# %% [6] Attack sanity checks (Carlini et al. red flags)
EPS = [0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.0]
rows = []
for e in EPS:
    a_f = accuracy_score(y_te, np.argmax(model.predict(fgsm(model, Xte, y_te, e), batch_size=4096, verbose=0), 1)) if e else clean["acc"]
    a_p = accuracy_score(y_te, np.argmax(model.predict(pgd(model, Xte, y_te, e, steps=10, restarts=1), batch_size=4096, verbose=0), 1)) if e else clean["acc"]
    rows.append({"eps": e, "fgsm_acc": float(a_f), "pgd10_acc": float(a_p)})
    print(f"eps={e:<5} FGSM acc={a_f:.4f}   PGD-10 acc={a_p:.4f}")

flags = []
for i in range(len(rows) - 1):
    for j in range(i + 1, len(rows)):
        if rows[j]["fgsm_acc"] > rows[i]["fgsm_acc"] + 0.01:
            flags.append(f"FGSM accuracy RISES from eps={rows[i]['eps']} to {rows[j]['eps']}")
            break
for r in rows:
    if r["pgd10_acc"] > r["fgsm_acc"] + 0.02:
        flags.append(f"PGD weaker than FGSM at eps={r['eps']}")
if rows[-1]["pgd10_acc"] > 0.05:
    flags.append("PGD at eps=1.0 does not drive accuracy near 0")
print("RED FLAGS:", flags if flags else "none")

json.dump(jsonable({"seed": SEED, "params": model.count_params(), "clean": clean,
                    "attack_sanity": rows, "flags": flags,
                    "test_rows_also_in_train": overlap, "unseen_category_rows": unseen}),
          open(f"{RES}/stage1_undefended_seed{SEED}.json", "w"), indent=1)
print("saved:", f"{RES}/stage1_undefended_seed{SEED}.json")
