# CICIDS2017 v2 - Stage 1: corrected preprocessing.
# Fixes vs the old pipeline: features are actually normalised (old pipeline had values up to 2.07e9, 54.8% outside [0,1]);
# a stratified row-level split so BENIGN is ~80.3% in every split and all 5 classes appear in test (reviewers' complaint);
# exact-duplicate removal before splitting (CICIDS2017 has many near-identical rows per attack session - Engelen et al. [40]);
# a second, blocked/contiguous split kept as a leakage robustness check, reported alongside the primary result, not instead of it.
# Paste into Colab. Needs the raw CICIDS2017 CSVs already available at data_raw/CICIDS2017 (same as your original pipeline).

# %% [0] Setup
from google.colab import drive
drive.mount('/content/drive')
import os, glob, json, random, hashlib
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import joblib

BASE = "/content/drive/MyDrive/A3IDPS_COLAB"
RAW  = f"{BASE}/data_raw/CICIDS2017"
OUT  = f"{BASE}/data_processed/cicids_v2"
os.makedirs(OUT, exist_ok=True)

SPLIT_SEED = 42
CLASS_NAMES = ["BENIGN", "Brute Force", "DoS/DDoS", "PortScan", "Other"]
DROP_COLS = {"Flow ID", "Src IP", "Source IP", "Dst IP", "Destination IP", "Timestamp", "__source_file__"}

def map_5class(label):
    if label == "BENIGN": return 0
    if label in ("FTP-Patator", "SSH-Patator"): return 1
    if label in ("DoS Hulk", "DoS GoldenEye", "DoS slowloris", "DoS Slowhttptest", "DDoS"): return 2
    if label == "PortScan": return 3
    return 4   # Bot, Web Attack *, Infiltration, Heartbleed

# %% [1] Load all raw CSVs, clean, label -- NO leakage columns, NO scaling yet (fit must happen on train only)
csvs = sorted(glob.glob(RAW + "/**/*.csv", recursive=True))
print("raw CSVs found:", len(csvs))
assert csvs, f"no CSVs under {RAW}"

chunks = []
for path in csvs:
    print("reading:", os.path.basename(path))
    try:
        df = pd.read_csv(path, low_memory=False, encoding_errors="ignore")
    except TypeError:
        df = pd.read_csv(path, low_memory=False, encoding="latin1")
    df.columns = [c.strip() for c in df.columns]
    if "Label" not in df.columns:
        print("  skipped (no Label column)"); continue
    df = df.drop(columns=[c for c in df.columns if c in DROP_COLS], errors="ignore")
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df["label_5class"] = df["Label"].map(map_5class).astype("int8")
    chunks.append(df.drop(columns=["Label"]))

full = pd.concat(chunks, ignore_index=True)
del chunks
print("total rows after cleaning:", len(full))
print("class counts:", full["label_5class"].value_counts().sort_index().to_dict())

feature_cols = [c for c in full.columns if c != "label_5class"]
for c in feature_cols:
    full[c] = pd.to_numeric(full[c], errors="coerce")
before = len(full)
full = full.dropna()
print(f"dropped {before - len(full)} rows with non-numeric feature values")

# %% [2] Remove exact duplicate feature rows (keep first occurrence; report how many were dropped, per class)
before = len(full)
dup_mask = full.duplicated(subset=feature_cols, keep="first")
dup_by_class = full.loc[dup_mask, "label_5class"].value_counts().sort_index().to_dict()
full = full.loc[~dup_mask].reset_index(drop=True)
print(f"removed {before - len(full)} exact-duplicate rows ({100*(before-len(full))/before:.1f}%); "
      f"by class: {dup_by_class}")
print("class counts after dedup:", full["label_5class"].value_counts().sort_index().to_dict())

y = full["label_5class"].values.astype("int32")
X_raw = full[feature_cols].values.astype("float64")   # float64 during log-transform to avoid overflow on huge raw values
del full

# %% [3] PRIMARY split: stratified, row-level, 70/15/15 (answers the reviewers' "describe the actual partitioning")
idx = np.arange(len(y))
idx_tr, idx_tmp, y_tr, y_tmp = train_test_split(idx, y, test_size=0.30, random_state=SPLIT_SEED, stratify=y)
idx_va, idx_te, y_va, y_te = train_test_split(idx_tmp, y_tmp, test_size=0.50, random_state=SPLIT_SEED, stratify=y_tmp)
print("primary split sizes:", len(idx_tr), len(idx_va), len(idx_te))
for name, yy in (("train", y_tr), ("val", y_va), ("test", y_te)):
    props = {CLASS_NAMES[c]: f"{100*np.mean(yy==c):.2f}%" for c in range(5)}
    print(f"  {name}: {props}")

# %% [4] Log-transform heavy-tailed features, then min-max scale, fit on TRAIN only
# log1p on the shifted-nonnegative feature (features can be negative here only due to upstream tool artefacts in a few columns)
shift = np.minimum(X_raw[idx_tr].min(axis=0), 0.0)          # per-feature: 0 if already >=0, else the negative min on train
X_shifted = X_raw - shift
X_log = np.log1p(X_shifted)
assert np.isfinite(X_log).all(), "non-finite values after log-transform - check DROP_COLS / raw data"

scaler = MinMaxScaler().fit(X_log[idx_tr])
X_all = scaler.transform(X_log).astype("float32")
frac_clip_va = float(((X_all[idx_va] < 0) | (X_all[idx_va] > 1)).mean())
frac_clip_te = float(((X_all[idx_te] < 0) | (X_all[idx_te] > 1)).mean())
print(f"share of val values outside [0,1] before clipping: {frac_clip_va:.6f}")
print(f"share of test values outside [0,1] before clipping: {frac_clip_te:.6f}")
X_all = np.clip(X_all, 0.0, 1.0)
print("post-scaling range check: min", float(X_all.min()), "max", float(X_all.max()))
assert 0.0 <= X_all.min() and X_all.max() <= 1.0

Xtr, Xva, Xte = X_all[idx_tr], X_all[idx_va], X_all[idx_te]

def hashes(X): return {hashlib.md5(r.tobytes()).hexdigest() for r in np.round(X, 6)}
overlap_te = len(hashes(Xte) & (hashes(Xtr) | hashes(Xva)))
print("exact test rows (post-scaling) also present in train/val (informational, dedup was pre-split):", overlap_te, "of", len(Xte))

np.savez_compressed(f"{OUT}/cicids_v2_primary.npz", Xtr=Xtr, ytr=y_tr, Xva=Xva, yva=y_va, Xte=Xte, yte=y_te)
joblib.dump(scaler, f"{OUT}/log_minmax_scaler.joblib")
np.save(f"{OUT}/log_shift.npy", shift)
json.dump(feature_cols, open(f"{OUT}/feature_names.json", "w"), indent=1)
print(f"input dim: {len(feature_cols)}")

# %% [5] SECONDARY split: blocked/contiguous per class, on ORIGINAL file order - the leakage robustness check
# Within each class, take the rows in the order they appear in the concatenated raw files (a rough time proxy),
# split 70/15/15 in blocks with a gap discarded between train and test, so temporally adjacent (near-duplicate) flows
# cannot straddle the train/test boundary the way a random split allows.
GAP_FRAC = 0.02   # discard 2% of each class's rows at each block boundary
blocked_tr, blocked_va, blocked_te = [], [], []
for c in range(5):
    rows = np.where(y == c)[0]         # already in original file order
    n = len(rows)
    if n < 20:
        print(f"class {CLASS_NAMES[c]}: only {n} rows, too few to block-split; assigning all to test only")
        blocked_te.extend(rows.tolist()); continue
    gap = max(1, int(GAP_FRAC * n))
    t_end = int(0.70 * n)
    v_end = t_end + gap + int(0.15 * n)
    blocked_tr.extend(rows[:t_end].tolist())
    blocked_va.extend(rows[t_end + gap: v_end].tolist())
    blocked_te.extend(rows[v_end + gap:].tolist())

blocked_tr, blocked_va, blocked_te = map(np.array, (blocked_tr, blocked_va, blocked_te))
print("\nblocked split sizes:", len(blocked_tr), len(blocked_va), len(blocked_te))
for name, ii in (("train", blocked_tr), ("val", blocked_va), ("test", blocked_te)):
    yy = y[ii]
    props = {CLASS_NAMES[c]: f"{100*np.mean(yy==c):.2f}%" for c in range(5)} if len(yy) else {}
    print(f"  {name}: n={len(ii)} {props}")

# Fit a SEPARATE scaler for the blocked split (its train rows differ from the primary split's train rows)
shift_b = np.minimum(X_raw[blocked_tr].min(axis=0), 0.0)
X_log_b = np.log1p(X_raw - shift_b)
scaler_b = MinMaxScaler().fit(X_log_b[blocked_tr])
X_all_b = np.clip(scaler_b.transform(X_log_b), 0.0, 1.0).astype("float32")

np.savez_compressed(f"{OUT}/cicids_v2_blocked.npz",
                    Xtr=X_all_b[blocked_tr], ytr=y[blocked_tr],
                    Xva=X_all_b[blocked_va], yva=y[blocked_va],
                    Xte=X_all_b[blocked_te], yte=y[blocked_te])
joblib.dump(scaler_b, f"{OUT}/log_minmax_scaler_blocked.joblib")
np.save(f"{OUT}/log_shift_blocked.npy", shift_b)

print("\nSaved:")
print(" ", f"{OUT}/cicids_v2_primary.npz", " (main protocol: stratified row-level)")
print(" ", f"{OUT}/cicids_v2_blocked.npz", " (leakage robustness check: blocked/contiguous)")
print(" ", f"{OUT}/feature_names.json")
