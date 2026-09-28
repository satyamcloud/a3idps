# Read-only checks. Run in a fresh Colab notebook in the account that owns A3IDPS_COLAB.
from google.colab import drive
drive.mount('/content/drive')

import json, joblib, glob
import numpy as np, pandas as pd

BASE = "/content/drive/MyDrive/A3IDPS_COLAB"
NSL = f"{BASE}/data_processed/nslkdd"

# 1) Is the label an input column in NSL-KDD?
cols = json.load(open(f"{NSL}/nsl_feature_list.json"))
print("NSL input dim:", len(cols), "| label_5class in inputs:", "label_5class" in cols)
if "label_5class" in cols:
    i = cols.index("label_5class")
    sc = joblib.load(f"{NSL}/nsl_scaler.joblib")
    X_te = np.load(f"{NSL}/X_test.npy"); y_te = np.load(f"{NSL}/y_test.npy")
    recovered = np.round(X_te[:, i] * sc.scale_[i] + sc.mean_[i])
    print("column index:", i, "| test rows where that column != true label:",
          int((recovered != y_te).sum()), "of", len(y_te))

# 2) mailbomb rows in raw KDDTest+ (mapped to U2R by the notebook's label function)
te = pd.read_csv(f"{BASE}/data_raw/NSL-KDD/KDDTest+.txt", header=None)
print("mailbomb rows in KDDTest+:", int((te[41] == "mailbomb").sum()))

# 3) Are CICIDS features in [0,1]? Read one processed shard directly.
feat = json.load(open(f"{BASE}/artifacts/feature_list.json"))
shard = sorted(glob.glob(f"{BASE}/data_processed/cicids_clean_shards_full/*.csv"))[0]
X = pd.read_csv(shard, usecols=feat, nrows=200000).values.astype("float32")
print("CICIDS shard min/max:", float(X.min()), float(X.max()),
      "| share of values outside [0,1]:", float(((X < 0) | (X > 1)).mean()))
