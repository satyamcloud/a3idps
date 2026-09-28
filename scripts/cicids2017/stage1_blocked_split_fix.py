# CICIDS Stage 1 - FIX for the blocked-split scaler in cell [5] (same bug as cell [4]: values below the
# blocked-train minimum went negative enough that log1p produced NaN). Run this to redo just the blocked-split
# scaling and re-save cicids_v2_blocked.npz correctly. Assumes X_raw, y, blocked_tr/va/te are still in memory.

import numpy as np, json, joblib

BASE = "/content/drive/MyDrive/A3IDPS_COLAB"
OUT  = f"{BASE}/data_processed/cicids_v2"

shift_b = np.minimum(X_raw[blocked_tr].min(axis=0), 0.0)
X_shifted_b = np.maximum(X_raw - shift_b, 0.0)          # <-- the same fix: floor at 0 before log1p
X_log_b = np.log1p(X_shifted_b)
assert np.isfinite(X_log_b).all(), "still non-finite in the blocked split - inspect further"

n_floored = int((X_raw[blocked_te] - shift_b < 0).sum())
print(f"blocked-test cells floored to blocked-train-min: {n_floored} (out of {X_raw[blocked_te].size})")

from sklearn.preprocessing import MinMaxScaler
scaler_b = MinMaxScaler().fit(X_log_b[blocked_tr])
X_all_b = scaler_b.transform(X_log_b).astype("float32")
frac_out = float(((X_all_b[blocked_te] < 0) | (X_all_b[blocked_te] > 1)).mean())
print(f"share of blocked-test values outside [0,1] before clipping: {frac_out:.6f}")
X_all_b = np.clip(X_all_b, 0.0, 1.0)
assert 0.0 <= X_all_b.min() and X_all_b.max() <= 1.0
print("blocked post-scaling range check OK: min", float(X_all_b.min()), "max", float(X_all_b.max()))

np.savez_compressed(f"{OUT}/cicids_v2_blocked.npz",
                    Xtr=X_all_b[blocked_tr], ytr=y[blocked_tr],
                    Xva=X_all_b[blocked_va], yva=y[blocked_va],
                    Xte=X_all_b[blocked_te], yte=y[blocked_te])
joblib.dump(scaler_b, f"{OUT}/log_minmax_scaler_blocked.joblib")
np.save(f"{OUT}/log_shift_blocked.npy", shift_b)
print("re-saved cicids_v2_blocked.npz, log_minmax_scaler_blocked.joblib, log_shift_blocked.npy correctly.")

# sanity re-check: reload from disk and confirm no NaN made it into the saved file
chk = np.load(f"{OUT}/cicids_v2_blocked.npz")
for k in ("Xtr", "Xva", "Xte"):
    assert np.isfinite(chk[k]).all(), f"{k} in the saved file still has non-finite values!"
print("re-load check passed: saved blocked-split arrays are all finite.")
