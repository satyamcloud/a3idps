# CICIDS Stage 1 - FIX for cell [4]. Run this in place of the failing cell (everything before it, i.e. cells [0]-[3],
# has already run successfully and X_raw / idx_tr / idx_va / idx_te / y_tr / y_va / y_te are still in memory).
# Fix: clip X_shifted to >= 0 BEFORE log1p. Values below the train-observed minimum (only possible in val/test, since
# shift is computed from train) are floored to 0 post-shift, i.e. treated as equal to the smallest value seen in
# training for that feature - the same kind of out-of-range handling as the [0,1] clip after MinMaxScaler, just earlier
# in the pipeline. This does not use any val/test information to choose the shift itself.

import numpy as np

shift = np.minimum(X_raw[idx_tr].min(axis=0), 0.0)
X_shifted = np.maximum(X_raw - shift, 0.0)          # <-- the fix: floor at 0 instead of letting it go negative
X_log = np.log1p(X_shifted)
assert np.isfinite(X_log).all(), "still non-finite after clipping - inspect X_raw for inf/extremely large values"

n_floored_va = int((X_raw[idx_va] - shift < 0).sum())
n_floored_te = int((X_raw[idx_te] - shift < 0).sum())
print(f"val cells floored to train-min (out of {X_raw[idx_va].size}): {n_floored_va} ({100*n_floored_va/X_raw[idx_va].size:.4f}%)")
print(f"test cells floored to train-min (out of {X_raw[idx_te].size}): {n_floored_te} ({100*n_floored_te/X_raw[idx_te].size:.4f}%)")
print("log-transform OK, X_log is finite. Continue from cell [4]'s scaler.fit(...) line onward.")

from sklearn.preprocessing import MinMaxScaler
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
print("shapes:", Xtr.shape, Xva.shape, Xte.shape)
print("\nNow re-run the ORIGINAL cell [4]'s save lines (np.savez_compressed / joblib.dump / json.dump), "
      "then continue to cell [5] (the blocked split) as originally written.")
