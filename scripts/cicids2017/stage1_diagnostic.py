# Diagnostic - run this BEFORE re-attempting the log-transform fix, to find out exactly what's non-finite.
# Assumes X_raw, idx_tr, feature_cols are still in memory from the earlier cells.
import numpy as np

print("X_raw dtype:", X_raw.dtype, "shape:", X_raw.shape)
print("any +inf in X_raw:", bool(np.isinf(X_raw).any()), "| count:", int(np.isinf(X_raw).sum()))
print("any nan in X_raw:", bool(np.isnan(X_raw).any()), "| count:", int(np.isnan(X_raw).sum()))
print("X_raw min:", float(np.nanmin(X_raw)), "| X_raw max:", float(np.nanmax(X_raw)))

if np.isinf(X_raw).any():
    rows_inf, cols_inf = np.where(np.isinf(X_raw))
    bad_cols = sorted(set(cols_inf.tolist()))
    print("\ncolumns containing inf:", [feature_cols[c] for c in bad_cols[:20]], "..." if len(bad_cols) > 20 else "")
    print("total inf cells:", len(rows_inf))

shift = np.minimum(X_raw[idx_tr].min(axis=0), 0.0)
X_shifted = X_raw - shift
neg_after_shift = X_shifted < 0
print("\ncells still negative after shifting by train-min (expected only in val/test, if any):", int(neg_after_shift.sum()))
print("most negative value after shift:", float(X_shifted.min()))

# Which is it: inf in X_raw itself, or just values below the train shift (which my fix handles)?
if np.isinf(X_raw).any():
    print("\n==> DIAGNOSIS: X_raw itself contains literal inf. The earlier inf->nan->dropna cleaning step did not remove all of them.")
    print("    This needs a different fix: re-check the cleaning step, or clip X_raw to a large finite value before log-transform.")
elif np.isnan(X_shifted).any() or (X_shifted < -1).any():
    print("\n==> DIAGNOSIS: some val/test values are far enough below the train minimum that shifting still leaves them < -1,")
    print("    so log1p(x) for x < -1 is NaN even before my floor-at-0 fix would apply (the floor-at-0 fix WOULD handle this -")
    print("    if you are still seeing the error, you are running the OLD unfixed cell, not the corrected one).")
else:
    print("\n==> No inf/nan found under this check - if the assertion is still firing, you are almost certainly running the old,")
    print("    unfixed version of the cell. Search your notebook for 'X_shifted = X_raw - shift' WITHOUT 'np.maximum' around it.")
