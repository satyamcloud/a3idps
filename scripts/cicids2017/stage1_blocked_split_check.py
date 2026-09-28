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
