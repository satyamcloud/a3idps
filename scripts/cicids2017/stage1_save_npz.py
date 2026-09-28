np.savez_compressed(f"{OUT}/cicids_v2_primary.npz", Xtr=Xtr, ytr=y_tr, Xva=Xva, yva=y_va, Xte=Xte, yte=y_te)
joblib.dump(scaler, f"{OUT}/log_minmax_scaler.joblib")
np.save(f"{OUT}/log_shift.npy", shift)
json.dump(feature_cols, open(f"{OUT}/feature_names.json", "w"), indent=1)
print(f"input dim: {len(feature_cols)}")


