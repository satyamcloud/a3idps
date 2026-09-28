# CICIDS Stage 2 - FIX: add SMOTE (as originally agreed in the spec, which I omitted the first time), and
# reduce the initial learning rate to guard against the same early-training instability. SMOTE is applied to
# the TRAINING split only, on the already log+minmax-scaled features (never touches val/test).
# Run this as a replacement for cells [2] onward of cicids_stage2_undefended.py (cell [0]/[1] - data loading and
# model/attack definitions - should already be in memory; if not, re-run those first).

# %% [A] Install imblearn if needed, then apply SMOTE to the training split only
import subprocess
subprocess.run(["pip", "install", "-q", "imbalanced-learn"], check=True)
from imblearn.over_sampling import SMOTE
import numpy as np

print("train class counts before SMOTE:", np.bincount(ytr))

# Target: bring the two smallest classes (Brute Force, Other) up to a workable size; leave BENIGN, DoS/DDoS,
# PortScan alone (already large enough that the model should be able to learn them from class weights alone).
TARGET_PER_CLASS = 50_000
counts = np.bincount(ytr)
strategy = {c: TARGET_PER_CLASS for c in range(len(counts)) if counts[c] < TARGET_PER_CLASS}
print("SMOTE strategy (classes to oversample -> target count):", strategy)

sm = SMOTE(random_state=0, k_neighbors=5, sampling_strategy=strategy)
Xtr_sm, ytr_sm = sm.fit_resample(Xtr, ytr)
print("train class counts after SMOTE:", np.bincount(ytr_sm))
print("SMOTE output range check: min", float(Xtr_sm.min()), "max", float(Xtr_sm.max()))
Xtr_sm = np.clip(Xtr_sm, 0.0, 1.0).astype("float32")   # SMOTE interpolation can occasionally land just outside [0,1]
ytr_sm = ytr_sm.astype("int32")

# %% [B] Recompute class weights on the SMOTE'd training set (should now need a much smaller cap, since the
# extreme classes are no longer extreme) and retrain with a lower initial learning rate
from sklearn.utils.class_weight import compute_class_weight
cw_sm = np.minimum(compute_class_weight("balanced", classes=np.arange(NCLS), y=ytr_sm), CLASS_WEIGHT_CAP).astype("float32")
print("class weights after SMOTE:", cw_sm.tolist())

set_seed(SEED)
model2 = build_model(D)
model2.compile(optimizer=keras.optimizers.Adam(3e-4),           # lower LR than before (was 1e-3)
               loss="sparse_categorical_crossentropy", metrics=["accuracy"])
cw_dict2 = {i: float(w) for i, w in enumerate(cw_sm)}
hist2 = model2.fit(Xtr_sm, ytr_sm, validation_data=(Xva, yva), epochs=MAX_EPOCHS, batch_size=BATCH,
                   class_weight=cw_dict2, verbose=2,
                   callbacks=[keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True),
                              keras.callbacks.ReduceLROnPlateau(patience=4, factor=0.5)])

# %% [C] Evaluate: check the model no longer collapses to majority-only before trusting anything else
clean2 = clean_eval(model2, Xte, yte)
print("\nCLEAN test (SMOTE + lower LR):", {k: clean2[k] for k in ("acc", "macro_f1")})
print("per-class F1:", clean2["per_class_f1"])

collapsed = all(clean2["per_class_f1"][c] == 0.0 for c in ("Brute Force", "DoS/DDoS", "PortScan", "Other"))
if collapsed:
    print("\n*** STILL COLLAPSED: every minority class F1 is 0.0. Do NOT proceed - paste this output back before continuing. ***")
else:
    print("\nModel is no longer predicting a single class - proceeding looks reasonable. Per-class support was:",
          clean2["per_class_support"])
    model2.save(f"{MOD}/undefended_smote_seed{SEED}.keras")
    print("saved:", f"{MOD}/undefended_smote_seed{SEED}.keras")
