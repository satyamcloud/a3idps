# CICIDS Stage 2 - completion: attack sanity checks + blocked-split check on the CORRECTED (SMOTE'd) undefended
# model, replacing the stale results from the collapsed run. Needs Xte, yte, MASK_ALL, make_attacks, run_attack,
# predict, clean_eval, robust_eval, CLASS_NAMES, EPS_EVAL, RES, SEED still in memory from cicids_stage2_undefended.py,
# plus model2 (or reload from disk) from the SMOTE fix.

# %% [0] Load the corrected model (reload from disk in case the session was restarted)
import json, numpy as np
from tensorflow import keras

BASE = "/content/drive/MyDrive/A3IDPS_COLAB"
MOD = f"{BASE}/models/cicids_v2"; RES = f"{BASE}/results/cicids_v2"
model2 = keras.models.load_model(f"{MOD}/undefended_smote_seed{SEED}.keras", compile=False)

clean2 = clean_eval(model2, Xte, yte)
print("re-confirmed CLEAN test:", {k: clean2[k] for k in ("acc", "macro_f1")})
assert clean2["per_class_f1"]["Brute Force"] > 0.0 and clean2["per_class_f1"]["DoS/DDoS"] > 0.0, \
    "loaded model looks collapsed again - stop and check which model file this is"

# %% [1] Attack sanity checks (Carlini et al. red flags), same eps grid as before, on the corrected model
rows = []
for e in [0.0] + EPS_EVAL:
    if e == 0:
        rows.append({"eps": 0.0, "fgsm": clean2["acc"], "pgd_ce": clean2["acc"], "pgd_margin": clean2["acc"], "worst": clean2["acc"]}); continue
    A = make_attacks(model2, MASK_ALL)
    import tensorflow as tf
    c_f = predict(model2, run_attack(A["fgsm"], Xte, yte, eps=tf.constant(e, tf.float32))) == yte
    c_c = predict(model2, run_attack(A["pgd"], Xte, yte, eps=tf.constant(e, tf.float32), steps=20, restarts=3, margin=False)) == yte
    c_m = predict(model2, run_attack(A["pgd"], Xte, yte, eps=tf.constant(e, tf.float32), steps=20, restarts=3, margin=True)) == yte
    w = c_f & c_c & c_m
    rows.append({"eps": e, "fgsm": float(c_f.mean()), "pgd_ce": float(c_c.mean()), "pgd_margin": float(c_m.mean()), "worst": float(w.mean())})
    print(f"eps={e:<5} FGSM={rows[-1]['fgsm']:.4f}  PGD-CE={rows[-1]['pgd_ce']:.4f}  PGD-margin={rows[-1]['pgd_margin']:.4f}  worst={rows[-1]['worst']:.4f}")

flags = []
for i in range(len(rows) - 1):
    if rows[i + 1]["worst"] > rows[i]["worst"] + 0.01:
        flags.append(f"worst-case accuracy rises from eps={rows[i]['eps']} to {rows[i+1]['eps']}")
    if rows[i + 1]["pgd_ce"] > rows[i]["fgsm"] + 0.02:
        flags.append(f"PGD-CE weaker than FGSM going from eps={rows[i]['eps']} to {rows[i+1]['eps']}")
if rows[-1]["worst"] > 0.20:
    flags.append(f"worst-case at eps={rows[-1]['eps']} (largest tested) still above 20% - may need a larger eps, or this may just reflect the BENIGN majority floor")
print("RED FLAGS:", flags if flags else "none")

# %% [2] Robustness eval at the standard eps grid, with per-class recall (the number that actually matters for an IDS)
primary_robust = robust_eval(model2, Xte, yte, MASK_ALL, EPS_EVAL)
for e in EPS_EVAL:
    r = primary_robust[str(e)]
    print(f"eps={e}: worst={r['worst']:.4f} | per-class recall under worst-case attack: {r['worst_recall']}")

# %% [3] Blocked-split check on the corrected model
db = np.load(f"{BASE}/data_processed/cicids_v2/cicids_v2_blocked.npz")
Xte_b, yte_b = db["Xte"].astype("float32"), db["yte"].astype("int32")
clean_b = clean_eval(model2, Xte_b, yte_b)
print("\nCLEAN on blocked test:", {k: clean_b[k] for k in ("acc", "macro_f1")}, "| per-class F1:", clean_b["per_class_f1"])
blocked_robust = robust_eval(model2, Xte_b, yte_b, MASK_ALL, [0.05, 0.10])
for e in (0.05, 0.10):
    print(f"blocked test, eps={e}: worst={blocked_robust[str(e)]['worst']:.4f}  (primary test at same eps: {primary_robust[str(e)]['worst']:.4f})")
gap = abs(clean_b["acc"] - clean2["acc"])
macro_gap = abs(clean_b["macro_f1"] - clean2["macro_f1"])
print(f"clean accuracy gap (primary vs blocked): {gap:.4f} | macro-F1 gap: {macro_gap:.4f}")
if gap > 0.05 or macro_gap > 0.05:
    print("NOTE: meaningful gap between primary and blocked test performance - discuss in Limitations "
          "(possible temporal / near-duplicate leakage in the row-level primary split).")
else:
    print("No strong evidence of leakage inflating the primary split's numbers.")

# %% [4] Save (overwrites the stale collapsed-model JSON)
def jsonable(o):
    if isinstance(o, dict): return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, np.floating): return float(o)
    return o

out = {
    "note": "corrected: trained with SMOTE (Brute Force, Other -> 50000) + lr=3e-4, replacing the collapsed run that used class weights only",
    "params": model2.count_params(),
    "clean_primary": clean2,
    "attack_sanity_primary": rows,
    "flags": flags,
    "robust_primary": primary_robust,
    "clean_blocked": clean_b,
    "robust_blocked": blocked_robust,
    "clean_accuracy_gap_primary_vs_blocked": gap,
    "macro_f1_gap_primary_vs_blocked": macro_gap,
}
json.dump(jsonable(out), open(f"{RES}/stage2_undefended_seed{SEED}.json", "w"), indent=1)
print("\nsaved (overwritten):", f"{RES}/stage2_undefended_seed{SEED}.json")
