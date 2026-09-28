# Faithful port of scripts/cicids2017/stage4_final_correction.py to NSL-KDD's schema:
#   - NSL's per-seed worst-case lives under tag -> "numeric_only" -> eps -> "worst" (not "robust")
#   - NSL's AT baseline tags are literally "fgsm_at" / "pgd_at" (no eps suffix)
#   - NSL's stage4b sweep keys are flat "tag|seed|epsE|config" strings, not nested dicts
import json
import numpy as np

SEEDS = [0, 1, 2, 3, 4]
BASE = "/tmp/nsl_agg"

s4a = {s: json.load(open(f"{BASE}/stage4a_seed{s}.json")) for s in SEEDS}
s4b = json.load(open(f"{BASE}/stage4b_advgan_attack.json"))

S4B_TAGS = {"undefended", "fgsm_at", "pgd_at", "kd_feedback_r5"}
CORRECTED_EPS = ["0.05", "0.1"]
EPS_KEY_TO_S4B = {"0.05": "eps0.05", "0.1": "eps0.1"}

ALL_TAGS = list(s4a[0].keys())
EPS_ALL = sorted(s4a[0][ALL_TAGS[0]]["numeric_only"].keys(), key=float)

def s4a_worst(seed, tag, eps):
    return s4a[seed][tag]["numeric_only"][eps]["worst"]

def best_advgan_acc(seed, tag, eps):
    s4b_eps = EPS_KEY_TO_S4B[eps]
    prefix = f"{tag}|seed{seed}|{s4b_eps}|"
    vals = [v["acc_under_generator"] for k, v in s4b.items() if k.startswith(prefix)]
    assert len(vals) == 8, f"expected 8 configs, got {len(vals)} for {prefix}"
    return min(vals)

corrections = []
corrected = {tag: {eps: [] for eps in EPS_ALL} for tag in ALL_TAGS}

for tag in ALL_TAGS:
    s4b_tag = tag if tag in S4B_TAGS else None
    for eps in EPS_ALL:
        for seed in SEEDS:
            wb = s4a_worst(seed, tag, eps)
            if s4b_tag is not None and eps in CORRECTED_EPS:
                adv = best_advgan_acc(seed, s4b_tag, eps)
                final = min(wb, adv)
                if adv < wb - 1e-9:
                    corrections.append({"seed": seed, "tag": tag, "eps": eps,
                                        "white_box_worst": wb, "advgan_acc": adv, "corrected_worst": final})
            else:
                final = wb
            corrected[tag][eps].append(final)

print(f"=== NSL-KDD FINAL 5-SEED TEST RESULTS, corrected worst-case = min(white-box, AdvGAN-sweep) where both exist ===")
print(f"{'model':<24}{'clean':>12}" + "".join(f"{('e='+e):>13}" for e in EPS_ALL))
agg_table = {}
for tag in ALL_TAGS:
    clean_vals = [s4a[s][tag]["clean"]["acc"] for s in SEEDS]
    cm, cs = float(np.mean(clean_vals)), float(np.std(clean_vals))
    row = {"clean": (cm, cs)}
    line = f"{tag:<24}{cm*100:>8.1f}+-{cs*100:<3.1f}"
    for eps in EPS_ALL:
        vals = corrected[tag][eps]
        m_, s_ = float(np.mean(vals)), float(np.std(vals))
        row[eps] = (m_, s_)
        line += f"  {m_*100:>6.1f}+-{s_*100:<3.1f}"
    print(line)
    agg_table[tag] = row

print(f"\n=== Cells actually changed by the Stage 4b correction ({len(corrections)} of {len(S4B_TAGS)*len(CORRECTED_EPS)*len(SEEDS)} checked) ===")
if corrections:
    for c in corrections:
        print(f"  seed{c['seed']} {c['tag']:<18} eps={c['eps']}: white-box={c['white_box_worst']*100:.2f}%  "
              f"-> AdvGAN found {c['advgan_acc']*100:.2f}%  (corrected worst-case: {c['corrected_worst']*100:.2f}%)")
else:
    print("  none - AdvGAN sweep never beat the white-box worst-case for any checked cell.")

def paired(tag_a, tag_b, eps):
    diffs = [corrected[tag_a][eps][i] - corrected[tag_b][eps][i] for i in range(len(SEEDS))]
    return float(np.mean(diffs)), float(np.std(diffs)), sum(1 for x in diffs if x > 0)

print(f"\n=== Paired differences on CORRECTED worst-case (kd_feedback_r5 vs others, eps=0.05 and 0.1) ===")
comparisons = [("kd_feedback_r5", "undefended"), ("kd_feedback_r5", "pgd_at"),
               ("kd_feedback_r5", "kd_fixed_gen_r5"), ("kd_feedback_r5", "nokd_feedback_r5")]
paired_table = {}
for a, b in comparisons:
    entry = {}
    for eps in ("0.05", "0.1"):
        m_, s_, npos = paired(a, b, eps)
        entry[eps] = {"mean": m_, "std": s_, "n_positive": npos}
        print(f"{a} - {b}  @eps={eps}: {m_*100:+.1f}+-{s_*100:.1f} pp  ({npos}/{len(SEEDS)} seeds positive)")
    paired_table[f"{a}__minus__{b}"] = entry

def jsonable(o):
    if isinstance(o, dict): return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, np.floating): return float(o)
    return o

json.dump(jsonable({"aggregate_corrected": agg_table, "corrections_applied": corrections,
                    "paired_differences_corrected": paired_table}),
          open(f"{BASE}/stage4_final_corrected.json", "w"), indent=1)
print(f"\nSaved: {BASE}/stage4_final_corrected.json")
