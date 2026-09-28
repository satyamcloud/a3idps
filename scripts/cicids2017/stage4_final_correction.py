# CICIDS2017 v2 - Final correction: fold the Stage 4b AdvGAN-attack sweep into the Stage 4a worst-case numbers.
# Rationale: Stage 4b found 3/320 cases (out of 5 seeds x 4 models x 2 eps x 8 generator configs) where the
# AdvGAN-attack sweep found a slightly stronger attack (lower accuracy) than the white-box FGSM/PGD ensemble
# Stage 4a reports as "worst-case". To make every worst-case number in the paper a true minimum over every
# attack actually tried (the strongest possible answer to "did you use a strong enough attack"), this script
# recomputes corrected_worst = min(stage4a_worst, best_advgan_attack_accuracy) for the 4 models Stage 4b covers
# (undefended, fgsm_at, pgd_at, kd_feedback_r5) at the 2 eps values Stage 4b covers (0.05, 0.1). Every other
# cell (other models, other eps) is copied through unchanged - Stage 4b never attacked them, so there is
# nothing to correct. Read-only over existing results; writes one new summary file, does not touch stage4a_*.

import json, os
import numpy as np

BASE = "/content/drive/MyDrive/A3IDPS_COLAB"
RES = f"{BASE}/results/cicids_v2"

SEEDS = [0, 1, 2, 3, 4]
sel_at = json.load(open(f"{RES}/stage2b_selected_seed0.json"))
EPS_FGSM, EPS_PGD = sel_at["fgsm"], sel_at["pgd"]

# maps Stage 4b's short model tags -> the tag names used in stage4a_seed{s}.json
S4B_TO_S4A_TAG = {
    "undefended": "undefended",
    "fgsm_at": f"fgsm_at_eps{EPS_FGSM}",
    "pgd_at": f"pgd_at_eps{EPS_PGD}",
    "kd_feedback_r5": "kd_feedback_r5",
}
CORRECTED_EPS = ["0.05", "0.1"]

s4a = {s: json.load(open(f"{RES}/stage4a_seed{s}.json")) for s in SEEDS}
s4b = json.load(open(f"{RES}/stage4b_results.json"))

# every tag present in seed 0's stage4a output, in a sensible print order
ALL_TAGS = list(s4a[0].keys())
EPS_ALL = sorted(s4a[0][ALL_TAGS[0]]["robust"].keys(), key=float)

def s4a_worst(seed, tag, eps):
    return s4a[seed][tag]["robust"][eps]["worst"]

def best_advgan_acc(seed, s4b_tag, eps):
    cfgs = s4b[str(seed)][s4b_tag]["configs"]
    return min(cfgs[c][eps]["acc_under_generator"] for c in cfgs if eps in cfgs.get(c, {}))

# %% Build the corrected per-seed, per-tag, per-eps worst-case table
corrections = []   # log of every cell actually changed, for transparency
corrected = {tag: {eps: [] for eps in EPS_ALL} for tag in ALL_TAGS}

for tag in ALL_TAGS:
    s4b_tag = next((k for k, v in S4B_TO_S4A_TAG.items() if v == tag), None)
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
                final = wb   # untouched - Stage 4b never attacked this (tag, eps) combination
            corrected[tag][eps].append(final)

# %% Aggregate mean +- std across seeds, and print the final corrected table (same format as Stage 4a's)
print(f"=== FINAL 5-SEED TEST RESULTS, corrected worst-case = min(white-box, AdvGAN-sweep) where both exist ===")
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

print(f"\n=== Cells actually changed by the Stage 4b correction ({len(corrections)} of {len(ALL_TAGS)*len(CORRECTED_EPS)*len(SEEDS)} checked) ===")
if corrections:
    for c in corrections:
        print(f"  seed{c['seed']} {c['tag']:<18} eps={c['eps']}: white-box={c['white_box_worst']*100:.2f}%  "
              f"-> AdvGAN found {c['advgan_acc']*100:.2f}%  (corrected worst-case: {c['corrected_worst']*100:.2f}%)")
else:
    print("  none - AdvGAN sweep never beat the white-box worst-case for any checked cell.")

# %% Paired differences on the CORRECTED numbers (same comparisons as Stage 4a, recomputed)
def paired(tag_a, tag_b, eps):
    diffs = [corrected[tag_a][eps][i] - corrected[tag_b][eps][i] for i in range(len(SEEDS))]
    return float(np.mean(diffs)), float(np.std(diffs)), sum(1 for x in diffs if x > 0)

print(f"\n=== Paired differences on CORRECTED worst-case (kd_feedback_r5 vs others, eps=0.05 and 0.1) ===")
comparisons = [("kd_feedback_r5", "undefended"), ("kd_feedback_r5", f"pgd_at_eps{EPS_PGD}"),
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
          open(f"{RES}/stage4_final_corrected.json", "w"), indent=1)
print(f"\nSaved: {RES}/stage4_final_corrected.json")
