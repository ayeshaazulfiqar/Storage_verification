#!/usr/bin/env python3
"""
Rebuilds the combined table and both figures from results/. No API calls.

Run from the repository root:
    python reproduce_tables.py

Each check compares a headline number with the value in the report.
A FAIL line means a file or a number does not match the report.
"""
import os
import subprocess
import sys

import pandas as pd

R = "results"


def need(name):
    path = os.path.join(R, name)
    if not os.path.exists(path):
        sys.exit(f"missing {path}: copy it into results/ before running")
    return pd.read_csv(path)


def num(value):
    """'0.115 [0.070, 0.160]' -> 0.115. Plain numbers pass through unchanged."""
    return float(str(value).split()[0])


gem_a = need("gemma_extended_summary.csv")   # Task A, Gemma, 50 episodes
dsk_a = need("llm_honest_summary.csv")       # Task A, DeepSeek, 50 episodes
final_b = need("taskb_final.csv")            # Task B, Gemma and DeepSeek finals
pilot_b = need("taskb_pilot.csv")            # Task B pilots
split_b = need("taskb_split.csv")            # Task B cause split (replayed, no API)

gem_b = final_b[final_b.model == "gemma"].reset_index(drop=True)
dsk_b = pilot_b[pilot_b.model == "deepseek"].reset_index(drop=True)

checks = []


def check(label, got, want):
    ok = got == want
    checks.append(ok)
    print(f"{'OK  ' if ok else 'FAIL'} {label}: got {got}, expected {want}")


# ---- Headline numbers used in the report ----
trust_a = gem_a[gem_a.policy == "trust"].iloc[0]
check("Task A Gemma trust decisions", int(trust_a.decisions), 1899)
check("Task A Gemma trust wrong", int(trust_a.wrong), 365)
check("Task A DeepSeek wrong, all policies", int(dsk_a.model_errors.sum()), 0)
check("Task B Gemma trust wrong (final)", int(gem_b[gem_b.policy == "trust"].iloc[0].wrong), 276)
check("Task B Gemma trust wrong (split)", int(split_b[split_b.policy == "trust"].iloc[0].wrong), 276)
check("Task B DeepSeek pilot wrong, all policies", int(dsk_b.wrong.sum()), 0)

# The cause split must come from the same run as the final table
for _, s in split_b.iterrows():
    f = gem_b[gem_b.policy == s.policy].iloc[0]
    check(f"Task B split decisions match final ({s.policy})", int(s.decisions), int(f.decisions))
    check(f"Task B split wrong match final ({s.policy})", int(s.wrong), int(f.wrong))

# ---- Combined table ----
rows = []

for _, r in gem_a.iterrows():
    rows.append(dict(task="A", model="Gemma", episodes=50, policy=r.policy,
                     decisions=r.decisions, wrong=r.wrong, error_rate=r.error_rate,
                     error_ci=r.error_ci, silent_per_order=r.silent_per_order,
                     escalated_per_order=r.escalated_per_order,
                     esc_rule=r.esc_rule_per_order, esc_fault=r.esc_fault_per_order,
                     esc_both=r.esc_both_per_order))

for _, r in dsk_a.iterrows():
    esc = float(r.escalated_per_order)     # no wrong decisions, so every escalation is a storage escalation
    rows.append(dict(task="A", model="DeepSeek", episodes=50, policy=r.policy,
                     decisions=r.decisions, wrong=r.model_errors, error_rate=r.error_rate,
                     error_ci=r.exact_95_ci, silent_per_order=r.silent_wrong_per_order,
                     escalated_per_order=esc, esc_rule=0.0, esc_fault=esc, esc_both=0.0))

for _, s in split_b.iterrows():
    f = gem_b[gem_b.policy == s.policy].iloc[0]
    rows.append(dict(task="B", model="Gemma", episodes=50, policy=s.policy,
                     decisions=s.decisions, wrong=s.wrong, error_rate=f.error_rate,
                     error_ci=f.error_ci, silent_per_order=f.silent_per_order,
                     escalated_per_order=f.escalated_per_order,
                     esc_rule=s.esc_rule_per_order, esc_fault=s.esc_fault_per_order,
                     esc_both=s.esc_both_per_order))

for _, r in dsk_b.iterrows():
    esc = num(r.escalated_per_order)
    rows.append(dict(task="B", model="DeepSeek", episodes=int(r.episodes), policy=r.policy,
                     decisions=r.decisions, wrong=r.wrong, error_rate=r.error_rate,
                     error_ci=r.error_ci, silent_per_order=r.silent_per_order,
                     escalated_per_order=esc, esc_rule=0.0, esc_fault=esc, esc_both=0.0))

table = pd.DataFrame(rows)
check("combined table rows", len(table), 12)

cols = list(table.columns)
table.to_csv(os.path.join(R, "combined_table.csv"), index=False)
with open(os.path.join(R, "combined_table.md"), "w") as f:
    f.write("| " + " | ".join(cols) + " |\n")
    f.write("|" + "---|" * len(cols) + "\n")
    for _, r in table.iterrows():
        f.write("| " + " | ".join(str(r[c]) for c in cols) + " |\n")
print(f"wrote {R}/combined_table.csv and {R}/combined_table.md")

# ---- Figures (these scripts read results/ by default) ----
for script in ["plot_results.py", "plot_recompute.py"]:
    if os.path.exists(script):
        subprocess.run([sys.executable, script], check=False)
    else:
        print(f"skipped {script}: not found in the repository root")

print("\nALL CHECKS PASSED" if all(checks) else "\nSOME CHECKS FAILED: see FAIL lines above")
sys.exit(0 if all(checks) else 1)