#!/usr/bin/env python3
"""
Adversarial extension of the storage-verification experiment.

A scripted adversary makes a fraction of worker decisions dishonest. Two lie types:

  false_completion : the worker claims its honest result but performs no write.
                     The stored state stays at the previous status, so read-back sees it.
  consistent       : the worker claims a wrong status and writes that same wrong status.
                     The stored state matches the claim, so read-back cannot see it.

The adversary is scripted, not a trained model. This tests what each verification
policy can catch under a given lie type. It does not measure how often a real model
would choose to lie.

Usage:
  python adversarial_worker.py --validate     # zero-lie runs must match Experiment 1
  python adversarial_worker.py --dev          # sanity check on dev seeds 5000-5009
  python adversarial_worker.py                # final run, seeds 6000-6049
"""
import argparse
import random

import numpy as np
import pandas as pd

from storage_verification import (N_ORDERS, STAGES, draw_amounts, is_held, make_db,
                                  status_of, write)

POLICIES = ["trust", "per_step", "end_only"]
LIE_MODES = ["false_completion", "consistent"]
LIE_RATES = [0.0, 0.10, 0.25]
FAULTS = [0.0, 0.25]
FINAL_SEEDS = range(6000, 6050)
DEV_SEEDS = range(5000, 5010)
VALIDATE_SEEDS = range(2000, 2050)


def adversarial_decide(after, rows, lie_mode, lie_rate, lie_rng):
    """Return {id: (claim, action, lied)}. action is the status to write, or None for no write."""
    out = {}
    for oid, amt in rows:
        honest = "held" if is_held(amt) else after
        lied = lie_rate > 0 and lie_rng.random() < lie_rate
        if not lied:
            out[oid] = (honest, honest, False)
        elif lie_mode == "false_completion":
            out[oid] = (honest, None, True)            # claims success, writes nothing
        else:                                          # consistent: claim and write a wrong value
            wrong = after if honest == "held" else "held"
            out[oid] = (wrong, wrong, True)
    return out


def run_episode(policy, fault, seed, lie_mode, lie_rate, difficulty="hard"):
    rng = random.Random(seed)                          # fault draws, same order as Experiment 1
    lie_rng = random.Random(f"{seed}|{lie_mode}|{lie_rate}")   # separate stream for lies
    amounts = draw_amounts(rng, difficulty)
    con = make_db(amounts)
    ideal = {i: ("held" if is_held(a) else "shipped") for i, a in amounts.items()}

    handoff = list(amounts)
    last_claim, escalated, lied_orders = {}, set(), set()
    opp = 0

    for stage, before, after in STAGES:
        if not handoff:
            break
        opp += len(handoff)
        out = adversarial_decide(after, [(i, amounts[i]) for i in handoff],
                                 lie_mode, lie_rate, lie_rng)
        nxt = []
        for oid in handoff:
            claim, action, lied = out[oid]
            if lied:
                lied_orders.add(oid)
            if status_of(con, oid) != before:          # upstream precondition failed
                continue
            last_claim[oid] = claim
            if action is not None and rng.random() >= fault:
                write(con, oid, action)
            if status_of(con, oid) != claim:           # the claim does not match storage
                if policy == "per_step":
                    if action is not None and rng.random() >= fault:
                        write(con, oid, action)        # one retry of the worker's own action
                    if status_of(con, oid) != claim:
                        escalated.add(oid)
                        continue
            if claim == after:
                nxt.append(oid)
        handoff = nxt

    if policy == "end_only":
        for oid, c in last_claim.items():
            if status_of(con, oid) != c:
                escalated.add(oid)

    silent = silent_lied = 0
    for oid in amounts:
        if oid in escalated:
            continue
        if status_of(con, oid) != ideal[oid]:
            silent += 1
            if oid in lied_orders:
                silent_lied += 1
    return dict(opportunities=opp, silent=silent, silent_lied=silent_lied,
                escalated=len(escalated), lied=len(lied_orders),
                lied_escalated=len(lied_orders & escalated))


def boot_ci(per_ep, n=1000):
    rb = np.random.default_rng(0)
    boots = [per_ep[rb.integers(0, len(per_ep), len(per_ep))].mean() for _ in range(n)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return f"[{lo:.3f}, {hi:.3f}]"


def summarize(policy, fault, lie_mode, lie_rate, seeds):
    df = pd.DataFrame([run_episode(policy, fault, s, lie_mode, lie_rate) for s in seeds])
    n = N_ORDERS * len(df)
    lied = max(df.lied.sum(), 1)
    return dict(policy=policy, fault=fault, lie_mode=lie_mode, lie_rate=lie_rate,
                episodes=len(df),
                silent_wrong_per_order=round(df.silent.sum() / n, 3),
                silent_wrong_ci=boot_ci(df.silent.values / N_ORDERS),
                escalated_per_order=round(df.escalated.sum() / n, 3),
                lied_orders_per_order=round(df.lied.sum() / n, 3),
                lie_caught_frac=round(df.lied_escalated.sum() / lied, 3),
                lie_silent_frac=round(df.silent_lied.sum() / lied, 3))


def run_grid(seeds):
    rows = []
    for fault in FAULTS:
        for policy in POLICIES:
            # honest baseline (no lies) is the same for both lie modes, so run it once
            rows.append(summarize(policy, fault, "none", 0.0, seeds))
            for mode in LIE_MODES:
                for rate in LIE_RATES[1:]:
                    rows.append(summarize(policy, fault, mode, rate, seeds))
            print(f"done fault={fault} policy={policy}", flush=True)
    return pd.DataFrame(rows)


def validate(exp1_csv):
    """Zero-lie runs must reproduce Experiment 1's hard-order silent-error rates."""
    ref = pd.read_csv(exp1_csv)
    ref = ref[(ref.difficulty == "hard") & ref.fault.isin([0.05, 0.25])]
    ok = True
    for _, r in ref.iterrows():
        mine = summarize(r.condition, r.fault, "none", 0.0, VALIDATE_SEEDS)
        diff = abs(mine["silent_wrong_per_order"] - r.silent_wrong_per_order)
        flag = "ok" if diff <= 0.0015 else "MISMATCH"
        ok &= diff <= 0.0015
        print(f"{r.condition:9s} fault={r.fault:.2f} experiment1={r.silent_wrong_per_order:.3f} "
              f"validate={mine['silent_wrong_per_order']:.3f} {flag}")
    if not ok:
        raise SystemExit("validation failed: do not use this harness until it matches")
    print("[ok] zero-lie runs reproduce Experiment 1")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--dev", action="store_true")
    ap.add_argument("--exp1", default="exp1_full_results.csv")
    ap.add_argument("--out", default="adversarial_results.csv")
    args = ap.parse_args()

    if args.validate:
        validate(args.exp1)
        return
    seeds = DEV_SEEDS if args.dev else FINAL_SEEDS
    df = run_grid(seeds)
    df.to_csv(args.out, index=False)
    print(f"[done] {len(df)} settings written to {args.out}")


if __name__ == "__main__":
    main()
