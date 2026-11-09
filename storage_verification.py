#!/usr/bin/env python3
"""
Experiment 1: does read-back verification catch storage-induced silent errors
in an LLM-driven order pipeline?

Three stages (pick, pack, ship) run one LLM worker each. The worker returns a
status per order as JSON. A harness, not the worker, performs each write and
drops it with probability `fault`. Three verification policies are compared:

  trust    - the next stage acts on every claim
  per_step - read back after each write; retry once; escalate if still wrong
  end_only - run as trust, then compare stored state with last claim at the end

Outcomes per order: correct, escalated, or silently wrong (wrong and unflagged).
Silent errors are split into storage errors (stored state != last claim) and
decision errors (stored state == last claim, but wrong).

Usage:
  python storage_verification.py --stub          # offline check, no API calls
  python storage_verification.py                 # real run (needs ADAPTION_API_KEY)

Outputs: results/full_results.csv and results/summary.txt
"""
import argparse
import json
import os
import random
import re
import sqlite3
import time

import numpy as np
import pandas as pd

# ---- Configuration (the values used in the report) -------------------------
N_ORDERS = 20
TAX = 0.08
REVIEW_AMOUNT = 450.0
MODEL = "deepseek-v4-flash"
STAGES = [("pick", "new", "picked"),
          ("pack", "picked", "packed"),
          ("ship", "packed", "shipped")]
FAULTS = [0.0, 0.05, 0.10, 0.25]
CONDITIONS = ["trust", "per_step", "end_only"]
DIFFICULTIES = ["easy", "hard"]
FULL_SEEDS = range(2000, 2050)          # final seeds; do not use for tuning
PILOT_SEEDS = range(1000, 1010)         # pilot seeds; used for design checks only


def is_held(amount):
    """Held if the tax-inclusive amount is strictly above the review threshold."""
    return amount * (1 + TAX) > REVIEW_AMOUNT


def draw_amounts(rng, difficulty):
    """Easy: uniform 10-500. Hard: half the orders in 405-430, where tax decides."""
    amounts = {}
    for i in range(1, N_ORDERS + 1):
        if difficulty == "hard" and rng.random() < 0.5:
            a = rng.uniform(405, 430)
        else:
            a = rng.uniform(10, 500)
        amounts[i] = round(a, 2)
    return amounts


def make_db(amounts):
    con = sqlite3.connect(":memory:")
    con.execute("CREATE TABLE orders (id INTEGER PRIMARY KEY, status TEXT, amount REAL)")
    con.executemany("INSERT INTO orders VALUES (?, 'new', ?)",
                    [(i, a) for i, a in amounts.items()])
    con.commit()
    return con


def status_of(con, oid):
    return con.execute("SELECT status FROM orders WHERE id=?", (oid,)).fetchone()[0]


def write(con, oid, status):
    con.execute("UPDATE orders SET status=? WHERE id=?", (status, oid))
    con.commit()


# ---- Workers ----------------------------------------------------------------
PROMPT = ("You are the '{stage}' worker in an order pipeline. For each order, set status to "
          "'{after}', except set status to 'held' if the amount multiplied by {mult:.2f} "
          "(tax included) is strictly greater than {limit:.2f}. Reply with JSON only: "
          '{{"updates":[{{"id":<int>,"status":"<string>"}}]}}. No other text.')


class ApiWorker:
    """Calls the hosted model. Caches outputs by prompt so reruns are free."""

    def __init__(self):
        from adaption import Adaption          # pip install adaption
        self.client = Adaption()
        self.cache, self.raw_log, self.calls = {}, [], 0

    def __call__(self, stage, after, rows):
        key = (stage, after, tuple(rows))
        if key not in self.cache:
            instr = PROMPT.format(stage=stage, after=after, mult=1 + TAX, limit=REVIEW_AMOUNT)
            user = "Orders (id, amount): " + "; ".join(f"{i}, {a:.2f}" for i, a in rows)
            text = None
            for attempt in range(3):
                try:
                    text = self.client.responses.create(
                        model=MODEL, instructions=instr, input=user).output_text
                    break
                except Exception:
                    time.sleep(2 * (attempt + 1))
            self.calls += 1
            self.raw_log.append(dict(stage=stage, after=after, user=user, output=text))
            self.cache[key] = text
        text = self.cache[key]
        if text is None:
            return None
        try:
            data = json.loads(re.search(r"\{.*\}", text, re.S).group(0))
            return {int(u["id"]): str(u["status"]) for u in data["updates"]}
        except Exception:
            return None


class StubWorker:
    """Offline stand-in that always decides correctly. Tests the harness only."""

    def __init__(self):
        self.calls, self.raw_log = 0, []

    def __call__(self, stage, after, rows):
        self.calls += 1
        return {i: ("held" if is_held(a) else after) for i, a in rows}


# ---- One episode ------------------------------------------------------------
def run_episode(worker, condition, fault, seed, difficulty):
    rng = random.Random(seed)
    amounts = draw_amounts(rng, difficulty)
    con = make_db(amounts)
    ideal = {i: ("held" if is_held(a) else "shipped") for i, a in amounts.items()}

    m = dict(opportunities=0, claims=0, false_claims=0, caught=0, retry_fixed=0,
             downstream_err=0, worker_wrong=0, parse_err=0)
    handoff = list(amounts)
    last_claim, escalated = {}, set()

    for stage, before, after in STAGES:
        if not handoff:
            break
        m["opportunities"] += len(handoff)
        out = worker(stage, after, tuple((i, amounts[i]) for i in handoff))
        if out is None:
            m["parse_err"] += 1
            break
        next_handoff = []
        for oid in handoff:
            if oid not in out:
                m["worker_wrong"] += 1
                continue
            claim = out[oid]
            expected = "held" if is_held(amounts[oid]) else after
            if claim != expected:
                m["worker_wrong"] += 1
            if status_of(con, oid) != before:           # upstream precondition failed
                m["downstream_err"] += 1
                continue
            m["claims"] += 1
            last_claim[oid] = claim
            if rng.random() >= fault:                   # harness write, may be dropped
                write(con, oid, claim)
            if status_of(con, oid) != claim:            # the claim was false
                m["false_claims"] += 1
                if condition == "per_step":
                    m["caught"] += 1
                    if rng.random() >= fault:           # one retry, same fault model
                        write(con, oid, claim)
                    if status_of(con, oid) == claim:
                        m["retry_fixed"] += 1
                    else:
                        escalated.add(oid)
                        continue
            if claim == after:
                next_handoff.append(oid)
        handoff = next_handoff

    if condition == "end_only":                         # check only after the pipeline
        for oid, c in last_claim.items():
            if status_of(con, oid) != c:
                escalated.add(oid)

    silent_llm = silent_write = 0
    for oid in amounts:
        if oid in escalated:
            continue
        status = status_of(con, oid)
        if status != ideal[oid]:
            if status == last_claim.get(oid):
                silent_llm += 1                         # stored faithfully, but wrong
            else:
                silent_write += 1                       # stored state disagrees with claim
    m.update(escalated=len(escalated), silent_wrong=silent_llm + silent_write,
             silent_llm=silent_llm, silent_write=silent_write)
    return m


# ---- Summaries --------------------------------------------------------------
def summarize(worker, condition, fault, difficulty, seeds):
    df = pd.DataFrame([run_episode(worker, condition, fault, s, difficulty) for s in seeds])
    n_orders = N_ORDERS * len(df)
    opp = max(df.opportunities.sum(), 1)
    per_ep = df.silent_wrong.values / N_ORDERS
    rb = np.random.default_rng(0)
    boots = [per_ep[rb.integers(0, len(per_ep), len(per_ep))].mean() for _ in range(1000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    fc = df.false_claims.sum()
    return dict(
        condition=condition, difficulty=difficulty, fault=fault, episodes=len(df),
        silent_wrong_per_order=round(df.silent_wrong.sum() / n_orders, 3),
        silent_wrong_ci=f"[{lo:.3f}, {hi:.3f}]",
        silent_llm_per_order=round(df.silent_llm.sum() / n_orders, 3),
        silent_write_per_order=round(df.silent_write.sum() / n_orders, 3),
        escalated_per_order=round(df.escalated.sum() / n_orders, 3),
        retry_fixed_frac=round(df.retry_fixed.sum() / max(df.caught.sum(), 1), 3),
        downstream_err_per_opp=round(df.downstream_err.sum() / opp, 3),
        worker_wrong_per_opp=round(df.worker_wrong.sum() / opp, 4),
        false_claim_rate=round(fc / max(df.claims.sum(), 1), 3),
    )


def run_all(worker, seeds, out_dir):
    """Runs every setting. Saves the CSV after each setting, so an interruption loses little."""
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for diff in DIFFICULTIES:
        for fault in FAULTS:
            for cond in CONDITIONS:
                r = summarize(worker, cond, fault, diff, seeds)
                rows.append(r)
                print(f"{diff:4s} {fault:.2f} {cond:9s} silent={r['silent_wrong_per_order']:.3f} "
                      f"escalated={r['escalated_per_order']:.3f}", flush=True)
                pd.DataFrame(rows).to_csv(os.path.join(out_dir, "full_results.csv"), index=False)
                if hasattr(worker, "raw_log"):
                    with open(os.path.join(out_dir, "raw_llm_log.jsonl"), "w") as f:
                        for entry in worker.raw_log:
                            f.write(json.dumps(entry) + "\n")
    return pd.DataFrame(rows)


def self_check(worker):
    """Sanity checks that run in seconds on any worker."""
    for diff in DIFFICULTIES:
        r = summarize(worker, "trust", 0.0, diff, range(2000, 2005))
        assert r["silent_wrong_per_order"] == 0.0, "trust with no faults must be error-free"
        p = summarize(worker, "per_step", 0.0, diff, range(2000, 2005))
        assert p["silent_wrong_per_order"] == 0.0, "per_step with no faults must be error-free"
    print("[ok] self-checks passed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stub", action="store_true", help="offline run with a correct stub worker")
    ap.add_argument("--seeds", choices=["final", "pilot"], default="final")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    worker = StubWorker() if args.stub else ApiWorker()
    self_check(worker)
    seeds = FULL_SEEDS if args.seeds == "final" else PILOT_SEEDS
    df = run_all(worker, seeds, args.out)
    print(f"[done] {len(df)} settings written to {args.out}/full_results.csv")
    if args.stub:
        print("NOTE: stub run. Silent errors come only from injected faults; "
              "numbers are not the report's results.")


if __name__ == "__main__":
    main()
