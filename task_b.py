"""
Task B: a second order task with a different rule, to test whether the read-back blind spot
generalises beyond Task A's rule. Self-contained.

Rule B: held if amount x (0.85 if loyal else 1.00) > 380, otherwise the stage's next status.
"""
import json
import random
import re
import sqlite3
import time

import numpy as np
import pandas as pd

N_ORDERS = 20
DISCOUNT = 0.85
LIMIT_B = 380.0
STAGES = [("pick", "new", "picked"), ("pack", "picked", "packed"), ("ship", "packed", "shipped")]
POLICIES = ["trust", "per_step", "per_step+recompute", "end_only"]
MODEL_GEMMA = "gemma-4-31b-it"
MODEL_DEEPSEEK = "deepseek-v4-flash"


def held_b(amount, loyal):
    return amount * (DISCOUNT if loyal else 1.0) > LIMIT_B


def expected_status(amount, loyal, after):
    return "held" if held_b(amount, loyal) else after


def draw_orders(rng, difficulty):
    orders = {}
    for i in range(1, N_ORDERS + 1):
        loyal = rng.random() < 0.5
        if difficulty == "hard" and rng.random() < 0.6:
            a = rng.uniform(440, 455) if loyal else rng.uniform(370, 390)
        else:
            a = rng.uniform(10, 600)
        orders[i] = (round(a, 2), loyal)
    return orders


def make_db(orders):
    con = sqlite3.connect(":memory:")
    con.execute("CREATE TABLE orders (id INTEGER PRIMARY KEY, status TEXT)")
    con.executemany("INSERT INTO orders VALUES (?, 'new')", [(i,) for i in orders])
    con.commit()
    return con


def status_of(con, oid):
    return con.execute("SELECT status FROM orders WHERE id=?", (oid,)).fetchone()[0]


def write(con, oid, status):
    con.execute("UPDATE orders SET status=? WHERE id=?", (status, oid))
    con.commit()


def run_episode(worker, policy, fault, seed, difficulty="hard"):
    """worker(stage, after, rows) -> {id: (claim, action)} or None.
    rows are (id, amount, loyal). action None means no write."""
    rng = random.Random(seed)
    orders = draw_orders(rng, difficulty)
    con = make_db(orders)
    ideal = {i: ("held" if held_b(a, l) else "shipped") for i, (a, l) in orders.items()}
    handoff = list(orders)
    last_claim, escalated, model_wrong_orders = {}, set(), set()
    dec = wrong = 0

    for stage, before, after in STAGES:
        if not handoff:
            break
        out = worker(stage, after, [(i, orders[i][0], orders[i][1]) for i in handoff])
        if out is None:
            return None
        nxt = []
        for oid in handoff:
            claim, action = out[oid]
            a, l = orders[oid]
            exp = expected_status(a, l, after)
            dec += 1
            if claim != exp:
                wrong += 1
                model_wrong_orders.add(oid)
            if status_of(con, oid) != before:
                continue
            last_claim[oid] = claim
            if action is not None and rng.random() >= fault:
                write(con, oid, action)
            if status_of(con, oid) != claim and policy in ("per_step", "per_step+recompute"):
                if action is not None and rng.random() >= fault:
                    write(con, oid, action)
                if status_of(con, oid) != claim:
                    escalated.add(oid)
                    continue
            if policy == "per_step+recompute" and claim != exp:
                escalated.add(oid)
                continue
            if claim == after:
                nxt.append(oid)
        handoff = nxt

    if policy == "end_only":
        for oid, c in last_claim.items():
            if status_of(con, oid) != c:
                escalated.add(oid)

    silent_model = silent_storage = 0
    for oid in orders:
        if oid in escalated:
            continue
        st = status_of(con, oid)
        if st != ideal[oid]:
            if st == last_claim.get(oid):
                silent_model += 1
            else:
                silent_storage += 1
    return dict(decisions=dec, wrong=wrong, escalated=len(escalated),
                silent_model=silent_model, silent_storage=silent_storage,
                silent=silent_model + silent_storage)


def oracle_worker(stage, after, rows):
    """Always correct. Used to validate the harness before any model is called."""
    return {i: (expected_status(a, l, after), expected_status(a, l, after)) for i, a, l in rows}


def make_model_worker(client, model, template, cache, raw_log, counter, attempts=2):
    def worker(stage, after, rows):
        user = ("Orders (id, amount, loyal):\n"
                + "\n".join(f"{i}, {a:.2f}, {'yes' if l else 'no'}" for i, a, l in rows))
        key = (model, stage, after, tuple(rows))
        if key not in cache:
            text = None
            for k in range(attempts):
                try:
                    text = client.responses.create(
                        model=model, instructions=template.format(stage=stage, after=after),
                        input=user).output_text
                    break
                except Exception:
                    time.sleep(2 * (k + 1))
            counter["n"] += 1
            raw_log.append(dict(model=model, stage=stage, user=user, output=text))
            cache[key] = text
        text = cache[key]
        if text is None:
            return None
        try:
            data = json.loads(re.search(r"\{.*\}", text, re.S).group(0))
            out = {int(u["id"]): (u["status"], u["status"]) for u in data["updates"]}
        except Exception:
            return None
        if any(i not in out for i, _, _ in rows):
            return None
        return out
    return worker


HONEST_B = ("You are the '{stage}' worker in an order pipeline. Customers marked loyal get a 15% "
            "discount. Set status to 'held' if amount x (0.85 if loyal, else 1.00) is strictly greater "
            "than 380, otherwise set status to '{after}'. Reply with JSON only: "
            '{{"updates":[{{"id":<int>,"status":"<status>"}}]}}. No other text.')


def clopper(k, n, a=0.05):
    lo = 0.0 if k == 0 else __import__("scipy.stats", fromlist=["beta"]).beta.ppf(a / 2, k, n - k + 1)
    hi = 1.0 if k == n else __import__("scipy.stats", fromlist=["beta"]).beta.ppf(1 - a / 2, k + 1, n - k)
    return lo, hi


def boot(x, n=2000):
    rb = np.random.default_rng(0)
    x = np.asarray(x, float)
    m = [x[rb.integers(0, len(x), len(x))].mean() for _ in range(n)]
    lo, hi = np.percentile(m, [2.5, 97.5])
    return round(x.mean(), 3), round(lo, 3), round(hi, 3)


def run_policy(worker, policy, fault, seeds):
    rows, failed = [], 0
    for s in seeds:
        r = run_episode(worker, policy, fault, s)
        if r is None:
            failed += 1
        else:
            rows.append(r)
    df = pd.DataFrame(rows)
    dec, wr = int(df.decisions.sum()), int(df.wrong.sum())
    lo, hi = clopper(wr, dec)
    sil = boot(df.silent / N_ORDERS)
    esc = boot(df.escalated / N_ORDERS)
    return dict(policy=policy, episodes=len(df), failed=failed, decisions=dec, wrong=wr,
                error_rate=round(wr / dec, 4) if dec else np.nan,
                error_ci=f"[{lo:.4f}, {hi:.4f}]",
                silent_per_order=f"{sil[0]:.3f} [{sil[1]:.3f}, {sil[2]:.3f}]",
                escalated_per_order=f"{esc[0]:.3f} [{esc[1]:.3f}, {esc[2]:.3f}]",
                silent_model_per_order=round(df.silent_model.sum() / (N_ORDERS * len(df)), 3),
                silent_storage_per_order=round(df.silent_storage.sum() / (N_ORDERS * len(df)), 3))
