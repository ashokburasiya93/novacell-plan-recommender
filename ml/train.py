"""Train and compare plan recommenders for NovaCell Plan Recommender. ALL DATA IS SYNTHETIC.

Task: for a customer about to change plan or buy an add-on, rank the eligible catalogue items so the one they
choose is near the top. Train on choices made April-August 2026; test on September-October (later months,
after customers' usage has grown and many misfits have already been fixed).

Recommenders compared (all see the same eligible candidates):
  popularity   most-adopted items in training (what a one-size banner does)
  rules        today's agent playbook (over the data cap -> Unlimited; abroad -> Day Pass; calls abroad -> Calling Pack; else Unlimited)
  item-knn     collaborative filtering: customers who own what you own went on to adopt...
  hybrid       gradient-boosted ranker on usage, holdings, estimated bill impact, popularity and the item-knn score

Then the chosen recommender gets a guardrail ("fit first"): drop any suggestion that would raise the expected
monthly bill by more than $15 without removing at least as much in charges or slowed data.

Writes ml/results/metrics.json, public/model.json and public/customers.json (top 3 for 1,500 current customers).
Usage: python3 ml/train.py
"""
import json


def _np(o):
    return o.item() if hasattr(o, "item") else str(o)

import os
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "ml"))
from generate_data import ADDONS, C, INTL_PAYG_PER_MIN, ITEMS, PLANS, ROAM_PAYG_PER_DAY, apply, eligible, monthly_cost, plan_price  # noqa: E402

D = os.path.join(ROOT, "data")
RES = os.path.join(ROOT, "ml", "results")
os.makedirs(RES, exist_ok=True)
TRAIN_MONTHS = ["2026-04", "2026-05", "2026-06", "2026-07", "2026-08"]
TEST_MONTHS = ["2026-09", "2026-10"]
K = 3
SHOCK_LIMIT = 15.0

cust = pd.read_csv(os.path.join(D, "customers.csv")).set_index("customer_id")
usage = pd.read_csv(os.path.join(D, "usage_monthly.csv"))
hold = pd.read_csv(os.path.join(D, "holdings_monthly.csv"), keep_default_na=False)
adopt = pd.read_csv(os.path.join(D, "adoptions.csv"))
U = {m: g.set_index("customer_id") for m, g in usage.groupby("month")}
H = {m: g.set_index("customer_id") for m, g in hold.groupby("month")}
ALLM = sorted(U)


def prev(m, k):
    i = ALLM.index(m)
    return ALLM[max(0, i - k):i]


def profile(cid, month):
    """What we can observe before a decision made at the end of the month before `month`:
    3-month average usage and the holdings at the start of the previous month."""
    ms = prev(month, 3)
    u = pd.concat([U[m].loc[[cid]] for m in ms])
    last = prev(month, 1)[0]
    h = H[last].loc[cid]
    addons = set(a for a in h["addons"].split("|") if a)
    c = cust.loc[cid]
    avg = {"data_gb": u.data_gb.mean(), "hotspot_gb": u.hotspot_gb.mean(), "abroad_days": u.abroad_days.mean(), "intl_minutes": u.intl_minutes.mean()}
    return {"cid": cid, "plan": h["plan"], "addons": addons, "usage": avg, "max_data": u.data_gb.max(), "paid": u.paid_usd.mean(), "c": c}


def bill(p, plan, addons):
    paid, pain = monthly_cost(plan, addons, p["usage"], p["c"])
    cap = C[plan]["data_cap_gb"]
    over = max(0.0, p["usage"]["data_gb"] - cap) if cap is not None else 0.0
    return paid, pain, over


def candidates(p):
    return [it for it in ITEMS if eligible(it, p["c"], p["plan"], p["addons"])]


def item_feats(p, it):
    paid0, pain0, over0 = bill(p, p["plan"], p["addons"])
    p2, a2 = apply(it, p["plan"], p["addons"])
    paid1, pain1, over1 = bill(p, p2, a2)
    return paid1 - paid0, pain1 - pain0, over0 - over1


# ---------- build decision sets ----------
def decisions(months):
    rows = []
    for _, r in adopt[adopt.month.isin(months)].iterrows():
        p = profile(r.customer_id, r.month)
        cands = candidates(p)
        if r["item"] not in cands or len(cands) < 2:
            continue
        rows.append((p, cands, r["item"], r.month))
    return rows


print("building decision sets ...")
train = decisions(TRAIN_MONTHS)
test = decisions(TEST_MONTHS)
print("train decisions", len(train), "test decisions", len(test))

# popularity
pop = Counter(t[2] for t in train)
pop_share = {it: pop.get(it, 0) / len(train) for it in ITEMS}

# item-knn: P(adopt j | own i) from training decisions
co = defaultdict(Counter)
own_n = Counter()
for p, cands, y, _ in train:
    owned = {p["plan"], *p["addons"]}
    for i in owned:
        own_n[i] += 1
        co[i][y] += 1


def knn_score(p, it):
    owned = {p["plan"], *p["addons"]}
    return float(np.mean([co[i][it] / own_n[i] if own_n[i] else 0 for i in owned]))


def rules_score(p, it):
    u = p["usage"]
    cap = C[p["plan"]]["data_cap_gb"]
    s = 0.0
    if cap is not None and u["data_gb"] > cap and it == "POST-UNLIMITED":
        s += 3
    if u["abroad_days"] > 0 and it == "ADDON-INTL-DAYPASS":
        s += 2
    if u["intl_minutes"] > 60 and it == "ADDON-INTL-CALLING":
        s += 2
    if it == "POST-UNLIMITED":
        s += 1  # the banner default
    return s + pop_share[it] * 0.1


PLAN_IDX = {it: k for k, it in enumerate(ITEMS)}
FEATS = ["item", "cur_plan", "lines", "prepaid", "student", "eligible_55_plus", "credit_ok", "has_tablet", "streamer", "data_gb", "max_data", "hotspot_gb",
         "abroad_days", "intl_minutes", "paid_now", "d_paid", "d_pain", "gb_unslowed", "net_gain", "pop", "knn", "n_addons"]


def featurize(p, it):
    c = p["c"]
    d_paid, d_pain, unslowed = item_feats(p, it)
    return [PLAN_IDX[it], PLAN_IDX[p["plan"]], c.lines, int(bool(C[p["plan"]].get("prepaid"))), int(c.student), int(c.age_55_plus and c.lines <= 2), int(c.credit_ok),
            int(c.has_tablet), int(c.streamer), p["usage"]["data_gb"], p["max_data"], p["usage"]["hotspot_gb"], p["usage"]["abroad_days"], p["usage"]["intl_minutes"],
            p["paid"], d_paid, d_pain, unslowed, -(d_paid + d_pain), pop_share[it], knn_score(p, it), len(p["addons"])]


def pairs(rows):
    X, y, g = [], [], []
    for k, (p, cands, target, _) in enumerate(rows):
        for it in cands:
            X.append(featurize(p, it)); y.append(int(it == target)); g.append(k)
    return np.array(X, dtype=float), np.array(y), np.array(g)


Xtr, ytr, _ = pairs(train)
model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_leaf_nodes=31, min_samples_leaf=30, l2_regularization=1.0,
                                       categorical_features=[0, 1], random_state=7)
model.fit(Xtr, ytr)


def rank(p, cands, method):
    if method == "popularity":
        s = [pop_share[it] for it in cands]
    elif method == "rules":
        s = [rules_score(p, it) for it in cands]
    elif method == "item-knn":
        s = [knn_score(p, it) + 1e-3 * pop_share[it] for it in cands]
    else:
        s = model.predict_proba(np.array([featurize(p, it) for it in cands], dtype=float))[:, 1]
    order = np.argsort(-np.asarray(s), kind="stable")
    return [cands[i] for i in order], [float(np.asarray(s)[i]) for i in order]


def shock(p, it):
    d_paid, d_pain, _ = item_feats(p, it)
    return d_paid > SHOCK_LIMIT and -(d_pain) < d_paid  # raises the bill by > $15 and removes less than that in slowed data / charges


MIN_GAIN = 3.0  # show nothing unless the customer is at least $3 a month better off (money + slowed data)


def gain_of(p, it):
    d_paid, d_pain, _ = item_feats(p, it)
    return -(d_paid + d_pain)


def guarded(p, ranked):
    keep = [it for it in ranked if not shock(p, it)]
    return keep or ranked[:1]


def evaluate(rows, method, guard=False):
    hits1 = hits3 = ndcg = mrr = 0.0
    shocks = 0
    saves = []
    per = []
    for p, cands, target, _ in rows:
        r, _ = rank(p, cands, method)
        if guard:
            r = guarded(p, r)
        pos = r.index(target) if target in r else None
        h3 = pos is not None and pos < K
        hits1 += pos == 0
        hits3 += h3
        ndcg += 1 / np.log2(pos + 2) if h3 else 0
        mrr += 1 / (pos + 1) if pos is not None else 0
        top = r[0]
        shocks += shock(p, top)
        d_paid, d_pain, _ = item_feats(p, top)
        saves.append(-(d_paid + d_pain))
        per.append((p, h3, shock(p, top)))
    n = len(rows)
    return {"hit_at_1": round(hits1 / n, 3), "hit_at_3": round(hits3 / n, 3), "ndcg_at_3": round(ndcg / n, 3), "mrr": round(mrr / n, 3),
            "bill_shock_top1": round(shocks / n, 3), "avg_customer_gain_top1_usd": round(float(np.mean(saves)), 2)}, per


metrics = {"data": {"customers": int(len(cust)), "train_decisions": len(train), "test_decisions": len(test), "train_months": TRAIN_MONTHS, "test_months": TEST_MONTHS,
                    "items": len(ITEMS), "note": "Time-based split: trained on choices made April-August, tested on September-October."}, "k": K, "shock_limit_usd": SHOCK_LIMIT}
per_store = {}
for m in ["popularity", "rules", "item-knn", "hybrid"]:
    metrics[m], per_store[m] = evaluate(test, m)
metrics["hybrid+guardrail"], per_store["hybrid+guardrail"] = evaluate(test, "hybrid", guard=True)
metrics["train_fit_hybrid"] = evaluate(train[-600:], "hybrid")[0]
for m in ["popularity", "rules", "item-knn", "hybrid", "hybrid+guardrail"]:
    print(f"{m:18s}", json.dumps(metrics[m], default=_np))

# feature importance by permutation on test pairs (hit@3 drop is costly; use log-loss proxy on pairs)
Xte, yte, gte = pairs(test)
base_ll = -np.mean(np.log(np.clip(model.predict_proba(Xte)[np.arange(len(yte)), yte], 1e-9, 1)))
rng = np.random.default_rng(0)
imp = {}
for j, f in enumerate(FEATS):
    Xp = Xte.copy(); Xp[:, j] = rng.permutation(Xp[:, j])
    ll = -np.mean(np.log(np.clip(model.predict_proba(Xp)[np.arange(len(yte)), yte], 1e-9, 1)))
    imp[f] = round(float(ll - base_ll), 4)
metrics["feature_importance"] = dict(sorted(imp.items(), key=lambda kv: -kv[1])[:10])

# fairness: hit@3 and bill-shock by group for the chosen policy (age band and gender are never model inputs)
fair = {}
per = per_store["hybrid+guardrail"]
for col in ["age_band", "gender", "region"]:
    grp = defaultdict(lambda: [0, 0, 0])
    for p, h3, sh in per:
        g = p["c"][col]; grp[g][0] += 1; grp[g][1] += h3; grp[g][2] += sh
    tab = np.array([[v[1], v[0] - v[1]] for v in grp.values()])
    pval = float(chi2_contingency(tab)[1]) if tab.shape[0] > 1 else 1.0
    fair[col] = {"groups": {g: {"n": v[0], "hit_at_3": round(v[1] / v[0], 3), "bill_shock": round(v[2] / v[0], 3)} for g, v in sorted(grp.items())},
                 "hit_p_value": round(pval, 3)}
metrics["fairness"] = fair

# when to show: at the start of September, would the gate have picked the customers who then changed in Sep-Oct?
print("evaluating the show/no-show gate ...")
changed = set(adopt[adopt.month.isin(TEST_MONTHS)].customer_id)
shown_ids, gate_rows = set(), 0
for cid in cust.index:
    p = profile(cid, "2026-09")
    cands = candidates(p)
    if not cands:
        continue
    gate_rows += 1
    r, _ = rank(p, cands, "hybrid")
    if any(gain_of(p, it) >= MIN_GAIN for it in guarded(p, r)):
        shown_ids.add(cid)
tp = len(shown_ids & changed)
metrics["gate"] = {"min_gain_usd": MIN_GAIN, "customers": gate_rows, "shown_share": round(len(shown_ids) / gate_rows, 3),
                   "precision": round(tp / max(1, len(shown_ids)), 3), "recall": round(tp / max(1, len(changed)), 3),
                   "base_rate_banner_to_all": round(len(changed) / gate_rows, 3)}
print("gate", metrics["gate"])

# coverage + business view on current customers (snapshot for November recommendations)
cur_month = "2026-11"
U[cur_month] = None  # placeholder so prev() works
ALLM.append(cur_month)
ids = cust.sample(1500, random_state=11).index
out, cover, cover_pop = [], Counter(), Counter()
rev, saved, n_rec = [], [], 0
for cid in ids:
    p = profile(cid, cur_month)
    cands = candidates(p)
    if not cands:
        continue
    r, s = rank(p, cands, "hybrid")
    rg = [it for it in guarded(p, r) if gain_of(p, it) >= MIN_GAIN][:K]
    if not rg:
        out.append({"customer_id": cid, "plan": p["plan"], "plan_name": C[p["plan"]]["name"], "addons": sorted(p["addons"]), "lines": int(p["c"].lines),
                    "usage": {k: round(v, 1) for k, v in p["usage"].items()}, "paid_now_usd": round(p["paid"], 2), "recommendations": []})
        continue
    cover.update(rg)
    cover_pop.update(rank(p, cands, "popularity")[0][:K])
    paid0, pain0, over0 = bill(p, p["plan"], p["addons"])
    recs = []
    for it in rg:
        d_paid, d_pain, unslowed = item_feats(p, it)
        reasons = []
        u = p["usage"]
        cap = C[p["plan"]]["data_cap_gb"]
        if cap is not None and u["data_gb"] > cap:
            reasons.append(f"Uses {u['data_gb']:.1f} GB a month on a {cap} GB plan: about {u['data_gb']-cap:.1f} GB slowed each month")
        elif cap is None and u["data_gb"] < 12 and C[it]["kind"] == "plan" and C[it]["data_cap_gb"]:
            reasons.append(f"Uses {u['data_gb']:.1f} GB a month on an unlimited plan")
        if u["abroad_days"] >= 0.3 and it in ("ADDON-INTL-DAYPASS", "ADDON-INTL-MONTHLY", "POST-UNLIMITED-PLUS"):
            reasons.append(f"Abroad about {u['abroad_days']:.1f} days a month")
        if u["intl_minutes"] >= 30 and it == "ADDON-INTL-CALLING":
            reasons.append(f"Calls international numbers about {u['intl_minutes']:.0f} minutes a month (about ${u['intl_minutes']*INTL_PAYG_PER_MIN:.0f} at pay-as-you-go)")
        hs = C[p["plan"]]["hotspot_gb"] + (15 if "ADDON-HOTSPOT-BOOST" in p["addons"] else 0)
        if u["hotspot_gb"] > hs and it == "ADDON-HOTSPOT-BOOST":
            reasons.append(f"Hotspot use {u['hotspot_gb']:.1f} GB a month, allowance {hs} GB")
        if it == "DATA-ONLY-30" and p["c"].has_tablet:
            reasons.append("Has a tablet that is not connected")
        if C[it].get("senior"):
            reasons.append("Eligible for the 55+ plan (2 unlimited lines for $40)")
        if C[it].get("student"):
            reasons.append("Verified student: eligible for Student Unlimited")
        if C[it].get("family"):
            reasons.append(f"{p['c'].lines} lines on the account")
        if not reasons:
            reasons.append("Customers with similar usage and plans chose this")
        recs.append({"item": it, "name": C[it]["name"], "kind": C[it]["kind"], "score": round(float(s[r.index(it)]), 3), "bill_change_usd": round(d_paid, 2),
                     "slowed_gb_removed": round(unslowed, 1), "customer_gain_usd": round(-(d_paid + d_pain), 2), "reasons": reasons[:3]})
    top = recs[0]
    rev.append(top["bill_change_usd"]); saved.append(top["customer_gain_usd"]); n_rec += 1
    out.append({"customer_id": cid, "plan": p["plan"], "plan_name": C[p["plan"]]["name"], "addons": sorted(p["addons"]), "lines": int(p["c"].lines),
                "usage": {k: round(v, 1) for k, v in p["usage"].items()}, "paid_now_usd": round(p["paid"], 2), "recommendations": recs})
metrics["current"] = {"customers": len(out), "shown": n_rec, "share_shown": round(n_rec / len(out), 3), "catalogue_coverage_top3": round(len(cover) / len(ITEMS), 3), "popularity_coverage_top3": round(len(cover_pop) / len(ITEMS), 3),
                      "avg_bill_change_top1_usd": round(float(np.mean(rev)), 2), "share_top1_lowers_bill": round(float(np.mean(np.array(rev) < 0)), 3),
                      "avg_customer_gain_top1_usd": round(float(np.mean(saved)), 2), "top1_mix": dict(Counter(o["recommendations"][0]["item"] for o in out if o["recommendations"]).most_common())}
# two in-app policies on the same recommendations: fit first, and balanced (big downgrades go to the retention team instead)
BIG_DOWN = -15.0
def policy_stats(rows, balanced):
    shown, gain, rev = 0, [], []
    for o in rows:
        rs = [r for r in o["recommendations"] if not (balanced and r["bill_change_usd"] < BIG_DOWN)]
        if rs:
            shown += 1; gain.append(rs[0]["customer_gain_usd"]); rev.append(rs[0]["bill_change_usd"])
    return {"share_shown": round(shown / len(rows), 3), "avg_customer_gain_top1_usd": round(float(np.mean(gain)), 2),
            "avg_revenue_change_top1_usd": round(float(np.mean(rev)), 2), "share_top1_lowers_bill": round(float(np.mean(np.array(rev) < 0)), 3),
            "share_top1_raises_bill": round(float(np.mean(np.array(rev) > 0)), 3)}
metrics["policies"] = {"fit_first": policy_stats(out, False), "balanced": policy_stats(out, True), "big_downgrade_usd": BIG_DOWN}
print("policies", metrics["policies"])
metrics["chosen"] = "hybrid+guardrail"
json.dump(metrics, open(os.path.join(RES, "metrics.json"), "w"), indent=1, default=_np)
json.dump({"metrics": {k: metrics[k] for k in ["popularity", "rules", "item-knn", "hybrid", "hybrid+guardrail"]}, "data": metrics["data"], "fairness": fair,
           "feature_importance": metrics["feature_importance"], "current": metrics["current"], "gate": metrics["gate"], "policies": metrics["policies"], "catalogue": [C[i] for i in ITEMS]},
          open(os.path.join(ROOT, "public", "model.json"), "w"), indent=1, default=_np)
json.dump(out, open(os.path.join(ROOT, "public", "customers.json"), "w"), default=_np)
json.dump({"snapshot": "2026-10-31", "catalogue": {i: C[i] for i in ITEMS}, "customers": out}, open(os.path.join(D, "current_recs.json"), "w"), default=_np)
print(json.dumps(metrics["current"], indent=1, default=_np))
print(json.dumps(metrics["feature_importance"]))
print({k: v["hit_p_value"] for k, v in fair.items()})
