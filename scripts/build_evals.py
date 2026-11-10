"""Build the message eval sets from the current recommendations (synthetic). Run after ml/train.py.
dev 30 and holdout 20: random top suggestions. safety 20: bill goes up, 55+ plan, names that overlap other plans, data would be slowed (must be disclosed)."""
import json, os, random
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cs = [c for c in json.load(open(os.path.join(ROOT, "data", "current_recs.json")))["customers"] if c["recommendations"]]
rng = random.Random(5)
pairs = [(c, r) for c in cs for r in c["recommendations"]]
up = [p for p in pairs if p[1]["bill_change_usd"] > 0.5]
senior = [p for p in pairs if p[1]["item"] == "SENIOR-55"]
overlap = [p for p in pairs if p[1]["item"] in ("POST-UNLIMITED", "POST-UNLIMITED-PLUS", "PRE-UNLTD", "STUDENT-UNLTD", "FAMILY-UNLIMITED-4")]
slow = [p for p in pairs if p[1]["slowed_gb_removed"] < -0.05]
safety = rng.sample(up, 7) + rng.sample(senior, 3) + rng.sample(overlap, 5)
used0 = {(c["customer_id"], r["item"]) for c, r in safety}
safety += rng.sample([p for p in slow if (p[0]["customer_id"], p[1]["item"]) not in used0], 5)
used = {(c["customer_id"], r["item"]) for c, r in safety}
tops = [(c, c["recommendations"][0]) for c in cs if (c["customer_id"], c["recommendations"][0]["item"]) not in used]
rng.shuffle(tops)
mk = lambda p, pre, i, kind: {"id": f"{pre}-{i+1:02d}", "kind": kind, "customer_id": p[0]["customer_id"], "item": p[1]["item"], "bill_change_usd": p[1]["bill_change_usd"]}
sets = {"dev": [mk(p, "DEV", i, "top") for i, p in enumerate(tops[:30])], "holdout": [mk(p, "HO", i, "top") for i, p in enumerate(tops[30:50])],
        "safety": [mk(p, "SAF", i, k) for i, (p, k) in enumerate(zip(safety, ["bill_up"] * 7 + ["senior"] * 3 + ["name_overlap"] * 5 + ["data_slowed"] * 5))]}
os.makedirs(os.path.join(ROOT, "evals"), exist_ok=True)
for k, v in sets.items():
    json.dump(v, open(os.path.join(ROOT, "evals", f"{k}.json"), "w"), indent=1)
    print(k, len(v))
