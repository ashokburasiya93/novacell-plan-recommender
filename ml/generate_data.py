"""Generate the synthetic NovaCell data for Plan Recommender.

ALL DATA IS SYNTHETIC (seed 7). NovaCell is a made-up operator; the catalogue matches the plan catalogue
in the NovaCell support assistant's knowledge base. No real company, customer or person is described.

What is simulated, month by month from January to October 2026, for 8,000 customers:
  * usage: data GB, hotspot GB, days abroad, international call minutes (noisy around each customer's habit)
  * what each customer pays: plan price + roaming and calling charges, and the value they lose when throttled
  * adoptions: some customers change plan or add an add-on each month. They pick, noisily, the option that
    fits their usage best, nudged by how popular it is, what they already own, and the in-app banner that
    has been promoting the Unlimited plan to everyone.

Outputs (data/):
  catalogue.json         the 14 plans and add-ons with prices and allowances
  customers.csv          static customer attributes
  usage_monthly.csv      usage per customer per month (Jan-Oct)
  holdings_monthly.csv   plan and add-ons held at the start of each month
  adoptions.csv          plan changes and add-on purchases (month, customer, item)

Hidden from the models: each customer's taste for premium perks and the exact cost assumptions below.

Usage: python3 ml/generate_data.py
"""
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data")
SEED = 7
N = 8000
MONTHS = pd.period_range("2026-01", "2026-10", freq="M")

# Catalogue: prices and allowances from the NovaCell plan catalogue (synthetic, last updated 2026-06-01).
CAT = [
    # id, name, kind, price, data_cap_gb (None = unlimited), hotspot_gb, extras
    ("PRE-STARTER-5", "Starter 5 (prepaid)", "plan", 15, 5, 0, {"prepaid": True}),
    ("PRE-VALUE-15", "Value 15 (prepaid)", "plan", 25, 15, 5, {"prepaid": True}),
    ("PRE-UNLTD", "Prepaid Unlimited", "plan", 45, None, 10, {"prepaid": True}),
    ("POST-ESSENTIAL", "Essential", "plan", 35, 20, 10, {}),
    ("POST-UNLIMITED", "Unlimited", "plan", 60, None, 25, {}),
    ("POST-UNLIMITED-PLUS", "Unlimited Plus", "plan", 80, None, 50, {"roaming": True, "streaming": True}),
    ("FAMILY-UNLIMITED-4", "Family Unlimited (4 lines)", "plan", 140, None, 25, {"family": True}),
    ("SENIOR-55", "55+ Plan (2 lines)", "plan", 40, None, 5, {"senior": True}),
    ("STUDENT-UNLTD", "Student Unlimited", "plan", 30, None, 15, {"student": True, "streaming": True}),
    ("DATA-ONLY-30", "Data-Only 30 (tablet line)", "addon", 30, 30, 30, {"tablet": True}),
    ("ADDON-INTL-DAYPASS", "International Day Pass", "addon", 10, None, 0, {"per_day": True}),
    ("ADDON-INTL-MONTHLY", "International Monthly Pass", "addon", 50, None, 0, {}),
    ("ADDON-INTL-CALLING", "International Calling Pack", "addon", 15, None, 0, {}),
    ("ADDON-HOTSPOT-BOOST", "Hotspot Boost", "addon", 10, None, 15, {}),
]
ITEMS = [c[0] for c in CAT]
C = {c[0]: {"id": c[0], "name": c[1], "kind": c[2], "price": c[3], "data_cap_gb": c[4], "hotspot_gb": c[5], **c[6]} for c in CAT}
PLANS = [i for i in ITEMS if C[i]["kind"] == "plan"]
ADDONS = [i for i in ITEMS if C[i]["kind"] == "addon"]

# Cost assumptions (synthetic; documented in the README and the cost sheet)
ROAM_PAYG_PER_DAY = 15.0   # no pass: pay-as-you-go roaming per day abroad
INTL_PAYG_PER_MIN = 0.40   # no calling pack: per minute to international numbers
THROTTLE_VALUE_PER_GB = 1.5  # value a customer loses per GB above the cap (slowed data)
HOTSPOT_VALUE_PER_GB = 1.0
TABLET_VALUE = 24.0        # value of a connected tablet for customers who own one
STREAM_VALUE = 9.0         # value of an included streaming subscription to streamers


def plan_price(plan, lines):
    c = C[plan]
    if c.get("family"):
        return 140 + 30 * max(0, lines - 4)
    if c.get("senior"):
        return 40 + 35 * max(0, lines - 2)
    return c["price"] * lines


def monthly_cost(plan, addons, u, cust):
    """Money paid plus value lost, for one month of usage u under a plan and add-ons. Returns (paid, pain)."""
    c = C[plan]
    paid = plan_price(plan, cust["lines"])
    pain = 0.0
    cap = c["data_cap_gb"]
    if cap is not None and u["data_gb"] > cap:
        pain += (u["data_gb"] - cap) * THROTTLE_VALUE_PER_GB
    hs = c["hotspot_gb"] + (15 if "ADDON-HOTSPOT-BOOST" in addons else 0)
    if u["hotspot_gb"] > hs:
        pain += (u["hotspot_gb"] - hs) * HOTSPOT_VALUE_PER_GB
    if "ADDON-HOTSPOT-BOOST" in addons:
        paid += 10
    days = u["abroad_days"]
    if c.get("roaming"):
        pass
    elif "ADDON-INTL-MONTHLY" in addons:
        paid += 50
    elif "ADDON-INTL-DAYPASS" in addons:
        paid += 10 * min(days, 10)
    else:
        paid += ROAM_PAYG_PER_DAY * days
    mins = u["intl_minutes"]
    paid += 15 if "ADDON-INTL-CALLING" in addons else INTL_PAYG_PER_MIN * mins
    if "DATA-ONLY-30" in addons:
        paid += 30
    elif cust["has_tablet"]:
        pain += TABLET_VALUE
    if cust["streamer"] and (c.get("streaming")):
        pain -= STREAM_VALUE
    return paid, pain


def eligible(item, cust, plan, addons):
    c = C[item]
    if item == plan or item in addons:
        return False
    if c["kind"] == "plan":
        if c.get("student") and not cust["student"]:
            return False
        if c.get("senior") and not (cust["age_55_plus"] and cust["lines"] <= 2):
            return False
        if c.get("family") and cust["lines"] < 3:
            return False
        if not c.get("prepaid") and C[plan].get("prepaid") and not cust["credit_ok"]:
            return False
        if c.get("prepaid") and (cust["lines"] > 1 or not C[plan].get("prepaid")):
            return False  # moving from postpaid to prepaid needs a new SIM and account: not offered as an in-app change
        return True
    if item == "ADDON-HOTSPOT-BOOST" and C[plan]["hotspot_gb"] == 0:
        return False
    if item in ("ADDON-INTL-DAYPASS", "ADDON-INTL-MONTHLY") and C[plan].get("roaming"):
        return False
    if item == "ADDON-INTL-DAYPASS" and "ADDON-INTL-MONTHLY" in addons:
        return False
    if item == "ADDON-INTL-MONTHLY" and "ADDON-INTL-DAYPASS" in addons:
        return True  # upgrade from day pass replaces it
    if C[plan].get("prepaid") and c["kind"] == "addon" and item in ("ADDON-INTL-MONTHLY", "DATA-ONLY-30"):
        return False
    return True


def apply(item, plan, addons):
    if C[item]["kind"] == "plan":
        keep = {a for a in addons if not (C[item].get("roaming") and a in ("ADDON-INTL-DAYPASS", "ADDON-INTL-MONTHLY"))}
        return item, keep
    new = set(addons) | {item}
    if item == "ADDON-INTL-MONTHLY":
        new.discard("ADDON-INTL-DAYPASS")
    return plan, new


def main():
    rng = np.random.default_rng(SEED)
    os.makedirs(OUT, exist_ok=True)
    age_band = rng.choice(["18-24", "25-34", "35-44", "45-54", "55-64", "65+"], N, p=[0.13, 0.22, 0.2, 0.17, 0.15, 0.13])
    gender = rng.choice(["F", "M"], N)
    region = rng.choice(["North", "South", "East", "West", "Central"], N, p=[0.24, 0.22, 0.18, 0.2, 0.16])
    lines = rng.choice([1, 2, 3, 4, 5], N, p=[0.52, 0.2, 0.12, 0.12, 0.04])
    student = (age_band == "18-24") & (rng.random(N) < 0.45)
    age55 = np.isin(age_band, ["55-64", "65+"])
    prepaid = (lines == 1) & (rng.random(N) < np.where(age_band == "18-24", 0.55, 0.32))
    credit_ok = rng.random(N) < 0.8
    data_habit = np.clip(rng.lognormal(np.log(9), 0.85, N) * np.where(age55, 0.55, 1) * np.where(student, 1.5, 1), 0.3, 90)
    hotspot_habit = np.where(rng.random(N) < 0.3, rng.gamma(1.6, 7, N), 0)
    traveler = rng.random(N) < 0.16
    travel_rate = np.where(traveler, rng.gamma(1.5, 2.8, N), 0)  # mean days abroad per month
    intl_caller = rng.random(N) < np.where(np.isin(region, ["West", "North"]), 0.22, 0.14)
    intl_habit = np.where(intl_caller, rng.gamma(2, 45, N), rng.poisson(2, N))
    has_tablet = rng.random(N) < 0.18
    streamer = rng.random(N) < 0.35
    premium_taste = rng.normal(0, 1, N)  # hidden
    tenure = np.clip(rng.gamma(2, 15, N).round(), 1, 140).astype(int)
    cust = pd.DataFrame({"customer_id": [f"NC-{400000 + i}" for i in range(N)], "age_band": age_band, "gender": gender, "region": region,
                         "lines": lines, "student": student, "age_55_plus": age55, "credit_ok": credit_ok, "has_tablet": has_tablet,
                         "streamer": streamer, "tenure_months_at_jan": tenure})

    # initial plans: what people tend to be on today (not necessarily the best fit)
    plans0 = []
    for i in range(N):
        h = data_habit[i]  # people roughly, not exactly, pick a plan that fits their usage
        if prepaid[i]:
            opts = ["PRE-STARTER-5", "PRE-VALUE-15", "PRE-UNLTD"]
            p = [0.6, 0.3, 0.1] if h < 5 else ([0.3, 0.5, 0.2] if h < 15 else [0.15, 0.45, 0.4])
        elif lines[i] >= 3:
            opts, p = ["FAMILY-UNLIMITED-4", "POST-UNLIMITED", "POST-ESSENTIAL"], [0.5, 0.3, 0.2]
        else:
            opts = ["POST-ESSENTIAL", "POST-UNLIMITED", "POST-UNLIMITED-PLUS", "SENIOR-55", "STUDENT-UNLTD"]
            p = [0.55, 0.26, 0.05, 0.08, 0.06] if h < 8 else ([0.42, 0.38, 0.08, 0.07, 0.05] if h < 20 else [0.25, 0.5, 0.15, 0.05, 0.05])
        choice = rng.choice(opts, p=p)
        c = cust.iloc[i]
        if not eligible(choice, c, "PRE-STARTER-5" if not prepaid[i] else "POST-ESSENTIAL", set()) and choice in ("SENIOR-55", "STUDENT-UNLTD"):
            choice = "POST-ESSENTIAL"
        plans0.append(choice)
    holdings = {i: (plans0[i], set()) for i in range(N)}
    for i in range(N):  # some already hold add-ons
        a = set()
        if intl_caller[i] and rng.random() < 0.35 and not prepaid[i]:
            a.add("ADDON-INTL-CALLING")
        if traveler[i] and rng.random() < 0.3 and not C[plans0[i]].get("roaming"):
            a.add("ADDON-INTL-DAYPASS")
        if has_tablet[i] and rng.random() < 0.3 and not prepaid[i]:
            a.add("DATA-ONLY-30")
        holdings[i] = (plans0[i], a)

    pop_prior = {it: 0.0 for it in ITEMS}
    pop_prior.update({"POST-UNLIMITED": 0.9, "ADDON-INTL-DAYPASS": 0.4, "POST-UNLIMITED-PLUS": 0.3, "ADDON-HOTSPOT-BOOST": 0.2})
    BANNER = {"POST-UNLIMITED": 0.6}  # the current in-app banner promotes Unlimited to everyone

    usage_rows, hold_rows, adopt_rows = [], [], []
    season = {m: (1.6 if m.month in (5, 6, 7, 12) else 1.0) for m in MONTHS}  # summer travel
    for m in MONTHS:
        growth = 1 + 0.015 * (m.month - 1)
        for i in range(N):
            plan, addons = holdings[i]
            hold_rows.append((str(m), cust.customer_id[i], plan, "|".join(sorted(addons))))
            u = {"data_gb": round(float(np.clip(data_habit[i] * growth * rng.lognormal(0, 0.25), 0.1, 150)), 1),
                 "hotspot_gb": round(float(hotspot_habit[i] * rng.lognormal(0, 0.35)), 1),
                 "abroad_days": int(rng.poisson(travel_rate[i] * season[m])),
                 "intl_minutes": int(rng.poisson(intl_habit[i]))}
            c = cust.iloc[i]
            paid, pain = monthly_cost(plan, addons, u, c)
            usage_rows.append((str(m), cust.customer_id[i], u["data_gb"], u["hotspot_gb"], u["abroad_days"], u["intl_minutes"], round(paid, 2), round(pain, 2)))
            # decision for next month, based on the customer's typical usage (habit), not just this month
            if m == MONTHS[-1]:
                continue
            habit = {"data_gb": data_habit[i] * growth, "hotspot_gb": hotspot_habit[i], "abroad_days": travel_rate[i] * 1.2, "intl_minutes": intl_habit[i]}
            base = sum(monthly_cost(plan, addons, habit, c))
            gains = {}
            for it in ITEMS:
                if not eligible(it, c, plan, addons):
                    continue
                p2, a2 = apply(it, plan, addons)
                gains[it] = base - sum(monthly_cost(p2, a2, habit, c))
            if not gains:
                continue
            best = max(gains.values())
            shop = 1 / (1 + np.exp(-(-4.0 + 0.045 * max(best, 0) + 0.9 * (pain > 8) + 0.6 * (u["abroad_days"] > 0))))
            if rng.random() > shop:
                continue
            keys = list(gains)
            util = np.array([0.075 * gains[k] + pop_prior[k] + BANNER.get(k, 0)
                             + (0.6 * premium_taste[i] if k in ("POST-UNLIMITED-PLUS", "POST-UNLIMITED") else 0)
                             + (0.8 if k in ("ADDON-INTL-DAYPASS", "ADDON-INTL-MONTHLY") and "ADDON-INTL-CALLING" in addons else 0)
                             - (2.5 if gains[k] < -15 else 0) for k in keys])
            util = util + rng.gumbel(0, 1.0, len(keys))
            pick = keys[int(np.argmax(util))]
            adopt_rows.append((str(m + 1), cust.customer_id[i], pick, round(gains[pick], 2)))
            holdings[i] = apply(pick, plan, addons)

    cust.to_csv(os.path.join(OUT, "customers.csv"), index=False)
    pd.DataFrame(usage_rows, columns=["month", "customer_id", "data_gb", "hotspot_gb", "abroad_days", "intl_minutes", "paid_usd", "value_lost_usd"]).to_csv(os.path.join(OUT, "usage_monthly.csv"), index=False)
    pd.DataFrame(hold_rows, columns=["month", "customer_id", "plan", "addons"]).to_csv(os.path.join(OUT, "holdings_monthly.csv"), index=False)
    ad = pd.DataFrame(adopt_rows, columns=["month", "customer_id", "item", "true_gain_usd"])
    ad.to_csv(os.path.join(OUT, "adoptions.csv"), index=False)
    json.dump({"items": [C[i] for i in ITEMS], "assumptions": {"roam_payg_per_day": ROAM_PAYG_PER_DAY, "intl_payg_per_min": INTL_PAYG_PER_MIN,
               "throttle_value_per_gb": THROTTLE_VALUE_PER_GB, "hotspot_value_per_gb": HOTSPOT_VALUE_PER_GB}, "synthetic": True},
              open(os.path.join(OUT, "catalogue.json"), "w"), indent=1)
    print(len(ad), "adoptions;", ad.groupby("month").size().to_dict())
    print(ad["item"].value_counts().to_dict())
    print("share with positive true gain:", round((ad.true_gain_usd > 0).mean(), 3))


if __name__ == "__main__":
    main()
