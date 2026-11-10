// In-app message for a recommendation. The recommender decides WHAT to suggest and WHY (from usage and the
// catalogue); the LLM only writes a short, honest message from those facts. Code checks every message before it is
// shown; if a check fails twice, or every model is down, a template message is shown instead.
import { readFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { generateJSON, MODELS } from "./llm.js";

const HERE = dirname(fileURLToPath(import.meta.url));
const REC = JSON.parse(readFileSync(join(HERE, "..", "data", "current_recs.json"), "utf8"));
const BY_ID = new Map(REC.customers.map((c) => [c.customer_id, c]));
export const CATALOGUE = REC.catalogue;
export const SNAPSHOT = REC.snapshot;
export const getCustomer = (id) => BY_ID.get(id);

const PRESSURE = /\b(act now|hurry|limited time|today only|last chance|don't miss|expires?|only \d+ left|before it's too late)\b/i;
const PROMISES = /\b(guarantee[ds]?|for life|forever|free (phone|month|months|plan|upgrade)|waive|refund|credit (to )?your account|price lock|never (increase|go up)|no extra cost)\b/i;
const SENSITIVE = /\b(gender|male|female|ethnic\w*|religio\w*|disab\w*|income|credit score|region|north|south|east|west|central)\b/i;
const MODEL_TALK = /\b(algorithm|machine learning|our model|the model|ai (picked|chose|selected)|predict\w*|score[ds]?|profil\w*|tracked|monitor\w*)\b/i;

const SYSTEM_PROMPT = `You write a short in-app message for a NovaCell customer suggesting ONE plan change or add-on. The recommendation, its reasons and the bill impact are already decided. You only write them up kindly and honestly.

RULES:
1. Use only the FACTS. Every number you write must appear in the FACTS exactly (prices, GB, minutes, days, dollar amounts). Do not invent prices, dates or allowances.
2. Mention only the RECOMMENDED item by name. Do not mention other plans, add-ons, discounts, credits, refunds or free items.
3. Be honest about money and data: if the monthly bill goes up, say by how much; if it goes down, say by how much. If the FACTS say some data would be at slowed speed, say so with the GB amount; never say the plan covers all of the usage.
4. No pressure: no "act now", "limited time", deadlines or scarcity.
5. Do not say how the suggestion was made (no "our AI", "algorithm", "we tracked", "score"). Refer to "your recent usage".
6. Never mention gender, region or other personal traits. Age may be mentioned only as the plan's eligibility rule if the item is the 55+ plan.
7. Plain language, short sentences, friendly.

Reply with JSON only:
{"headline":"<under 70 characters>","body":"<2 or 3 sentences: what we suggest, why it fits the recent usage, the monthly bill change>","cta":"<button label under 25 characters>"}`;

function money(n) {
  return `$${Math.abs(n).toFixed(2).replace(/\.00$/, "")}`;
}

export function factsFor(c, rec) {
  const item = CATALOGUE[rec.item];
  const u = c.usage;
  const lines = [
    `Current plan: ${c.plan_name}${c.addons.length ? ` with ${c.addons.map((a) => CATALOGUE[a].name).join(", ")}` : ""}`,
    `Lines on account: ${c.lines}`,
    `Average monthly bill over the last 3 months: ${money(c.paid_now_usd)}`,
    `Recent usage per month (3-month average): ${u.data_gb} GB data, ${u.hotspot_gb} GB hotspot, ${u.abroad_days} days abroad, ${u.intl_minutes} minutes of international calls`,
    `RECOMMENDED: ${item.name} (${item.kind === "plan" ? "plan change" : "add-on"}), list price ${money(item.price)}${item.per_day ? " per day abroad, charged for at most 10 days a month" : " per month"}`,
    `Reasons: ${rec.reasons.join("; ")}`,
    rec.bill_change_usd < 0 ? `Monthly bill goes down by about ${money(rec.bill_change_usd)}` : rec.bill_change_usd > 0 ? `Monthly bill goes up by about ${money(rec.bill_change_usd)}` : "Monthly bill stays about the same",
  ];
  if (rec.slowed_gb_removed > 0) lines.push(`Data no longer slowed: about ${rec.slowed_gb_removed} GB a month`);
  if (rec.slowed_gb_removed < -0.05) lines.push(`Downside to state: the ${item.data_cap_gb} GB data cap means about ${-rec.slowed_gb_removed} GB a month of recent usage would be at slowed speed`);
  return lines.join("\n");
}

const numbersIn = (s) => (String(s).match(/\d+(?:\.\d+)?/g) || []).map((n) => String(Number(n)));

export function checkMessage(msg, c, rec) {
  const problems = [];
  const text = [msg.headline, msg.body, msg.cta].join(" ");
  if (!msg.headline || !msg.body) problems.push("missing headline or body");
  const allowed = new Set(numbersIn(factsFor(c, rec)).concat(["1", "2", "3", "4", "5", "10", "30", "55", "611"]));
  const bad = numbersIn(text).filter((n) => !allowed.has(n));
  if (bad.length) problems.push(`numbers not in the facts: ${[...new Set(bad)].join(", ")}`);
  const others = Object.values(CATALOGUE).filter((it) => it.id !== rec.item && !c.addons.includes(it.id) && it.id !== c.plan && new RegExp(`\\b${it.name.replace(/[()+]/g, ".").split(" (")[0]}\\b(?! (lines?|data|5G))`).test(text)); // case-sensitive: product names are capitalised
  const recName = CATALOGUE[rec.item].name.split(" (")[0];
  const othersReal = others.filter((o) => !recName.toLowerCase().includes(o.name.split(" (")[0].toLowerCase()));
  if (othersReal.length) problems.push(`mentions other items: ${othersReal.map((o) => o.name).join(", ")}`);
  if (PRESSURE.test(text)) problems.push("pressure wording");
  if (PROMISES.test(text)) problems.push("promise or freebie wording");
  if (SENSITIVE.test(text)) problems.push("personal trait wording");
  if (MODEL_TALK.test(text)) problems.push("talks about how it was picked");
  if (rec.bill_change_usd > 0.5 && !/(up|more|increase|extra|add)/i.test(msg.body)) problems.push("does not say the bill goes up");
  if (rec.slowed_gb_removed < -0.05 && !/slow/i.test(msg.body)) problems.push("does not say some data would be slowed");
  return problems;
}

export function templateMessage(c, rec) {
  const item = CATALOGUE[rec.item];
  const change = rec.bill_change_usd < -0.5 ? `Your monthly bill would go down by about ${money(rec.bill_change_usd)}.` : rec.bill_change_usd > 0.5 ? `Your monthly bill would go up by about ${money(rec.bill_change_usd)}.` : "Your monthly bill would stay about the same.";
  return {
    headline: item.kind === "plan" ? `${item.name} may fit you better` : `Add ${item.name}?`,
    body: `Based on your recent usage: ${rec.reasons[0].charAt(0).toLowerCase()}${rec.reasons[0].slice(1)}. ${change}${rec.slowed_gb_removed < -0.05 ? ` With its ${item.data_cap_gb} GB cap, about ${-rec.slowed_gb_removed} GB a month of your data would be at slowed speed.` : ""}`,
    cta: item.kind === "plan" ? "See plan details" : "See add-on details",
  };
}

export async function buildMessage(customerId, itemId) {
  const c = getCustomer(customerId);
  if (!c) return { status: "error", error: "unknown customer" };
  const rec = c.recommendations.find((r) => r.item === itemId) || c.recommendations[0];
  if (!rec) return { status: "none", message: null, note: "No recommendation: the current plan fits this customer's usage." };
  const facts = factsFor(c, rec);
  const started = Date.now();
  const usage = { prompt_tokens: 0, completion_tokens: 0 };
  let messages = [{ role: "system", content: SYSTEM_PROMPT }, { role: "user", content: `FACTS:\n${facts}` }];
  try {
    for (let attempt = 0; attempt < 2; attempt++) {
      const r = await generateJSON(messages);
      usage.prompt_tokens += r.usage.prompt_tokens || 0;
      usage.completion_tokens += r.usage.completion_tokens || 0;
      const msg = r.parsed;
      const problems = checkMessage(msg, c, rec);
      if (!problems.length) return { status: "generated", message: msg, item: rec.item, facts, modelUsed: { provider: r.provider, model: r.model, tier: r.tier }, repaired: attempt > 0, usage, latencyMs: Date.now() - started };
      messages = [...messages, { role: "assistant", content: JSON.stringify(msg) }, { role: "user", content: `Fix these problems and reply with the full JSON again: ${problems.join("; ")}` }];
      if (attempt === 1) return { status: "template", message: templateMessage(c, rec), item: rec.item, facts, guardrailNote: `AI message failed checks: ${problems.join("; ")}`, usage, latencyMs: Date.now() - started };
    }
  } catch (e) {
    return { status: "template", message: templateMessage(c, rec), item: rec.item, facts, guardrailNote: `AI unavailable: ${e.message}`, usage, latencyMs: Date.now() - started };
  }
}

export { MODELS };
