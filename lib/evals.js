// Live evals for the in-app message. Each case calls buildMessage() and scores what would be SHOWN.
import { buildMessage, checkMessage, getCustomer } from "./message.js";

export async function runCase(tc) {
  const r = await buildMessage(tc.customer_id, tc.item);
  const c = getCustomer(tc.customer_id);
  const rec = c.recommendations.find((x) => x.item === tc.item);
  const shownProblems = r.message ? checkMessage(r.message, c, rec) : ["no message"];
  const text = r.message ? `${r.message.headline} ${r.message.body}` : "";
  return {
    id: tc.id, kind: tc.kind, customer_id: tc.customer_id, item: tc.item, status: r.status, model: r.modelUsed?.model || null, tier: r.modelUsed?.tier || null,
    repaired: !!r.repaired, latency_ms: r.latencyMs, usage: r.usage || null, guardrailNote: r.guardrailNote || null,
    safe_shown: shownProblems.length === 0,
    names_item: new RegExp(c && rec ? rec.name.split(" (")[0].replace(/[+]/g, ".") : "^$", "i").test(text),
    bill_direction_right: rec.bill_change_usd > 0.5 ? /(up|more|increase|extra|add)/i.test(text) : rec.bill_change_usd < -0.5 ? /(down|less|lower|save|reduce|cheaper)/i.test(text) : true,
  };
}

export function summarise(rows) {
  const n = rows.length;
  const pct = (a, b) => (b ? Math.round((1000 * a) / b) / 10 : null);
  const ai = rows.filter((r) => r.status === "generated");
  const lat = rows.map((r) => r.latency_ms).filter(Boolean).sort((a, b) => a - b);
  const q = (p) => (lat.length ? lat[Math.min(lat.length - 1, Math.floor(p * lat.length))] : null);
  return {
    cases: n, ai_message_rate: pct(ai.length, n), safe_shown: pct(rows.filter((r) => r.safe_shown).length, n),
    names_item: pct(rows.filter((r) => r.names_item).length, n), bill_direction_right: pct(rows.filter((r) => r.bill_direction_right).length, n),
    repaired: rows.filter((r) => r.repaired).length, primary_model_share: pct(rows.filter((r) => r.tier === "primary").length, n),
    latency_p50_ms: q(0.5), latency_p95_ms: q(0.95),
    avg_tokens_in: ai.length ? Math.round(ai.reduce((s, r) => s + (r.usage?.prompt_tokens || 0), 0) / ai.length) : null,
    avg_tokens_out: ai.length ? Math.round(ai.reduce((s, r) => s + (r.usage?.completion_tokens || 0), 0) / ai.length) : null,
  };
}
