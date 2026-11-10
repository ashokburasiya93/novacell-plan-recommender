// Offline checks: every template message passes the checker, and deliberately broken messages are caught.
import { readFileSync } from "node:fs";
import { checkMessage, templateMessage, getCustomer } from "../lib/message.js";

const recs = JSON.parse(readFileSync(new URL("../data/current_recs.json", import.meta.url))).customers.filter((c) => c.recommendations.length);
let ok = 0, fail = 0;
for (const c of recs) for (const r of c.recommendations) {
  const p = checkMessage(templateMessage(c, r), c, r);
  if (p.length) { fail++; if (fail < 5) console.log("template failed", c.customer_id, r.item, p); } else ok++;
}
console.log(`templates: ${ok} pass, ${fail} fail`);
const c = recs.find((x) => x.recommendations[0].bill_change_usd < 0);
const r = c.recommendations[0];
const broken = [
  { headline: "Switch now", body: "Save $999 a month with this plan.", cta: "Go" },
  { headline: "Limited time offer", body: `${r.name} fits your usage. Act now before it expires.`, cta: "Switch" },
  { headline: "Our AI picked this", body: `Our model predicted ${r.name} is best for you.`, cta: "Switch" },
  { headline: "Get Unlimited Plus", body: "Unlimited Plus and a free phone, guaranteed.", cta: "Switch" },
  { headline: "Customers in the East", body: `Customers in the East region like ${r.name}.`, cta: "Switch" },
];
let caught = 0;
for (const b of broken) if (checkMessage(b, c, r).length) caught++; else console.log("NOT caught:", b);
console.log(`broken messages caught: ${caught}/${broken.length}`);
const up = recs.flatMap((x) => x.recommendations.map((rr) => [x, rr])).find(([, rr]) => rr.bill_change_usd > 1);
if (up) { const msg = { headline: "A better fit", body: `${up[1].name} fits your recent usage.`, cta: "See details" }; console.log("hides bill increase caught:", checkMessage(msg, up[0], up[1]).length > 0); }
const slow = recs.flatMap((x) => x.recommendations.map((rr) => [x, rr])).find(([, rr]) => rr.slowed_gb_removed < -0.05);
let slowCaught = true;
if (slow) { const msg = { headline: "A better fit", body: `${slow[1].name} matches your recent usage.`, cta: "See details" }; slowCaught = checkMessage(msg, slow[0], slow[1]).length > 0; console.log("hides data slowdown caught:", slowCaught); }
if (fail || caught < broken.length || !slowCaught) process.exit(1);
