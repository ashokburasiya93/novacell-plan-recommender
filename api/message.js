import { buildMessage } from "../lib/message.js";

export default async function handler(req, res) {
  const { customer_id: id, item } = req.method === "POST" ? req.body || {} : req.query || {};
  if (!id) return res.status(400).json({ error: "customer_id required" });
  res.status(200).json(await buildMessage(String(id), item ? String(item) : undefined));
}
