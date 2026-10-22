// SENTRA demo bot (always online, runs on Vercel).
//
// Lets anyone try the resident side of SENTRA on Telegram without the
// on-site system running: register, Away Mode, then a simulated intruder
// that sends a real SENTRA clip as a verified alert.
//
// Vercel > Project > Settings > Environment Variables:
//   TELEGRAM_BOT_TOKEN   token of the DEMO bot from @BotFather
//   TELEGRAM_SECRET      any random word (letters, numbers, _ or -), e.g. sentra_demo_83kq
// Then open once in a browser:  https://<your-site>/api/telegram?setup=<TELEGRAM_SECRET>

const TOKEN = process.env.TELEGRAM_BOT_TOKEN;
const SECRET = process.env.TELEGRAM_SECRET;
const HOUSE = "12";                       // the demo house has a porch camera (CAM 02)
const CLIP = "/media/clip_1348f69f.mp4";  // real SENTRA Away Mode breach clip

const KEYBOARD = {
  keyboard: [[{ text: "/register 12" }, { text: "/away_on" }], [{ text: "/status" }, { text: "/away_off" }]],
  resize_keyboard: true,
  is_persistent: true,
};
const SIMULATE = { inline_keyboard: [[{ text: "🚶 Simulate an intruder at House 12", callback_data: "simulate" }]] };

const HELP =
  "👋 Welcome to SENTRA, the AI night guard for your township.\n\n" +
  "SENTRA watches the township CCTV. When it sees something wrong, a guard checks a 10-second clip first. " +
  "Only confirmed alerts reach you. No facial recognition.\n\n" +
  "Try it (demo):\n" +
  "1. Link your house:  /register 12\n" +
  "2. Going away? Turn on Away Mode:  /away_on\n" +
  "3. Tap \"Simulate an intruder\" and receive a verified alert\n" +
  "4. Back home:  /away_off\n\n" +
  "This is the always-online demo. In the pilot, alerts come from live cameras through the guard console.";

async function tg(method, params) {
  const r = await fetch(`https://api.telegram.org/bot${TOKEN}/${method}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });
  const data = await r.json();
  if (!data.ok) console.error(`Telegram ${method} failed: ${data.description}`);
  return data;
}

const reply = (chat_id, text, extra = {}) => tg("sendMessage", { chat_id, text, ...extra });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function nowMY() {
  return new Date().toLocaleTimeString("en-MY", { timeZone: "Asia/Kuala_Lumpur", hour: "2-digit", minute: "2-digit", hour12: true }).toUpperCase();
}

function siteUrl(req) {
  const host = req.headers["x-forwarded-host"] || req.headers.host;
  return `https://${host}`;
}

async function simulate(chat_id, req) {
  await reply(chat_id,
    "👀 SENTRA detected a person at House 12 while Away Mode is ON.\n" +
    "The 10-second clip has been sent to the guard on duty for checking...");
  await tg("sendChatAction", { chat_id, action: "upload_video" });
  await sleep(3000);
  const caption =
    "🚨 SENTRA VERIFIED ALERT\n" +
    "Away Mode breach: Person at House 12, which is in Away Mode\n" +
    `CAM 02 Porch, House 12, ${nowMY()}\n` +
    "Confirmed by the guard on duty.\n" +
    "Your house is in Away Mode. Reply /away_off when you are back home.";
  const sent = await tg("sendVideo", { chat_id, video: siteUrl(req) + CLIP, caption, supports_streaming: true });
  if (!sent.ok) await reply(chat_id, caption);   // still deliver the text if the video fails
}

async function handleMessage(msg, req) {
  const chat_id = msg.chat.id;
  const text = (msg.text || "").trim();
  const name = msg.from?.first_name || "there";
  const [raw, arg] = text.split(/\s+/);
  const cmd = (raw || "").split("@")[0].toLowerCase();

  if (!text.startsWith("/")) return reply(chat_id, "Send /start to see what I can do.", { reply_markup: KEYBOARD });
  if (cmd === "/start" || cmd === "/help") return reply(chat_id, HELP, { reply_markup: KEYBOARD });
  if (cmd === "/register") {
    if (!arg) return reply(chat_id, "Tell me your house number, e.g. /register 12");
    const note = arg === HOUSE ? "" : `\n(Demo: House ${HOUSE} is the house with a camera, so we use House ${HOUSE}.)`;
    return reply(chat_id, `Done, ${name}. This chat is linked to House ${HOUSE}. Verified alerts for your house will come here.${note}\n\nNext: /away_on`);
  }
  if (cmd === "/away_on") {
    const days = parseInt(arg, 10);
    let until = "";
    if (days > 0) {
      const d = new Date(Date.now() + days * 86400000);
      until = " until " + d.toLocaleDateString("en-CA", { timeZone: "Asia/Kuala_Lumpur" });
    }
    return reply(chat_id,
      `🏠 Away Mode is ON for House ${HOUSE}${until}.\n` +
      "SENTRA will now alert the guard on any person at your house, day or night.\n\n" +
      "Tap below to see what happens when someone comes to your house.",
      { reply_markup: SIMULATE });
  }
  if (cmd === "/demo_alert") return simulate(chat_id, req);
  if (cmd === "/away_off") return reply(chat_id, `Welcome home, ${name}. Away Mode is OFF for House ${HOUSE}.`, { reply_markup: KEYBOARD });
  if (cmd === "/status")
    return reply(chat_id,
      `House ${HOUSE} (demo)\nCameras watching your area: CAM 02 Porch, CAM 01 Main Gate\n` +
      "Guard on duty: online\nTry /away_on, then tap \"Simulate an intruder\".", { reply_markup: KEYBOARD });
  if (cmd === "/unregister") return reply(chat_id, "You will no longer receive SENTRA alerts. Send /register 12 to join again.");
  return reply(chat_id, "I don't know that command. Send /start for help.", { reply_markup: KEYBOARD });
}

async function setup(req, res) {
  const url = siteUrl(req) + "/api/telegram";
  const results = {
    webhook: await tg("setWebhook", { url, secret_token: SECRET, allowed_updates: ["message", "callback_query"], drop_pending_updates: true }),
    commands: await tg("setMyCommands", { commands: [
      { command: "start", description: "How SENTRA works" },
      { command: "register", description: "Link your house, e.g. /register 12" },
      { command: "away_on", description: "Turn on Away Mode" },
      { command: "away_off", description: "Turn off Away Mode" },
      { command: "status", description: "Your house and cameras" },
      { command: "demo_alert", description: "Simulate a verified alert" },
    ] }),
    description: await tg("setMyDescription", { description:
      "SENTRA sends you verified security alerts from your township's CCTV. Only alerts confirmed by the guard reach you. No facial recognition. Press Start to try the demo." }),
    me: await tg("getMe", {}),
  };
  const ok = results.webhook.ok;
  res.status(ok ? 200 : 500).json({
    ok,
    bot: results.me.ok ? "@" + results.me.result.username : null,
    webhook: url,
    message: ok ? "Demo bot is live. Open it in Telegram and press Start." : "Setup failed, see details.",
    details: Object.fromEntries(Object.entries(results).map(([k, v]) => [k, v.ok ? "ok" : v.description])),
  });
}

module.exports = async (req, res) => {
  if (!TOKEN || !SECRET) return res.status(500).json({ ok: false, message: "Set TELEGRAM_BOT_TOKEN and TELEGRAM_SECRET in Vercel, then redeploy." });

  if (req.method === "GET") {
    if (req.query.setup && req.query.setup === SECRET) return setup(req, res);
    return res.status(200).json({ ok: true, message: "SENTRA demo bot endpoint." });
  }

  if (req.headers["x-telegram-bot-api-secret-token"] !== SECRET) return res.status(401).end();

  try {
    const u = req.body || {};
    if (u.message) await handleMessage(u.message, req);
    else if (u.callback_query) {
      const q = u.callback_query;
      await tg("answerCallbackQuery", { callback_query_id: q.id });
      if (q.data === "simulate" && q.message) await simulate(q.message.chat.id, req);
    }
  } catch (e) {
    console.error(e);
  }
  res.status(200).json({ ok: true });   // always 200 so Telegram does not retry
};
