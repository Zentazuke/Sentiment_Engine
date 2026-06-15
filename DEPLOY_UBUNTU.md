# Deploying the Sentiment Engine on the Ubuntu server

Runs the API + live feeds as systemd services (auto-start, auto-restart, survive
reboots) plus a nightly calibration timer — alongside the trading bot, as the same
`kronos` user, in its own sibling directory (NOT nested inside the bot repo).

```
/home/kronos/
├── kronos-scalping-bot/      ← the bot (existing)
└── Sentiment_Engine/         ← the engine (this repo, its own clone + venv)
```

The bot reads the engine at `http://127.0.0.1:8787/signals/<SYMBOL>`, so the engine
binds to localhost only and never touches the internet.

---

## 0. Bot integration contract — already wired
The bot calls `GET /signals/BTC-USDT` and expects flat JSON
(`sentiment_score`, `fear_greed`, `funding_rate`, …). This route now ships in
`sentiment_engine/api.py`, mapped to this engine's real endpoints
(snapshot/context/positioning/derivatives/outlook). Just verify it after deploy with
the `curl` in step 6. Nothing to add.

## 1. Get the code (as the kronos user)
```bash
cd ~
git clone https://github.com/Zentazuke/Sentiment_Engine.git
cd Sentiment_Engine
```

## 2. Virtualenv + dependencies (its own venv — don't share the bot's)
```bash
python3 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 3. ⚠️ Check the box can reach Binance (do this BEFORE starting)
Microstructure, long/short, and funding feeds all need Binance. Some VPS regions
geo-block it. Test:
```bash
curl -s -o /dev/null -w "spot:  %{http_code}\n" https://api.binance.com/api/v3/ping
curl -s -o /dev/null -w "fapi:  %{http_code}\n" "https://fapi.binance.com/fapi/v1/ping"
```
- Both `200` → keep all sources.
- `451`/`403`/timeout → region geo-blocks Binance. Drop `binance lsr deriv` from
  `kronos-sentiment-feeds.service` (you'll lose microstructure + funding), or move the VPS.

## 4. Environment file
```bash
cp deploy/sentiment.env.example sentiment.env
nano sentiment.env   # keep DB under the repo, e.g. SENTIMENT_JOURNAL_DB=/home/kronos/Sentiment_Engine/data/sentiment_journal.db
```
Light mode (recommended on the shared 4 GB box): leave the scorer vars commented out
(fast lexicon, no CryptoBERT). The feeds service already omits `bluesky`.

## 5. Install + start the services
The unit files are preset to `User=kronos`, `/home/kronos/Sentiment_Engine`, light-mode feeds,
and named in the `kronos-*` family to sit alongside `kronos-bot` / `kronos-dashboard`.
```bash
sudo cp deploy/kronos-sentiment-engine.service    /etc/systemd/system/
sudo cp deploy/kronos-sentiment-feeds.service     /etc/systemd/system/
sudo cp deploy/kronos-sentiment-calibrate.service /etc/systemd/system/
sudo cp deploy/kronos-sentiment-calibrate.timer   /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now kronos-sentiment-engine.service kronos-sentiment-feeds.service
sudo systemctl enable --now kronos-sentiment-calibrate.timer
```

## 6. Verify it's alive
```bash
systemctl status kronos-sentiment-engine kronos-sentiment-feeds --no-pager
journalctl -u kronos-sentiment-feeds -f   # look for "trade feed: connected", "lsr feed", "news feed"
python check_db.py                         # healthy = micro_history newest ~0.0 min ago
```
Then verify the bot contract BEFORE wiring the bot:
```bash
curl http://127.0.0.1:8787/signals/BTC-USDT
```
You should get flat JSON (`sentiment_score`, `fear_greed`, `funding_rate`, `outlook_1h`, …).
Fields read `null` until each feed has data — that's expected and the bot treats null as
"unavailable".

## 7. (Optional) one-time history backfill
```bash
python -m sentiment_engine.ingestion.backfill --sources news --hours 48
```

## 8. Always-on dashboard (over Tailscale, like the bot dashboard)
The engine API stays private on `127.0.0.1:8787`. A tiny stdlib server
(`serve_dashboard.py`) serves the built React app on **port 8788** and proxies
`/api` to the engine — no Node needed on the server.

Build the dashboard once on your PC (where npm works), then copy the output up:
```powershell
# on your PC
cd C:\...\sentiment_dashboard_react\sentiment_dashboard_react
npm run build                                   # produces dist/
scp -r dist/* kronos@SERVER:~/Sentiment_Engine/dashboard_dist/
```
On the server:
```bash
mkdir -p ~/Sentiment_Engine/dashboard_dist      # (target for the scp above)
sudo cp deploy/kronos-sentiment-dashboard.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now kronos-sentiment-dashboard
systemctl status kronos-sentiment-dashboard --no-pager
```
Open `http://<server-tailscale-ip>:8788` from any device on your tailnet — same
access model as the bot dashboard on 8765. Rebuild + re-scp `dist/*` to update the UI;
`sudo systemctl restart kronos-sentiment-dashboard` to pick up a new `serve_dashboard.py`.

---

## Managing it
```bash
sudo systemctl restart kronos-sentiment-feeds
journalctl -u kronos-sentiment-feeds --since "1 hour ago"
git -C ~/Sentiment_Engine pull && sudo systemctl restart kronos-sentiment-engine kronos-sentiment-feeds   # update from GitHub
systemctl list-timers kronos-sentiment-calibrate    # nightly 07:00 calibration
```

## Resource expectations (shared 4 GB box, bot already ~2 GB)
- Engine + light feeds (lexicon): ~0.5–0.8 GB RAM, near-idle CPU.
- ONNX-int8 scorer: +~350 MB. PyTorch transformer: +~1.5–2 GB — do NOT use here.
- Disk: SQLite journal ~1.5–2 GB at 8 coins / 7-day retention.
- Keep the swap file you added — cheap OOM insurance with the bot co-resident.
