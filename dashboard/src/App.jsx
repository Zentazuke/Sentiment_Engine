import React, { useEffect, useMemo, useState } from "react";
import {
  Activity,
  AlertTriangle,
  BarChart3,
  CheckCircle2,
  HeartPulse,
  Play,
  Radio,
  RefreshCcw,
  Send,
  ShieldCheck,
  TerminalSquare,
  TrendingDown,
  TrendingUp,
  XCircle
} from "lucide-react";

const API_BASE = "/api";

const SCENARIOS = {
  bullish_breakout: [
    "BTC breakout looks strong, bulls taking control",
    "BTC pumping hard, support held perfectly",
    "Massive squeeze incoming, buyers stepping in",
    "Clean reclaim, momentum looks bullish"
  ],
  panic_dump: [
    "BTC dumping hard, support lost",
    "Panic selloff, longs getting rekt",
    "Bearish rejection, buyers disappeared",
    "Crash vibes, this looks ugly"
  ],
  mixed_chop: [
    "BTC might break out but not sure",
    "Choppy price action, no clean direction",
    "Some buyers but rejection risk is high",
    "Market feels undecided right now"
  ],
  euphoria_top: [
    "BTC moon mission, parabolic pump",
    "Everyone is bullish, insane green candles",
    "This is going vertical, no bears left",
    "Massive pump, send it higher"
  ],
  bearish_continuation: [
    "Another rejection, bearish continuation likely",
    "Support lost, short setup looks clean",
    "Sellers still in control, red candles stacking",
    "Weak bounce, likely more downside"
  ]
};

const SYMBOLS = ["BTC/USDT", "ADA/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT", "DOGE/USDT", "LINK/USDT"];

function symbolKey(symbol) {
  return symbol.replace("/", "-");
}

async function apiGet(path) {
  const response = await fetch(`${API_BASE}${path}`);
  const text = await response.text();
  let data;
  try {
    data = JSON.parse(text);
  } catch {
    data = { raw: text };
  }
  if (!response.ok) {
    throw new Error(data?.detail || data?.message || `HTTP ${response.status}`);
  }
  return data;
}

async function apiPost(path, payload) {
  const response = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
  const text = await response.text();
  let data;
  try {
    data = JSON.parse(text);
  } catch {
    data = { raw: text };
  }
  if (!response.ok) {
    throw new Error(data?.detail || data?.message || `HTTP ${response.status}`);
  }
  return data;
}

function randomChoice(items) {
  return items[Math.floor(Math.random() * items.length)];
}

function formatNumber(value, digits = 3) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "—";
  return Number(value).toFixed(digits);
}

function formatSigned(value, digits = 3) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "—";
  const n = Number(value);
  return `${n > 0 ? "+" : ""}${n.toFixed(digits)}`;
}

function formatCompact(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "—";
  const n = Number(value);
  if (Math.abs(n) >= 1e6) return `${(n / 1e6).toFixed(2)}M`;
  if (Math.abs(n) >= 1e3) return `${(n / 1e3).toFixed(1)}k`;
  return n.toFixed(1);
}

function formatPrice(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "—";
  const n = Number(value);
  return n >= 100 ? n.toFixed(2) : n.toFixed(5);
}

function signClass(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "";
  if (Number(value) > 0.0001) return "pos";
  if (Number(value) < -0.0001) return "neg";
  return "";
}

function longPctFromRatio(ratio) {
  if (ratio === null || ratio === undefined || Number.isNaN(Number(ratio))) return null;
  const r = Number(ratio);
  return r < 0 ? null : (100 * r) / (1 + r);
}

function formatUsdCompact(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "—";
  const n = Number(value);
  if (Math.abs(n) >= 1e9) return `$${(n / 1e9).toFixed(2)}B`;
  if (Math.abs(n) >= 1e6) return `$${(n / 1e6).toFixed(1)}M`;
  if (Math.abs(n) >= 1e3) return `$${(n / 1e3).toFixed(1)}k`;
  return `$${n.toFixed(0)}`;
}

function formatFundingPct(rate) {
  if (rate === null || rate === undefined || Number.isNaN(Number(rate))) return "—";
  const n = Number(rate) * 100;
  return `${n > 0 ? "+" : ""}${n.toFixed(4)}%`;
}

function crowdRead(longPct) {
  if (longPct === null) return "Awaiting the lsr feed.";
  if (longPct >= 65) return `Crowd heavily long (${longPct.toFixed(0)}%) — classic contrarian-watch zone.`;
  if (longPct >= 55) return `Crowd leaning long (${longPct.toFixed(0)}%).`;
  if (longPct > 45) return `Positioning roughly balanced (${longPct.toFixed(0)}% long).`;
  if (longPct > 35) return `Crowd leaning short (${(100 - longPct).toFixed(0)}% short).`;
  return `Crowd heavily short (${(100 - longPct).toFixed(0)}%) — classic contrarian-watch zone.`;
}

function actionMeta(action) {
  switch (action) {
    case "confirm":
      return { label: "CONFIRM", icon: CheckCircle2, className: "pill confirm" };
    case "veto":
      return { label: "VETO", icon: XCircle, className: "pill veto" };
    default:
      return { label: "NEUTRAL", icon: ShieldCheck, className: "pill neutral" };
  }
}

function MetricCard({ label, value, hint, icon: Icon }) {
  return (
    <div className="metric-card">
      <div className="metric-header">
        {Icon ? <Icon size={18} /> : null}
        <span>{label}</span>
      </div>
      <div className="metric-value">{value}</div>
      {hint ? <div className="metric-hint">{hint}</div> : null}
    </div>
  );
}

function MicroMetric({ label, value, hint, tone }) {
  return (
    <div className="metric-card micro">
      <div className="metric-header"><span>{label}</span></div>
      <div className={`metric-value micro-value ${tone || ""}`}>{value}</div>
      {hint ? <div className="metric-hint">{hint}</div> : null}
    </div>
  );
}

function TimelineChart({ points }) {
  const data = (points || []).filter((p) => typeof p.score === "number");
  if (data.length < 2) {
    return <div className="empty">Not enough outlook history yet — it accumulates every 30s while auto-refresh is on.</div>;
  }
  const W = 800, H = 230, P = 34, PR = 56; // PR = right padding for the price axis
  const t0 = data[0].t, t1 = data[data.length - 1].t;
  const x = (t) => P + (W - P - PR) * (t1 === t0 ? 0.5 : (t - t0) / (t1 - t0));
  const yScore = (s) => H / 2 - s * (H / 2 - P);
  // Robustly drop price outliers (e.g. a manual test-injection spike) via the median.
  const rawPriced = data.filter((p) => p.price != null);
  const fmtP = (v) => (v >= 1000 ? formatUsdCompact(v) : `$${v.toFixed(v < 1 ? 4 : 2)}`);
  let pricePath = "", priced = [], pmin = 0, pmax = 0;
  if (rawPriced.length >= 2) {
    const sorted = rawPriced.map((p) => p.price).sort((a, b) => a - b);
    const med = sorted[Math.floor(sorted.length / 2)] || 1;
    priced = rawPriced.filter((p) => p.price >= med * 0.5 && p.price <= med * 2);
    if (priced.length >= 2) {
      pmin = Math.min(...priced.map((p) => p.price));
      pmax = Math.max(...priced.map((p) => p.price));
      const yPrice = (v) => P + (H - 2 * P) * (pmax === pmin ? 0.5 : (pmax - v) / (pmax - pmin));
      pricePath = priced.map((p, i) => `${i ? "L" : "M"}${x(p.t).toFixed(1)},${yPrice(p.price).toFixed(1)}`).join(" ");
    }
  }
  const scorePath = data.map((p, i) => `${i ? "L" : "M"}${x(p.t).toFixed(1)},${yScore(p.score).toFixed(1)}`).join(" ");
  const fmtTime = (t) => new Date(t * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  const priceColor = "#d2a23a";
  return (
    <div>
      <div className="chart-legend">
        <span className="legend-item"><span className="legend-swatch swatch-score" /> outlook score (left)</span>
        <span className="legend-item"><span className="legend-swatch swatch-price" /> price (right)</span>
        <span className="legend-time">{fmtTime(t0)} → {fmtTime(t1)}</span>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} className="timeline-svg" preserveAspectRatio="none">
        <line x1={P} y1={H / 2} x2={W - PR} y2={H / 2} className="axis-zero" />
        <text x={P - 4} y={yScore(1) + 4} className="axis-label">+1</text>
        <text x={P - 4} y={H / 2 + 4} className="axis-label">0</text>
        <text x={P - 4} y={yScore(-1) + 4} className="axis-label">-1</text>
        {priced.length >= 2 ? (
          <>
            <text x={W - 4} y={P + 4} className="axis-label" textAnchor="end" style={{ fill: priceColor }}>{fmtP(pmax)}</text>
            <text x={W - 4} y={H / 2 + 4} className="axis-label" textAnchor="end" style={{ fill: priceColor }}>{fmtP((pmax + pmin) / 2)}</text>
            <text x={W - 4} y={H - P + 4} className="axis-label" textAnchor="end" style={{ fill: priceColor }}>{fmtP(pmin)}</text>
          </>
        ) : null}
        {pricePath ? <path d={pricePath} className="line-price" /> : null}
        <path d={scorePath} className="line-score" />
      </svg>
    </div>
  );
}

function JsonBlock({ data }) {
  return (
    <pre className="json-block">
      {JSON.stringify(data, null, 2)}
    </pre>
  );
}

export default function App() {
  const [symbol, setSymbol] = useState("BTC/USDT");
  const [scenario, setScenario] = useState("bullish_breakout");
  const [count, setCount] = useState(25);
  const [price, setPrice] = useState(104000);
  const [priceChangePct, setPriceChangePct] = useState(0.3);
  const [botConfidence, setBotConfidence] = useState(0.7);
  const [customText, setCustomText] = useState("BTC breakout looks strong, bulls taking control");
  const [customPrice, setCustomPrice] = useState(104250);
  const [health, setHealth] = useState(null);
  const [snapshot, setSnapshot] = useState(null);
  const [evaluation, setEvaluation] = useState(null);
  const [log, setLog] = useState([]);
  const [busy, setBusy] = useState(false);
  const [autoRefresh, setAutoRefresh] = useState(false);
  const [showTools, setShowTools] = useState(false);
  const [showFeeds, setShowFeeds] = useState(false);
  const [socialFeed, setSocialFeed] = useState([]);
  const [outlook, setOutlook] = useState(null);
  const [positioningAll, setPositioningAll] = useState({});
  const [derivativesAll, setDerivativesAll] = useState({});
  const [shockAlert, setShockAlert] = useState(null);
  const [timeline, setTimeline] = useState(null);
  const [insights, setInsights] = useState(null);

  const latestAction = useMemo(() => actionMeta(evaluation?.action), [evaluation]);
  const ActionIcon = latestAction.icon;
  const micro = snapshot?.microstructure ?? null;

  useEffect(() => {
    if (!autoRefresh) return undefined;
    let cancelled = false;
    const tick = async () => {
      try {
        const [snap, healthData, social, alertData] = await Promise.all([
          apiGet(`/snapshot/${symbolKey(symbol)}`),
          apiGet("/health"),
          apiGet(`/social/${symbolKey(symbol)}/recent?limit=15`),
          apiGet(`/alerts/${symbolKey(symbol)}`)
        ]);
        if (!cancelled) {
          setSnapshot(snap);
          setHealth(healthData);
          setSocialFeed(social.events || []);
          setShockAlert(alertData?.shock ? alertData : null);
        }
      } catch {
        // Engine offline: keep the last values, the feed badges will go stale.
      }
    };
    tick();
    const id = setInterval(tick, 2000);

    // Outlook is medium-term: refresh every 30s (each fetch is journaled
    // server-side for backtesting, so we keep the cadence modest).
    const tickOutlook = async () => {
      try {
        const [data, history, insightData] = await Promise.all([
          apiGet(`/outlook/${symbolKey(symbol)}`),
          apiGet(`/outlook/${symbolKey(symbol)}/history?hours=12`),
          apiGet(`/insights/${symbolKey(symbol)}?hours=24`)
        ]);
        if (!cancelled) {
          setOutlook(data);
          setTimeline(history.points || []);
          setInsights(insightData);
        }
      } catch {
        // engine offline; keep last values
      }
      try {
        const [posResults, derivResults] = await Promise.all([
          Promise.all(SYMBOLS.map((c) => apiGet(`/positioning/${symbolKey(c)}`))),
          Promise.all(SYMBOLS.map((c) => apiGet(`/derivatives/${symbolKey(c)}`)))
        ]);
        if (!cancelled) {
          const pmap = {};
          posResults.forEach((res, i) => { pmap[SYMBOLS[i]] = res.positioning; });
          setPositioningAll(pmap);
          const dmap = {};
          derivResults.forEach((res, i) => { dmap[SYMBOLS[i]] = res.derivatives; });
          setDerivativesAll(dmap);
        }
      } catch {
        // positioning/derivatives unavailable
      }
    };
    tickOutlook();
    const outlookId = setInterval(tickOutlook, 30000);

    return () => {
      cancelled = true;
      clearInterval(id);
      clearInterval(outlookId);
    };
  }, [autoRefresh, symbol]);

  function addLog(kind, payload) {
    setLog((items) => [
      {
        id: (globalThis.crypto?.randomUUID?.() ?? `${Date.now()}-${Math.random().toString(36).slice(2)}`),
        time: new Date().toLocaleTimeString(),
        kind,
        payload
      },
      ...items
    ].slice(0, 20));
  }

  async function runTask(kind, task) {
    setBusy(true);
    try {
      const result = await task();
      addLog(kind, result);
      return result;
    } catch (error) {
      const err = { error: error.message || String(error) };
      addLog(`${kind}:error`, err);
      alert(err.error);
      return null;
    } finally {
      setBusy(false);
    }
  }

  async function checkHealth() {
    const result = await runTask("health", () => apiGet("/health"));
    if (result) setHealth(result);
  }

  async function getSnapshot() {
    const result = await runTask("snapshot", () => apiGet(`/snapshot/${symbolKey(symbol)}`));
    if (result) setSnapshot(result);
  }

  async function injectPrice(value = customPrice) {
    return runTask("inject-price", () =>
      apiPost("/ingest/price", {
        symbol,
        price: Number(value),
        timestamp: Date.now() / 1000
      })
    );
  }

  async function injectSocial() {
    return runTask("inject-social", () =>
      apiPost("/ingest/social", {
        symbol,
        source: "dashboard",
        text: customText,
        author: "manual-ui",
        timestamp: Date.now() / 1000
      })
    );
  }

  async function simulateScenario() {
    await runTask("simulate", async () => {
      const now = Date.now() / 1000;
      const baselinePhrases = SCENARIOS.mixed_chop;
      const activePhrases = SCENARIOS[scenario];

      // A previous baseline helps the engine calculate velocity.
      const baselineCount = Math.max(5, Math.floor(Number(count) / 2));
      for (let i = 0; i < baselineCount; i += 1) {
        await apiPost("/ingest/social", {
          symbol,
          source: "dashboard-baseline",
          text: randomChoice(baselinePhrases),
          author: "baseline",
          timestamp: now - 100 + i
        });
      }

      const startPrice = Number(price);
      const endPrice = startPrice * (1 + Number(priceChangePct) / 100);
      for (let i = 0; i < 5; i += 1) {
        const p = startPrice + (endPrice - startPrice) * (i / 4);
        await apiPost("/ingest/price", {
          symbol,
          price: p,
          timestamp: now - 20 + i * 4
        });
      }

      for (let i = 0; i < Number(count); i += 1) {
        await apiPost("/ingest/social", {
          symbol,
          source: "dashboard-scenario",
          text: randomChoice(activePhrases),
          author: scenario,
          timestamp: now - Math.max(0, Number(count) - i)
        });
      }

      const snap = await apiGet(`/snapshot/${symbolKey(symbol)}`);
      setSnapshot(snap);
      return {
        injected: true,
        symbol,
        scenario,
        social_messages: Number(count) + baselineCount,
        start_price: startPrice,
        end_price: endPrice,
        snapshot: snap
      };
    });
  }

  async function evaluate(direction) {
    const result = await runTask(`evaluate-${direction}`, () =>
      apiPost("/evaluate", {
        symbol,
        direction,
        bot_confidence: Number(botConfidence),
        trigger_price: Number(customPrice),
        timestamp: Date.now() / 1000
      })
    );
    if (result) {
      setEvaluation(result);
      if (result.snapshot) setSnapshot(result.snapshot);
    }
  }

  return (
    <div className="app">
      <header className="hero">
        <div>
          <div className="eyebrow">Standalone Test Dashboard</div>
          <h1>Crypto Sentiment Engine</h1>
          <p>
            Test the sentiment layer independently before connecting it to your scalping bot.
            It can inject mock social activity, track rolling snapshots, and evaluate fake LONG/SHORT signals.
          </p>
        </div>
        <div className="status-card status-compact">
          <button className="status-head" onClick={() => setShowFeeds((v) => !v)} aria-expanded={showFeeds}>
            <span className={`dot ${health?.status ? "dot-live" : "dot-stale"}`} />
            <strong>{health?.status ? "Online" : "Not checked"}</strong>
            {health?.feeds ? (
              <span className="status-count">{Object.keys(health.feeds).length} feeds {showFeeds ? "▾" : "▸"}</span>
            ) : null}
          </button>
          {showFeeds && health?.feeds ? (
            <div className="feed-list">
              {Object.entries(health.feeds).map(([key, feed]) => (
                <div key={key} className="feed-row">
                  <span className={`dot ${feed.live ? "dot-live" : "dot-stale"}`} />
                  <span className="feed-symbol">{key}</span>
                  <span className="feed-age">
                    {feed.live
                      ? `live · ${feed.microstructure_age_seconds}s`
                      : feed.microstructure_age_seconds == null
                        ? "no feed"
                        : `stale · ${feed.microstructure_age_seconds}s`}
                  </span>
                </div>
              ))}
            </div>
          ) : null}
          <button className="secondary small" onClick={checkHealth} disabled={busy}>
            <RefreshCcw size={14} />
            Check health
          </button>
        </div>
      </header>

      <div className="topbar">
        <div className="coin-tabs">
          {SYMBOLS.map((item) => (
            <button
              key={item}
              className={`coin-tab ${symbol === item ? "active" : ""}`}
              onClick={() => setSymbol(item)}
            >
              {item}
            </button>
          ))}
        </div>
        <button
          className={`refresh-toggle ${autoRefresh ? "on" : "off"}`}
          onClick={() => setAutoRefresh((v) => !v)}
          aria-pressed={autoRefresh}
        >
          <span className="refresh-dot" />
          {autoRefresh ? "Live" : "Paused"}
        </button>
      </div>

      <section className="panel timeline-panel" style={{ marginBottom: "1rem" }}>
        <h2><BarChart3 size={19} /> {symbol} · Outlook vs Price (12h)</h2>
        <TimelineChart points={timeline} />
      </section>

      <section className="panel positioning-band" style={{ marginBottom: "1rem" }}>
        <h2><Activity size={19} /> Crowd Positioning &amp; Derivatives — Binance</h2>
        <p className="metric-hint" style={{ marginTop: "-4px", marginBottom: "12px" }}>
          What traders are actually doing across all four coins: long/short crowd, funding (leverage cost), and open interest (conviction). Crowd and funding extremes are a contrarian tell; the calibrator learns the true direction from outcomes.
        </p>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(210px, 1fr))", gap: "12px" }}>
          {SYMBOLS.map((coin) => {
            const p = positioningAll[coin];
            const d = derivativesAll[coin];
            const longPct = p ? longPctFromRatio(p.global_account_ratio) : null;
            const sig = p ? p.signal : null;
            const sigColor = sig > 0.0001 ? "#4ade80" : sig < -0.0001 ? "#f87171" : "#cbd5e1";
            return (
              <div key={coin} style={{ border: "1px solid rgba(148,163,184,0.18)", borderRadius: "14px", padding: "0.9rem", background: "rgba(15,23,42,0.5)" }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: "8px" }}>
                  <strong style={{ fontSize: "16px" }}>{coin.replace("/USDT", "")}</strong>
                  <span style={{ fontSize: "12px", color: "#94a3b8" }}>{longPct != null ? `${longPct.toFixed(1)}% long` : "no data"}</span>
                </div>
                <div style={{ display: "flex", height: "9px", borderRadius: "5px", overflow: "hidden", background: "#334155", marginBottom: "10px" }}>
                  <div style={{ width: `${longPct != null ? longPct : 50}%`, background: "#4ade80" }} />
                  <div style={{ width: `${longPct != null ? 100 - longPct : 50}%`, background: "#f87171" }} />
                </div>
                {p ? (
                  <>
                    <div style={{ display: "flex", justifyContent: "space-between", fontSize: "12.5px", color: "#cbd5e1", marginBottom: "4px" }}><span>Top trader pos L:S</span><strong>{formatNumber(p.top_position_ratio, 2)}</strong></div>
                    <div style={{ display: "flex", justifyContent: "space-between", fontSize: "12.5px", color: "#cbd5e1", marginBottom: "4px" }}><span>Taker buy/sell</span><strong>{formatNumber(p.taker_ratio, 2)}</strong></div>
                    <div style={{ display: "flex", justifyContent: "space-between", fontSize: "12.5px", color: "#cbd5e1", marginBottom: "4px" }}><span>Funding (8h)</span><strong style={{ color: d && d.funding_rate > 0 ? "#4ade80" : d && d.funding_rate < 0 ? "#f87171" : "#cbd5e1" }}>{formatFundingPct(d ? d.funding_rate : null)}</strong></div>
                    <div style={{ display: "flex", justifyContent: "space-between", fontSize: "12.5px", color: "#cbd5e1", marginBottom: "4px" }}><span>Open interest</span><strong>{formatUsdCompact(d ? d.open_interest_usd : null)}</strong></div>
                    <div style={{ display: "flex", justifyContent: "space-between", fontSize: "12.5px", color: "#cbd5e1" }}><span>Signal</span><strong style={{ color: sigColor }}>{formatSigned(p.signal, 2)}</strong></div>
                    <div style={{ marginTop: "9px", fontSize: "12px", color: "#94a3b8", lineHeight: 1.5 }}>{crowdRead(longPct)}</div>
                  </>
                ) : (
                  <div style={{ fontSize: "12px", color: "#64748b", lineHeight: 1.5 }}>Awaiting data — run the live feeds with <code>lsr</code> in sources.</div>
                )}
              </div>
            );
          })}
        </div>
      </section>

      <main className={`grid ${showTools ? "" : "tools-hidden"}`}>
        <section className="tools-toggle">
          <button onClick={() => setShowTools((v) => !v)}>
            <TerminalSquare size={16} />
            {showTools ? "Hide tools" : "Tools — manual testing · scenario sim · trade-signal eval · event log"}
          </button>
        </section>

        <section className="panel controls">
          <h2><TerminalSquare size={19} /> Controls (testing)</h2>

          <label>
            Scenario
            <select value={scenario} onChange={(e) => setScenario(e.target.value)}>
              {Object.keys(SCENARIOS).map((item) => (
                <option key={item} value={item}>{item.replaceAll("_", " ")}</option>
              ))}
            </select>
          </label>

          <div className="two-col">
            <label>
              Messages
              <input type="number" min="1" value={count} onChange={(e) => setCount(e.target.value)} />
            </label>
            <label>
              Price change %
              <input type="number" step="0.1" value={priceChangePct} onChange={(e) => setPriceChangePct(e.target.value)} />
            </label>
          </div>

          <div className="two-col">
            <label>
              Start price
              <input type="number" step="1" value={price} onChange={(e) => setPrice(e.target.value)} />
            </label>
            <label>
              Trigger price
              <input type="number" step="1" value={customPrice} onChange={(e) => setCustomPrice(e.target.value)} />
            </label>
          </div>

          <label>
            Bot confidence
            <input type="number" min="0" max="1" step="0.01" value={botConfidence} onChange={(e) => setBotConfidence(e.target.value)} />
          </label>

          <button className="primary" onClick={simulateScenario} disabled={busy}>
            <Play size={16} />
            Simulate scenario
          </button>

          <div className="button-row">
            <button onClick={getSnapshot} disabled={busy}>
              <BarChart3 size={16} />
              Snapshot
            </button>
            <button onClick={() => evaluate("STRAT_LONG")} disabled={busy}>
              <TrendingUp size={16} />
              Evaluate LONG
            </button>
            <button onClick={() => evaluate("STRAT_SHORT")} disabled={busy}>
              <TrendingDown size={16} />
              Evaluate SHORT
            </button>
          </div>
        </section>

        <section className="panel snapshot-panel">
          <h2><Activity size={19} /> Current Snapshot</h2>
          {snapshot ? (
            <div className="metrics">
              <MetricCard label="Messages" value={snapshot.message_count} hint={`Prev: ${snapshot.previous_message_count}`} icon={Activity} />
              <MetricCard label="Avg Sentiment" value={formatNumber(snapshot.average_sentiment)} hint={`Prev: ${formatNumber(snapshot.previous_average_sentiment)}`} icon={HeartPulse} />
              <MetricCard label="Sentiment Velocity" value={formatNumber(snapshot.sentiment_velocity)} hint="Mood acceleration" icon={TrendingUp} />
              <MetricCard label="Attention Spike" value={formatNumber(snapshot.attention_spike)} hint="Mention velocity" icon={AlertTriangle} />
              <MetricCard label="Price Change %" value={formatNumber(snapshot.price_change_pct)} hint="Recent price window" icon={BarChart3} />
              <MetricCard label="Data Quality" value={formatNumber(snapshot.data_quality)} hint={snapshot.data_quality >= 0.6 ? "Usable" : "Low confidence"} icon={ShieldCheck} />
            </div>
          ) : (
            <div className="empty">No snapshot yet. Simulate a scenario or inject messages first.</div>
          )}
        </section>

        <section className="panel micro-panel">
          <h2>
            <Radio size={19} /> Microstructure
            {micro ? (
              <span className={`feed-badge ${micro.stale ? "badge-stale" : "badge-live"}`}>
                {micro.stale ? "STALE" : "LIVE"}
              </span>
            ) : null}
          </h2>
          {micro ? (
            <>
              {micro.stale ? (
                <div className="stale-banner">
                  Feed stale — metrics are older than the freshness limit. The engine ignores them and leans neutral.
                </div>
              ) : null}
              <div className="micro-grid">
                <MicroMetric label="Last Price" value={formatPrice(micro.last_price)} />
                <MicroMetric label="Imbalance 10s" tone={signClass(micro.trade_imbalance_10s)} value={formatSigned(micro.trade_imbalance_10s)} hint="buy − sell flow" />
                <MicroMetric label="Imbalance 30s" tone={signClass(micro.trade_imbalance_30s)} value={formatSigned(micro.trade_imbalance_30s)} />
                <MicroMetric label="Imbalance 60s" tone={signClass(micro.trade_imbalance_60s)} value={formatSigned(micro.trade_imbalance_60s)} />
                <MicroMetric label="Book Imbalance" tone={signClass(micro.book_imbalance)} value={formatSigned(micro.book_imbalance)} hint="±0.25% of mid" />
                <MicroMetric label="Spread" value={micro.spread_bps != null ? `${formatNumber(micro.spread_bps, 2)} bps` : "—"} />
                <MicroMetric label="Bid Depth" value={formatCompact(micro.bid_depth_quote)} hint="quote value near mid" />
                <MicroMetric label="Ask Depth" value={formatCompact(micro.ask_depth_quote)} hint="quote value near mid" />
                <MicroMetric label="Rel. Volume" value={micro.relative_volume != null ? `${formatNumber(micro.relative_volume, 2)}×` : "—"} hint="vs 15-min baseline" />
                <MicroMetric label="Momentum 30s" tone={signClass(micro.momentum_pct_30s)} value={micro.momentum_pct_30s != null ? `${formatSigned(micro.momentum_pct_30s, 3)}%` : "—"} />
                <MicroMetric label="Volatility 30s" value={micro.volatility_bps_30s != null ? `${formatNumber(micro.volatility_bps_30s, 1)} bps` : "—"} />
                <MicroMetric label="VWAP Dist" tone={signClass(micro.vwap_distance_bps)} value={micro.vwap_distance_bps != null ? `${formatSigned(micro.vwap_distance_bps, 1)} bps` : "—"} hint="5-min VWAP" />
                <MicroMetric label="Trades 60s" value={micro.trade_count_60s ?? "—"} />
              </div>
            </>
          ) : (
            <div className="empty">
              No microstructure data yet. Start the live feed:
              <code className="cmd">python -m sentiment_engine.ingestion.run_live --sources binance --symbols BTC/USDT ADA/USDT ETH/USDT BNB/USDT</code>
              then enable auto-refresh.
            </div>
          )}
        </section>

        <section className="panel evaluation">
          <h2><ShieldCheck size={19} /> Latest Evaluation</h2>
          {evaluation ? (
            <>
              <div className={latestAction.className}>
                <ActionIcon size={18} />
                {latestAction.label}
              </div>

              <div className="eval-grid">
                <MetricCard label="Sentiment Score" value={formatNumber(evaluation.sentiment_score)} />
                <MetricCard label="Modifier" value={formatNumber(evaluation.confidence_modifier)} />
                <MetricCard label="Final Confidence" value={formatNumber(evaluation.final_confidence_estimate)} />
                <MetricCard label="Safe to Use" value={evaluation.safe_to_use ? "Yes" : "No"} />
              </div>

              <div className="reason">{evaluation.reason}</div>
              <details>
                <summary>Raw evaluation JSON</summary>
                <JsonBlock data={evaluation} />
              </details>
            </>
          ) : (
            <div className="empty">No evaluation yet. Simulate sentiment, then evaluate LONG or SHORT.</div>
          )}
        </section>

        <section className="panel outlook-panel">
          <h2>
            <TrendingUp size={19} /> Sentiment Outlook
            {outlook ? (
              <span className={`feed-badge ${
                outlook.outlook_score > 0.12 ? "badge-live" : outlook.outlook_score < -0.12 ? "badge-stale" : "tag-social"
              }`}>
                {outlook.label?.toUpperCase()}
              </span>
            ) : null}
          </h2>
          {shockAlert ? (
            <div className={`shock-banner ${shockAlert.type === "panic" ? "shock-panic" : "shock-euphoria"}`}>
              <AlertTriangle size={16} />
              {shockAlert.type?.toUpperCase()} SHOCK — intensity {formatNumber(shockAlert.intensity, 2)} ·{" "}
              {shockAlert.window_events} events/{Math.round(shockAlert.window_seconds / 60)}min · mood{" "}
              {formatSigned(shockAlert.mean_sentiment, 2)}
            </div>
          ) : null}
          {outlook ? (
            <>
              <div className="outlook-head">
                <div>
                  <div className={`outlook-score ${signClass(outlook.outlook_score)}`}>
                    {formatSigned(outlook.outlook_score, 3)}
                  </div>
                  <div className="metric-hint">composite · confidence {formatNumber(outlook.confidence, 2)}</div>
                  <div style={{ marginTop: "12px", maxWidth: "260px" }}>
                    <div style={{ fontSize: "11px", color: "#94a3b8", marginBottom: "4px", display: "flex", justifyContent: "space-between" }}>
                      <span>6h up-probability</span>
                      <strong style={{ color: outlook.direction_prob_6h == null ? "#64748b" : outlook.direction_prob_6h >= 0.5 ? "#4ade80" : "#f87171" }}>
                        {outlook.direction_prob_6h == null ? "—" : `${(outlook.direction_prob_6h * 100).toFixed(0)}%`}
                      </strong>
                    </div>
                    {outlook.direction_prob_6h == null ? (
                      <div style={{ fontSize: "11px", color: "#64748b" }}>multi-feature model not live yet — needs more data</div>
                    ) : (
                      <div style={{ position: "relative", height: "9px", borderRadius: "5px", background: "#334155" }}>
                        <div style={{ position: "absolute", left: 0, top: 0, bottom: 0, width: `${outlook.direction_prob_6h * 100}%`, background: outlook.direction_prob_6h >= 0.5 ? "#4ade80" : "#f87171", borderRadius: "5px" }} />
                        <div style={{ position: "absolute", left: "50%", top: "-2px", bottom: "-2px", width: "1px", background: "#94a3b8" }} title="50% (coin flip)" />
                      </div>
                    )}
                  </div>
                </div>
                {outlook.context ? (
                  <div className="context-strip">
                    <div className="context-item">
                      <span>Fear &amp; Greed</span>
                      <strong>{outlook.context.fear_greed_value ?? "—"} {outlook.context.fear_greed_label || ""}</strong>
                    </div>
                    <div className="context-item">
                      <span>BTC Dominance</span>
                      <strong>{outlook.context.btc_dominance_pct != null ? `${formatNumber(outlook.context.btc_dominance_pct, 1)}%` : "—"}</strong>
                    </div>
                    <div className="context-item">
                      <span>Mcap 24h</span>
                      <strong className={signClass(outlook.context.market_cap_change_24h_pct)}>
                        {outlook.context.market_cap_change_24h_pct != null ? `${formatSigned(outlook.context.market_cap_change_24h_pct, 2)}%` : "—"}
                      </strong>
                    </div>
                    <div className="context-item">
                      <span>Context tilt</span>
                      <strong className={signClass(outlook.context_tilt)}>{formatSigned(outlook.context_tilt, 3)}</strong>
                    </div>
                  </div>
                            ) : (
                  <div className="metric-hint">No market context yet — run the feed with <code>context</code> in sources.</div>
                )}
              </div>
              <div className="horizon-list">
                {outlook.horizons.map((h) => {
                  const label = h.horizon_seconds >= 86400 ? `${h.horizon_seconds / 86400}d` : `${h.horizon_seconds / 3600}h`;
                  const pct = Math.min(100, Math.abs(h.score) * 100);
                  return (
                    <div key={h.horizon_seconds} className="horizon-row">
                      <span className="horizon-label">{label}</span>
                      <div className="horizon-bar">
                        <div
                          className={`horizon-fill ${h.score >= 0 ? "fill-pos" : "fill-neg"}`}
                          style={{ width: `${pct}%` }}
                        />
                      </div>
                      <span className={`horizon-score ${signClass(h.score)}`}>{formatSigned(h.score, 2)}</span>
                      <span className="horizon-meta">{h.event_count} ev · {h.news_count} news · conf {formatNumber(h.confidence, 2)}</span>
                    </div>
                  );
                })}
              </div>
              <div className="hint">{outlook.disclaimer}</div>
            </>
          ) : (
            <div className="empty">
              No outlook yet. Enable auto-refresh, and run the feed with all sources:
              <code className="cmd">python -m sentiment_engine.ingestion.run_live --sources binance news context --symbols BTC/USDT ADA/USDT ETH/USDT BNB/USDT</code>
            </div>
          )}
        </section>

        <section className="panel drivers-panel">
          <h2><Activity size={19} /> Mood Drivers (24h)</h2>
          {insights && insights.event_count > 0 ? (
            <>
              <div className="drivers-sources">
                {["news", "reddit"].map((cls) => {
                  const s = insights.sources?.[cls];
                  return (
                    <div key={cls} className="source-stat">
                      <span className={`source-tag ${cls === "news" ? "tag-news" : "tag-social"}`}>{cls}</span>
                      <strong className={signClass(s?.avg_sentiment)}>{formatSigned(s?.avg_sentiment, 2)}</strong>
                      <span className="metric-hint">{s?.count ?? 0} items</span>
                    </div>
                  );
                })}
                {insights.sources?.divergence != null ? (
                  <div className="source-stat">
                    <span className="source-tag tag-social">divergence</span>
                    <strong className={insights.sources.divergence > 0.4 ? "neg" : ""}>
                      {formatNumber(insights.sources.divergence, 2)}
                    </strong>
                    <span className="metric-hint">press vs retail</span>
                  </div>
                ) : null}
              </div>
              <div className="term-cloud">
                {(insights.terms || []).map((t) => (
                  <span
                    key={t.term}
                    className={`term-chip ${signClass(t.avg_sentiment)}`}
                    title={`${t.count} mentions · avg ${formatSigned(t.avg_sentiment, 2)}`}
                  >
                    {t.term} <em>{t.count}</em>
                  </span>
                ))}
              </div>
            </>
          ) : (
            <div className="empty">No real social/news events in the last 24h yet. Run the backfill and keep the feeds running.</div>
          )}
        </section>

        <section className="panel social-panel">
          <h2><Send size={19} /> News &amp; Social Feed</h2>
          {socialFeed.length ? (
            <div className="social-list">
              {socialFeed.map((item, index) => (
                <div key={`${item.timestamp}-${index}`} className="social-item">
                  <div className="social-meta">
                    <span className={`source-tag ${item.source?.startsWith("news:") ? "tag-news" : "tag-social"}`}>
                      {item.source || "unknown"}
                    </span>
                    <span className={`social-sentiment ${signClass(item.sentiment)}`}>
                      {formatSigned(item.sentiment, 2)}
                    </span>
                    <span className="social-time">
                      {new Date(item.timestamp * 1000).toLocaleTimeString()}
                    </span>
                  </div>
                  <div className="social-text">{item.text}</div>
                </div>
              ))}
            </div>
          ) : (
            <div className="empty">
              No social/news events yet. Run the feed with <code>--sources binance news</code> to capture
              live headlines, or inject a manual message below. Enable auto-refresh to see new items arrive.
            </div>
          )}
        </section>

        <section className="panel manual">
          <h2><Send size={19} /> Manual Injection</h2>
          <label>
            Social text
            <textarea rows="4" value={customText} onChange={(e) => setCustomText(e.target.value)} />
          </label>
          <div className="button-row">
            <button onClick={injectSocial} disabled={busy}>Inject social</button>
            <button onClick={() => injectPrice()} disabled={busy}>Inject price</button>
          </div>
          <p className="hint">
            Use this to manually test phrases like "BTC support lost, panic dump" or "ADA breakout, strong buyers".
          </p>
        </section>

        <section className="panel log-panel">
          <h2><TerminalSquare size={19} /> Event Log</h2>
          {log.length ? (
            <div className="log-list">
              {log.map((item) => (
                <details key={item.id} className="log-item">
                  <summary>
                    <span>{item.time}</span>
                    <strong>{item.kind}</strong>
                  </summary>
                  <JsonBlock data={item.payload} />
                </details>
              ))}
            </div>
          ) : (
            <div className="empty">No events yet.</div>
          )}
        </section>
      </main>
    </div>
  );
}
