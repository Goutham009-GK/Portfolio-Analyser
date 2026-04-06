"""
================================
STOCK PIPELINE V8.2 — WATCHLIST EDITION
================================
Changes over V8.1:
- Replaced Nifty 50 universe with a user-defined watchlist
- GET  /watchlist          → returns current watchlist
- POST /watchlist          → saves full watchlist (replaces)
- PUT  /watchlist/{symbol} → add one stock
- DELETE /watchlist/{symbol} → remove one stock
- Watchlist persisted to watchlist.json (survives Railway restarts)
- Each stock: { symbol, exchange, name }
- Supports both NSE and BSE exchanges
================================
"""

import os
import sys
import json
import time
import logging
import threading
import requests
import pandas as pd
import pandas_ta as ta
from datetime import datetime, timedelta
from pathlib import Path
from SmartApi import SmartConnect
from openai import OpenAI
import pyotp
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

# ================================
# LOGGING
# ================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("pipeline.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ================================
# CONFIG
# ================================
ANGEL_API_KEY     = os.getenv("ANGEL_API_KEY", "")
ANGEL_CLIENT_ID   = os.getenv("ANGEL_CLIENT_ID", "")
ANGEL_PASSWORD    = os.getenv("ANGEL_PASSWORD", "")
ANGEL_TOTP_SECRET = os.getenv("ANGEL_TOTP_SECRET", "")
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY", "")

RESEND_API_KEY  = os.getenv("RESEND_API_KEY", "")
EMAIL_SENDER    = os.getenv("EMAIL_SENDER", "")
EMAIL_RECIPIENT = os.getenv("EMAIL_RECIPIENT", "")

INITIAL_CAPITAL    = 100_000
RISK_PER_TRADE_PCT = 1.0
MAX_HOLD_DAYS      = 10
BROKERAGE_PCT      = 0.05
MAX_POSITION_PCT   = 20.0

OPENAI_MODEL = "gpt-4o"

BUY_SCORE_THRESHOLD  =  4
SELL_SCORE_THRESHOLD = -4
ATR_SL_MULT          = 1.5
ATR_TARGET_MULT      = 2.0

TOP_N          = int(os.getenv("TOP_N", "50"))   # show all watchlist stocks by default
FETCH_DELAY    = 1.2
MIN_AVG_VOLUME = 100_000   # lower threshold for custom watchlists (user picks their own stocks)
OPENAI_DELAY   = 1.5

# ================================
# WATCHLIST PERSISTENCE
# ================================
WATCHLIST_FILE = Path(os.getenv("WATCHLIST_FILE", "watchlist.json"))

DEFAULT_WATCHLIST = [
    {"symbol": "RELIANCE-EQ",  "exchange": "NSE", "name": "Reliance Industries"},
    {"symbol": "HDFCBANK-EQ",  "exchange": "NSE", "name": "HDFC Bank"},
    {"symbol": "INFY-EQ",      "exchange": "NSE", "name": "Infosys"},
    {"symbol": "TCS-EQ",       "exchange": "NSE", "name": "Tata Consultancy Services"},
    {"symbol": "ICICIBANK-EQ", "exchange": "NSE", "name": "ICICI Bank"},
]

_watchlist_lock = threading.Lock()


def load_watchlist() -> list[dict]:
    """Load watchlist from disk. Returns default list if file doesn't exist."""
    try:
        if WATCHLIST_FILE.exists():
            data = json.loads(WATCHLIST_FILE.read_text())
            if isinstance(data, list) and len(data) > 0:
                log.info(f"Watchlist loaded: {len(data)} stocks from {WATCHLIST_FILE}")
                return data
    except Exception as exc:
        log.warning(f"Could not load watchlist from {WATCHLIST_FILE}: {exc}")
    log.info("Using default watchlist (5 stocks). Add your own via the app.")
    return list(DEFAULT_WATCHLIST)


def save_watchlist(watchlist: list[dict]) -> None:
    """Persist watchlist to disk."""
    try:
        WATCHLIST_FILE.write_text(json.dumps(watchlist, indent=2))
        log.info(f"Watchlist saved: {len(watchlist)} stocks to {WATCHLIST_FILE}")
    except Exception as exc:
        log.error(f"Could not save watchlist: {exc}")


# ================================
# CAPABILITY FLAGS
# ================================
AI_ENABLED    = False
EMAIL_ENABLED = False
openai_client: "OpenAI | None" = None


# ================================
# STARTUP VALIDATION
# ================================
def validate_env() -> None:
    global AI_ENABLED, EMAIL_ENABLED, openai_client

    required = {
        "ANGEL_API_KEY":     ANGEL_API_KEY,
        "ANGEL_CLIENT_ID":   ANGEL_CLIENT_ID,
        "ANGEL_PASSWORD":    ANGEL_PASSWORD,
        "ANGEL_TOTP_SECRET": ANGEL_TOTP_SECRET,
    }
    missing = [k for k, v in required.items() if not v]
    if missing:
        log.critical(
            f"STARTUP FAILED — missing required Angel One env vars: "
            f"{', '.join(missing)}. Set them before running."
        )
        sys.exit(1)

    if not OPENAI_API_KEY:
        log.warning("OPENAI_API_KEY not set — AI analysis will be skipped.")
    else:
        try:
            probe = OpenAI(api_key=OPENAI_API_KEY)
            probe.chat.completions.create(
                model=OPENAI_MODEL,
                max_tokens=1,
                messages=[{"role": "user", "content": "hi"}],
            )
            AI_ENABLED    = True
            openai_client = probe
            log.info("OpenAI quota verified — AI analysis enabled.")
        except Exception as exc:
            err = str(exc)
            if "insufficient_quota" in err or "quota" in err.lower():
                log.warning("OpenAI quota exhausted — AI analysis will be skipped.")
            elif "invalid_api_key" in err or "401" in err:
                log.warning("OpenAI API key is invalid — AI analysis will be skipped.")
            else:
                log.warning(f"OpenAI probe failed ({exc}) — AI analysis will be skipped.")

    if RESEND_API_KEY and EMAIL_SENDER and EMAIL_RECIPIENT:
        EMAIL_ENABLED = True
        log.info("Resend enabled — report will be emailed.")
    else:
        missing_email = [
            k for k, v in {
                "RESEND_API_KEY":   RESEND_API_KEY,
                "EMAIL_SENDER":     EMAIL_SENDER,
                "EMAIL_RECIPIENT":  EMAIL_RECIPIENT,
            }.items() if not v
        ]
        log.warning(f"Email disabled — missing: {', '.join(missing_email)}.")


# ================================
# LOGIN
# ================================
def login() -> SmartConnect:
    try:
        totp = pyotp.TOTP(ANGEL_TOTP_SECRET).now()
        obj  = SmartConnect(api_key=ANGEL_API_KEY)
        data = obj.generateSession(ANGEL_CLIENT_ID, ANGEL_PASSWORD, totp)
        if not data.get("status"):
            raise RuntimeError(f"Angel One session failed: {data}")
        log.info("Angel One login successful.")
        return obj
    except Exception as exc:
        log.critical(f"Login failed: {exc}")
        raise


# ================================
# SYMBOL TOKEN LOOKUP
# ================================
def get_token(obj: SmartConnect, symbol: str, exchange: str) -> str | None:
    try:
        resp = obj.searchScrip(exchange, symbol)
        if resp and resp.get("data"):
            return resp["data"][0]["symboltoken"]
        log.warning(f"No token found for {symbol} on {exchange}.")
        return None
    except Exception as exc:
        log.error(f"Token lookup failed for {symbol}: {exc}")
        return None


# ================================
# DATA FETCH
# ================================
def get_data(
    obj: SmartConnect,
    symbol: str,
    exchange: str = "NSE",
    days: int = 300,
    retries: int = 3,
) -> pd.DataFrame | None:
    token = get_token(obj, symbol, exchange)
    if not token:
        return None

    params = {
        "exchange":    exchange,
        "symboltoken": token,
        "interval":    "ONE_DAY",
        "fromdate":    (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d %H:%M"),
        "todate":      datetime.now().strftime("%Y-%m-%d %H:%M"),
    }

    for attempt in range(1, retries + 1):
        try:
            data = obj.getCandleData(params)
            if not data or "data" not in data or not data["data"]:
                raise ValueError("getCandleData returned empty response.")

            df = pd.DataFrame(
                data["data"],
                columns=["ts", "open", "high", "low", "close", "volume"],
            )
            df["ts"] = pd.to_datetime(df["ts"])

            if len(df) < 50:
                raise ValueError(f"Only {len(df)} rows — need ≥ 50.")

            return df.sort_values("ts").reset_index(drop=True)

        except Exception as exc:
            wait = 2 ** attempt
            if attempt < retries:
                log.warning(f"Fetch attempt {attempt}/{retries} for {symbol}: {exc}. Retrying in {wait}s…")
                time.sleep(wait)
            else:
                log.error(f"All {retries} fetch attempts failed for {symbol}: {exc}")

    return None


# ================================
# VOLUME PRE-SCREEN
# ================================
def passes_volume_screen(df: pd.DataFrame, symbol: str) -> bool:
    avg_vol = df["volume"].tail(20).mean()
    if avg_vol < MIN_AVG_VOLUME:
        log.info(f"Skipping {symbol} — avg 20-day volume {avg_vol:,.0f} < {MIN_AVG_VOLUME:,}.")
        return False
    return True


# ================================
# INDICATORS
# ================================
def add_indicators(df: pd.DataFrame) -> pd.DataFrame | None:
    try:
        c, h, l = df["close"], df["high"], df["low"]
        df["rsi"]       = ta.rsi(c, length=14)
        macd            = ta.macd(c)
        df["macd_hist"] = macd["MACDh_12_26_9"]
        df["macd_line"] = macd["MACD_12_26_9"]
        df["ema9"]      = ta.ema(c, length=9)
        df["ema21"]     = ta.ema(c, length=21)
        df["ma200"]     = c.rolling(200).mean()
        df["atr"]       = ta.atr(h, l, c, length=14)
        df = df.dropna().reset_index(drop=True)
        if df.empty:
            raise ValueError("Empty after dropna.")
        return df
    except Exception as exc:
        log.error(f"Indicator computation failed: {exc}")
        return None


# ================================
# SIGNAL ENGINE
# ================================
def generate_signal(df: pd.DataFrame, i: int) -> dict:
    if i < 1 or i >= len(df):
        return {"signal": "HOLD", "score": 0, "entry": 0.0, "sl": 0.0, "target": 0.0, "atr": 0.0, "rsi": 50.0}

    row  = df.iloc[i]
    prev = df.iloc[i - 1]
    score = 0

    if row.close > row.ma200:
        score += 2
    elif row.close < row.ma200:
        score -= 2

    if prev.macd_hist <= 0 and row.macd_hist > 0:
        score += 3
    elif prev.macd_hist >= 0 and row.macd_hist < 0:
        score -= 3

    if row.ema9 > row.ema21:
        score += 1
    elif row.ema9 < row.ema21:
        score -= 1

    if row.rsi > 70:
        score -= 1
    elif row.rsi < 30:
        score += 1

    if score >= BUY_SCORE_THRESHOLD:
        signal = "BUY"
    elif score <= SELL_SCORE_THRESHOLD:
        signal = "SELL"
    else:
        signal = "HOLD"

    atr   = float(row.atr)
    close = float(row.close)
    return {
        "signal": signal,
        "score":  score,
        "entry":  round(close, 2),
        "sl":     round(close - ATR_SL_MULT * atr, 2),
        "target": round(close + ATR_TARGET_MULT * atr, 2),
        "atr":    round(atr, 2),
        "rsi":    round(float(row.rsi), 1),
    }


# ================================
# POSITION SIZING
# ================================
def calc_position_size(capital: float, entry: float, sl: float) -> int:
    per_share = entry - sl
    if per_share <= 0:
        return 0
    risk_qty = int((capital * RISK_PER_TRADE_PCT / 100) / per_share)
    cap_qty  = int((capital * MAX_POSITION_PCT  / 100) / entry)
    return min(risk_qty, cap_qty)


# ================================
# BACKTEST
# ================================
def backtest(df: pd.DataFrame) -> dict:
    capital = float(INITIAL_CAPITAL)
    peak    = capital
    max_dd  = 0.0
    trades  = []
    gross_win = gross_loss = 0.0

    for i in range(max(50, 1), len(df) - 1):
        sig = generate_signal(df, i)
        if sig["signal"] != "BUY":
            continue

        entry  = float(df.iloc[i + 1]["open"])
        atr    = sig["atr"]
        sl     = round(entry - ATR_SL_MULT    * atr, 2)
        target = round(entry + ATR_TARGET_MULT * atr, 2)
        qty    = calc_position_size(capital, entry, sl)
        if qty == 0:
            continue

        exit_price = None
        for j in range(1, MAX_HOLD_DAYS + 1):
            idx = i + 1 + j
            if idx >= len(df):
                exit_price = float(df.iloc[-1]["close"])
                break
            candle = df.iloc[idx]
            if float(candle["open"]) < sl:
                exit_price = float(candle["open"])
                break
            if float(candle["low"]) <= sl:
                exit_price = sl
                break
            if float(candle["high"]) >= target:
                exit_price = target
                break

        if exit_price is None:
            final_idx = min(i + 1 + MAX_HOLD_DAYS, len(df) - 1)
            exit_price = float(df.iloc[final_idx]["close"])

        brokerage = (entry + exit_price) * qty * (BROKERAGE_PCT / 100)
        pnl       = (exit_price - entry) * qty - brokerage
        capital  += pnl
        trades.append(pnl)
        gross_win  += max(pnl, 0)
        gross_loss += abs(min(pnl, 0))

        if capital > peak:
            peak = capital
        dd = (peak - capital) / peak * 100
        if dd > max_dd:
            max_dd = dd

    total = len(trades)
    wins  = sum(1 for p in trades if p > 0)
    return {
        "final_capital":    round(capital, 2),
        "trades":           total,
        "win_rate_pct":     round(wins / total * 100, 1) if total else 0.0,
        "avg_pnl":          round(sum(trades) / total, 2) if total else 0.0,
        "net_return_pct":   round((capital - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "profit_factor":    round(gross_win / gross_loss, 2) if gross_loss else None,
    }


# ================================
# AI ANALYSIS
# ================================
def analyze(stock_name: str, sig: dict, bt: dict) -> str:
    if not AI_ENABLED:
        return (
            "Signal: {signal} | Score: {score} | RSI: {rsi} | "
            "BT return: {bt_return}% — AI analysis not configured."
        ).format(signal=sig["signal"], score=sig["score"], rsi=sig["rsi"], bt_return=bt["net_return_pct"])

    prompt = (
        f"You are a concise stock analyst. Analyse {stock_name} in exactly 2 sentences: "
        f"one for the signal rationale, one for the key risk. Under 60 words total.\n\n"
        f"Signal: {sig['signal']} (score {sig['score']}) | "
        f"Entry ₹{sig['entry']} | SL ₹{sig['sl']} | Target ₹{sig['target']} | "
        f"RSI {sig['rsi']} | BT return {bt['net_return_pct']}% over {bt['trades']} trades."
    )

    try:
        time.sleep(OPENAI_DELAY)
        res = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            max_tokens=120,
            messages=[{"role": "user", "content": prompt}],
        )
        return res.choices[0].message.content.strip()
    except Exception as exc:
        log.warning(f"AI analysis unavailable for {stock_name}: {exc}")
        return (
            "AI analysis unavailable ({signal} | RSI {rsi} | BT return {bt_return}%)."
        ).format(signal=sig["signal"], rsi=sig["rsi"], bt_return=bt["net_return_pct"])


# ================================
# SINGLE STOCK WORKER
# ================================
def process_stock(obj: SmartConnect, s: dict) -> dict | None:
    time.sleep(FETCH_DELAY)

    df = get_data(obj, s["symbol"], s["exchange"])
    if df is None:
        return None

    if not passes_volume_screen(df, s["symbol"]):
        return None

    df = add_indicators(df)
    if df is None:
        return None

    sig = generate_signal(df, len(df) - 1)

    if sig["signal"] == "HOLD":
        bt       = {"net_return_pct": 0.0, "win_rate_pct": 0.0}
        analysis = "HOLD — no actionable signal."
    else:
        bt       = backtest(df)
        analysis = analyze(s["name"], sig, bt)

    log.info(f"{s['name']}: {sig['signal']} (score {sig['score']}, RSI {sig['rsi']})")

    return {
        "name":      s["name"],
        "symbol":    s["symbol"],
        "exchange":  s["exchange"],
        "signal":    sig["signal"],
        "score":     sig["score"],
        "entry":     sig["entry"],
        "sl":        sig["sl"],
        "target":    sig["target"],
        "rsi":       sig["rsi"],
        "atr":       sig["atr"],
        "bt_return": bt["net_return_pct"],
        "win_rate":  bt["win_rate_pct"],
        "analysis":  analysis,
    }


# ================================
# TOP-N FILTER
# ================================
def top_n_signals(results: list[dict], n: int = TOP_N) -> list[dict]:
    def sort_key(r):
        is_hold = 1 if r["signal"] == "HOLD" else 0
        return (is_hold, -abs(r["score"]))
    return sorted(results, key=sort_key)[:n]


# ================================
# HTML EMAIL
# ================================
SIGNAL_COLOR = {"BUY": "#16a34a", "SELL": "#dc2626", "HOLD": "#d97706"}
SIGNAL_BG    = {"BUY": "#f0fdf4", "SELL": "#fef2f2", "HOLD": "#fffbeb"}
SIGNAL_EMOJI = {"BUY": "▲", "SELL": "▼", "HOLD": "●"}


def build_html_email(results: list[dict]) -> str:
    date_str = datetime.now().strftime("%A, %d %B %Y")
    rows = ""
    for r in results:
        sig    = r["signal"]
        color  = SIGNAL_COLOR.get(sig, "#6b7280")
        bg     = SIGNAL_BG.get(sig, "#f9fafb")
        emoji  = SIGNAL_EMOJI.get(sig, "●")
        rows += (
            '<tr style="background:' + bg + '; border-bottom:1px solid #e5e7eb;">'
            '<td style="padding:14px 16px; font-weight:600; color:#111827;">' + r["name"] +
            '<br><span style="font-size:11px; color:#6b7280; font-weight:400;">' + r["symbol"] + " · " + r["exchange"] + "</span></td>"
            '<td style="padding:14px 16px; text-align:center;">'
            '<span style="display:inline-block; padding:4px 10px; border-radius:9999px; background:' + color + '; color:#fff; font-size:12px; font-weight:700;">' + emoji + " " + sig + "</span>"
            '<br><span style="font-size:11px; color:#6b7280;">score ' + str(r["score"]) + "</span></td>"
            '<td style="padding:14px 16px; text-align:right; font-family:monospace;">&#8377;{:,.2f}</td>'.format(r["entry"]) +
            '<td style="padding:14px 16px; text-align:right; font-family:monospace; color:#dc2626;">&#8377;{:,.2f}</td>'.format(r["sl"]) +
            '<td style="padding:14px 16px; text-align:right; font-family:monospace; color:#16a34a;">&#8377;{:,.2f}</td>'.format(r["target"]) +
            '<td style="padding:14px 16px; text-align:center;">' + str(r["rsi"]) + "</td>"
            '<td style="padding:14px 16px; font-size:12px; color:#374151; max-width:260px;">' + r["analysis"] + "</td>"
            "</tr>"
        )

    time_str  = datetime.now().strftime("%H:%M IST")
    top_count = len(results)
    html = (
        "<!DOCTYPE html><html><head>"
        '<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">'
        "</head><body style=\"margin:0;padding:0;background:#f3f4f6;font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;\">"
        '<div style="max-width:900px;margin:32px auto;background:#fff;border-radius:12px;overflow:hidden;box-shadow:0 1px 3px rgba(0,0,0,0.1);">'
        '<div style="background:#1e3a5f;padding:28px 32px;">'
        '<h1 style="margin:0;color:#fff;font-size:22px;font-weight:700;">&#128200; Watchlist Signal Report</h1>'
        '<p style="margin:6px 0 0;color:#93c5fd;font-size:14px;">' + date_str + " &nbsp;&middot;&nbsp; Top " + str(top_count) + " Signals</p></div>"
        '<div style="overflow-x:auto;padding:0 0 8px;">'
        '<table style="width:100%;border-collapse:collapse;font-size:13px;"><thead>'
        '<tr style="background:#f8fafc;border-bottom:2px solid #e2e8f0;">'
        '<th style="padding:12px 16px;text-align:left;color:#64748b;font-weight:600;text-transform:uppercase;font-size:11px;">Stock</th>'
        '<th style="padding:12px 16px;text-align:center;color:#64748b;font-weight:600;text-transform:uppercase;font-size:11px;">Signal</th>'
        '<th style="padding:12px 16px;text-align:right;color:#64748b;font-weight:600;text-transform:uppercase;font-size:11px;">Entry</th>'
        '<th style="padding:12px 16px;text-align:right;color:#64748b;font-weight:600;text-transform:uppercase;font-size:11px;">Stop Loss</th>'
        '<th style="padding:12px 16px;text-align:right;color:#64748b;font-weight:600;text-transform:uppercase;font-size:11px;">Target</th>'
        '<th style="padding:12px 16px;text-align:center;color:#64748b;font-weight:600;text-transform:uppercase;font-size:11px;">RSI</th>'
        '<th style="padding:12px 16px;text-align:left;color:#64748b;font-weight:600;text-transform:uppercase;font-size:11px;">Analysis</th>'
        "</tr></thead><tbody>" + rows + "</tbody></table></div>"
        '<div style="padding:20px 32px;background:#f8fafc;border-top:1px solid #e5e7eb;">'
        '<p style="margin:0;font-size:11px;color:#9ca3af;line-height:1.6;">'
        "&#9888;&#65039; <strong>Disclaimer:</strong> This report is for informational purposes only. Not financial advice."
        "</p><p style=\"margin:8px 0 0;font-size:11px;color:#d1d5db;\">Generated by SignalEdge Watchlist &nbsp;&middot;&nbsp;" + time_str + "</p></div>"
        "</div></body></html>"
    )
    return html


def send_email(results: list[dict]) -> bool:
    try:
        date_str = datetime.now().strftime("%d %b %Y")
        subject  = "Watchlist Signals — " + date_str + " (Top " + str(len(results)) + ")"
        html     = build_html_email(results)
        resp = requests.post(
            "https://api.resend.com/emails",
            headers={"Authorization": "Bearer " + RESEND_API_KEY, "Content-Type": "application/json"},
            json={"from": EMAIL_SENDER, "to": [EMAIL_RECIPIENT], "subject": subject, "html": html},
            timeout=15,
        )
        if resp.ok:
            log.info(f"Email sent to {EMAIL_RECIPIENT} via Resend ✓")
            return True
        log.error(f"Resend rejected — status {resp.status_code}: {resp.text[:300]}")
        return False
    except Exception as exc:
        log.error(f"Email send failed: {exc}")
        return False


# ================================
# FASTAPI APP
# ================================
app = FastAPI(title="SignalEdge API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# ── In-memory cache ──
_cache: dict = {
    "signals":  [],
    "last_run": None,
    "status":   "idle",
    "error":    None,
}
_cache_lock = threading.Lock()


# ── Pydantic models ──
class StockItem(BaseModel):
    symbol:   str      # e.g. "INFY-EQ"  or  "INFY" (normalised below)
    exchange: str      # "NSE" or "BSE"
    name:     str      # Display name e.g. "Infosys"


def normalise_symbol(symbol: str, exchange: str) -> str:
    """Ensure symbol has the -EQ suffix for equity tokens."""
    sym = symbol.upper().strip()
    if exchange.upper() == "NSE" and not sym.endswith("-EQ"):
        sym = sym + "-EQ"
    return sym


# ================================
# WATCHLIST ENDPOINTS
# ================================
@app.get("/watchlist")
def get_watchlist_endpoint():
    """Return the current watchlist."""
    with _watchlist_lock:
        return {"watchlist": load_watchlist()}


@app.post("/watchlist")
def replace_watchlist(items: list[StockItem]):
    """Replace the entire watchlist."""
    stocks = [
        {
            "symbol":   normalise_symbol(i.symbol, i.exchange),
            "exchange": i.exchange.upper(),
            "name":     i.name.strip(),
        }
        for i in items
    ]
    with _watchlist_lock:
        save_watchlist(stocks)
    return {"message": f"Watchlist updated — {len(stocks)} stocks.", "watchlist": stocks}


@app.put("/watchlist/{symbol}")
def add_to_watchlist(symbol: str, item: StockItem):
    """Add a single stock to the watchlist (no duplicates)."""
    with _watchlist_lock:
        wl = load_watchlist()
        norm = normalise_symbol(item.symbol, item.exchange)
        if any(s["symbol"] == norm for s in wl):
            raise HTTPException(status_code=409, detail=f"{norm} already in watchlist.")
        wl.append({"symbol": norm, "exchange": item.exchange.upper(), "name": item.name.strip()})
        save_watchlist(wl)
    return {"message": f"{norm} added.", "watchlist": wl}


@app.delete("/watchlist/{symbol}")
def remove_from_watchlist(symbol: str):
    """Remove a stock from the watchlist by symbol."""
    norm = symbol.upper()
    if not norm.endswith("-EQ"):
        norm_eq = norm + "-EQ"
    else:
        norm_eq = norm
    with _watchlist_lock:
        wl = load_watchlist()
        new_wl = [s for s in wl if s["symbol"] not in (norm, norm_eq)]
        if len(new_wl) == len(wl):
            raise HTTPException(status_code=404, detail=f"{symbol} not found in watchlist.")
        save_watchlist(new_wl)
    return {"message": f"{symbol} removed.", "watchlist": new_wl}


# ================================
# PIPELINE ENDPOINTS
# ================================
def _run_pipeline_bg() -> None:
    with _cache_lock:
        _cache["status"] = "running"
        _cache["error"]  = None

    try:
        validate_env()
        obj     = login()
        stocks  = load_watchlist()

        if not stocks:
            raise ValueError("Watchlist is empty. Add stocks via the app before running.")

        log.info(f"Running pipeline for {len(stocks)} watchlist stocks…")
        all_results = []
        for s in stocks:
            try:
                result = process_stock(obj, s)
                if result:
                    all_results.append(result)
            except Exception as exc:
                log.error(f"Error processing {s['name']}: {exc}")

        top = top_n_signals(all_results, TOP_N)

        if EMAIL_ENABLED:
            send_email(top)

        with _cache_lock:
            _cache["signals"]  = top
            _cache["last_run"] = datetime.now().isoformat()
            _cache["status"]   = "done"
            _cache["error"]    = None

        log.info("Background pipeline run complete — cache updated.")

    except Exception as exc:
        log.error(f"Background pipeline failed: {exc}")
        with _cache_lock:
            _cache["status"] = "error"
            _cache["error"]  = str(exc)


@app.get("/")
def root():
    return {"service": "SignalEdge Watchlist API", "status": "ok"}


@app.get("/signals")
def get_signals():
    with _cache_lock:
        return JSONResponse({
            "status":   _cache["status"],
            "last_run": _cache["last_run"],
            "count":    len(_cache["signals"]),
            "signals":  _cache["signals"],
        })


@app.post("/run")
def trigger_run(background_tasks: BackgroundTasks):
    with _cache_lock:
        if _cache["status"] == "running":
            return JSONResponse({"message": "Pipeline already running."}, status_code=202)

    background_tasks.add_task(_run_pipeline_bg)
    return JSONResponse({"message": "Pipeline started. Poll GET /signals for results."}, status_code=202)


@app.get("/status")
def get_status():
    with _cache_lock:
        return {"status": _cache["status"], "last_run": _cache["last_run"], "error": _cache["error"]}


# ================================
# MAIN
# ================================
if __name__ == "__main__":
    mode = os.getenv("MODE", "api")
    if mode == "pipeline":
        validate_env()
        obj     = login()
        stocks  = load_watchlist()
        results = []
        for s in stocks:
            r = process_stock(obj, s)
            if r:
                results.append(r)
        top = top_n_signals(results, TOP_N)
        if EMAIL_ENABLED:
            send_email(top)
        else:
            for r in top:
                log.info(
                    f"{r['signal']:4s} | {r['name']:<30s} | Score {r['score']:+d} | "
                    f"RSI {r['rsi']} | Entry ₹{r['entry']:,.2f} | SL ₹{r['sl']:,.2f} | Target ₹{r['target']:,.2f}"
                )
    else:
        port = int(os.getenv("PORT", "8000"))
        log.info(f"Starting SignalEdge Watchlist API on port {port}…")
        uvicorn.run(app, host="0.0.0.0", port=port)
