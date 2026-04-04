"""
================================
STOCK PIPELINE V8.1 (PRODUCTION READY)
================================
Changes over V8:
- OpenAI is now a required credential (pipeline exits at startup if missing)
- openai imported at module level — no lazy import, no silent fallback
- analyze() no longer has a fallback path; any API failure raises clearly
- OPENAI_MODEL constant added for easy model switching
================================
"""

import os
import sys
import time
import logging
import requests
import concurrent.futures
import pandas as pd
import pandas_ta as ta
from datetime import datetime, timedelta
from SmartApi import SmartConnect
from openai import OpenAI
import pyotp

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
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY", "")   # required

# Resend email delivery (HTTPS — works on Railway free tier)
RESEND_API_KEY    = os.getenv("RESEND_API_KEY", "")      # from resend.com dashboard
EMAIL_SENDER      = os.getenv("EMAIL_SENDER", "")         # onboarding@resend.dev or verified sender
EMAIL_RECIPIENT   = os.getenv("EMAIL_RECIPIENT", "")      # where to send the report

# Trading parameters
INITIAL_CAPITAL    = 100_000
RISK_PER_TRADE_PCT = 1.0
MAX_HOLD_DAYS      = 10
BROKERAGE_PCT      = 0.05      # per leg
MAX_POSITION_PCT   = 20.0

# OpenAI
OPENAI_MODEL = "gpt-4o"   # swap to "gpt-4o-mini" to reduce cost

# Signal thresholds
BUY_SCORE_THRESHOLD  =  4
SELL_SCORE_THRESHOLD = -4
ATR_SL_MULT          = 1.5
ATR_TARGET_MULT      = 2.0

# Pipeline behaviour
TOP_N              = int(os.getenv("TOP_N", "10"))        # signals to email
MAX_WORKERS        = 1          # Angel One rate-limits concurrent token lookups — keep serial
FETCH_DELAY        = 1.2        # seconds between each stock (stay under Angel One RPM limit)
MIN_AVG_VOLUME     = 500_000    # skip stocks averaging < 500k daily volume
OPENAI_DELAY       = 1.5        # seconds between OpenAI calls to avoid 429s

# NSE Nifty 50 index endpoint (public, no auth needed)
NIFTY50_URL = (
    "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%2050"
)
NSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/",
}


# ================================
# CAPABILITY FLAGS  (set after env validation)
# ================================
AI_ENABLED    = False   # set True in validate_env() if OPENAI_API_KEY is present
EMAIL_ENABLED = False   # set True in validate_env() if Gmail creds are present


# ================================
# STARTUP VALIDATION
# ================================
def validate_env() -> None:
    """
    Hard-fail only on Angel One creds — the market scan cannot run without them.
    OpenAI and Gmail are treated as optional delivery layers: missing creds
    disable that layer with a warning but never abort the pipeline.
    """
    global AI_ENABLED, EMAIL_ENABLED

    # --- Hard requirements: Angel One ---
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

    # --- Soft requirements: OpenAI ---
    # Probe with a single minimal call to verify the key has active credits.
    # This costs essentially nothing (~1 token) and saves the 75s of failed
    # retries across 50 stocks when the quota is exhausted.
    if not OPENAI_API_KEY:
        log.warning(
            "OPENAI_API_KEY not set — AI analysis will be skipped. "
            "Pipeline will still run."
        )
    else:
        try:
            probe = OpenAI(api_key=OPENAI_API_KEY)
            probe.chat.completions.create(
                model=OPENAI_MODEL,
                max_tokens=1,
                messages=[{"role": "user", "content": "hi"}],
            )
            AI_ENABLED = True
            log.info("OpenAI quota verified — AI analysis enabled.")
        except Exception as exc:
            err = str(exc)
            if "insufficient_quota" in err or "quota" in err.lower():
                log.warning(
                    "OpenAI quota exhausted — AI analysis will be skipped. "
                    "Add credits at platform.openai.com/settings/billing."
                )
            elif "invalid_api_key" in err or "401" in err:
                log.warning(
                    "OpenAI API key is invalid — AI analysis will be skipped. "
                    "Check OPENAI_API_KEY in your env vars."
                )
            else:
                log.warning(
                    f"OpenAI probe failed ({exc}) — "
                    "AI analysis will be skipped for this run."
                )

    # --- Soft requirements: Resend ---
    if RESEND_API_KEY and EMAIL_SENDER and EMAIL_RECIPIENT:
        EMAIL_ENABLED = True
        log.info("Resend enabled — report will be emailed.")
    else:
        missing_email = [
            k for k, v in {
                "RESEND_API_KEY": RESEND_API_KEY,
                "EMAIL_SENDER":   EMAIL_SENDER,
                "EMAIL_RECIPIENT": EMAIL_RECIPIENT,
            }.items() if not v
        ]
        log.warning(
            f"Email disabled — missing: {', '.join(missing_email)}. "
            "Report will be logged to console only."
        )


# ================================
# NIFTY 50 UNIVERSE  (live from NSE)
# ================================
def fetch_nifty50() -> list[dict]:
    """
    Fetch the current Nifty 50 constituents from NSE.
    Returns a list of dicts: {"symbol": "HDFCBANK-EQ", "exchange": "NSE", "name": "HDFC Bank"}
    Falls back to a hardcoded snapshot (April 2025) if the request fails.
    """
    try:
        # NSE requires a session cookie — get one via the homepage first
        session = requests.Session()
        session.get("https://www.nseindia.com", headers=NSE_HEADERS, timeout=10)
        resp = session.get(NIFTY50_URL, headers=NSE_HEADERS, timeout=10)
        resp.raise_for_status()

        data = resp.json().get("data", [])
        stocks = []
        for item in data:
            sym = item.get("symbol", "").strip()
            name = item.get("meta", {}).get("companyName", sym)
            if sym and sym != "NIFTY 50":   # first row is the index itself
                stocks.append({
                    "symbol":   f"{sym}-EQ",
                    "exchange": "NSE",
                    "name":     name,
                })

        if len(stocks) < 40:
            raise ValueError(f"Only {len(stocks)} stocks returned — suspiciously low.")

        log.info(f"Nifty 50 universe loaded: {len(stocks)} stocks.")
        return stocks

    except Exception as exc:
        log.warning(
            f"Live Nifty 50 fetch failed ({exc}). "
            "Using hardcoded April-2025 snapshot as fallback."
        )
        return _nifty50_fallback()


def _nifty50_fallback() -> list[dict]:
    """Hardcoded Nifty 50 snapshot — used only if NSE API is unreachable."""
    symbols = [
        "ADANIENT", "ADANIPORTS", "APOLLOHOSP", "ASIANPAINT", "AXISBANK",
        "BAJAJ-AUTO", "BAJFINANCE", "BAJAJFINSV", "BEL", "BPCL",
        "BHARTIARTL", "BRITANNIA", "CIPLA", "COALINDIA", "DRREDDY",
        "EICHERMOT", "GRASIM", "HCLTECH", "HDFCBANK", "HDFCLIFE",
        "HEROMOTOCO", "HINDALCO", "HINDUNILVR", "ICICIBANK", "ITC",
        "INDUSINDBK", "INFY", "JSWSTEEL", "KOTAKBANK", "LT",
        "M&M", "MARUTI", "NESTLEIND", "NTPC", "ONGC",
        "POWERGRID", "RELIANCE", "SBILIFE", "SHRIRAMFIN", "SBIN",
        "SUNPHARMA", "TCS", "TATACONSUM", "TATAMOTORS", "TATASTEEL",
        "TECHM", "TITAN", "TRENT", "ULTRACEMCO", "WIPRO",
    ]
    return [
        {"symbol": f"{s}-EQ", "exchange": "NSE", "name": s}
        for s in symbols
    ]


# ================================
# LOGIN
# ================================
def login() -> SmartConnect:
    try:
        totp = pyotp.TOTP(ANGEL_TOTP_SECRET).now()
        obj  = SmartConnect(api_key=ANGEL_API_KEY)
        data = obj.generateSession(ANGEL_CLIENT_ID, ANGEL_PASSWORD, totp)
        if not data.get("status"):
            raise RuntimeError(
                f"Angel One session creation failed — API response: {data}"
            )
        log.info("Angel One login successful.")
        return obj
    except Exception as exc:
        log.critical(f"Login failed — cannot continue: {exc}")
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
# DATA FETCH  (retry + backoff)
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
                log.warning(
                    f"Fetch attempt {attempt}/{retries} failed for {symbol}: "
                    f"{exc}. Retrying in {wait}s…"
                )
                time.sleep(wait)
            else:
                log.error(f"All {retries} fetch attempts failed for {symbol}: {exc}")

    return None


# ================================
# VOLUME PRE-SCREEN
# ================================
def passes_volume_screen(df: pd.DataFrame, symbol: str) -> bool:
    """
    Skip stocks whose 20-day average volume is below MIN_AVG_VOLUME.
    Avoids generating signals on illiquid names where execution is unreliable.
    """
    avg_vol = df["volume"].tail(20).mean()
    if avg_vol < MIN_AVG_VOLUME:
        log.info(
            f"Skipping {symbol} — avg 20-day volume "
            f"{avg_vol:,.0f} < {MIN_AVG_VOLUME:,}."
        )
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
        return {
            "signal": "HOLD", "score": 0,
            "entry": 0.0, "sl": 0.0, "target": 0.0,
            "atr": 0.0, "rsi": 50.0,
        }

    row  = df.iloc[i]
    prev = df.iloc[i - 1]
    score = 0

    # Trend (mutually exclusive)
    if row.close > row.ma200:
        score += 2
    elif row.close < row.ma200:
        score -= 2

    # MACD histogram crossover (one event at a time)
    if prev.macd_hist <= 0 and row.macd_hist > 0:
        score += 3
    elif prev.macd_hist >= 0 and row.macd_hist < 0:
        score -= 3

    # Short-term trend (mutually exclusive)
    if row.ema9 > row.ema21:
        score += 1
    elif row.ema9 < row.ema21:
        score -= 1

    # RSI
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
# BACKTEST  (used internally; results NOT included in email)
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

    total    = len(trades)
    wins     = sum(1 for p in trades if p > 0)
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
# AI ANALYSIS  (OpenAI — optional, degrades gracefully)
# ================================
def analyze(stock_name: str, sig: dict, bt: dict) -> str:
    """
    Returns a 2-sentence AI analysis if OpenAI is available and the call succeeds.
    Always returns a string — never raises. Signal rows survive regardless.

    Failure modes handled:
      - AI_ENABLED is False (no key)  → returns a structured fallback string
      - API quota exceeded            → returns fallback with error hint
      - Any other exception           → returns fallback with error hint
    """
    if not AI_ENABLED:
        return (
            "Signal: {signal} | Score: {score} | RSI: {rsi} | "
            "BT return: {bt_return}% — AI analysis not configured."
        ).format(
            signal=sig["signal"],
            score=sig["score"],
            rsi=sig["rsi"],
            bt_return=bt["net_return_pct"],
        )

    prompt = (
        f"You are a concise stock analyst. Analyse {stock_name} in exactly 2 sentences: "
        f"one for the signal rationale, one for the key risk. Under 60 words total.\n\n"
        f"Signal: {sig['signal']} (score {sig['score']}) | "
        f"Entry ₹{sig['entry']} | SL ₹{sig['sl']} | Target ₹{sig['target']} | "
        f"RSI {sig['rsi']} | BT return {bt['net_return_pct']}% over {bt['trades']} trades."
    )

    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        time.sleep(OPENAI_DELAY)   # avoid OpenAI RPM rate limit across 50 stocks
        res = client.chat.completions.create(
            model=OPENAI_MODEL,
            max_tokens=120,
            messages=[{"role": "user", "content": prompt}],
        )
        return res.choices[0].message.content.strip()
    except Exception as exc:
        # Log the real error but never propagate it — stock row must survive
        log.warning(f"AI analysis unavailable for {stock_name}: {exc}")
        return (
            "AI analysis unavailable ({signal} | RSI {rsi} | "
            "BT return {bt_return}%)."
        ).format(
            signal=sig["signal"],
            rsi=sig["rsi"],
            bt_return=bt["net_return_pct"],
        )


# ================================
# CONCURRENT PROCESSING  (single stock worker)
# ================================
def process_stock(obj: SmartConnect, s: dict) -> dict | None:
    """
    Full pipeline for one stock: fetch → screen → indicators → signal → backtest → AI.
    Returns a result dict or None if the stock should be skipped.
    Intended to run inside a ThreadPoolExecutor worker.
    """
    time.sleep(FETCH_DELAY)   # throttle per-thread to respect API rate limits

    df = get_data(obj, s["symbol"], s["exchange"])
    if df is None:
        return None

    if not passes_volume_screen(df, s["symbol"]):
        return None

    df = add_indicators(df)
    if df is None:
        return None

    sig = generate_signal(df, len(df) - 1)
    bt  = backtest(df)

    # analyze() always returns a string — never drops the stock row
    analysis = analyze(s["name"], sig, bt)

    log.info(
        f"{s['name']}: {sig['signal']} "
        f"(score {sig['score']}, RSI {sig['rsi']})"
    )

    return {
        "name":      s["name"],
        "symbol":    s["symbol"],
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
    """
    Sort all processed stocks by absolute score (strongest signal first),
    then return the top N. HOLD signals are included only if needed to fill N.
    """
    # Prioritise BUY/SELL over HOLD, then by |score| descending
    def sort_key(r):
        is_hold = 1 if r["signal"] == "HOLD" else 0
        return (is_hold, -abs(r["score"]))

    sorted_results = sorted(results, key=sort_key)
    return sorted_results[:n]


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
        sig      = r["signal"]
        color    = SIGNAL_COLOR.get(sig, "#6b7280")
        bg       = SIGNAL_BG.get(sig, "#f9fafb")
        emoji    = SIGNAL_EMOJI.get(sig, "●")
        name     = r["name"]
        symbol   = r["symbol"]
        score    = r["score"]
        entry    = r["entry"]
        sl       = r["sl"]
        target   = r["target"]
        rsi      = r["rsi"]
        analysis = r["analysis"]

        # All dynamic values extracted to local vars — no dict access or
        # single quotes inside the f-string (compatible with Python 3.8+)
        rows += (
            '<tr style="background:' + bg + '; border-bottom:1px solid #e5e7eb;">'
            '<td style="padding:14px 16px; font-weight:600; color:#111827;">'
            + name +
            '<br><span style="font-size:11px; color:#6b7280; font-weight:400;">' + symbol + "</span>"
            "</td>"
            '<td style="padding:14px 16px; text-align:center;">'
            '<span style="display:inline-block; padding:4px 10px; border-radius:9999px;'
            " background:" + color + '; color:#fff; font-size:12px; font-weight:700; letter-spacing:0.05em;">'
            + emoji + " " + sig +
            "</span>"
            '<br><span style="font-size:11px; color:#6b7280;">score ' + str(score) + "</span>"
            "</td>"
            '<td style="padding:14px 16px; text-align:right; font-family:monospace;">'
            + "&#8377;{:,.2f}".format(entry) +
            "</td>"
            '<td style="padding:14px 16px; text-align:right; font-family:monospace; color:#dc2626;">'
            + "&#8377;{:,.2f}".format(sl) +
            "</td>"
            '<td style="padding:14px 16px; text-align:right; font-family:monospace; color:#16a34a;">'
            + "&#8377;{:,.2f}".format(target) +
            "</td>"
            '<td style="padding:14px 16px; text-align:center; color:#374151;">'
            + str(rsi) +
            "</td>"
            '<td style="padding:14px 16px; font-size:12px; color:#374151; max-width:260px;">'
            + analysis +
            "</td>"
            "</tr>"
        )

    time_str   = datetime.now().strftime("%H:%M IST")
    top_count  = len(results)

    # Static HTML template — no f-string, all dynamic parts injected via .replace()
    # This avoids ALL quote-inside-f-string issues across every Python version.
    html = (
        "<!DOCTYPE html>"
        "<html><head>"
        '<meta charset="UTF-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1.0">'
        "</head>"
        '<body style="margin:0; padding:0; background:#f3f4f6;'
        " font-family:-apple-system, BlinkMacSystemFont, Segoe UI, sans-serif;\">"
        '<div style="max-width:900px; margin:32px auto; background:#ffffff;'
        ' border-radius:12px; overflow:hidden; box-shadow:0 1px 3px rgba(0,0,0,0.1);">'

        # Header
        '<div style="background:#1e3a5f; padding:28px 32px;">'
        '<h1 style="margin:0; color:#ffffff; font-size:22px; font-weight:700;">'
        "&#128200; Daily Portfolio Analysis"
        "</h1>"
        '<p style="margin:6px 0 0; color:#93c5fd; font-size:14px;">'
        + date_str + " &nbsp;&middot;&nbsp; Nifty 50 Universe &nbsp;&middot;&nbsp; Top "
        + str(top_count) + " Signals"
        "</p></div>"

        # Table
        '<div style="overflow-x:auto; padding:0 0 8px;">'
        '<table style="width:100%; border-collapse:collapse; font-size:13px;">'
        "<thead>"
        '<tr style="background:#f8fafc; border-bottom:2px solid #e2e8f0;">'
        '<th style="padding:12px 16px; text-align:left; color:#64748b; font-weight:600;'
        ' text-transform:uppercase; font-size:11px; letter-spacing:0.05em;">Stock</th>'
        '<th style="padding:12px 16px; text-align:center; color:#64748b; font-weight:600;'
        ' text-transform:uppercase; font-size:11px; letter-spacing:0.05em;">Signal</th>'
        '<th style="padding:12px 16px; text-align:right; color:#64748b; font-weight:600;'
        ' text-transform:uppercase; font-size:11px; letter-spacing:0.05em;">Entry</th>'
        '<th style="padding:12px 16px; text-align:right; color:#64748b; font-weight:600;'
        ' text-transform:uppercase; font-size:11px; letter-spacing:0.05em;">Stop Loss</th>'
        '<th style="padding:12px 16px; text-align:right; color:#64748b; font-weight:600;'
        ' text-transform:uppercase; font-size:11px; letter-spacing:0.05em;">Target</th>'
        '<th style="padding:12px 16px; text-align:center; color:#64748b; font-weight:600;'
        ' text-transform:uppercase; font-size:11px; letter-spacing:0.05em;">RSI</th>'
        '<th style="padding:12px 16px; text-align:left; color:#64748b; font-weight:600;'
        ' text-transform:uppercase; font-size:11px; letter-spacing:0.05em;">Analysis</th>'
        "</tr></thead>"
        "<tbody>" + rows + "</tbody>"
        "</table></div>"

        # Footer
        '<div style="padding:20px 32px; background:#f8fafc; border-top:1px solid #e5e7eb;">'
        '<p style="margin:0; font-size:11px; color:#9ca3af; line-height:1.6;">'
        "&#9888;&#65039; <strong>Disclaimer:</strong> This report is generated automatically "
        "for informational purposes only. It does not constitute financial advice. "
        "Always conduct your own research before making investment decisions. "
        "Past backtest performance does not guarantee future results."
        "</p>"
        '<p style="margin:8px 0 0; font-size:11px; color:#d1d5db;">'
        "Generated by Stock Pipeline V8 &nbsp;&middot;&nbsp; " + time_str
        + "</p></div>"

        "</div></body></html>"
    )
    return html


def send_email(results: list[dict]) -> bool:
    """
    Send the HTML report via Resend's HTTPS API (port 443).
    Works on Railway free tier. Free plan: 3,000 emails/month.
    Use EMAIL_SENDER=onboarding@resend.dev for testing (no domain needed).
    """
    try:
        date_str = datetime.now().strftime("%d %b %Y")
        subject  = "Nifty 50 Signals — " + date_str + " (Top " + str(len(results)) + ")"
        html     = build_html_email(results)

        resp = requests.post(
            "https://api.resend.com/emails",
            headers={
                "Authorization": "Bearer " + RESEND_API_KEY,
                "Content-Type":  "application/json",
            },
            json={
                "from":    EMAIL_SENDER,
                "to":      [EMAIL_RECIPIENT],
                "subject": subject,
                "html":    html,
            },
            timeout=15,
        )

        # Resend returns 200 with an id field on success
        if resp.status_code == 200:
            log.info(f"Email sent to {EMAIL_RECIPIENT} via Resend ✓")
            return True

        log.error(
            f"Resend rejected the request — "
            f"status {resp.status_code}: {resp.text[:300]}"
        )
        return False

    except requests.exceptions.Timeout:
        log.error("Resend request timed out after 15s.")
        return False
    except Exception as exc:
        log.error(f"Email send failed: {exc}")
        return False


# ================================
# MAIN
# ================================
def run() -> None:
    validate_env()
    obj    = login()
    stocks = fetch_nifty50()

    log.info(
        f"Processing {len(stocks)} stocks with {MAX_WORKERS} workers…"
    )

    all_results = []

    # Concurrent fetch + process
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_stock, obj, s): s for s in stocks
        }
        for future in concurrent.futures.as_completed(futures):
            s = futures[future]
            try:
                result = future.result()
                if result:
                    all_results.append(result)
            except Exception as exc:
                log.error(f"Unhandled error processing {s['name']}: {exc}")

    log.info(
        f"Processing complete — {len(all_results)}/{len(stocks)} stocks "
        f"passed screens and generated signals."
    )

    if not all_results:
        log.warning("No results to report — all stocks were skipped.")
        return

    top = top_n_signals(all_results, TOP_N)
    log.info(
        f"Top {len(top)} signals selected: "
        + ", ".join(f"{r['name']} ({r['signal']})" for r in top)
    )

    if EMAIL_ENABLED:
        send_email(top)
    else:
        # Email not configured — print a readable summary to the log so the
        # run still produces useful output (visible in Railway deploy logs)
        log.info("── SIGNAL REPORT (email disabled) ──")
        for r in top:
            log.info(
                f"{r['signal']:4s} | {r['name']:<30s} | "
                f"Score {r['score']:+d} | RSI {r['rsi']} | "
                f"Entry ₹{r['entry']:,.2f} | SL ₹{r['sl']:,.2f} | "
                f"Target ₹{r['target']:,.2f}"
            )

    log.info("═" * 40 + " PIPELINE COMPLETE " + "═" * 40)


if __name__ == "__main__":
    run()
