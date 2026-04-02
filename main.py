"""
Advanced Stock Analysis & Signal Pipeline v2.0
─────────────────────────────────────────────
- Angel One SmartAPI for live data
- Technical indicators via pandas-ta
- Smart multi-factor signal engine (trend-filtered)
- Realistic backtesting with stop loss, hold period, risk/reward
- Sharpe ratio, max drawdown, profit factor stats
- Claude AI for 360° analysis
- WhatsApp delivery via CallMeBot
"""
import os
import time
import math
import requests
import pandas as pd
import pandas_ta as ta
from datetime import datetime, timedelta
from SmartApi import SmartConnect
import pyotp
import anthropic
# ─────────────────────────────────────────────────────
# CONFIG — set all these in Railway environment variables
# ─────────────────────────────────────────────────────
ANGEL_API_KEY = os.getenv("ANGEL_API_KEY")
ANGEL_CLIENT_ID = os.getenv("ANGEL_CLIENT_ID")
ANGEL_PASSWORD = os.getenv("ANGEL_PASSWORD")
ANGEL_TOTP_SECRET = os.getenv("ANGEL_TOTP_SECRET")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
CALLMEBOT_PHONE = os.getenv("CALLMEBOT_PHONE")
CALLMEBOT_APIKEY = os.getenv("CALLMEBOT_APIKEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
# ── Backtest settings ──
STOP_LOSS_PCT = 2.0 # Exit if price drops 2% from entry
TARGET_PCT = 4.0 # Exit if price rises 4% from entry (2:1 R/R)
MAX_HOLD_DAYS = 10 # Max days to hold a trade
BROKERAGE_PCT = 0.05 # 0.05% per side (typical discount broker)
# ── Stocks to analyze ──
STOCKS = [
{"symbol": "HDFCBANK-EQ", "exchange": "NSE", "name": "HDFC Bank"},
{"symbol": "RELIANCE-EQ", "exchange": "NSE", "name": "Reliance Industries"},
{"symbol": "TCS-EQ", "exchange": "NSE", "name": "TCS"},
{"symbol": "INFY-EQ", "exchange": "NSE", "name": "Infosys"},
]
# ═══════════════════════════════════════════════════
# 1. LOGIN
# ═══════════════════════════════════════════════════
def login():
print("Logging into Angel One...")
totp = pyotp.TOTP(ANGEL_TOTP_SECRET).now()
obj = SmartConnect(api_key=ANGEL_API_KEY)
data = obj.generateSession(ANGEL_CLIENT_ID, ANGEL_PASSWORD, totp)
if not data.get("status"):
raise Exception(f"Login failed: {data}")
print("Login successful")
return obj
# ═══════════════════════════════════════════════════
# 2. DATA FETCHING
# ═══════════════════════════════════════════════════
def get_token(obj, symbol, exchange):
token_map = {
"HDFCBANK-EQ": "1333",
"RELIANCE-EQ": "2885",
"TCS-EQ": "11536",
"INFY-EQ": "1594",
"ICICIBANK-EQ":"4963",
"SBIN-EQ": "3045",
}
try:
result = obj.searchScrip(exchange, symbol)
return result["data"][0]["symboltoken"]
except:
return token_map.get(symbol, "")
def get_data(obj, symbol, exchange="NSE", days=300):
to_date = datetime.now().strftime("%Y-%m-%d %H:%M")
from_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d %H:%M")
token = get_token(obj, symbol, exchange)
params = {
"exchange": exchange,
"symboltoken": token,
"interval": "ONE_DAY",
"fromdate": from_date,
"todate": to_date,
}
data = obj.getCandleData(params)
df = pd.DataFrame(data["data"], columns=["ts","open","high","low","close","volume"])
df["ts"] = pd.to_datetime(df["ts"])
return df.sort_values("ts").reset_index(drop=True)
# ═══════════════════════════════════════════════════
# 3. INDICATORS
# ═══════════════════════════════════════════════════
def add_indicators(df):
c, h, l, v = df["close"], df["high"], df["low"], df["volume"]
# Momentum
df["rsi"] = ta.rsi(c, length=14)
macd = ta.macd(c)
df["macd"] = macd["MACD_12_26_9"]
df["macd_sig"] = macd["MACDs_12_26_9"]
df["macd_hist"] = macd["MACDh_12_26_9"]
stoch = ta.stoch(h, l, c)
df["stoch_k"] = stoch["STOCHk_14_3_3"]
df["stoch_d"] = stoch["STOCHd_14_3_3"]
# Trend
df["ma20"] = c.rolling(20).mean()
df["ma50"] = c.rolling(50).mean()
df["ma100"] = c.rolling(100).mean()
df["ma200"] = c.rolling(200).mean()
df["ema9"] = ta.ema(c, length=9)
df["ema21"] = ta.ema(c, length=21)
# Volatility
bb = ta.bbands(c, length=20)
df["bb_upper"] = bb["BBU_20_2.0"]
df["bb_mid"] = bb["BBM_20_2.0"]
df["bb_lower"] = bb["BBL_20_2.0"]
df["atr"] = ta.atr(h, l, c, length=14)
# Volume
df["vol_avg20"] = v.rolling(20).mean()
df["vol_ratio"] = v / df["vol_avg20"]
return df.dropna().reset_index(drop=True)
# ═══════════════════════════════════════════════════
# 4. SMART SIGNAL ENGINE (trend-filtered)
# ═══════════════════════════════════════════════════
def generate_signal(row):
score = 0
reasons = []
# Trend filter — the most important gate
in_uptrend = row.close > row.ma200 and row.ma50 > row.ma200
in_downtrend = row.close < row.ma200 and row.ma50 < row.ma200
if in_uptrend:
score += 2
reasons.append("Strong uptrend (above MA200, MA50 > MA200)")
elif in_downtrend:
score -= 2
reasons.append("Downtrend (below MA200)")
# RSI — trend-aware interpretation
# In uptrend: RSI pullback to 40-55 is a good entry, NOT a warning
# In downtrend: RSI < 30 is a falling knife, not a buy
if in_uptrend and 40 <= row.rsi <= 55:
score += 2
reasons.append(f"RSI pullback in uptrend ({row.rsi:.1f}) — good entry zone")
elif in_uptrend and row.rsi < 40:
score += 1
reasons.append(f"RSI oversold in uptrend ({row.rsi:.1f}) — bounce potential")
elif row.rsi > 70:
score -= 2
reasons.append(f"RSI overbought ({row.rsi:.1f})")
elif not in_uptrend and row.rsi < 30:
score -= 1
reasons.append(f"RSI oversold in downtrend ({row.rsi:.1f}) — falling knife risk")
# MACD
if row.macd_hist > 0 and row.macd > row.macd_sig:
score += 2
reasons.append("MACD bullish (above signal line)")
elif row.macd_hist < 0 and row.macd < row.macd_sig:
score -= 2
reasons.append("MACD bearish (below signal line)")
# Fresh MACD crossover bonus
if abs(row.macd_hist) < 0.5 and row.macd_hist > 0:
score += 1
reasons.append("Fresh MACD bullish crossover")
# EMA crossover
if row.ema9 > row.ema21:
score += 1
reasons.append("EMA9 > EMA21 (short-term bullish)")
else:
score -= 1
reasons.append("EMA9 < EMA21 (short-term bearish)")
# Bollinger Band position
bb_range = row.bb_upper - row.bb_lower
bb_pos = (row.close - row.bb_lower) / bb_range if bb_range > 0 else 0.5
if bb_pos < 0.2 and in_uptrend:
score += 1
reasons.append("Near BB lower band in uptrend — bounce potential")
elif bb_pos > 0.85:
score -= 1
reasons.append("Near BB upper band — price extended")
# Volume confirmation
if row.vol_ratio > 1.5:
score += 1
reasons.append(f"High volume confirmation ({row.vol_ratio:.1f}x avg)")
elif row.vol_ratio < 0.6:
score -= 1
reasons.append("Low volume — weak conviction")
# Stochastic
if row.stoch_k < 20 and row.stoch_k > row.stoch_d and in_uptrend:
score += 1
reasons.append("Stochastic bullish crossover in oversold zone")
elif row.stoch_k > 80:
score -= 1
reasons.append("Stochastic overbought")
# Signal classification
if score >= 6:
signal = "STRONG BUY"
elif score >= 3:
signal = "BUY"
elif score <= -6:
signal = "STRONG SELL"
elif score <= -3:
signal = "SELL"
else:
signal = "HOLD"
# Realistic confidence — capped at 90%, never 100%
max_possible = 11
confidence = round(min((abs(score) / max_possible) * 100, 90), 1)
# ATR-based entry/targets
atr = row.atr if not math.isnan(row.atr) else row.close * 0.015
entry = round(row.close, 2)
stop_loss = round(entry - (1.5 * atr), 2)
target_1 = round(entry + (2.0 * atr), 2) # 1:2 R/R minimum
target_2 = round(entry + (3.0 * atr), 2) # stretch target
rr = round((target_1 - entry) / (entry - stop_loss), 2) if entry != stop_loss else 0
return {
"signal": signal,
"score": score,
"confidence": confidence,
"reasons": reasons,
"entry": entry,
"stop_loss": stop_loss,
"target_1": target_1,
"target_2": target_2,
"risk_reward":rr,
}
# ═══════════════════════════════════════════════════
# 5. REALISTIC BACKTESTER
# ═══════════════════════════════════════════════════
def backtest(df):
"""
Realistic backtest:
- Enters at NEXT DAY open (avoids look-ahead bias)
- Applies stop loss using candle low
- Applies target using candle high
- Holds max MAX_HOLD_DAYS
- Deducts brokerage both sides
- No overlapping trades
"""
trades = []
i = 50
while i < len(df) - MAX_HOLD_DAYS - 1:
row = df.iloc[i]
signal = generate_signal(row)
if signal["signal"] not in ["BUY", "STRONG BUY"]:
i += 1
continue
# Enter next day's open
entry_price = df.iloc[i + 1]["open"]
sl_price = entry_price * (1 - STOP_LOSS_PCT / 100)
tgt_price = entry_price * (1 + TARGET_PCT / 100)
brokerage = entry_price * (BROKERAGE_PCT / 100) * 2
exit_price = None
exit_reason = "MAX_HOLD"
hold_days = 0
for j in range(1, MAX_HOLD_DAYS + 1):
if i + 1 + j >= len(df):
break
candle = df.iloc[i + 1 + j]
hold_days = j
if candle["low"] <= sl_price:
exit_price = sl_price
exit_reason = "STOP_LOSS"
break
if candle["high"] >= tgt_price:
exit_price = tgt_price
exit_reason = "TARGET"
break
if exit_price is None:
exit_price = df.iloc[min(i + 1 + MAX_HOLD_DAYS, len(df)-1)]["close"]
pnl_pct = ((exit_price - entry_price) / entry_price * 100) - brokerage
trades.append({
"pnl_pct": round(pnl_pct, 3),
"hold_days": hold_days,
"exit_reason": exit_reason,
})
i += hold_days + 1
if not trades:
return {"error": "No trades generated in backtest period"}
df_t = pd.DataFrame(trades)
wins = df_t[df_t["pnl_pct"] > 0]
losses= df_t[df_t["pnl_pct"] <= 0]
# Max drawdown
cumret = (1 + df_t["pnl_pct"] / 100).cumprod()
rollmax = cumret.cummax()
max_dd = round(((cumret - rollmax) / rollmax * 100).min(), 2)
# Sharpe (annualised)
mean_r = df_t["pnl_pct"].mean()
std_r = df_t["pnl_pct"].std()
sharpe = round((mean_r / std_r) * (252 ** 0.5), 2) if std_r > 0 else 0
# Profit factor
gp = wins["pnl_pct"].sum()
gl = abs(losses["pnl_pct"].sum())
pf = round(gp / gl, 2) if gl > 0 else 99.0
return {
"total_trades": len(trades),
"win_rate": round(len(wins) / len(trades) * 100, 1),
"avg_pnl_pct": round(df_t["pnl_pct"].mean(), 3),
"avg_win_pct": round(wins["pnl_pct"].mean(), 3) if len(wins) > 0 else 0,
"avg_loss_pct": round(losses["pnl_pct"].mean(), 3) if len(losses) > 0 else 0,
"best_trade_pct": round(df_t["pnl_pct"].max(), 2),
"worst_trade_pct": round(df_t["pnl_pct"].min(), 2),
"max_drawdown_pct": max_dd,
"sharpe_ratio": sharpe,
"profit_factor": pf,
"target_hits": len(df_t[df_t["exit_reason"] == "TARGET"]),
"sl_hits": len(df_t[df_t["exit_reason"] == "STOP_LOSS"]),
"avg_hold_days": round(df_t["hold_days"].mean(), 1),
"total_return_pct": round(df_t["pnl_pct"].sum(), 2),
}
def interpret_backtest(bt):
wr = bt.get("win_rate", 0)
pf = bt.get("profit_factor", 0)
sr = bt.get("sharpe_ratio", 0)
dd = bt.get("max_drawdown_pct", 0)
if wr >= 55 and pf >= 1.5 and sr >= 1.0:
verdict = "Strategy looks VIABLE based on historical data"
elif wr >= 45 and pf >= 1.2:
verdict = "Strategy shows MODERATE promise"
else:
verdict = "Strategy needs improvement before live trading"
warnings = []
if pf < 1.0:
warnings.append("Profit Factor < 1 (losses exceed gains)")
if dd < -15:
warnings.append(f"High drawdown {dd}% — use strict position sizing")
if sr > 1.5:
warnings.append(f"Strong Sharpe {sr}")
return verdict, warnings
# ═══════════════════════════════════════════════════
# 6. NEWS FETCHER
# ═══════════════════════════════════════════════════
def fetch_news(stock_name, max_articles=6):
if not NEWS_API_KEY:
return ["NewsAPI key not configured"]
try:
r = requests.get("https://newsapi.org/v2/everything", params={
"q": f"{stock_name} NSE India stock",
"language": "en",
"sortBy": "publishedAt",
"pageSize": max_articles,
"apiKey": NEWS_API_KEY,
}, timeout=10)
articles = r.json().get("articles", [])
return [
f"[{a.get('publishedAt','')[:10]}] {a.get('title','')} — {(a.get('description','')
or '')[:100]}"
for a in articles
] or ["No recent news found"]
except Exception as e:
return [f"News error: {e}"]
# ═══════════════════════════════════════════════════
# 7. CLAUDE 360 ANALYSIS
# ═══════════════════════════════════════════════════
def call_claude(stock_name, row, sig, bt, news_items):
news_text = "\n".join([f" {i+1}. {n}" for i, n in enumerate(news_items)])
verdict, warnings = interpret_backtest(bt)
warn_text = "\n".join(warnings) if warnings else "No major warnings"
prompt = f"""
You are a world-class equity research analyst. Analyze {stock_name} using the LIVE DATA
provided. Use ONLY this data for technical analysis.
LIVE TECHNICAL DATA:
Price: Rs{row.close:.2f}
MA20: Rs{row.ma20:.2f} ({('ABOVE' if row.close > row.ma20 else 'BELOW')}) | MA50:
Rs{row.ma50:.2f} ({('ABOVE' if row.close > row.ma50 else 'BELOW')}) | MA200: Rs{row.ma200:.2f}
({('ABOVE' if row.close > row.ma200 else 'BELOW')})
EMA9: Rs{row.ema9:.2f} | EMA21: Rs{row.ema21:.2f}
RSI: {row.rsi:.1f} | MACD hist: {row.macd_hist:.4f} ({('bullish' if row.macd_hist > 0 else
'bearish')})
Stochastic K/D: {row.stoch_k:.1f}/{row.stoch_d:.1f}
BB: Rs{row.bb_upper:.2f} / Rs{row.bb_mid:.2f} / Rs{row.bb_lower:.2f}
ATR: Rs{row.atr:.2f} | Volume ratio: {row.vol_ratio:.2f}x
SIGNAL ENGINE:
Signal: {sig['signal']} | Score: {sig['score']}/11 | Confidence: {sig['confidence']}%
Entry: Rs{sig['entry']} | Stop Loss: Rs{sig['stop_loss']} | Target 1: Rs{sig['target_1']} |
Target 2: Rs{sig['target_2']}
Risk/Reward: {sig['risk_reward']}:1
Reasons: {' | '.join(sig['reasons'])}
BACKTEST RESULTS (300 days, SL={STOP_LOSS_PCT}%, Target={TARGET_PCT}%, max {MAX_HOLD_DAYS} days
hold):
Trades: {bt.get('total_trades')} | Win Rate: {bt.get('win_rate')}% | Avg PnL:
{bt.get('avg_pnl_pct')}%
Avg Win: {bt.get('avg_win_pct')}% | Avg Loss: {bt.get('avg_loss_pct')}%
Profit Factor: {bt.get('profit_factor')} | Sharpe: {bt.get('sharpe_ratio')} | Max DD:
{bt.get('max_drawdown_pct')}%
Target hits: {bt.get('target_hits')} | SL hits: {bt.get('sl_hits')} | Avg hold:
{bt.get('avg_hold_days')} days
Interpretation: {verdict}
Warnings: {warn_text}
LATEST NEWS:
{news_text}
Write a structured 360 analysis with these sections:
1. TECHNICAL ANALYSIS
- Interpret each indicator and confirm or challenge the {sig['signal']} signal
- Validate entry Rs{sig['entry']}, SL Rs{sig['stop_loss']}, targets
- Identify trend and any chart patterns
- 3-sentence technical summary
2. BACKTEST INTERPRETATION
- Is this strategy reliable for {stock_name}?
- Interpret win rate, profit factor, Sharpe, drawdown
- Position sizing recommendation (what % of portfolio to risk?)
3. FUNDAMENTAL SNAPSHOT
- Business model and competitive moat (label as approximate, from general knowledge)
- Valuation vs sector peers
- Bull case vs Bear case
- 1-3 month outlook
4. NEWS SENTIMENT
- Summarize each news item above
- Mark each Positive / Negative / Neutral
- Overall sentiment score 0-10 with reasoning
KEY TAKEAWAYS: 5 bullet points combining technical + fundamental + sentiment
Be data-backed and clear for an intermediate investor.
"""
print(" Sending to Claude for analysis...")
client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
message = client.messages.create(
model="claude-opus-4-5",
max_tokens=4096,
messages=[{"role": "user", "content": prompt}]
)
return message.content[0].text
# ═══════════════════════════════════════════════════
# 8. WHATSAPP
# ═══════════════════════════════════════════════════
def send_whatsapp(message):
try:
r = requests.get("https://api.callmebot.com/whatsapp.php", params={
"phone": CALLMEBOT_PHONE,
"text": message,
"apikey": CALLMEBOT_APIKEY,
}, timeout=15)
print(f" WhatsApp: {'sent' if r.status_code == 200 else 'failed ' +
str(r.status_code)}")
time.sleep(5)
except Exception as e:
print(f" WhatsApp error: {e}")
def format_snapshot(name, row, sig, bt):
emoji = {"STRONG BUY":"🟢🟢","BUY":"🟢","HOLD":"🟡","SELL":"🔴","STRONG
SELL":"🔴🔴"}.get(sig["signal"],"⚪")
verdict, _ = interpret_backtest(bt)
return f"""{emoji} *{name} — {sig['signal']}* ({sig['confidence']}% confidence)
Date: {datetime.now().strftime('%d %b %Y')}
Price: Rs{row.close:.2f} | RSI: {row.rsi:.1f}
Trend: {'Uptrend' if row.close > row.ma200 else 'Downtrend'}
MACD: {'Bullish' if row.macd_hist > 0 else 'Bearish'}
Entry: Rs{sig['entry']}
Stop Loss: Rs{sig['stop_loss']}
Target 1: Rs{sig['target_1']}
Target 2: Rs{sig['target_2']}
R/R: {sig['risk_reward']}:1
Backtest (300d): Win {bt.get('win_rate')}% | Sharpe {bt.get('sharpe_ratio')} | DD
{bt.get('max_drawdown_pct')}%
{verdict}
Full AI analysis follows..."""
def split_message(text, max_len=4000):
chunks = []
while len(text) > max_len:
cut = text.rfind("\n", 0, max_len)
if cut == -1:
cut = max_len
chunks.append(text[:cut])
text = text[cut:].strip()
chunks.append(text)
return chunks
# ═══════════════════════════════════════════════════
# 9. MAIN
# ═══════════════════════════════════════════════════
def run():
print(f"\n{'='*55}")
print(f"Stock Analysis Pipeline v2.0 | {datetime.now().strftime('%d %b %Y %H:%M')}")
print(f"{'='*55}\n")
obj = login()
for stock in STOCKS:
name = stock["name"]
print(f"\nProcessing {name}...")
try:
df = get_data(obj, stock["symbol"], stock["exchange"], days=300)
df = add_indicators(df)
row = df.iloc[-1]
sig = generate_signal(row)
print(f" Signal: {sig['signal']} | Score: {sig['score']} | Conf:
{sig['confidence']}%")
bt = backtest(df)
verdict, _ = interpret_backtest(bt)
print(f" Backtest: WR={bt.get('win_rate')}% | Sharpe={bt.get('sharpe_ratio')} | PF=
{bt.get('profit_factor')}")
print(f" {verdict}")
news = fetch_news(name)
send_whatsapp(format_snapshot(name, row, sig, bt))
analysis = call_claude(name, row, sig, bt, news)
header = f"AI Analysis — {name}\n{'-'*40}\n"
for i, chunk in enumerate(split_message(header + analysis)):
prefix = f"[Part {i+1}]\n" if len(split_message(header + analysis)) > 1 else ""
send_whatsapp(prefix + chunk)
print(f" Done: {name}")
except Exception as e:
print(f" Error processing {name}: {e}")
import traceback
traceback.print_exc()
time.sleep(3)
print(f"\nPipeline complete — {len(STOCKS)} stocks analyzed")
if __name__ == "__main__":
run()

