"""
Stock Analysis & Signal Pipeline v4.0
──────────────────────────────────────
Best of v2 + v3 merged:
- Angel One SmartAPI for live data
- Full technical indicators (pandas-ta)
- Trend-filtered signal engine with TRUE MACD crossover detection
- ATR-based entry / stop loss / targets
- Position sizing (risk % of capital per trade)
- Realistic backtester: SL/target on candle H/L, no overlap, brokerage
- Full stats: win rate, Sharpe, profit factor, max drawdown, equity curve
- AI analysis via GPT-4 (fixed API call) with full data context
- WhatsApp delivery via CallMeBot (snapshot + full report)
- Per-stock error handling so one failure doesn't crash the run
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
from openai import OpenAI
# ─────────────────────────────────────────────────────
# CONFIG — set all in Railway environment variables
# ─────────────────────────────────────────────────────
ANGEL_API_KEY = os.getenv("ANGEL_API_KEY")
ANGEL_CLIENT_ID = os.getenv("ANGEL_CLIENT_ID")
ANGEL_PASSWORD = os.getenv("ANGEL_PASSWORD")
ANGEL_TOTP_SECRET = os.getenv("ANGEL_TOTP_SECRET")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") # GPT-4
NEWS_API_KEY = os.getenv("NEWS_API_KEY") # newsapi.org free tier
CALLMEBOT_PHONE = os.getenv("CALLMEBOT_PHONE") # e.g. 919876543210
CALLMEBOT_APIKEY = os.getenv("CALLMEBOT_APIKEY")
# ── Risk / backtest settings ──
INITIAL_CAPITAL = 100000 # Rs 1 lakh starting capital for backtest
RISK_PER_TRADE_PCT = 1.0 # Risk 1% of capital per trade
STOP_LOSS_PCT = 2.0 # Fallback SL if ATR unavailable
TARGET_PCT = 4.0 # Fallback target
MAX_HOLD_DAYS = 10 # Max days to hold before forced exit
BROKERAGE_PCT = 0.05 # 0.05% per side
# ── Stocks to analyze ──
STOCKS = [
{"symbol": "HDFCBANK-EQ", "exchange": "NSE", "name": "HDFC Bank"},
{"symbol": "RELIANCE-EQ", "exchange": "NSE", "name": "Reliance Industries"},
{"symbol": "TCS-EQ", "exchange": "NSE", "name": "TCS"},
{"symbol": "INFY-EQ", "exchange": "NSE", "name": "Infosys"},
]
# Token fallback map (used if searchScrip fails)
TOKEN_MAP = {
"HDFCBANK-EQ": "1333",
"RELIANCE-EQ": "2885",
"TCS-EQ": "11536",
"INFY-EQ": "1594",
"ICICIBANK-EQ": "4963",
"SBIN-EQ": "3045",
}
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
# 2. DATA
# ═══════════════════════════════════════════════════
def get_token(obj, symbol, exchange):
try:
return obj.searchScrip(exchange, symbol)["data"][0]["symboltoken"]
except:
return TOKEN_MAP.get(symbol, "")
def get_data(obj, symbol, exchange="NSE", days=300):
token = get_token(obj, symbol, exchange)
params = {
"exchange": exchange,
"symboltoken": token,
"interval": "ONE_DAY",
"fromdate": (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d %H:%M"),
"todate": datetime.now().strftime("%Y-%m-%d %H:%M"),
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
df["macd_hist"]= macd["MACDh_12_26_9"]
stoch = ta.stoch(h, l, c)
df["stoch_k"] = stoch["STOCHk_14_3_3"]
df["stoch_d"] = stoch["STOCHd_14_3_3"]
# Trend
df["ma20"] = c.rolling(20).mean()
df["ma50"] = c.rolling(50).mean()
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
# 4. SIGNAL ENGINE
# — trend-filtered RSI (v2)
# — TRUE MACD crossover detection (v3 fix)
# — ATR-based levels (v3)
# — position sizing (v3)
# ═══════════════════════════════════════════════════
def generate_signal(df, i):
row = df.iloc[i]
prev = df.iloc[i - 1]
score = 0
reasons = []
in_uptrend = row.close > row.ma200 and row.ma50 > row.ma200
in_downtrend = row.close < row.ma200 and row.ma50 < row.ma200
# ── Trend ──
if in_uptrend:
score += 2
reasons.append("Uptrend: price above MA200, MA50 > MA200")
elif in_downtrend:
score -= 2
reasons.append("Downtrend: price below MA200")
# ── RSI — trend-aware (v2 logic) ──
if in_uptrend and 40 <= row.rsi <= 55:
score += 2
reasons.append(f"RSI pullback in uptrend ({row.rsi:.1f}) — good entry")
elif in_uptrend and row.rsi < 40:
score += 1
reasons.append(f"RSI oversold in uptrend ({row.rsi:.1f}) — bounce possible")
elif row.rsi > 70:
score -= 2
reasons.append(f"RSI overbought ({row.rsi:.1f})")
elif in_downtrend and row.rsi < 30:
score -= 1
reasons.append(f"RSI oversold in downtrend ({row.rsi:.1f}) — falling knife risk")
# ── TRUE MACD crossover (v3 fix — checks previous bar) ──
if prev.macd_hist < 0 and row.macd_hist > 0:
score += 3
reasons.append("Bullish MACD crossover (confirmed this bar)")
elif prev.macd_hist > 0 and row.macd_hist < 0:
score -= 3
reasons.append("Bearish MACD crossover (confirmed this bar)")
elif row.macd_hist > 0:
score += 1
reasons.append("MACD histogram positive (bullish momentum)")
else:
score -= 1
reasons.append("MACD histogram negative (bearish momentum)")
# ── EMA short-term ──
if row.ema9 > row.ema21:
score += 1
reasons.append("EMA9 > EMA21 (short-term bullish)")
else:
score -= 1
reasons.append("EMA9 < EMA21 (short-term bearish)")
# ── Bollinger Band position ──
bb_range = row.bb_upper - row.bb_lower
if bb_range > 0:
bb_pos = (row.close - row.bb_lower) / bb_range
if bb_pos < 0.2 and in_uptrend:
score += 1
reasons.append("Near BB lower in uptrend — bounce zone")
elif bb_pos > 0.85:
score -= 1
reasons.append("Near BB upper band — extended")
# ── Volume ──
if row.vol_ratio > 1.5:
score += 1
reasons.append(f"High volume ({row.vol_ratio:.1f}x avg) — strong conviction")
elif row.vol_ratio < 0.6:
score -= 1
reasons.append("Low volume — weak move")
# ── Stochastic ──
if row.stoch_k < 20 and row.stoch_k > row.stoch_d and in_uptrend:
score += 1
reasons.append("Stochastic bullish crossover in oversold zone")
elif row.stoch_k > 80:
score -= 1
reasons.append("Stochastic overbought")
# ── Signal ──
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
# Capped confidence
confidence = round(min((abs(score) / 12) * 100, 90), 1)
# ATR-based levels
atr = row.atr if not math.isnan(row.atr) else row.close * 0.015
entry = round(row.close, 2)
stop_loss = round(entry - 1.5 * atr, 2)
target_1 = round(entry + 2.0 * atr, 2)
target_2 = round(entry + 3.0 * atr, 2)
rr = round((target_1 - entry) / max(entry - stop_loss, 0.01), 2)
return {
"signal": signal,
"score": score,
"confidence": confidence,
"entry": entry,
"stop_loss": stop_loss,
"target_1": target_1,
"target_2": target_2,
"risk_reward": rr,
"reasons": reasons,
"atr": round(atr, 2),
}
# ── Position sizing (v3) ──
def position_size(capital, entry, stop_loss):
risk_amount = capital * (RISK_PER_TRADE_PCT / 100)
per_share_risk = abs(entry - stop_loss)
if per_share_risk == 0:
return 0
return int(risk_amount / per_share_risk)
# ═══════════════════════════════════════════════════
# 5. REALISTIC BACKTESTER
# — enters at next day open (no look-ahead bias)
# — checks SL on candle low, target on candle high
# — proper position sizing applied
# — brokerage deducted both sides
# — full stats: win rate, Sharpe, profit factor, max DD
# ═══════════════════════════════════════════════════
def backtest(df):
capital = float(INITIAL_CAPITAL)
equity_curve = []
trades = []
i = 50
while i < len(df) - MAX_HOLD_DAYS - 1:
sig = generate_signal(df, i)
if sig["signal"] not in ["BUY", "STRONG BUY"]:
equity_curve.append(capital)
i += 1
continue
entry_price = df.iloc[i + 1]["open"]
sl_price = entry_price * (1 - STOP_LOSS_PCT / 100)
tgt_price = entry_price * (1 + TARGET_PCT / 100)
qty = position_size(capital, entry_price, sl_price)
if qty == 0:
i += 1
continue
brokerage = entry_price * qty * (BROKERAGE_PCT / 100) * 2
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
pnl = (exit_price - entry_price) * qty - brokerage
pnl_pct = (exit_price - entry_price) / entry_price * 100
capital += pnl
trades.append({
"pnl": round(pnl, 2),
"pnl_pct": round(pnl_pct, 3),
"exit_reason":exit_reason,
"hold_days": hold_days,
})
equity_curve.append(capital)
i += hold_days + 1
if not trades:
return {"error": "No trades generated"}
df_t = pd.DataFrame(trades)
wins = df_t[df_t["pnl"] > 0]
losses = df_t[df_t["pnl"] <= 0]
# Max drawdown
eq = pd.Series(equity_curve)
rollmax= eq.cummax()
max_dd = round(((eq - rollmax) / rollmax * 100).min(), 2)
# Sharpe (annualised, using % returns)
mean_r = df_t["pnl_pct"].mean()
std_r = df_t["pnl_pct"].std()
sharpe = round((mean_r / std_r) * (252 ** 0.5), 2) if std_r > 0 else 0
# Profit factor
gp = wins["pnl"].sum()
gl = abs(losses["pnl"].sum())
pf = round(gp / gl, 2) if gl > 0 else 99.0
return {
"initial_capital": INITIAL_CAPITAL,
"final_capital": round(capital, 2),
"total_return_pct": round((capital - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100, 2),
"total_trades": len(trades),
"win_rate": round(len(wins) / len(trades) * 100, 1),
"avg_pnl_pct": round(df_t["pnl_pct"].mean(), 3),
"avg_win_pct": round(wins["pnl_pct"].mean(), 3) if len(wins) > 0 else 0,
"avg_loss_pct": round(losses["pnl_pct"].mean(), 3) if len(losses) > 0 else 0,
"profit_factor": pf,
"sharpe_ratio": sharpe,
"max_drawdown_pct": max_dd,
"target_hits": len(df_t[df_t["exit_reason"] == "TARGET"]),
"sl_hits": len(df_t[df_t["exit_reason"] == "STOP_LOSS"]),
"avg_hold_days": round(df_t["hold_days"].mean(), 1),
}
def interpret_backtest(bt):
wr = bt.get("win_rate", 0)
pf = bt.get("profit_factor", 0)
sr = bt.get("sharpe_ratio", 0)
dd = bt.get("max_drawdown_pct", 0)
if wr >= 55 and pf >= 1.5 and sr >= 1.0:
return "Strategy looks VIABLE based on historical data"
elif wr >= 45 and pf >= 1.2:
return "Strategy shows MODERATE promise — validate further"
else:
return "Strategy needs improvement before live use"
# ═══════════════════════════════════════════════════
# 6. NEWS
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
# 7. GPT-4 ANALYSIS (FIXED API CALL)
# — full data context injected (not just signal+return)
# — correct: client.chat.completions.create
# — NOT client.responses.create (that was v3's bug)
# ═══════════════════════════════════════════════════
def analyze_with_gpt(stock_name, row, sig, bt, news_items):
client = OpenAI(api_key=OPENAI_API_KEY)
news_text = "\n".join([f" {i+1}. {n}" for i, n in enumerate(news_items)])
verdict = interpret_backtest(bt)
qty = position_size(INITIAL_CAPITAL, sig["entry"], sig["stop_loss"])
prompt = f"""
You are a world-class equity research analyst and technical analyst.
Analyze {stock_name} using the LIVE DATA below. Use ONLY this data for technical analysis.
LIVE TECHNICAL DATA:
Price: Rs{row.close:.2f}
MA20: Rs{row.ma20:.2f} | MA50: Rs{row.ma50:.2f} | MA200: Rs{row.ma200:.2f}
EMA9: Rs{row.ema9:.2f} | EMA21: Rs{row.ema21:.2f}
RSI: {row.rsi:.1f} | MACD hist: {row.macd_hist:.4f} ({'bullish' if row.macd_hist > 0 else
'bearish'})
Stochastic K/D: {row.stoch_k:.1f}/{row.stoch_d:.1f}
BB: Rs{row.bb_upper:.2f} / Rs{row.bb_mid:.2f} / Rs{row.bb_lower:.2f}
ATR: Rs{row.atr:.2f} | Volume ratio: {row.vol_ratio:.2f}x avg
SIGNAL:
Signal: {sig['signal']} | Score: {sig['score']} | Confidence: {sig['confidence']}%
Entry: Rs{sig['entry']} | Stop Loss: Rs{sig['stop_loss']}
Target 1: Rs{sig['target_1']} | Target 2: Rs{sig['target_2']}
Risk/Reward: {sig['risk_reward']}:1 | ATR: Rs{sig['atr']}
Position size (on Rs1L capital, 1% risk): {qty} shares
Reasons: {' | '.join(sig['reasons'])}
BACKTEST (300 days, SL={STOP_LOSS_PCT}%, Target={TARGET_PCT}%, max {MAX_HOLD_DAYS}d hold):
Trades: {bt.get('total_trades')} | Win Rate: {bt.get('win_rate')}%
Profit Factor: {bt.get('profit_factor')} | Sharpe: {bt.get('sharpe_ratio')}
Max Drawdown: {bt.get('max_drawdown_pct')}% | Avg Hold: {bt.get('avg_hold_days')} days
Total Return: {bt.get('total_return_pct')}% on Rs{INITIAL_CAPITAL:,}
Interpretation: {verdict}
LATEST NEWS:
{news_text}
Write a structured analysis with these sections:
1. TECHNICAL ANALYSIS
- Interpret each indicator precisely
- Confirm or challenge the {sig['signal']} signal
- Validate entry/SL/targets
- 3-sentence technical summary
2. BACKTEST INTERPRETATION
- Is this strategy historically reliable for {stock_name}?
- Interpret win rate, profit factor, Sharpe, drawdown
- Position sizing advice: is {qty} shares appropriate?
3. FUNDAMENTAL SNAPSHOT
- Business model, moat (label as approximate/general knowledge)
- Valuation context vs sector
- Bull case vs Bear case
- 1-3 month outlook
4. NEWS SENTIMENT
- Summarize each news item
- Mark Positive / Negative / Neutral
- Overall sentiment score 0-10
KEY TAKEAWAYS: 5 bullet points
Keep it clear and useful for an intermediate investor.
"""
print(" Sending to GPT-4...")
response = client.chat.completions.create(
model="gpt-4o",
messages=[{"role": "user", "content": prompt}],
max_tokens=2000,
)
return response.choices[0].message.content
# ═══════════════════════════════════════════════════
# 8. WHATSAPP
# ═══════════════════════════════════════════════════
def send_whatsapp(message):
if not CALLMEBOT_PHONE or not CALLMEBOT_APIKEY:
print(" WhatsApp: not configured, skipping")
return
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
verdict = interpret_backtest(bt)
qty = position_size(INITIAL_CAPITAL, sig["entry"], sig["stop_loss"])
return f"""{emoji} *{name} — {sig['signal']}* ({sig['confidence']}% conf)
{datetime.now().strftime('%d %b %Y')}
Price: Rs{row.close:.2f} | RSI: {row.rsi:.1f} | ATR: Rs{sig['atr']}
Trend: {'Uptrend' if row.close > row.ma200 else 'Downtrend'}
MACD: {'Bullish crossover!' if (row.macd_hist > 0) else 'Bearish'}
Entry: Rs{sig['entry']}
Stop Loss: Rs{sig['stop_loss']}
Target 1: Rs{sig['target_1']}
Target 2: Rs{sig['target_2']}
R/R: {sig['risk_reward']}:1
Qty (1% risk on Rs1L): {qty} shares
Backtest: WR {bt.get('win_rate')}% | Sharpe {bt.get('sharpe_ratio')} | DD
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
print(f" Stock Pipeline v4.0 | {datetime.now().strftime('%d %b %Y %H:%M')}")
print(f"{'='*55}\n")
obj = login()
for stock in STOCKS:
name = stock["name"]
print(f"\nProcessing {name}...")
try:
df = get_data(obj, stock["symbol"], stock["exchange"], days=300)
df = add_indicators(df)
row = df.iloc[-1]
sig = generate_signal(df, len(df) - 1)
print(f" Signal: {sig['signal']} | Score: {sig['score']} | Conf:
{sig['confidence']}%")
print(f" Entry: Rs{sig['entry']} | SL: Rs{sig['stop_loss']} | T1:
Rs{sig['target_1']}")
bt = backtest(df)
if "error" not in bt:
print(f" Backtest: WR={bt['win_rate']}% | Sharpe={bt['sharpe_ratio']} | PF=
{bt['profit_factor']} | DD={bt['max_drawdown_pct']}%")
print(f" Return on Rs1L: Rs{bt['final_capital']:,.0f}
({bt['total_return_pct']}%)")
else:
print(f" Backtest: {bt['error']}")
bt = {"win_rate":0,"sharpe_ratio":0,"profit_factor":0,"max_drawdown_pct":0,
"total_trades":0,"avg_hold_days":0,"total_return_pct":0,"final_capital":INITIAL_CAPITAL}
news = fetch_news(name)
# WhatsApp snapshot
send_whatsapp(format_snapshot(name, row, sig, bt))
# GPT-4 full analysis
analysis = analyze_with_gpt(name, row, sig, bt, news)
header = f"AI Analysis — {name}\n{'-'*40}\n"
chunks = split_message(header + analysis)
for i, chunk in enumerate(chunks):
prefix = f"[Part {i+1}/{len(chunks)}]\n" if len(chunks) > 1 else ""
send_whatsapp(prefix + chunk)
print(f" Done: {name}")
except Exception as e:
print(f" ERROR processing {name}: {e}")
import traceback
traceback.print_exc()
time.sleep(3)
print(f"\nDone — {len(STOCKS)} stocks processed")
if __name__ == "__main__":
run()
