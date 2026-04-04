# Stock Pipeline V8.1 - Automated Nifty 50 Trading Signal Generator

A production-ready Python pipeline that scans the Nifty 50 index daily, generates AI-powered trading signals, backtests them, and delivers results via email.

## Overview

This project automates the entire workflow of:
- **Market Scanning**: Fetches real-time data for all 50 Nifty constituents from NSE
- **Technical Analysis**: Computes RSI, MACD, EMA, SMA200, and ATR indicators
- **Signal Generation**: Generates BUY/SELL/HOLD signals using a weighted scoring system
- **Backtesting**: Validates signal quality with historical performance metrics
- **AI Analysis**: Uses OpenAI to generate concise trading insights for each signal
- **Report Delivery**: Sends formatted HTML email reports with top signals

## Key Features

✅ **Production-Ready** - Error handling, retry logic, rate limiting, graceful degradation  
✅ **Real-time NSE Data** - Live Nifty 50 constituents with fallback snapshot  
✅ **Intelligent Signal Engine** - Score-based system combining 4 technical indicators  
✅ **Backtesting Framework** - Validates signal quality with position sizing & P&L tracking  
✅ **AI-Powered Insights** - ChatGPT analysis for each signal (optional, graceful fallback)  
✅ **Email Delivery** - Beautiful HTML reports via Resend (works on Railway free tier)  
✅ **Concurrent Processing** - Efficient multi-threaded stock processing with rate-limit respect  

## Architecture

### Signal Scoring System
Each stock receives a score based on four technical conditions:
- **Trend (±2 pts)**: Close vs 200-day MA
- **MACD Crossover (±3 pts)**: Histogram crosses zero
- **Short-term Trend (±1 pt)**: EMA-9 vs EMA-21
- **RSI (±1 pt)**: Extreme conditions (<30 or >70)

**Signal Thresholds:**
- BUY: score ≥ 4
- SELL: score ≤ -4
- HOLD: -3 to +3

### Position Sizing
For each signal, calculates quantity based on:
- **Risk per trade**: 1% of ₹100,000 capital
- **ATR stop-loss**: Close - (1.5 × ATR)
- **Target**: Close + (2.0 × ATR)
- **Max position**: 20% of capital or risk-based qty (whichever is lower)

### Backtesting Engine
Simulates trading on historical data (300 days):
- Entry on signal generation
- Exit on SL hit, target hit, or after 10 days max hold
- Accounts for 0.05% brokerage per leg
- Tracks: final capital, win rate, P&L, max drawdown, profit factor

## Setup & Configuration

### Prerequisites
- Python 3.8+
- Angel One trading account (for market data)
- OpenAI API key (for AI analysis)
- Resend account (for email delivery, optional)

### Environment Variables

**Required (Market Data):**
```bash
ANGEL_API_KEY=your_angel_one_api_key
ANGEL_CLIENT_ID=your_client_id
ANGEL_PASSWORD=your_password
ANGEL_TOTP_SECRET=your_totp_secret
```

**Optional (AI Analysis):**
```bash
OPENAI_API_KEY=your_openai_key
OPENAI_MODEL=gpt-4o  # or gpt-4o-mini for cost savings
```

**Optional (Email Delivery):**
```bash
RESEND_API_KEY=your_resend_api_key
EMAIL_SENDER=onboarding@resend.dev  # or verified domain
EMAIL_RECIPIENT=your_email@example.com
```

### Installation

```bash
# Clone the repository
git clone https://github.com/Goutham009-GK/Portfolio-Analyser.git
cd Portfolio-Analyser

# Install dependencies
pip install -r requirements.txt

# Set environment variables (in .env or system env)
export ANGEL_API_KEY=...
export OPENAI_API_KEY=...
# etc.

# Run the pipeline
python main.py
```

## Usage

### Daily Execution
The pipeline is designed to run once daily (typically post-market close):

```bash
python main.py
```

**Output:**
1. **Console Logs** → Real-time processing status and errors
2. **pipeline.log** → Detailed execution log
3. **Email Report** → HTML with top 10 signals (if email is configured)

### Typical Flow
1. Validates all required credentials
2. Probes OpenAI quota (fails gracefully if unavailable)
3. Fetches live Nifty 50 constituents from NSE
4. For each stock:
   - Fetches 300-day OHLCV data
   - Screens by minimum volume (500k daily average)
   - Computes technical indicators
   - Generates signal & backtest metrics
   - Requests AI analysis (if enabled)
5. Selects top N signals (default: 10, sorted by score strength)
6. Sends email report or logs results

## Project Structure

```
.
├── main.py                 # Main pipeline entry point
├── requirements.txt        # Python dependencies
├── pipeline.log            # Runtime logs
├── README.md               # This file
└── .env                    # Environment variables (git-ignored)
```

## Signal Interpretation

**BUY Signals** (Green ▲)
- Multiple bullish indicators aligned
- Typically occur after MACD crossover + price above MA200
- Use Entry, Stop Loss, and Target for position management

**SELL Signals** (Red ▼)
- Multiple bearish indicators aligned
- Typically occur after MACD crossover below + price below MA200
- High-risk reversals; backtest validation crucial

**HOLD Signals** (Orange ●)
- Weak or conflicting signals
- Included in report only if needed to fill top-N slots
- Good for observing stocks without trading

## Email Report Contents

| Column | Description |
|--------|-------------|
| Stock | Company name and NSE symbol |
| Signal | BUY/SELL/HOLD with signal strength |
| Entry | Recommended entry price (₹) |
| Stop Loss | Risk management stop level (₹) |
| Target | Profit target (₹) |
| RSI | 14-period RSI (0-100) |
| Analysis | AI-generated 2-sentence rationale |

## Backtesting Metrics Included

- **Net Return %** → Total profit/loss from backtest
- **Win Rate %** → Percentage of winning trades
- **Profit Factor** → Gross wins / Gross losses
- **Max Drawdown %** → Peak-to-trough decline

## Rate Limiting & Reliability

- **Angel One**: 1.2s delay between stocks (respect API RPM)
- **OpenAI**: 1.5s delay between API calls (avoid 429 errors)
- **Retry Logic**: 3 attempts with exponential backoff on network failures
- **Graceful Degradation**: Signals survive even if AI/email fail

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "Invalid Angel One credentials" | Verify ANGEL_* env vars and broker account |
| "OpenAI quota exhausted" | Add credits at platform.openai.com/settings/billing |
| "Email send failed" | Check RESEND_API_KEY validity; test with onboarding@resend.dev |
| "No token found for SYMBOL" | Stock may be delisted or excluded from Angel One universe |

## Disclaimer

⚠️ This report is generated automatically for informational purposes only and does not constitute financial advice. Always conduct your own research before making investment decisions. Past backtest performance does not guarantee future results. Trading involves substantial risk of loss.

## License

This project is provided as-is for educational and research purposes.

## Author

Maintained by [@Goutham009-GK](https://github.com/Goutham009-GK)

---

**Version:** 8.1 (Production Ready)  
**Last Updated:** 2026-04-04