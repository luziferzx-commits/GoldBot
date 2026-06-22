# 🤖 GoldBot 4.0 - Advanced AI Trading System

GoldBot is an automated, AI-driven trading system designed specifically for the XAUUSD (Gold) pair. It combines a deep learning Bidirectional LSTM model with an Attention mechanism, multiple technical/fundamental filters, and an ensemble of traditional algorithmic strategies (Silver Bullet, Asian Range, SGE, PO3, Overlap Scalper) orchestrated by a Dynamic Strategy Selector.

## 🌟 Key Features
- **AI-Powered:** 3-layer Bi-LSTM with Attention mechanism trained on 10 years of MTF data.
- **Online Learning:** The model continuously learns from recent trades and features a rollback mechanism if performance drops.
- **Dynamic Strategy Selector:** Routes capital to the most profitable strategy based on current market regime and session.
- **Premium Dashboard:** Real-time web dashboard for tracking equity, performance, and current active signals.
- **Telegram Integration:** Complete control and real-time alerts via Telegram.
- **Resilient Architecture:** Automatic MT5 reconnections, daily SQLite backups, and Windows Service (NSSM) support.

---

## 🛠️ 1. Step-by-Step Installation

### Prerequisites
- Windows OS (Required for MetaTrader 5 Terminal)
- MetaTrader 5 installed and logged in
- Python 3.11+
- Git

### Installation Steps
1. **Clone the repository:**
   ```bash
   git clone https://github.com/luziferzx-commits/GoldBot.git
   cd GoldBot
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Variables Configuration:**
   Create a `.env` file in the root directory (where `README.md` is) and add your credentials:
   ```env
   MT5_LOGIN=your_mt5_account_number
   MT5_PASSWORD=your_mt5_password
   MT5_SERVER=your_broker_server_name

   TELEGRAM_BOT_TOKEN=your_telegram_bot_token
   TELEGRAM_CHAT_ID=your_telegram_chat_id
   ```

---

## 📱 2. How to Set Up Telegram Bot

1. Open Telegram and search for **@BotFather**.
2. Send `/newbot` and follow the instructions to create a new bot and get the **HTTP API Token**.
3. Copy the token and paste it into your `.env` file as `TELEGRAM_BOT_TOKEN`.
4. Open your new bot in Telegram and click **Start** (or send any message).
5. Search for **@userinfobot** or **@getmyid_bot** to get your personal Chat ID.
6. Copy the ID and paste it into your `.env` file as `TELEGRAM_CHAT_ID`.

---

## 🚀 3. How to Run Demo and Live Mode

The system relies on `config/settings.yaml` for behavioral configuration. 
To switch between Demo/Learning mode and Live trading mode, edit `learning_mode` in `config/settings.yaml`:

```yaml
ai:
  model_path: "models/live/model_current.pt"
  confidence_threshold: 0.52
  learning_mode: true  # Set to 'true' for Demo/Learning. Set to 'false' for Live Trading.
```

### Starting the Bot manually:
To run the Main Trading Loop:
```bash
.\venv\Scripts\python -m src.main
```

To run the Premium Dashboard:
```bash
.\venv\Scripts\python -m src.dashboard.app
```
Then open your browser to `http://localhost:5000`.

---

## 💬 4. Telegram Commands

You can control the bot directly from your Telegram chat. Just type these commands:

- `/status` — Check if the bot is running and view current mode (Live/Learning).
- `/summary` — Get a daily performance report (Equity, Daily PnL, Win Rate).
- `/bias` — View today's fundamental bias (DXY, VIX, US10Y).
- `/patterns` — View the top performing patterns in the last 30 days.
- `/calendar` — Check high-impact economic news for the next 24 hours.
- `/selector` — View current scores of all strategies in the Strategy Selector.
- `/history` — View recent trade history.
- `/rollback` — Check rollback status or manually revert the AI model to a previous stable state.
- `/retrain` — Force the AI model to start retraining immediately.
- `/close_all` — Emergency button: Closes all open positions instantly.
- `/stop` — Stops the bot and closes all open positions.

---

## ☁️ 5. VPS Installation (Windows Service)

To ensure the bot runs 24/7, survives VPS reboots, and automatically recovers from crashes, we use **NSSM (Non-Sucking Service Manager)**.

1. Download [NSSM](https://nssm.cc/release/nssm-2.24.zip).
2. Extract the ZIP and copy `win64\nssm.exe` to `C:\Windows\System32\` (or add it to your system PATH).
3. Right-click on `scripts\install_service.bat` and select **Run as Administrator**.
4. The script will automatically install both `GoldBot_Main` and `GoldBot_Dashboard` as Windows background services.

### Managing Services:
You can manage them via the standard Windows `Services.msc` app, or via Command Prompt (Admin):
```bash
nssm start GoldBot_Main
nssm stop GoldBot_Main
nssm restart GoldBot_Main
```
Logs for the services will be piped to `logs/service_main.log` and `logs/service_dashboard.log`.

---
*Disclaimer: Trading involves significant risk. This bot is for educational and experimental purposes.*
