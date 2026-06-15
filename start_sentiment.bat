@echo off
REM ============================================================
REM  Sentiment Engine - one-click launcher
REM  Starts: engine API, live feeds, dashboard (3 windows)
REM  Optionally runs the 48h history backfill.
REM ============================================================
title Sentiment Stack Launcher
set ENGINE_DIR=%~dp0
set DASH_DIR=C:\Users\Ricardo\Desktop\Work\PROJECTOS\sentiment_dashboard_react\sentiment_dashboard_react

REM ===================== OPT-IN FEATURES (see README) =====================
REM CryptoBERT transformer scorer. Requires one-time:  pip install transformers torch
REM Set blank to go back to the fast lexicon. Falls back to lexicon if deps missing.
set SENTIMENT_TRANSFORMER_MODEL=ElKulako/cryptobert
REM CryptoPanic is now pulled via its keyless RSS feed (in the news sources) -- no
REM token needed. Their free API is discontinued and votes are paywalled.
REM ========================================================================

echo [1/4] Starting engine API on port 8787...
start "Sentiment Engine (API :8787)" cmd /k "cd /d %ENGINE_DIR% && python -m uvicorn sentiment_engine.api:app --host 127.0.0.1 --port 8787"

echo       Waiting for engine to come online...
:wait_engine
timeout /t 2 /nobreak >nul
powershell -NoProfile -Command "try { Invoke-RestMethod http://127.0.0.1:8787/health -TimeoutSec 2 | Out-Null; exit 0 } catch { exit 1 }"
if errorlevel 1 goto wait_engine
echo       Engine online.

echo [2/4] Starting live feeds (binance + news + context)...
REM Reddit omitted: its public API now requires approved OAuth (see README).
REM Re-add "reddit" below once SENTIMENT_REDDIT_CLIENT_ID/SECRET are set.
start "Live Feeds" cmd /k "cd /d %ENGINE_DIR% && python -m sentiment_engine.ingestion.run_live --sources binance news bluesky context lsr deriv --symbols BTC/USDT ADA/USDT ETH/USDT BNB/USDT SOL/USDT XRP/USDT DOGE/USDT LINK/USDT"

echo [3/4] Starting dashboard...
start "Dashboard (http://127.0.0.1:5173)" cmd /k "cd /d %DASH_DIR% && npm run dev"

echo [4/4] Opening dashboard in browser...
timeout /t 4 /nobreak >nul
start http://127.0.0.1:5173

choice /C YN /T 15 /D N /M "Run 48h history backfill now (recommended after downtime)"
if errorlevel 2 goto done
echo Running backfill...
cd /d %ENGINE_DIR%
python -m sentiment_engine.ingestion.backfill --sources news --hours 48

:done
echo.
echo All running. Close the three opened windows to stop everything.
echo This launcher window can be closed.
pause
