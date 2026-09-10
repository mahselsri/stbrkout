from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
import numpy as np
import time
import requests
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# NIFTY 50 STOCK LIST
# ============================================================
NIFTY_50 = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
    "LT.NS", "AXISBANK.NS", "WIPRO.NS", "ASIANPAINT.NS", "HCLTECH.NS",
    "MARUTI.NS", "BAJFINANCE.NS", "TITAN.NS", "SUNPHARMA.NS", "TECHM.NS",
    "NESTLEIND.NS", "POWERGRID.NS", "ULTRACEMCO.NS", "ADANIENT.NS", "TATAMOTORS.NS",
    "ONGC.NS", "TATASTEEL.NS", "JSWSTEEL.NS", "NTPC.NS", "INDUSINDBK.NS",
    "M&M.NS", "COALINDIA.NS", "BAJAJFINSV.NS", "HINDALCO.NS", "DRREDDY.NS",
    "GRASIM.NS", "DIVISLAB.NS", "BAJAJ-AUTO.NS", "BRITANNIA.NS", "HEROMOTOCO.NS",
    "ADANIPORTS.NS", "CIPLA.NS", "UPL.NS", "SBILIFE.NS", "EICHERMOT.NS",
    "BPCL.NS", "TATACONSUM.NS", "APOLLOHOSP.NS", "SHREECEM.NS", "HDFC.NS"
]

# ============================================================
# Models
# ============================================================
class ResistanceLevels(BaseModel):
    pivot: float
    r1: float
    r2: float
    r3: float

class BreakoutSignal(BaseModel):
    level: str
    value: float
    breakout_percent: float

class BreakoutResponse(BaseModel):
    symbol: str
    current_price: float
    previous_close: float
    resistance_levels: ResistanceLevels
    breakouts: List[BreakoutSignal]
    is_breakout: bool
    volume_confirmation: bool
    timestamp: str
    data_source: str = "unknown"
    date_used: str = ""

class StockBreakout(BaseModel):
    symbol: str
    current_price: float
    previous_close: float
    change_percent: float
    resistance_levels: ResistanceLevels
    breakouts: List[BreakoutSignal]
    is_breakout: bool
    volume_confirmation: bool
    volume_ratio: float
    date_used: str
    data_source: str = "unknown"

class ScanResponse(BaseModel):
    scan_time: str
    total_stocks: int
    breakout_count: int
    breakouts: List[StockBreakout]
    failed_stocks: List[str]
    scan_duration_seconds: float

cache = {}
cache_expiry = {}

def get_from_cache(key, max_age=300):
    if key in cache and key in cache_expiry:
        if datetime.now() < cache_expiry[key]:
            return cache[key]
    return None

def set_in_cache(key, data, ttl=300):
    cache[key] = data
    cache_expiry[key] = datetime.now() + timedelta(seconds=ttl)

# --- Data Source 1: Yahoo Finance ---
def fetch_from_yahoo(symbol: str, period: str = '3mo'):
    try:
        print(f"Yahoo Finance: {symbol}")
        ticker = yf.Ticker(symbol)
        info = ticker.info
        currency = info.get('currency', 'INR') if info else 'INR'
        today = datetime.now()

        data = None
        try:
            data = ticker.history(period=period)
        except Exception as e:
            print(f"  Period fetch error: {e}")

        if data is None or data.empty:
            try:
                end_date = today.strftime('%Y-%m-%d')
                start_date = (today - timedelta(days=90)).strftime('%Y-%m-%d')
                data = ticker.history(start=start_date, end=end_date)
            except Exception as e:
                print(f"  Date fetch error: {e}")

        if data is None or data.empty:
            try:
                data = ticker.history(period='1mo')
            except Exception as e:
                print(f"  1mo fetch error: {e}")

        if data is not None and not data.empty:
            latest_price = float(data['Close'].iloc[-1])
            if currency == 'USD' and 1 < latest_price < 1000:
                data['Open'] = data['Open'] * 83.5
                data['High'] = data['High'] * 83.5
                data['Low'] = data['Low'] * 83.5
                data['Close'] = data['Close'] * 83.5
            return data, 'yahoo'

        return None, None
    except Exception as e:
        print(f"  Yahoo error: {str(e)}")
        return None, None

# --- Data Source 2: Info API ---
def fetch_from_info(symbol: str):
    try:
        print(f"Info API: {symbol}")
        ticker = yf.Ticker(symbol)
        info = ticker.info

        if info and 'regularMarketPrice' in info:
            current_price = info.get('regularMarketPrice', 0)
            previous_close = info.get('previousClose', current_price)
            currency = info.get('currency', 'INR')

            if currency == 'USD' and current_price < 1000:
                current_price = current_price * 83.5
                previous_close = previous_close * 83.5

            data = pd.DataFrame({
                'Open': [current_price * 0.995],
                'High': [current_price * 1.005],
                'Low': [current_price * 0.995],
                'Close': [current_price],
                'Volume': [info.get('regularMarketVolume', 0)]
            }, index=[pd.Timestamp.now()])

            return data, 'info'
        return None, None
    except Exception as e:
        print(f"  Info error: {str(e)}")
        return None, None

# --- Data Source 3: Alpha Vantage (BSE only) ---
def fetch_from_alphavantage(symbol: str):
    try:
        api_key = os.environ.get('ALPHA_VANTAGE_KEY', '')
        if not api_key:
            print("  Alpha Vantage: No API key")
            return None, None

        print(f"Alpha Vantage: {symbol}")

        # Alpha Vantage supports BSE, not NSE
        if symbol.endswith('.NS'):
            av_symbol = symbol.replace('.NS', '.BSE')
        elif symbol.endswith('.BO'):
            av_symbol = symbol.replace('.BO', '.BSE')
        else:
            av_symbol = symbol + '.BSE'

        print(f"  AV symbol: {av_symbol}")

        url = "https://www.alphavantage.co/query"
        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": av_symbol,
            "apikey": api_key,
            "outputsize": "compact"
        }

        response = requests.get(url, params=params, timeout=10)
        data = response.json()

        if "Time Series (Daily)" in data:
            time_series = data["Time Series (Daily)"]
            df_data = []
            for date_str, values in list(time_series.items())[:90]:
                try:
                    date = pd.to_datetime(date_str)
                    df_data.append({
                        'Date': date,
                        'Open': float(values['1. open']),
                        'High': float(values['2. high']),
                        'Low': float(values['3. low']),
                        'Close': float(values['4. close']),
                        'Volume': float(values['5. volume'])
                    })
                except Exception:
                    continue

            if df_data:
                df = pd.DataFrame(df_data)
                df.set_index('Date', inplace=True)
                df.sort_index(inplace=True)
                latest = float(df['Close'].iloc[-1])
                print(f"  AV success: {len(df)} rows, latest: ₹{latest}")
                return df, 'alphavantage'

        if "Note" in data:
            print(f"  AV rate limit: {data['Note'][:100]}")
        elif "Information" in data:
            print(f"  AV info: {data['Information'][:100]}")
        else:
            print(f"  AV unexpected response: {list(data.keys())}")

        return None, None
    except Exception as e:
        print(f"  Alpha Vantage error: {str(e)}")
        return None, None

# --- Main Fetch Function ---
def fetch_stock_data(symbol: str, period: str = '3mo'):
    cache_key = f"{symbol}_{period}"
    cached = get_from_cache(cache_key)
    if cached is not None:
        return cached

    symbol = symbol.strip().upper()
    if '.' not in symbol:
        symbols_to_try = [f"{symbol}.NS", f"{symbol}.BO"]
    else:
        symbols_to_try = [symbol]

    for try_symbol in symbols_to_try:
        data, source = fetch_from_yahoo(try_symbol, period)
        if data is not None and not data.empty:
            latest = float(data['Close'].iloc[-1])
            if 100 < latest < 50000:
                set_in_cache(cache_key, data)
                return data

        data, source = fetch_from_info(try_symbol)
        if data is not None and not data.empty:
            latest = float(data['Close'].iloc[-1])
            if 100 < latest < 50000:
                set_in_cache(cache_key, data)
                return data

        data, source = fetch_from_alphavantage(try_symbol)
        if data is not None and not data.empty:
            latest = float(data['Close'].iloc[-1])
            if 100 < latest < 50000:
                set_in_cache(cache_key, data)
                return data

    raise HTTPException(
        status_code=404,
        detail=f"Could not fetch valid data for {symbol}. Please try again later."
    )

# --- Batch Download with Alpha Vantage Fallback ---
def batch_download_stocks(symbols: List[str], period: str = '3mo') -> Dict[str, pd.DataFrame]:
    """
    Batch download stocks via Yahoo Finance.
    Falls back to Alpha Vantage (BSE) for any that fail.
    """
    results = {}
    failed_symbols = []

    # --- Yahoo batch download ---
    try:
        print(f"\n📥 Batch downloading {len(symbols)} stocks via Yahoo...")
        start = time.time()

        data = yf.download(
            tickers=symbols,
            period=period,
            progress=False,
            threads=True,
            auto_adjust=True,
            group_by='ticker'
        )

        if len(symbols) == 1:
            if not data.empty:
                results[symbols[0]] = data
        else:
            for symbol in symbols:
                try:
                    if symbol in data.columns.get_level_values(0):
                        df = data[symbol].dropna()
                        if not df.empty and len(df) >= 5:
                            results[symbol] = df
                        else:
                            failed_symbols.append(symbol)
                    else:
                        failed_symbols.append(symbol)
                except Exception:
                    failed_symbols.append(symbol)

        elapsed = round(time.time() - start, 2)
        print(f"✅ Yahoo batch: {len(results)}/{len(symbols)} stocks in {elapsed}s")
        if failed_symbols:
            print(f"⚠️ Failed from Yahoo: {len(failed_symbols)} stocks")

    except Exception as e:
        print(f"❌ Yahoo batch failed: {e}")
        failed_symbols = list(symbols)

    # --- Alpha Vantage fallback for failed stocks ---
    if failed_symbols and os.environ.get('ALPHA_VANTAGE_KEY'):
        print(f"\n🔄 Alpha Vantage fallback for {len(failed_symbols)} stocks...")

        # AV free tier: 5 requests/minute → limit to 5 per scan
        max_av_attempts = min(len(failed_symbols), 5)
        av_success = 0

        for i, symbol in enumerate(failed_symbols[:max_av_attempts]):
            try:
                # Delay between AV requests to avoid rate limits
                if i > 0:
                    time.sleep(2)

                df, source = fetch_from_alphavantage(symbol)
                if df is not None and not df.empty:
                    results[symbol] = df
                    av_success += 1
                    print(f"  ✅ AV added: {symbol}")
                else:
                    print(f"  ❌ AV failed: {symbol}")
            except Exception as e:
                print(f"  ❌ AV error {symbol}: {e}")
                continue

        print(f"✅ Alpha Vantage added: {av_success} stocks")
    elif failed_symbols:
        print(f"⚠️ {len(failed_symbols)} stocks failed — no ALPHA_VANTAGE_KEY set")

    print(f"\n📊 Total stocks ready: {len(results)}/{len(symbols)}\n")
    return results

# --- Analysis Functions ---
def calculate_pivot_points(data):
    try:
        if len(data) < 2:
            row = data.iloc[-1]
            high = float(row['High'])
            low = float(row['Low'])
            close = float(row['Close'])
        else:
            prev_day = data.iloc[-2]
            high = float(prev_day['High'])
            low = float(prev_day['Low'])
            close = float(prev_day['Close'])

        pivot = (high + low + close) / 3
        r1 = 2 * pivot - low
        r2 = pivot + (high - low)
        r3 = high + 2 * (pivot - low)

        return {
            'pivot': round(pivot, 2),
            'r1': round(r1, 2),
            'r2': round(r2, 2),
            'r3': round(r3, 2)
        }
    except Exception as e:
        print(f"Pivot calculation error: {e}")
        return {'pivot': 0, 'r1': 0, 'r2': 0, 'r3': 0}

def calculate_dynamic_resistance(data, lookback=14):
    try:
        if len(data) < lookback:
            lookback = max(len(data) // 2, 5)
        if len(data) < 3:
            return {'r1': 0, 'r2': 0, 'r3': 0}

        rolling_mean = data['Close'].rolling(window=lookback).mean()
        rolling_std = data['Close'].rolling(window=lookback).std()

        r1 = float(rolling_mean.iloc[-1] + rolling_std.iloc[-1]) if not pd.isna(rolling_mean.iloc[-1]) else 0
        r2 = float(rolling_mean.iloc[-1] + 1.5 * rolling_std.iloc[-1]) if not pd.isna(rolling_mean.iloc[-1]) else 0
        r3 = float(rolling_mean.iloc[-1] + 2 * rolling_std.iloc[-1]) if not pd.isna(rolling_mean.iloc[-1]) else 0

        return {'r1': round(r1, 2), 'r2': round(r2, 2), 'r3': round(r3, 2)}
    except Exception as e:
        print(f"Dynamic resistance error: {e}")
        return {'r1': 0, 'r2': 0, 'r3': 0}

def detect_breakout(data, symbol):
    try:
        if data.empty:
            raise ValueError("No data available")

        pivot_levels = calculate_pivot_points(data)
        dynamic_levels = calculate_dynamic_resistance(data)

        resistance = {
            'pivot': pivot_levels['pivot'],
            'r1': max(pivot_levels['r1'], dynamic_levels['r1']),
            'r2': max(pivot_levels['r2'], dynamic_levels['r2']),
            'r3': max(pivot_levels['r3'], dynamic_levels['r3'])
        }

        current_price = round(float(data['Close'].iloc[-1]), 2)
        previous_close = round(float(data['Close'].iloc[-2]), 2) if len(data) > 1 else current_price
        latest_date = data.index[-1].strftime('%Y-%m-%d')

        volume_confirmation = False
        volume_ratio = 0.0
        if len(data) >= 14:
            try:
                avg_volume = float(data['Volume'].rolling(window=14).mean().iloc[-1])
                current_volume = float(data['Volume'].iloc[-1])
                if avg_volume > 0:
                    volume_ratio = round(current_volume / avg_volume, 2)
                    volume_confirmation = volume_ratio > 1.5
            except:
                pass

        breakouts = []
        if resistance['r1'] > 0 and current_price > resistance['r1']:
            breakouts.append({
                'level': 'R1',
                'value': resistance['r1'],
                'breakout_percent': round(((current_price - resistance['r1']) / resistance['r1']) * 100, 2)
            })
        if resistance['r2'] > 0 and current_price > resistance['r2']:
            breakouts.append({
                'level': 'R2',
                'value': resistance['r2'],
                'breakout_percent': round(((current_price - resistance['r2']) / resistance['r2']) * 100, 2)
            })
        if resistance['r3'] > 0 and current_price > resistance['r3']:
            breakouts.append({
                'level': 'R3',
                'value': resistance['r3'],
                'breakout_percent': round(((current_price - resistance['r3']) / resistance['r3']) * 100, 2)
            })

        return {
            'symbol': symbol,
            'current_price': current_price,
            'previous_close': previous_close,
            'resistance_levels': resistance,
            'breakouts': breakouts,
            'is_breakout': len(breakouts) > 0,
            'volume_confirmation': volume_confirmation,
            'volume_ratio': volume_ratio,
            'date_used': latest_date
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")

# ============================================================
# Endpoints
# ============================================================

@app.get("/")
async def root():
    return {
        "message": "Stock Breakout Detection API",
        "status": "running",
        "data_sources": ["Yahoo Finance", "Info API", "Alpha Vantage (BSE)"],
        "endpoints": {
            "existing": ["/debug/{symbol}", "/analyze/{symbol}", "/scan?symbols=...", "/popular", "/health", "/cache/clear"],
            "new": ["/scan/nifty50", "/scan/custom?symbols=...", "/nifty50"]
        },
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "cache_size": len(cache),
        "alpha_vantage_configured": bool(os.environ.get('ALPHA_VANTAGE_KEY')),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/debug/{symbol}")
async def debug_stock(symbol: str):
    try:
        symbol = symbol.strip().upper()
        if '.' not in symbol:
            symbol = symbol + '.NS'

        results = {}

        data, source = fetch_from_yahoo(symbol)
        if data is not None and not data.empty:
            results['yahoo'] = {
                'status': 'success',
                'latest_price': float(data['Close'].iloc[-1]),
                'data_points': len(data),
                'first_date': data.index[0].strftime('%Y-%m-%d'),
                'last_date': data.index[-1].strftime('%Y-%m-%d')
            }
        else:
            results['yahoo'] = {'status': 'failed'}

        data, source = fetch_from_info(symbol)
        if data is not None and not data.empty:
            results['info'] = {
                'status': 'success',
                'latest_price': float(data['Close'].iloc[-1]),
                'data_points': len(data)
            }
        else:
            results['info'] = {'status': 'failed'}

        data, source = fetch_from_alphavantage(symbol)
        if data is not None and not data.empty:
            results['alphavantage'] = {
                'status': 'success',
                'latest_price': float(data['Close'].iloc[-1]),
                'data_points': len(data),
                'first_date': data.index[0].strftime('%Y-%m-%d'),
                'last_date': data.index[-1].strftime('%Y-%m-%d')
            }
        else:
            results['alphavantage'] = {'status': 'failed'}

        results['server_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        return {"symbol": symbol, "results": results, "timestamp": datetime.now().isoformat()}
    except Exception as e:
        return {"error": str(e)}

@app.get("/popular")
async def get_popular_indian_stocks():
    stocks = [
        {"symbol": "RELIANCE.NS", "name": "Reliance Industries"},
        {"symbol": "TCS.NS", "name": "Tata Consultancy Services"},
        {"symbol": "INFY.NS", "name": "Infosys"},
        {"symbol": "HDFCBANK.NS", "name": "HDFC Bank"},
        {"symbol": "ICICIBANK.NS", "name": "ICICI Bank"},
        {"symbol": "SBIN.NS", "name": "State Bank of India"},
        {"symbol": "BHARTIARTL.NS", "name": "Bharti Airtel"},
        {"symbol": "ITC.NS", "name": "ITC Ltd"},
        {"symbol": "WIPRO.NS", "name": "Wipro"},
        {"symbol": "HCLTECH.NS", "name": "HCL Technologies"}
    ]
    return {"stocks": stocks, "total": len(stocks), "timestamp": datetime.now().isoformat()}

@app.get("/analyze/{symbol}")
async def analyze_stock(symbol: str, period: str = Query('3mo')):
    try:
        symbol = symbol.strip().upper()
        if '.' not in symbol:
            symbol = symbol + '.NS'

        data = fetch_stock_data(symbol, period)
        result = detect_breakout(data, symbol)

        return BreakoutResponse(
            symbol=result['symbol'],
            current_price=result['current_price'],
            previous_close=result['previous_close'],
            resistance_levels=ResistanceLevels(
                pivot=result['resistance_levels']['pivot'],
                r1=result['resistance_levels']['r1'],
                r2=result['resistance_levels']['r2'],
                r3=result['resistance_levels']['r3']
            ),
            breakouts=[BreakoutSignal(**b) for b in result['breakouts']],
            is_breakout=result['is_breakout'],
            volume_confirmation=result['volume_confirmation'],
            timestamp=datetime.now().isoformat(),
            date_used=result.get('date_used', '')
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/scan")
async def scan_stocks(symbols: str = Query('RELIANCE.NS,TCS.NS,INFY.NS')):
    try:
        stock_list = [s.strip().upper() for s in symbols.split(',')]
        stock_list = [s if '.' in s else s + '.NS' for s in stock_list]
        stock_list = stock_list[:3]

        results = []
        failed_stocks = []

        for symbol in stock_list:
            try:
                data = fetch_stock_data(symbol, '3mo')
                result = detect_breakout(data, symbol)
                if result['is_breakout']:
                    results.append(BreakoutResponse(
                        symbol=result['symbol'],
                        current_price=result['current_price'],
                        previous_close=result['previous_close'],
                        resistance_levels=ResistanceLevels(
                            pivot=result['resistance_levels']['pivot'],
                            r1=result['resistance_levels']['r1'],
                            r2=result['resistance_levels']['r2'],
                            r3=result['resistance_levels']['r3']
                        ),
                        breakouts=[BreakoutSignal(**b) for b in result['breakouts']],
                        is_breakout=result['is_breakout'],
                        volume_confirmation=result['volume_confirmation'],
                        timestamp=datetime.now().isoformat(),
                        date_used=result.get('date_used', '')
                    ))
            except Exception as e:
                failed_stocks.append(f"{symbol}: {str(e)[:50]}")
                continue

        return {
            "total_scanned": len(stock_list),
            "breakout_stocks": results,
            "breakout_count": len(results),
            "failed_stocks": failed_stocks,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/cache/clear")
async def clear_cache():
    cache.clear()
    cache_expiry.clear()
    return {"message": "Cache cleared", "timestamp": datetime.now().isoformat()}

@app.get("/nifty50")
async def get_nifty50_list():
    return {
        "stocks": [s.replace('.NS', '') for s in NIFTY_50],
        "symbols": NIFTY_50,
        "total": len(NIFTY_50),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/scan/nifty50", response_model=ScanResponse)
async def scan_nifty50(period: str = Query('3mo', description='Data period: 1mo, 3mo, 6mo')):
    start_time = time.time()

    print(f"\n{'='*60}")
    print(f"SCANNING NIFTY 50 FOR BREAKOUTS")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    stock_data = batch_download_stocks(NIFTY_50, period)

    if not stock_data:
        raise HTTPException(
            status_code=503,
            detail="Could not fetch stock data. Yahoo Finance and Alpha Vantage both failed. Try again in 5 minutes."
        )

    breakouts = []
    failed_stocks = []

    for symbol, data in stock_data.items():
        try:
            result = detect_breakout(data, symbol)

            stock_breakout = StockBreakout(
                symbol=symbol.replace('.NS', '').replace('.BO', ''),
                current_price=result['current_price'],
                previous_close=result['previous_close'],
                change_percent=round(
                    ((result['current_price'] - result['previous_close']) / result['previous_close']) * 100, 2
                ) if result['previous_close'] > 0 else 0,
                resistance_levels=ResistanceLevels(
                    pivot=result['resistance_levels']['pivot'],
                    r1=result['resistance_levels']['r1'],
                    r2=result['resistance_levels']['r2'],
                    r3=result['resistance_levels']['r3']
                ),
                breakouts=[BreakoutSignal(**b) for b in result['breakouts']],
                is_breakout=result['is_breakout'],
                volume_confirmation=result['volume_confirmation'],
                volume_ratio=result.get('volume_ratio', 0.0),
                date_used=result.get('date_used', ''),
                data_source="yahoo" if symbol.endswith('.NS') else "alphavantage"
            )

            if result['is_breakout']:
                breakouts.append(stock_breakout)
        except Exception as e:
            failed_stocks.append(f"{symbol}: {str(e)[:50]}")
            continue

    not_downloaded = [s for s in NIFTY_50 if s not in stock_data]
    failed_stocks.extend(not_downloaded)

    breakouts.sort(key=lambda x: (len(x.breakouts), x.volume_ratio), reverse=True)

    scan_duration = round(time.time() - start_time, 2)

    print(f"\nSCAN COMPLETE: {len(breakouts)} breakouts in {scan_duration}s\n")

    return ScanResponse(
        scan_time=datetime.now().isoformat(),
        total_stocks=len(NIFTY_50),
        breakout_count=len(breakouts),
        breakouts=breakouts,
        failed_stocks=failed_stocks,
        scan_duration_seconds=scan_duration
    )

@app.get("/scan/custom", response_model=ScanResponse)
async def scan_custom(
    symbols: str = Query(..., description='Comma-separated list of stock symbols'),
    period: str = Query('3mo')
):
    start_time = time.time()

    symbol_list = [s.strip().upper() for s in symbols.split(',')]
    symbol_list = [s if '.' in s else s + '.NS' for s in symbol_list]
    symbol_list = symbol_list[:20]

    stock_data = batch_download_stocks(symbol_list, period)

    if not stock_data:
        raise HTTPException(status_code=503, detail="Could not fetch data. Try again later.")

    breakouts = []
    failed_stocks = []

    for symbol, data in stock_data.items():
        try:
            result = detect_breakout(data, symbol)

            stock_breakout = StockBreakout(
                symbol=symbol.replace('.NS', '').replace('.BO', ''),
                current_price=result['current_price'],
                previous_close=result['previous_close'],
                change_percent=round(
                    ((result['current_price'] - result['previous_close']) / result['previous_close']) * 100, 2
                ) if result['previous_close'] > 0 else 0,
                resistance_levels=ResistanceLevels(
                    pivot=result['resistance_levels']['pivot'],
                    r1=result['resistance_levels']['r1'],
                    r2=result['resistance_levels']['r2'],
                    r3=result['resistance_levels']['r3']
                ),
                breakouts=[BreakoutSignal(**b) for b in result['breakouts']],
                is_breakout=result['is_breakout'],
                volume_confirmation=result['volume_confirmation'],
                volume_ratio=result.get('volume_ratio', 0.0),
                date_used=result.get('date_used', ''),
                data_source="yahoo" if symbol.endswith('.NS') else "alphavantage"
            )

            if result['is_breakout']:
                breakouts.append(stock_breakout)
        except Exception as e:
            failed_stocks.append(f"{symbol}: {str(e)[:50]}")
            continue

    not_downloaded = [s for s in symbol_list if s not in stock_data]
    failed_stocks.extend(not_downloaded)

    breakouts.sort(key=lambda x: (len(x.breakouts), x.volume_ratio), reverse=True)

    scan_duration = round(time.time() - start_time, 2)

    return ScanResponse(
        scan_time=datetime.now().isoformat(),
        total_stocks=len(symbol_list),
        breakout_count=len(breakouts),
        breakouts=breakouts,
        failed_stocks=failed_stocks,
        scan_duration_seconds=scan_duration
    )

# For Vercel
app = app