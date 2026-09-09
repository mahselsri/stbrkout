from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
import numpy as np
import time
import json
import requests
import os
from functools import lru_cache

app = FastAPI(title="Stock Breakout API", version="2.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Cache System ---
cache = {}
cache_expiry = {}

def get_cache_key(symbol: str, period: str) -> str:
    return f"{symbol}_{period}"

def get_from_cache(key: str, max_age_seconds: int = 300):
    if key in cache and key in cache_expiry:
        if datetime.now() < cache_expiry[key]:
            return cache[key]
        else:
            del cache[key]
            del cache_expiry[key]
    return None

def set_in_cache(key: str, data, ttl_seconds: int = 300):
    cache[key] = data
    cache_expiry[key] = datetime.now() + timedelta(seconds=ttl_seconds)

# --- Rate Limiting ---
request_timestamps = []

def check_rate_limit(max_requests: int = 3, time_window: int = 60):
    now = datetime.now()
    request_timestamps[:] = [ts for ts in request_timestamps if now - ts < timedelta(seconds=time_window)]
    if len(request_timestamps) >= max_requests:
        return False
    request_timestamps.append(now)
    return True

# --- Data Source 1: Yahoo Finance (with retry) ---
def fetch_from_yahoo(symbol: str, period: str = '3mo'):
    """Fetch from Yahoo Finance with retry logic"""
    try:
        print(f"Attempting Yahoo Finance: {symbol}")
        
        # Add small delay
        time.sleep(0.3)
        
        ticker = yf.Ticker(symbol)
        
        # Try different methods
        data = None
        
        # Method 1: With period
        try:
            data = ticker.history(period=period)
        except:
            pass
        
        # Method 2: With dates
        if data is None or data.empty:
            try:
                end_date = datetime.now()
                start_date = end_date - timedelta(days=90)
                data = ticker.history(
                    start=start_date.strftime('%Y-%m-%d'),
                    end=end_date.strftime('%Y-%m-%d')
                )
            except:
                pass
        
        # Method 3: Only info
        if data is None or data.empty:
            try:
                info = ticker.info
                if info and 'regularMarketPrice' in info:
                    current_price = info.get('regularMarketPrice', 0)
                    data = pd.DataFrame({
                        'Open': [info.get('regularMarketOpen', current_price)],
                        'High': [info.get('regularMarketDayHigh', current_price)],
                        'Low': [info.get('regularMarketDayLow', current_price)],
                        'Close': [current_price],
                        'Volume': [info.get('regularMarketVolume', 0)]
                    }, index=[pd.Timestamp.now()])
            except:
                pass
        
        if data is not None and not data.empty:
            print(f"Yahoo Finance success: {len(data)} rows")
            return data
            
        return None
        
    except Exception as e:
        print(f"Yahoo Finance failed: {str(e)}")
        return None

# --- Data Source 2: Alpha Vantage (FREE API - Need API Key) ---
ALPHA_VANTAGE_KEY = os.environ.get('ALPHA_VANTAGE_KEY', 'demo')

def fetch_from_alphavantage(symbol: str):
    """Fetch from Alpha Vantage API (Free tier: 5 requests/min)"""
    try:
        print(f"Attempting Alpha Vantage: {symbol}")
        
        # Clean symbol for Alpha Vantage
        av_symbol = symbol.replace('.NS', '').replace('.BO', '')
        
        url = f"https://www.alphavantage.co/query"
        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": av_symbol,
            "apikey": ALPHA_VANTAGE_KEY,
            "outputsize": "compact"
        }
        
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        
        if "Time Series (Daily)" in data:
            time_series = data["Time Series (Daily)"]
            
            # Convert to DataFrame
            df_data = []
            for date, values in list(time_series.items())[:90]:  # Last 90 days
                df_data.append({
                    'Date': pd.to_datetime(date),
                    'Open': float(values['1. open']),
                    'High': float(values['2. high']),
                    'Low': float(values['3. low']),
                    'Close': float(values['4. close']),
                    'Volume': float(values['5. volume'])
                })
            
            df = pd.DataFrame(df_data)
            df.set_index('Date', inplace=True)
            df.sort_index(inplace=True)
            
            print(f"Alpha Vantage success: {len(df)} rows")
            return df
            
        elif "Note" in data:
            print(f"Alpha Vantage rate limit: {data['Note']}")
            return None
        else:
            print(f"Alpha Vantage error: {data}")
            return None
            
    except Exception as e:
        print(f"Alpha Vantage failed: {str(e)}")
        return None

# --- Data Source 3: Twelve Data (Free API) ---
TWELVE_DATA_KEY = os.environ.get('TWELVE_DATA_KEY', '')

def fetch_from_twelvedata(symbol: str):
    """Fetch from Twelve Data API (Free tier: 800 requests/day)"""
    try:
        if not TWELVE_DATA_KEY:
            print("Twelve Data API key not configured")
            return None
            
        print(f"Attempting Twelve Data: {symbol}")
        
        # Clean symbol
        td_symbol = symbol.replace('.NS', '').replace('.BO', '')
        
        url = f"https://api.twelvedata.com/time_series"
        params = {
            "symbol": f"NSE:{td_symbol}",
            "interval": "1day",
            "outputsize": "90",
            "apikey": TWELVE_DATA_KEY
        }
        
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        
        if "values" in data and data["values"]:
            values = data["values"]
            
            df_data = []
            for item in values[:90]:
                df_data.append({
                    'Date': pd.to_datetime(item['datetime']),
                    'Open': float(item['open']),
                    'High': float(item['high']),
                    'Low': float(item['low']),
                    'Close': float(item['close']),
                    'Volume': float(item['volume'])
                })
            
            df = pd.DataFrame(df_data)
            df.set_index('Date', inplace=True)
            df.sort_index(inplace=True)
            
            print(f"Twelve Data success: {len(df)} rows")
            return df
            
        else:
            print(f"Twelve Data error: {data.get('status', 'unknown')}")
            return None
            
    except Exception as e:
        print(f"Twelve Data failed: {str(e)}")
        return None

# --- Data Source 4: Mock/Simulated Data (Fallback) ---
def generate_mock_data(symbol: str, days: int = 90):
    """Generate simulated data when all APIs fail"""
    print(f"Generating mock data for {symbol}")
    
    end_date = datetime.now()
    dates = pd.date_range(end=end_date, periods=days, freq='D')
    
    # Start with a base price
    if 'RELIANCE' in symbol:
        base_price = 2450
    elif 'TCS' in symbol:
        base_price = 4200
    elif 'INFY' in symbol:
        base_price = 1800
    elif 'HDFCBANK' in symbol:
        base_price = 1600
    else:
        base_price = 1000
    
    np.random.seed(hash(symbol) % 2**32)
    
    # Generate random walk
    returns = np.random.normal(0.0005, 0.015, days)
    prices = base_price * np.exp(np.cumsum(returns))
    
    # Create OHLC data
    data = {
        'Open': prices * (1 + np.random.normal(0, 0.002, days)),
        'High': prices * (1 + np.random.normal(0.005, 0.005, days)),
        'Low': prices * (1 - np.random.normal(0.005, 0.005, days)),
        'Close': prices,
        'Volume': np.random.randint(100000, 1000000, days)
    }
    
    # Ensure High is always highest and Low is always lowest
    for i in range(days):
        data['High'][i] = max(data['Open'][i], data['High'][i], data['Close'][i])
        data['Low'][i] = min(data['Open'][i], data['Low'][i], data['Close'][i])
    
    df = pd.DataFrame(data, index=dates)
    print(f"Mock data generated: {len(df)} rows")
    return df

# --- Main Fetch Function with Fallbacks ---
def fetch_stock_data(symbol: str, period: str = '3mo'):
    """Fetch stock data with multiple fallback sources"""
    
    # Check cache first
    cache_key = get_cache_key(symbol, period)
    cached_data = get_from_cache(cache_key)
    if cached_data is not None:
        print(f"Using cached data for {symbol}")
        return cached_data
    
    # Check rate limit
    if not check_rate_limit(max_requests=2, time_window=30):
        # If rate limited, try mock data
        print("Rate limited, using mock data")
        data = generate_mock_data(symbol)
        set_in_cache(cache_key, data, ttl_seconds=600)  # 10 minutes cache for mock
        return data
    
    # Clean symbol
    symbol = symbol.strip().upper()
    if '.' not in symbol:
        symbols_to_try = [f"{symbol}.NS", f"{symbol}.BO"]
    else:
        symbols_to_try = [symbol]
    
    for try_symbol in symbols_to_try:
        # Try Yahoo Finance
        data = fetch_from_yahoo(try_symbol, period)
        if data is not None and not data.empty:
            set_in_cache(cache_key, data)
            return data
        
        # Try Alpha Vantage
        if ALPHA_VANTAGE_KEY and ALPHA_VANTAGE_KEY != 'demo':
            data = fetch_from_alphavantage(try_symbol)
            if data is not None and not data.empty:
                set_in_cache(cache_key, data)
                return data
        
        # Try Twelve Data
        if TWELVE_DATA_KEY:
            data = fetch_from_twelvedata(try_symbol)
            if data is not None and not data.empty:
                set_in_cache(cache_key, data)
                return data
    
    # If all APIs fail, use mock data
    print("All data sources failed, using mock data")
    data = generate_mock_data(symbol)
    set_in_cache(cache_key, data, ttl_seconds=600)
    return data

# --- Analysis Functions ---
def calculate_pivot_points(data):
    """Calculate pivot points and resistance levels"""
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
    """Calculate dynamic resistance"""
    try:
        if len(data) < lookback:
            lookback = max(len(data) // 2, 5)
        
        if len(data) < 3:
            return {'r1': 0, 'r2': 0, 'r3': 0}
            
        rolling_high = data['High'].rolling(window=lookback).max()
        rolling_mean = data['High'].rolling(window=lookback).mean()
        rolling_std = data['High'].rolling(window=lookback).std()
        
        r1 = float(rolling_mean.iloc[-1] + rolling_std.iloc[-1]) if not pd.isna(rolling_mean.iloc[-1]) else 0
        r2 = float(rolling_mean.iloc[-1] + 1.5 * rolling_std.iloc[-1]) if not pd.isna(rolling_mean.iloc[-1]) else 0
        r3 = float(rolling_mean.iloc[-1] + 2 * rolling_std.iloc[-1]) if not pd.isna(rolling_mean.iloc[-1]) else 0
        
        return {
            'r1': round(r1, 2),
            'r2': round(r2, 2),
            'r3': round(r3, 2)
        }
    except:
        return {'r1': 0, 'r2': 0, 'r3': 0}

def detect_breakout(data, symbol):
    """Detect breakout patterns"""
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
        
        # Volume confirmation
        volume_confirmation = False
        if len(data) >= 14:
            try:
                avg_volume = float(data['Volume'].rolling(window=14).mean().iloc[-1])
                current_volume = float(data['Volume'].iloc[-1])
                volume_confirmation = current_volume > avg_volume * 1.5 if avg_volume > 0 else False
            except:
                volume_confirmation = False
        
        # Check breakouts
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
            'data_source': 'cached' if get_from_cache(get_cache_key(symbol, '3mo')) is not None else 'api'
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")

# --- Models ---
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
    data_source: str = "api"

# --- API Endpoints ---

@app.get("/")
async def root():
    return {
        "message": "Stock Breakout Detection API v2",
        "status": "running",
        "data_sources": ["Yahoo Finance", "Alpha Vantage", "Twelve Data", "Mock Data (Fallback)"],
        "cache": "5 minutes TTL (10 min for mock data)",
        "rate_limit": "2 requests per 30 seconds",
        "endpoints": {
            "/popular": "Get popular Indian stocks",
            "/analyze/{symbol}": "Analyze a single stock",
            "/scan": "Scan multiple stocks (max 3)",
            "/health": "Health check",
            "/cache/clear": "Clear cache"
        },
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "cache_size": len(cache),
        "data_sources_available": {
            "yahoo": "limited (rate limited)",
            "alphavantage": bool(ALPHA_VANTAGE_KEY and ALPHA_VANTAGE_KEY != 'demo'),
            "twelvedata": bool(TWELVE_DATA_KEY),
            "mock": True
        },
        "timestamp": datetime.now().isoformat()
    }

@app.get("/cache/clear")
async def clear_cache():
    cache.clear()
    cache_expiry.clear()
    return {"message": "Cache cleared", "timestamp": datetime.now().isoformat()}

@app.get("/popular")
async def get_popular_indian_stocks():
    stocks = [
        {"symbol": "RELIANCE.NS", "name": "Reliance Industries", "sector": "Oil & Gas"},
        {"symbol": "TCS.NS", "name": "Tata Consultancy Services", "sector": "IT"},
        {"symbol": "INFY.NS", "name": "Infosys", "sector": "IT"},
        {"symbol": "HDFCBANK.NS", "name": "HDFC Bank", "sector": "Banking"},
        {"symbol": "ICICIBANK.NS", "name": "ICICI Bank", "sector": "Banking"},
        {"symbol": "SBIN.NS", "name": "State Bank of India", "sector": "Banking"},
        {"symbol": "BHARTIARTL.NS", "name": "Bharti Airtel", "sector": "Telecom"},
        {"symbol": "ITC.NS", "name": "ITC Ltd", "sector": "FMCG"},
        {"symbol": "WIPRO.NS", "name": "Wipro", "sector": "IT"},
        {"symbol": "HCLTECH.NS", "name": "HCL Technologies", "sector": "IT"},
        {"symbol": "ASIANPAINT.NS", "name": "Asian Paints", "sector": "Paint"},
        {"symbol": "MARUTI.NS", "name": "Maruti Suzuki", "sector": "Automobile"},
        {"symbol": "TATAMOTORS.NS", "name": "Tata Motors", "sector": "Automobile"},
        {"symbol": "TITAN.NS", "name": "Titan Company", "sector": "Retail"},
        {"symbol": "AXISBANK.NS", "name": "Axis Bank", "sector": "Banking"},
        {"symbol": "KOTAKBANK.NS", "name": "Kotak Mahindra Bank", "sector": "Banking"},
        {"symbol": "LT.NS", "name": "Larsen & Toubro", "sector": "Construction"},
        {"symbol": "HINDUNILVR.NS", "name": "Hindustan Unilever", "sector": "FMCG"},
        {"symbol": "SUNPHARMA.NS", "name": "Sun Pharma", "sector": "Pharma"},
        {"symbol": "BAJFINANCE.NS", "name": "Bajaj Finance", "sector": "Finance"}
    ]
    return {
        "stocks": stocks,
        "total": len(stocks),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/analyze/{symbol}")
async def analyze_stock(
    symbol: str,
    period: str = Query('3mo', description='Data period: 1d,5d,1mo,3mo,6mo,1y')
):
    """Analyze a single stock for breakout patterns"""
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
            data_source=result.get('data_source', 'api')
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/scan")
async def scan_stocks(
    symbols: str = Query(
        'RELIANCE.NS,TCS.NS,INFY.NS',
        description='Comma-separated list of stock symbols (max 3)'
    )
):
    """Scan multiple stocks for breakouts (max 3 to avoid rate limiting)"""
    try:
        stock_list = [s.strip().upper() for s in symbols.split(',')]
        stock_list = [s if '.' in s else s + '.NS' for s in stock_list]
        # Limit to 3 stocks
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
                        data_source=result.get('data_source', 'api')
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

# For Vercel
app = app