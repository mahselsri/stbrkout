from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
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
    """Fetch from Yahoo Finance with price validation"""
    try:
        print(f"Yahoo Finance: {symbol}")
        
        # Add delay
        time.sleep(0.3)
        
        ticker = yf.Ticker(symbol)
        
        # Get info first to check price
        info = ticker.info
        if info:
            currency = info.get('currency', '')
            market = info.get('market', '')
            print(f"  Currency: {currency}, Market: {market}")
        
        # Try to get history
        data = None
        try:
            data = ticker.history(period=period)
        except:
            pass
        
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
        
        if data is None or data.empty:
            try:
                data = ticker.history(period='1mo')
            except:
                pass
        
        if data is not None and not data.empty:
            # Check if price is in INR (should be < 50000 for Indian stocks)
            latest_price = float(data['Close'].iloc[-1])
            
            # If price seems like USD (too high for INR), try to get from info
            if latest_price > 10000 and 'NS' in symbol:
                print(f"  Price {latest_price} seems high, checking info...")
                if info and 'regularMarketPrice' in info:
                    real_price = info.get('regularMarketPrice', 0)
                    if 100 < real_price < 10000:
                        print(f"  Using info price: ₹{real_price}")
                        # Create dataframe with correct price
                        data = pd.DataFrame({
                            'Open': [info.get('regularMarketOpen', real_price)],
                            'High': [info.get('regularMarketDayHigh', real_price)],
                            'Low': [info.get('regularMarketDayLow', real_price)],
                            'Close': [real_price],
                            'Volume': [info.get('regularMarketVolume', 0)]
                        }, index=[pd.Timestamp.now()])
            
            print(f"  Yahoo success: {len(data)} rows, latest: ₹{data['Close'].iloc[-1]}")
            return data, 'yahoo'
        
        return None, None
        
    except Exception as e:
        print(f"  Yahoo error: {str(e)}")
        return None, None

# --- Data Source 2: Alpha Vantage ---
def fetch_from_alphavantage(symbol: str):
    """Fetch from Alpha Vantage with INR conversion"""
    try:
        api_key = os.environ.get('ALPHA_VANTAGE_KEY', 'demo')
        if api_key == 'demo':
            print("  Alpha Vantage: No API key")
            return None, None
            
        print(f"Alpha Vantage: {symbol}")
        
        # Clean symbol for Alpha Vantage
        av_symbol = symbol.replace('.NS', '').replace('.BO', '')
        
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
            for date, values in list(time_series.items())[:90]:
                close = float(values['4. close'])
                
                # Alpha Vantage returns USD for Indian stocks
                # Convert to INR (approximate rate ~83 INR/USD)
                # Check if price seems like USD (should be < 1000)
                if close < 1000:
                    close_inr = close * 83  # Convert to INR
                    print(f"  Converted USD ${close} to ₹{close_inr}")
                else:
                    close_inr = close  # Already in INR
                
                df_data.append({
                    'Date': pd.to_datetime(date),
                    'Open': float(values['1. open']) * 83 if float(values['1. open']) < 1000 else float(values['1. open']),
                    'High': float(values['2. high']) * 83 if float(values['2. high']) < 1000 else float(values['2. high']),
                    'Low': float(values['3. low']) * 83 if float(values['3. low']) < 1000 else float(values['3. low']),
                    'Close': close_inr,
                    'Volume': float(values['5. volume'])
                })
            
            df = pd.DataFrame(df_data)
            df.set_index('Date', inplace=True)
            df.sort_index(inplace=True)
            
            print(f"  Alpha Vantage success: {len(df)} rows")
            return df, 'alphavantage'
            
        elif "Note" in data:
            print(f"  Alpha Vantage rate limit: {data['Note']}")
            return None, None
        else:
            print(f"  Alpha Vantage error: {data}")
            return None, None
            
    except Exception as e:
        print(f"  Alpha Vantage error: {str(e)}")
        return None, None

# --- Data Source 3: Twelve Data ---
def fetch_from_twelvedata(symbol: str):
    """Fetch from Twelve Data with INR conversion"""
    try:
        api_key = os.environ.get('TWELVE_DATA_KEY', '')
        if not api_key:
            print("  Twelve Data: No API key")
            return None, None
            
        print(f"Twelve Data: {symbol}")
        
        td_symbol = symbol.replace('.NS', '').replace('.BO', '')
        
        url = "https://api.twelvedata.com/time_series"
        params = {
            "symbol": f"NSE:{td_symbol}",
            "interval": "1day",
            "outputsize": "90",
            "apikey": api_key
        }
        
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        
        if "values" in data and data["values"]:
            values = data["values"]
            
            df_data = []
            for item in values[:90]:
                close = float(item['close'])
                
                # Check if price seems like USD (should be < 1000 for Indian stocks in USD)
                if close < 1000:
                    close_inr = close * 83  # Convert to INR
                else:
                    close_inr = close
                
                df_data.append({
                    'Date': pd.to_datetime(item['datetime']),
                    'Open': float(item['open']) * 83 if float(item['open']) < 1000 else float(item['open']),
                    'High': float(item['high']) * 83 if float(item['high']) < 1000 else float(item['high']),
                    'Low': float(item['low']) * 83 if float(item['low']) < 1000 else float(item['low']),
                    'Close': close_inr,
                    'Volume': float(item['volume'])
                })
            
            df = pd.DataFrame(df_data)
            df.set_index('Date', inplace=True)
            df.sort_index(inplace=True)
            
            print(f"  Twelve Data success: {len(df)} rows")
            return df, 'twelvedata'
            
        else:
            print(f"  Twelve Data error: {data.get('status', 'unknown')}")
            return None, None
            
    except Exception as e:
        print(f"  Twelve Data error: {str(e)}")
        return None, None

# --- Main Fetch Function ---
def fetch_stock_data(symbol: str, period: str = '3mo'):
    """Fetch stock data from multiple sources with price validation"""
    
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
        print(f"\nTrying: {try_symbol}")
        
        # Try Yahoo Finance
        data, source = fetch_from_yahoo(try_symbol, period)
        if data is not None and not data.empty:
            print(f"✓ Using {source}")
            set_in_cache(cache_key, data)
            return data
        
        # Try Alpha Vantage
        data, source = fetch_from_alphavantage(try_symbol)
        if data is not None and not data.empty:
            print(f"✓ Using {source}")
            set_in_cache(cache_key, data)
            return data
        
        # Try Twelve Data
        data, source = fetch_from_twelvedata(try_symbol)
        if data is not None and not data.empty:
            print(f"✓ Using {source}")
            set_in_cache(cache_key, data)
            return data
    
    # If all fail
    raise HTTPException(
        status_code=404,
        detail=f"Could not fetch data for {symbol}. Please try again later."
    )

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
        
        # Volume confirmation
        volume_confirmation = False
        if len(data) >= 14:
            try:
                avg_volume = float(data['Volume'].rolling(window=14).mean().iloc[-1])
                current_volume = float(data['Volume'].iloc[-1])
                volume_confirmation = current_volume > avg_volume * 1.5 if avg_volume > 0 else False
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
            'volume_confirmation': volume_confirmation
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")

@app.get("/")
async def root():
    return {
        "message": "Stock Breakout Detection API",
        "status": "running",
        "data_sources": ["Yahoo Finance", "Alpha Vantage", "Twelve Data"],
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "cache_size": len(cache),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/debug/{symbol}")
async def debug_stock(symbol: str):
    """Debug endpoint to check data sources"""
    try:
        symbol = symbol.strip().upper()
        if '.' not in symbol:
            symbol = symbol + '.NS'
        
        results = {}
        
        # Test Yahoo
        data, source = fetch_from_yahoo(symbol)
        if data is not None and not data.empty:
            results['yahoo'] = {
                'status': 'success',
                'latest_price': float(data['Close'].iloc[-1]),
                'data_points': len(data)
            }
        else:
            results['yahoo'] = {'status': 'failed'}
        
        # Test Alpha Vantage
        data, source = fetch_from_alphavantage(symbol)
        if data is not None and not data.empty:
            results['alphavantage'] = {
                'status': 'success',
                'latest_price': float(data['Close'].iloc[-1]),
                'data_points': len(data)
            }
        else:
            results['alphavantage'] = {'status': 'failed'}
        
        # Test Twelve Data
        data, source = fetch_from_twelvedata(symbol)
        if data is not None and not data.empty:
            results['twelvedata'] = {
                'status': 'success',
                'latest_price': float(data['Close'].iloc[-1]),
                'data_points': len(data)
            }
        else:
            results['twelvedata'] = {'status': 'failed'}
        
        return {
            "symbol": symbol,
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
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
            timestamp=datetime.now().isoformat()
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
                        timestamp=datetime.now().isoformat()
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

app = app