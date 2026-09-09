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
import pytz

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
    date_used: str = ""

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

# --- Data Source 1: Yahoo Finance (with correct date handling) ---
def fetch_from_yahoo(symbol: str, period: str = '3mo'):
    """Fetch from Yahoo Finance with correct date handling"""
    try:
        print(f"📊 Yahoo Finance: {symbol}")
        
        ticker = yf.Ticker(symbol)
        
        # Get info first
        info = ticker.info
        currency = info.get('currency', 'INR') if info else 'INR'
        print(f"  Currency: {currency}")
        
        # Get current date for reference
        today = datetime.now()
        print(f"  Today's date: {today.strftime('%Y-%m-%d')}")
        
        # Try different date formats and periods
        data = None
        
        # Method 1: Standard period
        try:
            print(f"  Fetching with period: {period}")
            data = ticker.history(period=period)
            if not data.empty:
                print(f"  Data range: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
                print(f"  Latest price: ₹{data['Close'].iloc[-1]}")
        except Exception as e:
            print(f"  Period fetch error: {e}")
        
        # Method 2: With explicit dates (YYYY-MM-DD format)
        if data is None or data.empty:
            try:
                end_date = today.strftime('%Y-%m-%d')
                start_date = (today - timedelta(days=90)).strftime('%Y-%m-%d')
                print(f"  Fetching with dates: {start_date} to {end_date}")
                data = ticker.history(start=start_date, end=end_date)
                if not data.empty:
                    print(f"  Data range: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
                    print(f"  Latest price: ₹{data['Close'].iloc[-1]}")
            except Exception as e:
                print(f"  Date fetch error: {e}")
        
        # Method 3: Try with different date format (DD-MM-YYYY)
        if data is None or data.empty:
            try:
                end_date = today.strftime('%d-%m-%Y')
                start_date = (today - timedelta(days=90)).strftime('%d-%m-%Y')
                print(f"  Fetching with DD-MM-YYYY: {start_date} to {end_date}")
                data = ticker.history(start=start_date, end=end_date)
                if not data.empty:
                    print(f"  Data range: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
                    print(f"  Latest price: ₹{data['Close'].iloc[-1]}")
            except Exception as e:
                print(f"  DD-MM-YYYY fetch error: {e}")
        
        # Method 4: Try with timestamp
        if data is None or data.empty:
            try:
                end_date = int(today.timestamp())
                start_date = int((today - timedelta(days=90)).timestamp())
                print(f"  Fetching with timestamps: {start_date} to {end_date}")
                data = ticker.history(start=start_date, end=end_date)
                if not data.empty:
                    print(f"  Data range: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
                    print(f"  Latest price: ₹{data['Close'].iloc[-1]}")
            except Exception as e:
                print(f"  Timestamp fetch error: {e}")
        
        # Method 5: Just get 1 month if 3 months fails
        if data is None or data.empty:
            try:
                print(f"  Fetching with period: 1mo")
                data = ticker.history(period='1mo')
                if not data.empty:
                    print(f"  Data range: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
                    print(f"  Latest price: ₹{data['Close'].iloc[-1]}")
            except Exception as e:
                print(f"  1mo fetch error: {e}")
        
        if data is not None and not data.empty:
            # Get the latest price
            latest_price = float(data['Close'].iloc[-1])
            latest_date = data.index[-1]
            print(f"  ✅ Latest: ₹{latest_price} on {latest_date.strftime('%Y-%m-%d')}")
            
            # Check if price is in USD and convert
            if currency == 'USD' and 1 < latest_price < 1000:
                inr_price = latest_price * 83.5
                print(f"  Converting USD ${latest_price} to INR ₹{inr_price}")
                data['Open'] = data['Open'] * 83.5
                data['High'] = data['High'] * 83.5
                data['Low'] = data['Low'] * 83.5
                data['Close'] = data['Close'] * 83.5
            
            return data, 'yahoo'
        
        print(f"  ❌ No data found")
        return None, None
        
    except Exception as e:
        print(f"  Yahoo error: {str(e)}")
        return None, None

# --- Data Source 2: Info API (Current price only) ---
def fetch_from_info(symbol: str):
    """Fetch current price from info only"""
    try:
        print(f"📊 Info API: {symbol}")
        
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        if info and 'regularMarketPrice' in info:
            current_price = info.get('regularMarketPrice', 0)
            previous_close = info.get('previousClose', current_price)
            currency = info.get('currency', 'INR')
            market_time = info.get('regularMarketTime', datetime.now())
            
            print(f"  Market price: {current_price} ({currency})")
            print(f"  Market time: {market_time}")
            
            # Convert if USD
            if currency == 'USD' and current_price < 1000:
                current_price = current_price * 83.5
                previous_close = previous_close * 83.5
                print(f"  Converted to INR: ₹{current_price}")
            
            # Create dataframe with proper date
            data = pd.DataFrame({
                'Open': [current_price * 0.995],
                'High': [current_price * 1.005],
                'Low': [current_price * 0.995],
                'Close': [current_price],
                'Volume': [info.get('regularMarketVolume', 0)]
            }, index=[pd.Timestamp.now()])
            
            print(f"  ✅ Latest: ₹{current_price}")
            return data, 'info'
        
        return None, None
        
    except Exception as e:
        print(f"  Info error: {str(e)}")
        return None, None

# --- Data Source 3: Alpha Vantage ---
def fetch_from_alphavantage(symbol: str):
    """Fetch from Alpha Vantage with correct date handling"""
    try:
        api_key = os.environ.get('ALPHA_VANTAGE_KEY', '')
        if not api_key:
            print("  Alpha Vantage: No API key")
            return None, None
            
        print(f"📊 Alpha Vantage: {symbol}")
        
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
            print(f"  Alpha Vantage returned {len(time_series)} days of data")
            
            df_data = []
            for date_str, values in list(time_series.items())[:90]:
                try:
                    # Parse date correctly
                    date = pd.to_datetime(date_str)
                    
                    close = float(values['4. close'])
                    open_price = float(values['1. open'])
                    high = float(values['2. high'])
                    low = float(values['3. low'])
                    
                    # Convert if USD
                    if 1 < close < 1000:
                        close = close * 83.5
                        open_price = open_price * 83.5
                        high = high * 83.5
                        low = low * 83.5
                    
                    df_data.append({
                        'Date': date,
                        'Open': open_price,
                        'High': high,
                        'Low': low,
                        'Close': close,
                        'Volume': float(values['5. volume'])
                    })
                except Exception as e:
                    print(f"  Error parsing {date_str}: {e}")
                    continue
            
            if df_data:
                df = pd.DataFrame(df_data)
                df.set_index('Date', inplace=True)
                df.sort_index(inplace=True)
                
                latest_date = df.index[-1]
                latest_price = float(df['Close'].iloc[-1])
                print(f"  ✅ Alpha Vantage: {len(df)} rows, latest: ₹{latest_price} on {latest_date.strftime('%Y-%m-%d')}")
                return df, 'alphavantage'
        
        return None, None
            
    except Exception as e:
        print(f"  Alpha Vantage error: {str(e)}")
        return None, None

# --- Main Fetch Function ---
def fetch_stock_data(symbol: str, period: str = '3mo'):
    """Fetch stock data with correct date handling"""
    
    cache_key = f"{symbol}_{period}"
    cached = get_from_cache(cache_key)
    if cached is not None:
        return cached
    
    symbol = symbol.strip().upper()
    if '.' not in symbol:
        symbols_to_try = [f"{symbol}.NS", f"{symbol}.BO"]
    else:
        symbols_to_try = [symbol]
    
    print(f"\n🔍 Fetching data for {symbol}")
    print(f"   Period: {period}")
    print(f"   Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    for try_symbol in symbols_to_try:
        print(f"\n📈 Trying: {try_symbol}")
        
        # Try Yahoo Finance first
        data, source = fetch_from_yahoo(try_symbol, period)
        if data is not None and not data.empty:
            latest = float(data['Close'].iloc[-1])
            latest_date = data.index[-1]
            if 100 < latest < 50000:
                print(f"✅ Using {source} with price ₹{latest} on {latest_date.strftime('%Y-%m-%d')}")
                set_in_cache(cache_key, data)
                return data
            else:
                print(f"⚠️ Price ₹{latest} seems wrong, trying next source")
        
        # Try Info API
        data, source = fetch_from_info(try_symbol)
        if data is not None and not data.empty:
            latest = float(data['Close'].iloc[-1])
            if 100 < latest < 50000:
                print(f"✅ Using {source} with price ₹{latest}")
                set_in_cache(cache_key, data)
                return data
        
        # Try Alpha Vantage
        data, source = fetch_from_alphavantage(try_symbol)
        if data is not None and not data.empty:
            latest = float(data['Close'].iloc[-1])
            latest_date = data.index[-1]
            if 100 < latest < 50000:
                print(f"✅ Using {source} with price ₹{latest} on {latest_date.strftime('%Y-%m-%d')}")
                set_in_cache(cache_key, data)
                return data
            else:
                print(f"⚠️ Price ₹{latest} seems wrong")
    
    # If all fail
    raise HTTPException(
        status_code=404,
        detail=f"Could not fetch valid data for {symbol}. Please try again later."
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
        latest_date = data.index[-1].strftime('%Y-%m-%d')
        
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
            'volume_confirmation': volume_confirmation,
            'date_used': latest_date
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")

@app.get("/")
async def root():
    return {
        "message": "Stock Breakout Detection API",
        "status": "running",
        "data_sources": ["Yahoo Finance", "Info API", "Alpha Vantage"],
        "current_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
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
    """Debug endpoint to check all data sources and dates"""
    try:
        symbol = symbol.strip().upper()
        if '.' not in symbol:
            symbol = symbol + '.NS'
        
        results = {}
        
        print(f"\n🔍 Debug: {symbol}")
        print(f"   Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Test Yahoo
        data, source = fetch_from_yahoo(symbol)
        if data is not None and not data.empty:
            results['yahoo'] = {
                'status': 'success',
                'latest_price': float(data['Close'].iloc[-1]),
                'data_points': len(data),
                'currency': 'INR',
                'first_date': data.index[0].strftime('%Y-%m-%d'),
                'last_date': data.index[-1].strftime('%Y-%m-%d')
            }
        else:
            results['yahoo'] = {'status': 'failed'}
        
        # Test Info
        data, source = fetch_from_info(symbol)
        if data is not None and not data.empty:
            results['info'] = {
                'status': 'success',
                'latest_price': float(data['Close'].iloc[-1]),
                'data_points': len(data),
                'currency': 'INR'
            }
        else:
            results['info'] = {'status': 'failed'}
        
        # Test Alpha Vantage
        data, source = fetch_from_alphavantage(symbol)
        if data is not None and not data.empty:
            results['alphavantage'] = {
                'status': 'success',
                'latest_price': float(data['Close'].iloc[-1]),
                'data_points': len(data),
                'currency': 'INR',
                'first_date': data.index[0].strftime('%Y-%m-%d'),
                'last_date': data.index[-1].strftime('%Y-%m-%d')
            }
        else:
            results['alphavantage'] = {'status': 'failed'}
        
        # Actual price from Yahoo
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            info = ticker.info
            real_price = info.get('regularMarketPrice', 'N/A')
            currency = info.get('currency', 'N/A')
            market_time = info.get('regularMarketTime', 'N/A')
            results['actual'] = {
                'price': real_price,
                'currency': currency,
                'market_time': market_time
            }
        except:
            results['actual'] = {'price': 'N/A', 'currency': 'N/A'}
        
        results['server_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
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

app = app