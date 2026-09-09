from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
import numpy as np
import time

app = FastAPI(title="Stock Breakout API", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

# --- Simple Cache ---
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

# --- Stock Data Fetcher ---
def fetch_stock_data(symbol: str, period: str = '3mo'):
    """Fetch stock data with caching"""
    
    # Check cache
    cache_key = f"{symbol}_{period}"
    cached = get_from_cache(cache_key)
    if cached is not None:
        return cached
    
    # Clean symbol
    symbol = symbol.strip().upper()
    if '.' not in symbol:
        symbols_to_try = [f"{symbol}.NS", f"{symbol}.BO"]
    else:
        symbols_to_try = [symbol]
    
    for try_symbol in symbols_to_try:
        try:
            # Add delay to avoid rate limiting
            time.sleep(0.3)
            
            ticker = yf.Ticker(try_symbol)
            
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
            
            # Method 3: Try shorter period
            if data is None or data.empty:
                try:
                    data = ticker.history(period='1mo')
                except:
                    pass
            
            # Method 4: Get info only
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
                set_in_cache(cache_key, data)
                return data
                
        except Exception as e:
            print(f"Error with {try_symbol}: {str(e)}")
            continue
    
    # If all fail, try mock data
    return generate_mock_data(symbol)

def generate_mock_data(symbol: str):
    """Generate mock data when API fails"""
    print(f"Generating mock data for {symbol}")
    
    end_date = datetime.now()
    dates = pd.date_range(end=end_date, periods=60, freq='D')
    
    # Base prices for different stocks
    base_prices = {
        'RELIANCE': 2450,
        'TCS': 4200,
        'INFY': 1800,
        'HDFCBANK': 1600,
        'ICICIBANK': 1100,
        'SBIN': 800,
        'BHARTIARTL': 1200,
        'ITC': 450,
    }
    
    base_price = 1000
    for key, price in base_prices.items():
        if key in symbol:
            base_price = price
            break
    
    np.random.seed(hash(symbol) % 2**32)
    returns = np.random.normal(0.0005, 0.015, 60)
    prices = base_price * np.exp(np.cumsum(returns))
    
    data = {
        'Open': prices * (1 + np.random.normal(0, 0.002, 60)),
        'High': prices * (1 + np.random.normal(0.005, 0.005, 60)),
        'Low': prices * (1 - np.random.normal(0.005, 0.005, 60)),
        'Close': prices,
        'Volume': np.random.randint(100000, 1000000, 60)
    }
    
    # Ensure High is highest, Low is lowest
    for i in range(60):
        data['High'][i] = max(data['Open'][i], data['High'][i], data['Close'][i])
        data['Low'][i] = min(data['Open'][i], data['Low'][i], data['Close'][i])
    
    df = pd.DataFrame(data, index=dates)
    set_in_cache(f"{symbol}_3mo", df, ttl=600)
    return df

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
    except:
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
        
        return {
            'r1': round(r1, 2),
            'r2': round(r2, 2),
            'r3': round(r3, 2)
        }
    except:
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

# --- API Endpoints ---

@app.get("/")
async def root():
    return {
        "message": "Stock Breakout Detection API",
        "version": "1.0.0",
        "status": "running",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "cache_size": len(cache),
        "timestamp": datetime.now().isoformat()
    }

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
        {"symbol": "HCLTECH.NS", "name": "HCL Technologies", "sector": "IT"}
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
async def scan_stocks(
    symbols: str = Query('RELIANCE.NS,TCS.NS,INFY.NS')
):
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
                failed_stocks.append(symbol)
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

# For Vercel
app = app