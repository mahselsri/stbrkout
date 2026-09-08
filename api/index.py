from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime
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

# --- Improved Stock Data Fetcher ---
def fetch_stock_data(symbol: str, period: str = '6mo'):
    """Fetch stock data from Yahoo Finance with retry logic"""
    
    # Clean and validate symbol
    symbol = symbol.strip().upper()
    
    # If symbol doesn't have suffix, try both NSE and BSE
    if '.' not in symbol:
        symbols_to_try = [f"{symbol}.NS", f"{symbol}.BO"]
    else:
        symbols_to_try = [symbol]
    
    last_error = None
    
    for try_symbol in symbols_to_try:
        try:
            print(f"Trying to fetch: {try_symbol}")
            
            # Create ticker with proper user-agent
            stock = yf.Ticker(try_symbol)
            
            # Try to get info first to validate symbol
            info = stock.info
            if not info or 'regularMarketPrice' not in info:
                print(f"No market data for {try_symbol}")
                continue
            
            # Fetch historical data
            data = stock.history(period=period, progress=False)
            
            if not data.empty:
                print(f"Successfully fetched data for {try_symbol}")
                return data
            
        except Exception as e:
            last_error = str(e)
            print(f"Error with {try_symbol}: {last_error}")
            continue
    
    # If we get here, no data was found
    raise HTTPException(
        status_code=404, 
        detail=f"No data found for {symbol}. Please check the symbol. Try using format like 'RELIANCE.NS' or 'TCS.BO'"
    )

def fetch_stock_data_with_retry(symbol: str, max_retries: int = 3):
    """Fetch with retry logic"""
    for attempt in range(max_retries):
        try:
            return fetch_stock_data(symbol)
        except HTTPException as e:
            if attempt == max_retries - 1:
                raise e
            time.sleep(1)  # Wait before retry
    raise HTTPException(status_code=404, detail=f"Failed to fetch data for {symbol} after {max_retries} attempts")

# --- Analysis Functions ---
def calculate_pivot_points(data):
    """Calculate pivot points and resistance levels"""
    try:
        if len(data) < 2:
            raise ValueError("Insufficient data")
            
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
        return {'pivot': 0, 'r1': 0, 'r2': 0, 'r3': 0}

def calculate_dynamic_resistance(data, lookback=20):
    """Calculate dynamic resistance"""
    try:
        if len(data) < lookback:
            lookback = len(data) // 2
            
        rolling_high = data['High'].rolling(window=lookback).max()
        rolling_mean = data['High'].rolling(window=lookback).mean()
        rolling_std = data['High'].rolling(window=lookback).std()
        
        return {
            'r1': round(float(rolling_mean.iloc[-1] + rolling_std.iloc[-1]), 2),
            'r2': round(float(rolling_mean.iloc[-1] + 1.5 * rolling_std.iloc[-1]), 2),
            'r3': round(float(rolling_mean.iloc[-1] + 2 * rolling_std.iloc[-1]), 2)
        }
    except:
        return {'r1': 0, 'r2': 0, 'r3': 0}

def detect_breakout(data, symbol):
    """Detect breakout patterns"""
    try:
        pivot_levels = calculate_pivot_points(data)
        dynamic_levels = calculate_dynamic_resistance(data)
        
        # Use max of pivot and dynamic resistance
        resistance = {
            'pivot': pivot_levels['pivot'],
            'r1': max(pivot_levels['r1'], dynamic_levels['r1']),
            'r2': max(pivot_levels['r2'], dynamic_levels['r2']),
            'r3': max(pivot_levels['r3'], dynamic_levels['r3'])
        }
        
        current_price = round(float(data['Close'].iloc[-1]), 2)
        previous_close = round(float(data['Close'].iloc[-2]), 2)
        
        # Volume confirmation
        avg_volume = float(data['Volume'].rolling(window=20).mean().iloc[-1]) if len(data) >= 20 else float(data['Volume'].mean())
        current_volume = float(data['Volume'].iloc[-1])
        volume_confirmation = current_volume > avg_volume * 1.5 if avg_volume > 0 else False
        
        # Check breakouts
        breakouts = []
        
        if current_price > resistance['r1'] and resistance['r1'] > 0:
            breakouts.append({
                'level': 'R1',
                'value': resistance['r1'],
                'breakout_percent': round(((current_price - resistance['r1']) / resistance['r1']) * 100, 2)
            })
        
        if current_price > resistance['r2'] and resistance['r2'] > 0:
            breakouts.append({
                'level': 'R2',
                'value': resistance['r2'],
                'breakout_percent': round(((current_price - resistance['r2']) / resistance['r2']) * 100, 2)
            })
        
        if current_price > resistance['r3'] and resistance['r3'] > 0:
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
        "endpoints": {
            "/popular": "Get popular Indian stocks",
            "/analyze/{symbol}": "Analyze a single stock",
            "/scan": "Scan multiple stocks",
            "/health": "Health check"
        },
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/popular")
async def get_popular_indian_stocks():
    """Get a list of popular Indian stocks with NSE and BSE formats"""
    stocks = [
        {"symbol": "RELIANCE.NS", "name": "Reliance Industries", "sector": "Oil & Gas", "exchange": "NSE"},
        {"symbol": "TCS.NS", "name": "Tata Consultancy Services", "sector": "IT", "exchange": "NSE"},
        {"symbol": "INFY.NS", "name": "Infosys", "sector": "IT", "exchange": "NSE"},
        {"symbol": "HDFCBANK.NS", "name": "HDFC Bank", "sector": "Banking", "exchange": "NSE"},
        {"symbol": "ICICIBANK.NS", "name": "ICICI Bank", "sector": "Banking", "exchange": "NSE"},
        {"symbol": "SBIN.NS", "name": "State Bank of India", "sector": "Banking", "exchange": "NSE"},
        {"symbol": "BHARTIARTL.NS", "name": "Bharti Airtel", "sector": "Telecom", "exchange": "NSE"},
        {"symbol": "ITC.NS", "name": "ITC Ltd", "sector": "FMCG", "exchange": "NSE"},
        {"symbol": "WIPRO.NS", "name": "Wipro", "sector": "IT", "exchange": "NSE"},
        {"symbol": "HCLTECH.NS", "name": "HCL Technologies", "sector": "IT", "exchange": "NSE"},
        {"symbol": "ASIANPAINT.NS", "name": "Asian Paints", "sector": "Paint", "exchange": "NSE"},
        {"symbol": "MARUTI.NS", "name": "Maruti Suzuki", "sector": "Automobile", "exchange": "NSE"},
        {"symbol": "TATAMOTORS.NS", "name": "Tata Motors", "sector": "Automobile", "exchange": "NSE"},
        {"symbol": "TITAN.NS", "name": "Titan Company", "sector": "Retail", "exchange": "NSE"},
        {"symbol": "AXISBANK.NS", "name": "Axis Bank", "sector": "Banking", "exchange": "NSE"},
        {"symbol": "KOTAKBANK.NS", "name": "Kotak Mahindra Bank", "sector": "Banking", "exchange": "NSE"},
        {"symbol": "LT.NS", "name": "Larsen & Toubro", "sector": "Construction", "exchange": "NSE"},
        {"symbol": "HINDUNILVR.NS", "name": "Hindustan Unilever", "sector": "FMCG", "exchange": "NSE"},
        {"symbol": "SUNPHARMA.NS", "name": "Sun Pharma", "sector": "Pharma", "exchange": "NSE"},
        {"symbol": "BAJFINANCE.NS", "name": "Bajaj Finance", "sector": "Finance", "exchange": "NSE"}
    ]
    return {
        "stocks": stocks,
        "total": len(stocks),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/analyze/{symbol}")
async def analyze_stock(
    symbol: str,
    period: str = Query('6mo', description='Data period: 1d,5d,1mo,3mo,6mo,1y,2y,5y,max')
):
    """Analyze a single stock for breakout patterns"""
    try:
        # Clean symbol
        symbol = symbol.strip().upper()
        
        # If no exchange specified, try .NS (NSE) first
        if '.' not in symbol:
            # Try NSE format first
            try:
                data = fetch_stock_data(f"{symbol}.NS", period)
                symbol = f"{symbol}.NS"
            except:
                # Try BSE format
                data = fetch_stock_data(f"{symbol}.BO", period)
                symbol = f"{symbol}.BO"
        else:
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
    symbols: str = Query(
        'RELIANCE.NS,TCS.NS,INFY.NS,HDFCBANK.NS,ICICIBANK.NS,SBIN.NS,BHARTIARTL.NS,ITC.NS',
        description='Comma-separated list of stock symbols'
    )
):
    """Scan multiple stocks for breakouts"""
    try:
        stock_list = [s.strip().upper() for s in symbols.split(',')]
        # Limit to 10 stocks for performance
        stock_list = stock_list[:10]
        
        results = []
        failed_stocks = []
        
        for symbol in stock_list:
            try:
                # Ensure .NS for NSE stocks
                if '.' not in symbol:
                    symbol = symbol + '.NS'
                
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

# For Vercel
app = app