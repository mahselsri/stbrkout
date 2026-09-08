from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime
import yfinance as yf
import pandas as pd
import numpy as np
import time
import json

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

# --- Stock Data Fetcher (Compatible with all yfinance versions) ---
def fetch_stock_data(symbol: str, period: str = '3mo'):
    """Fetch stock data from Yahoo Finance - Compatible version"""
    
    # Clean symbol
    symbol = symbol.strip().upper()
    
    # If no exchange specified, add .NS (NSE)
    if '.' not in symbol:
        symbols_to_try = [f"{symbol}.NS", f"{symbol}.BO"]
    else:
        symbols_to_try = [symbol]
    
    for try_symbol in symbols_to_try:
        try:
            print(f"Trying to fetch: {try_symbol}")
            
            # Create ticker
            ticker = yf.Ticker(try_symbol)
            
            # Try different methods to get data (without progress parameter)
            try:
                # Method 1: Try with period only
                data = ticker.history(period=period)
            except TypeError:
                try:
                    # Method 2: Try without any parameters
                    data = ticker.history()
                except:
                    try:
                        # Method 3: Try with start and end dates
                        end_date = datetime.now()
                        start_date = end_date - pd.Timedelta(days=90)  # 3 months
                        data = ticker.history(start=start_date.strftime('%Y-%m-%d'), 
                                             end=end_date.strftime('%Y-%m-%d'))
                    except:
                        data = pd.DataFrame()
            
            # If data is empty, try a shorter period
            if data.empty:
                print(f"No data for {try_symbol}, trying 1mo...")
                try:
                    data = ticker.history(period='1mo')
                except:
                    data = ticker.history()
            
            if not data.empty:
                print(f"Successfully fetched {len(data)} rows for {try_symbol}")
                return data
            
            # If still empty, try to get info
            try:
                info = ticker.info
                if info and 'regularMarketPrice' in info:
                    print(f"Using info data for {try_symbol}")
                    current_price = info.get('regularMarketPrice', 0)
                    previous_close = info.get('previousClose', current_price)
                    
                    # Create a single row dataframe
                    data = pd.DataFrame({
                        'Open': [info.get('regularMarketOpen', current_price)],
                        'High': [info.get('regularMarketDayHigh', current_price)],
                        'Low': [info.get('regularMarketDayLow', current_price)],
                        'Close': [current_price],
                        'Volume': [info.get('regularMarketVolume', 0)]
                    }, index=[pd.Timestamp.now()])
                    
                    if current_price > 0:
                        return data
            except:
                pass
                
        except Exception as e:
            print(f"Error with {try_symbol}: {str(e)}")
            continue
    
    # If we get here, no data was found
    raise HTTPException(
        status_code=404, 
        detail=f"No data found for {symbol}. Please check the symbol. Try using format like 'RELIANCE.NS' or 'TCS.BO'"
    )

# --- Analysis Functions ---
def calculate_pivot_points(data):
    """Calculate pivot points and resistance levels"""
    try:
        if len(data) < 2:
            # If only one row, use that
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
    """Calculate dynamic resistance with shorter lookback for 3mo data"""
    try:
        if len(data) < lookback:
            lookback = max(len(data) // 2, 5)
        
        # Ensure we have enough data
        if len(data) < 3:
            return {'r1': 0, 'r2': 0, 'r3': 0}
            
        rolling_high = data['High'].rolling(window=lookback).max()
        rolling_mean = data['High'].rolling(window=lookback).mean()
        rolling_std = data['High'].rolling(window=lookback).std()
        
        # Get last valid values
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
        
        # Use max of pivot and dynamic resistance
        resistance = {
            'pivot': pivot_levels['pivot'],
            'r1': max(pivot_levels['r1'], dynamic_levels['r1']),
            'r2': max(pivot_levels['r2'], dynamic_levels['r2']),
            'r3': max(pivot_levels['r3'], dynamic_levels['r3'])
        }
        
        current_price = round(float(data['Close'].iloc[-1]), 2)
        previous_close = round(float(data['Close'].iloc[-2]), 2) if len(data) > 1 else current_price
        
        # Volume confirmation
        if len(data) >= 14:
            try:
                avg_volume = float(data['Volume'].rolling(window=14).mean().iloc[-1])
                current_volume = float(data['Volume'].iloc[-1])
                volume_confirmation = current_volume > avg_volume * 1.5 if avg_volume > 0 else False
            except:
                volume_confirmation = False
        else:
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
        "data_period": "3 months (optimized)",
        "endpoints": {
            "/popular": "Get popular Indian stocks",
            "/analyze/{symbol}": "Analyze a single stock",
            "/scan": "Scan multiple stocks",
            "/health": "Health check",
            "/test": "Test stock availability"
        },
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/test")
async def test_connection(symbol: str = Query("RELIANCE.NS")):
    """Test if a stock symbol is accessible"""
    try:
        data = fetch_stock_data(symbol, '3mo')
        return {
            "symbol": symbol,
            "status": "accessible",
            "data_points": len(data),
            "last_price": float(data['Close'].iloc[-1]) if not data.empty else None,
            "period": "3mo",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "symbol": symbol,
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.get("/popular")
async def get_popular_indian_stocks():
    """Get a list of popular Indian stocks"""
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
        # Clean symbol
        symbol = symbol.strip().upper()
        
        # Add .NS if no exchange specified
        if '.' not in symbol:
            symbol = symbol + '.NS'
        
        # Fetch data
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
        'RELIANCE.NS,TCS.NS,INFY.NS,HDFCBANK.NS,ICICIBANK.NS',
        description='Comma-separated list of stock symbols'
    )
):
    """Scan multiple stocks for breakouts"""
    try:
        stock_list = [s.strip().upper() for s in symbols.split(',')]
        # Add .NS if needed
        stock_list = [s if '.' in s else s + '.NS' for s in stock_list]
        # Limit to 5 stocks for performance
        stock_list = stock_list[:5]
        
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

# For Vercel
app = app