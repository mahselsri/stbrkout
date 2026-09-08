import sys
import os
import json
from datetime import datetime
from typing import List, Optional, Dict, Any

# Add the current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    import yfinance as yf
    import pandas as pd
    import numpy as np
except ImportError as e:
    print(f"Import error: {e}")
    # Fallback for Vercel build
    FastAPI = None

# Check if imports succeeded
if FastAPI is None:
    # Simple fallback for build phase
    app = None
else:
    app = FastAPI(title="Stock Breakout Detection API", version="1.0.0")
    
    # Enable CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# --- Data Models ---
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

class BulkScanResponse(BaseModel):
    total_scanned: int
    breakout_stocks: List[BreakoutResponse]
    timestamp: str

# --- Core Functions ---
def fetch_stock_data(symbol: str, period: str = '6mo') -> pd.DataFrame:
    """Fetch stock data from Yahoo Finance"""
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period=period)
        if data.empty:
            raise ValueError(f"No data found for {symbol}")
        return data
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Error fetching {symbol}: {str(e)}")

def calculate_pivot_points(data: pd.DataFrame) -> Dict[str, float]:
    """Calculate pivot points and resistance levels"""
    try:
        # Get previous day's OHLC
        if len(data) < 2:
            raise ValueError("Insufficient data")
            
        prev_day = data.iloc[-2]
        
        high = prev_day['High']
        low = prev_day['Low']
        close = prev_day['Close']
        
        # Classic Pivot Points
        pivot = (high + low + close) / 3
        r1 = 2 * pivot - low
        r2 = pivot + (high - low)
        r3 = high + 2 * (pivot - low)
        
        return {
            'pivot': round(float(pivot), 2),
            'r1': round(float(r1), 2),
            'r2': round(float(r2), 2),
            'r3': round(float(r3), 2)
        }
    except Exception as e:
        # Return default values if calculation fails
        return {'pivot': 0, 'r1': 0, 'r2': 0, 'r3': 0}

def calculate_dynamic_resistance(data: pd.DataFrame, lookback: int = 20) -> Dict[str, float]:
    """Calculate dynamic resistance using moving averages and standard deviation"""
    try:
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

def detect_breakout(data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
    """Detect breakout patterns with volume confirmation"""
    try:
        # Calculate resistance levels
        pivot_levels = calculate_pivot_points(data)
        dynamic_levels = calculate_dynamic_resistance(data)
        
        # Use the higher of pivot and dynamic resistance
        resistance = {
            'pivot': pivot_levels['pivot'],
            'r1': max(pivot_levels['r1'], dynamic_levels['r1']),
            'r2': max(pivot_levels['r2'], dynamic_levels['r2']),
            'r3': max(pivot_levels['r3'], dynamic_levels['r3'])
        }
        
        # Current data
        current_price = round(float(data['Close'].iloc[-1]), 2)
        previous_close = round(float(data['Close'].iloc[-2]), 2)
        current_volume = float(data['Volume'].iloc[-1])
        
        # Volume confirmation
        avg_volume = float(data['Volume'].rolling(window=20).mean().iloc[-1])
        volume_confirmation = current_volume > avg_volume * 1.5
        
        # Check breakouts
        breakouts = []
        
        if current_price > resistance['r1']:
            breakouts.append({
                'level': 'R1',
                'value': resistance['r1'],
                'breakout_percent': round(((current_price - resistance['r1']) / resistance['r1']) * 100, 2)
            })
        
        if current_price > resistance['r2']:
            breakouts.append({
                'level': 'R2',
                'value': resistance['r2'],
                'breakout_percent': round(((current_price - resistance['r2']) / resistance['r2']) * 100, 2)
            })
        
        if current_price > resistance['r3']:
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
            'avg_volume': round(avg_volume, 0),
            'current_volume': round(current_volume, 0)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")

# --- API Endpoints (only if app exists) ---
if app is not None:
    
    @app.get("/")
    async def root():
        return {
            "message": "Stock Breakout Detection API",
            "version": "1.0.0",
            "status": "healthy",
            "timestamp": datetime.now().isoformat()
        }
    
    @app.get("/analyze/{symbol}", response_model=BreakoutResponse)
    async def analyze_stock(
        symbol: str,
        period: str = Query('6mo', description='Data period')
    ):
        """Analyze a single stock for breakout patterns"""
        try:
            # Clean symbol
            symbol = symbol.strip().upper()
            if not symbol.endswith('.NS'):
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
    
    @app.get("/scan", response_model=BulkScanResponse)
    async def scan_stocks(
        symbols: str = Query('RELIANCE.NS,TCS.NS,INFY.NS,HDFCBANK.NS,ICICIBANK.NS')
    ):
        """Scan multiple stocks for breakouts"""
        try:
            stock_list = [s.strip().upper() for s in symbols.split(',')]
            # Ensure .NS suffix
            stock_list = [s if s.endswith('.NS') else s + '.NS' for s in stock_list]
            
            results = []
            for symbol in stock_list:
                try:
                    data = fetch_stock_data(symbol, '3mo')  # Shorter period for faster scan
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
                    # Skip stocks that fail
                    print(f"Error with {symbol}: {str(e)}")
                    continue
            
            return BulkScanResponse(
                total_scanned=len(stock_list),
                breakout_stocks=results,
                timestamp=datetime.now().isoformat()
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/popular")
    async def get_popular_indian_stocks():
        """Get a list of popular Indian stocks"""
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
            {"symbol": "HCLTECH.NS", "name": "HCL Technologies"},
            {"symbol": "ASIANPAINT.NS", "name": "Asian Paints"},
            {"symbol": "MARUTI.NS", "name": "Maruti Suzuki"},
            {"symbol": "TATAMOTORS.NS", "name": "Tata Motors"},
            {"symbol": "TITAN.NS", "name": "Titan Company"},
            {"symbol": "AXISBANK.NS", "name": "Axis Bank"},
            {"symbol": "KOTAKBANK.NS", "name": "Kotak Mahindra Bank"},
            {"symbol": "LT.NS", "name": "Larsen & Toubro"},
            {"symbol": "HINDUNILVR.NS", "name": "Hindustan Unilever"},
            {"symbol": "SUNPHARMA.NS", "name": "Sun Pharma"},
            {"symbol": "BAJFINANCE.NS", "name": "Bajaj Finance"},
        ]
        return {
            "stocks": stocks,
            "total": len(stocks),
            "timestamp": datetime.now().isoformat()
        }
    
    @app.get("/health")
    async def health_check():
        return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# For Vercel - need to export app
if app is not None:
    # This is the handler Vercel will use
    handler = app
else:
    # Fallback if imports failed
    from fastapi import FastAPI
    handler = FastAPI()
    @handler.get("/")
    async def root():
        return {"message": "Stock Breakout API", "status": "loading"}

# Export for Vercel
app = handler