from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import json

app = FastAPI(title="Stock Breakout API")

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

# --- Helper Functions ---
def fetch_stock_data(symbol: str, period: str = '6mo'):
    """Fetch stock data from Yahoo Finance"""
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period=period)
        if data.empty:
            raise ValueError(f"No data found for {symbol}")
        return data
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Error fetching {symbol}: {str(e)}")

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

def detect_breakout(data, symbol):
    """Detect breakout patterns"""
    try:
        pivot_levels = calculate_pivot_points(data)
        
        current_price = round(float(data['Close'].iloc[-1]), 2)
        previous_close = round(float(data['Close'].iloc[-2]), 2)
        
        # Volume confirmation
        avg_volume = float(data['Volume'].rolling(window=20).mean().iloc[-1])
        current_volume = float(data['Volume'].iloc[-1])
        volume_confirmation = current_volume > avg_volume * 1.5
        
        # Check breakouts
        breakouts = []
        
        if current_price > pivot_levels['r1']:
            breakouts.append({
                'level': 'R1',
                'value': pivot_levels['r1'],
                'breakout_percent': round(((current_price - pivot_levels['r1']) / pivot_levels['r1']) * 100, 2)
            })
        
        if current_price > pivot_levels['r2']:
            breakouts.append({
                'level': 'R2',
                'value': pivot_levels['r2'],
                'breakout_percent': round(((current_price - pivot_levels['r2']) / pivot_levels['r2']) * 100, 2)
            })
        
        if current_price > pivot_levels['r3']:
            breakouts.append({
                'level': 'R3',
                'value': pivot_levels['r3'],
                'breakout_percent': round(((current_price - pivot_levels['r3']) / pivot_levels['r3']) * 100, 2)
            })
        
        return {
            'symbol': symbol,
            'current_price': current_price,
            'previous_close': previous_close,
            'resistance_levels': pivot_levels,
            'breakouts': breakouts,
            'is_breakout': len(breakouts) > 0,
            'volume_confirmation': volume_confirmation
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")

# --- API Endpoints ---

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Stock Breakout Detection API",
        "version": "1.0.0",
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
    """Health check endpoint"""
    return {
        "status": "healthy",
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

@app.get("/analyze/{symbol}", response_model=BreakoutResponse)
async def analyze_stock(
    symbol: str,
    period: str = Query('6mo', description='Data period: 1d,5d,1mo,3mo,6mo,1y,2y,5y,max')
):
    """Analyze a single stock for breakout patterns"""
    try:
        # Clean symbol
        symbol = symbol.strip().upper()
        # Add .NS if not present (for NSE stocks)
        if not symbol.endswith('.NS') and '.' not in symbol:
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
    symbols: str = Query(
        'RELIANCE.NS,TCS.NS,INFY.NS,HDFCBANK.NS,ICICIBANK.NS,SBIN.NS,BHARTIARTL.NS,ITC.NS',
        description='Comma-separated list of stock symbols'
    )
):
    """Scan multiple stocks for breakouts"""
    try:
        stock_list = [s.strip().upper() for s in symbols.split(',')]
        # Ensure .NS suffix
        stock_list = [s if s.endswith('.NS') else s + '.NS' for s in stock_list]
        
        results = []
        for symbol in stock_list[:10]:  # Limit to 10 stocks for performance
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
                # Skip stocks that fail
                continue
        
        return {
            "total_scanned": len(stock_list),
            "breakout_stocks": results,
            "breakout_count": len(results),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# For Vercel - this is critical
app = app