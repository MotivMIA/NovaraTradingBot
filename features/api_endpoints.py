from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.security import APIKeyHeader
from slowapi import Limiter
from slowapi.util import get_remote_address
from pydantic import BaseModel
from typing import List
import logging
from features.exchange_interface import BloFinExchange
from features.config import API_KEY, DEMO_MODE
from features.database import DB_PATH
import sqlite3
import pandas as pd

logger = logging.getLogger(__name__)
app = FastAPI(title="NovaraTradingBot API")
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

API_KEY_HEADER = APIKeyHeader(name="X-API-Key")
VALID_API_KEYS = {API_KEY}

class TradeRequest(BaseModel):
    symbol: str
    side: str
    size: float
    price: float

class Position(BaseModel):
    symbol: str
    side: str
    size: float
    entry_price: float
    leverage: float
    stop_loss: float
    take_profit: float

@app.get("/health")
@limiter.limit("10/minute")
async def health_check(request: Request, api_key: str = Depends(API_KEY_HEADER)):
    if api_key not in VALID_API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API key")
    exchange = BloFinExchange("your_key", "your_secret", "your_passphrase")
    if not exchange.validate_credentials():
        raise HTTPException(status_code=503, detail="Credential validation failed")
    balance = exchange.get_account_balance(None)
    if balance is None:
        raise HTTPException(status_code=503, detail="Balance retrieval failed")
    logger.info("Health check passed via API")
    return {"status": "healthy", "balance": balance}

@app.get("/balance")
@limiter.limit("10/minute")
async def get_balance(request: Request, api_key: str = Depends(API_KEY_HEADER)):
    if api_key not in VALID_API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API key")
    exchange = BloFinExchange("your_key", "your_secret", "your_passphrase")
    balance = exchange.get_account_balance(None)
    if balance is None:
        raise HTTPException(status_code=500, detail="Failed to retrieve balance")
    logger.info(f"Balance retrieved: ${balance:.2f}")
    return {"balance": balance}

@app.post("/trade")
@limiter.limit("5/minute")
async def place_trade(trade: TradeRequest, request: Request, api_key: str = Depends(API_KEY_HEADER)):
    if api_key not in VALID_API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API key")
    exchange = BloFinExchange("your_key", "your_secret", "your_passphrase")
    try:
        order_response = await exchange.place_order(trade.symbol, trade.side, trade.size, trade.price)
        if not order_response:
            raise HTTPException(status_code=400, detail="Order placement failed")
        logger.info(f"Placed {trade.side} trade for {trade.symbol}: {trade.size} at ${trade.price}")
        return {"status": "success", "order": order_response}
    except Exception as e:
        logger.error(f"Trade placement failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/positions", response_model=List[Position])
@limiter.limit("10/minute")
async def get_positions(request: Request, api_key: str = Depends(API_KEY_HEADER)):
    if api_key not in VALID_API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API key")
    # Placeholder: Fetch from bot.open_orders
    positions = []
    logger.info("Retrieved positions via API")
    return positions

@app.get("/analytics")
@limiter.limit("5/minute")
async def get_analytics(request: Request, api_key: str = Depends(API_KEY_HEADER)):
    if api_key not in VALID_API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API key")
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql("SELECT * FROM trades", conn)
        conn.close()
        if df.empty:
            return {"message": "No trades to analyze"}
        win_rate = len(df[df["profit_loss"] > 0]) / len(df)
        sharpe_ratio = df["profit_loss"].mean() / df["profit_loss"].std() * np.sqrt(252) if df["profit_loss"].std() != 0 else 0
        logger.info("Analytics retrieved via API")
        return {
            "win_rate": win_rate,
            "sharpe_ratio": sharpe_ratio,
            "total_trades": len(df)
        }
    except Exception as e:
        logger.error(f"Analytics retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))