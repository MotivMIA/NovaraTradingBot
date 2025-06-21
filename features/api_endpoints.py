from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

class Signal(BaseModel):
    symbol: str
    signal: str
    confidence: float
    patterns: List[str]
    timeframes: List[str]
    bot_name: str

@router.get("/health")
async def health_check():
    return {"status": "healthy"}

@router.get("/balance")
async def get_balance():
    return {"balance": 50000}  # Placeholder

@router.get("/trades")
async def get_trades():
    return []  # Placeholder

@router.get("/analytics")
async def get_analytics():
    return {}  # Placeholder

@router.post("/receive")
async def receive_signal(signal: Signal):
    try:
        logger.info(f"Received signal from {signal.bot_name}: {signal.signal} for {signal.symbol}")
        bot = router.bot  # Injected at startup
        await bot.process_signal(signal)
        return {"status": "received"}
    except Exception as e:
        logger.error(f"Error processing signal: {e}")
        raise HTTPException(status_code=500, detail=str(e))