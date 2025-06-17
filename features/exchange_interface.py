from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple
from features.api_utils import APIUtils
import logging

logger = logging.getLogger(__name__)

class ExchangeInterface(ABC):
    @abstractmethod
    def validate_credentials(self) -> bool:
        pass
    
    @abstractmethod
    def get_account_balance(self, bot) -> Optional[float]:
        pass
    
    @abstractmethod
    def get_max_leverage(self, symbol: str, bot) -> Optional[float]:
        pass
    
    @abstractmethod
    def get_instrument_info(self, symbol: str) -> Optional[Dict]:
        pass
    
    @abstractmethod
    def get_candles(self, symbol: str, timeframe: str, limit: int) -> Optional[List[Dict]]:
        pass
    
    @abstractmethod
    def get_price(self, symbol: str) -> Optional[Tuple[float, float]]:
        pass
    
    @abstractmethod
    async def place_order(self, symbol: str, side: str, size: float, price: float) -> Optional[Dict]:
        pass

class BloFinExchange(ExchangeInterface):
    def __init__(self, api_key: str, api_secret: str, api_passphrase: str):
        self.api_utils = APIUtils(api_key, api_secret, api_passphrase)
    
    def validate_credentials(self) -> bool:
        return self.api_utils.validate_credentials()
    
    def get_account_balance(self, bot) -> Optional[float]:
        return self.api_utils.get_account_balance(bot)
    
    def get_max_leverage(self, symbol: str, bot) -> Optional[float]:
        return self.api_utils.get_max_leverage(symbol, bot)
    
    def get_instrument_info(self, symbol: str) -> Optional[Dict]:
        return self.api_utils.get_instrument_info(symbol)
    
    def get_candles(self, symbol: str, timeframe: str, limit: int) -> Optional[List[Dict]]:
        return self.api_utils.get_candles(symbol, limit, timeframe)
    
    def get_price(self, symbol: str) -> Optional[Tuple[float, float]]:
        return self.api_utils.get_price(symbol)
    
    async def place_order(self, symbol: str, side: str, size: float, price: float) -> Optional[Dict]:
        return await self.api_utils.place_order(symbol, side, size, price)