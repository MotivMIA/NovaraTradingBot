import redis
import json
from logging import getLogger
from typing import Dict, Callable
from features.config import REDIS_HOST, REDIS_PORT

logger = getLogger(__name__)

class EventBus:
    def __init__(self):
        try:
            self.redis = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
            logger.info("Redis connection established")
        except redis.RedisError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.redis = None
    
    def publish(self, channel: str, message: Dict):
        if not self.redis:
            logger.warning(f"Cannot publish to {channel}: Redis not connected")
            return
        try:
            self.redis.publish(channel, json.dumps(message))
            logger.debug(f"Published to {channel}: {message}")
        except redis.RedisError as e:
            logger.error(f"Failed to publish to {channel}: {e}")
    
    def subscribe(self, channel: str, callback: Callable[[Dict], None]):
        if not self.redis:
            logger.warning(f"Cannot subscribe to {channel}: Redis not connected")
            return
        try:
            pubsub = self.redis.pubsub()
            pubsub.subscribe(channel)
            logger.info(f"Subscribed to {channel}")
            for message in pubsub.listen():
                if message["type"] == "message":
                    try:
                        data = json.loads(message["data"])
                        callback(data)
                        logger.debug(f"Processed message from {channel}: {data}")
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to decode message from {channel}: {e}")
                    except Exception as e:
                        logger.error(f"Callback failed for {channel}: {e}")
        except redis.RedisError as e:
            logger.error(f"Subscription to {channel} failed: {e}")