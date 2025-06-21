import redis.asyncio as redis
import json
import logging
from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EventBus:
    def __init__(self):
        self.config = Config()
        self.redis_url = f"redis://{self.config.REDIS_HOST}:{self.config.REDIS_PORT}"
        self.client = None
        self.pubsub = None

    async def connect(self):
        try:
            self.client = redis.from_url(self.redis_url, password=self.config.REDIS_PASSWORD)
            self.pubsub = self.client.pubsub()
            logger.info("Connected to Redis event bus")
        except redis.RedisError as e:
            logger.error(f"Error connecting to Redis: {e}")

    async def publish(self, channel: str, message: dict):
        try:
            await self.client.publish(channel, json.dumps(message))
            logger.info(f"Published to {channel}: {message}")
        except redis.RedisError as e:
            logger.error(f"Error publishing to {channel}: {e}")

    async def subscribe(self, channel: str, callback):
        try:
            await self.pubsub.subscribe(channel)
            async for message in self.pubsub.listen():
                if message["type"] == "message":
                    data = json.loads(message["data"])
                    await callback(data)
                    logger.debug(f"Processed message from {channel}: {data}")
        except (redis.RedisError, json.JSONDecodeError) as e:
            logger.error(f"Error in subscription to {channel}: {e}")

    async def close(self):
        if self.client:
            await self.client.close()
            logger.info("Disconnected from Redis event bus")