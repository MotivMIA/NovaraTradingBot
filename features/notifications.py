import requests
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Notifications:
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url

    async def send_alert(self, message: str):
        try:
            payload = {"content": message}
            response = requests.post(self.webhook_url, json=payload)
            response.raise_for_status()
            logger.info(f"Alert sent: {message}")
        except Exception as e:
            logger.error(f"Error sending alert: {e}")