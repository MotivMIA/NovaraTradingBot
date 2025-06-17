# Webhook alerts, Logtail integration
import requests
from logging import getLogger

logger = getLogger(__name__)

class Notifications:
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url

    def send_webhook_alert(self, message: str):
        if not self.webhook_url:
            return
        try:
            payload = {"content": message}
            requests.post(self.webhook_url, json=payload, timeout=5)
            logger.debug(f"Sent webhook alert: {message}")
        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")