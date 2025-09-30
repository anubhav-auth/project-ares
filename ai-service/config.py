# ai-service/config.py
import os
from typing import Optional
from pydantic import BaseSettings

class Settings(BaseSettings):
    # API Keys
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    
    # Model Configuration
    ai_model: str = os.getenv("AI_MODEL", "gemini-pro")
    ai_max_retries: int = int(os.getenv("AI_MAX_RETRIES", "3"))
    ai_timeout: int = int(os.getenv("AI_TIMEOUT", "30"))
    
    # Redis Configuration
    redis_host: str = os.getenv("REDIS_HOST", "localhost")
    redis_port: int = int(os.getenv("REDIS_PORT", "6379"))
    redis_password: Optional[str] = os.getenv("REDIS_PASSWORD")
    
    # Service Configuration
    service_port: int = int(os.getenv("AI_SERVICE_PORT", "8001"))
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    debug_mode: bool = os.getenv("DEBUG_MODE", "false").lower() == "true"
    
    # Notification Services
    telegram_bot_token: Optional[str] = os.getenv("TELEGRAM_BOT_TOKEN")
    telegram_chat_id: Optional[str] = os.getenv("TELEGRAM_CHAT_ID")
    slack_webhook_url: Optional[str] = os.getenv("SLACK_WEBHOOK_URL")
    
    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()