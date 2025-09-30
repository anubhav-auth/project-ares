# ai-service/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, validator
from typing import Optional, List
import os

class Settings(BaseSettings):
    """Centralized configuration using Pydantic BaseSettings"""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # Gemini Configuration
    gemini_api_key: str = Field("", env="GEMINI_API_KEY")
    ai_model: str = Field("gemini-pro", env="AI_MODEL")
    ai_max_retries: int = Field(3, env="AI_MAX_RETRIES")
    ai_timeout: int = Field(30, env="AI_TIMEOUT")
    
    # Redis Configuration
    redis_host: str = Field("localhost", env="REDIS_HOST")
    redis_port: int = Field(6379, env="REDIS_PORT")
    redis_password: Optional[str] = Field(None, env="REDIS_PASSWORD")
    redis_ttl: int = Field(3600, env="REDIS_TTL")
    
    # Service Configuration
    service_port: int = Field(8001, env="AI_SERVICE_PORT")
    log_level: str = Field("INFO", env="LOG_LEVEL")
    debug_mode: bool = Field(False, env="DEBUG_MODE")
    
    # CORS Configuration
    cors_allowed_origins: str = Field(
        "http://localhost:5678,http://localhost:3000",
        env="CORS_ALLOWED_ORIGINS"
    )
    
    # Rate Limiting
    rate_limit_per_minute: int = Field(30, env="AI_RATE_LIMIT")
    
    # Notification Services
    telegram_bot_token: Optional[str] = Field(None, env="TELEGRAM_BOT_TOKEN")
    telegram_chat_id: Optional[str] = Field(None, env="TELEGRAM_CHAT_ID")
    slack_webhook_url: Optional[str] = Field(None, env="SLACK_WEBHOOK_URL")
    discord_webhook_url: Optional[str] = Field(None, env="DISCORD_WEBHOOK_URL")
    
    # Application Settings
    max_prompt_length: int = Field(4000, env="MAX_PROMPT_LENGTH")
    max_response_length: int = Field(2000, env="MAX_RESPONSE_LENGTH")
    cache_enabled: bool = Field(True, env="CACHE_ENABLED")
    
    @validator("gemini_api_key")
    def validate_api_key(cls, v):
        if not v:
            raise ValueError("GEMINI_API_KEY is required")
        return v
    
    @validator("cors_allowed_origins")
    def parse_cors_origins(cls, v):
        """Convert comma-separated string to list"""
        if isinstance(v, str):
            return v.split(",")
        return v
    
    @validator("log_level")
    def validate_log_level(cls, v):
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Invalid log level. Must be one of {valid_levels}")
        return v.upper()
    
    @property
    def redis_url(self) -> str:
        """Build Redis connection URL"""
        if self.redis_password:
            return f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}"
        return f"redis://{self.redis_host}:{self.redis_port}"

# Create singleton instance
settings = Settings()