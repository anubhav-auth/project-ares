# automation-service/config.py
import os
from typing import Optional

class Config:
    # Playwright Configuration
    PLAYWRIGHT_HEADLESS = os.getenv("PLAYWRIGHT_HEADLESS", "true").lower() == "true"
    SCREENSHOT_DIR = os.getenv("SCREENSHOT_DIR", "/app/screenshots")
    
    # Rate Limiting
    MAX_PARALLEL_APPLICATIONS = int(os.getenv("MAX_PARALLEL_APPLICATIONS", "5"))
    RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "10"))
    
    # Retry Configuration
    RETRY_ATTEMPTS = int(os.getenv("RETRY_ATTEMPTS", "3"))
    RETRY_DELAY = int(os.getenv("RETRY_DELAY", "2000"))
    
    # Redis Configuration
    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")
    
    # Service Configuration
    SERVICE_PORT = int(os.getenv("AUTOMATION_SERVICE_PORT", "8000"))
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    USER_AGENT = os.getenv("SCRAPING_USER_AGENT", "Mozilla/5.0")
    
    # Job Board URLs
    FOUNDIT_BASE_URL = os.getenv("FOUNDIT_BASE_URL", "https://www.foundit.in")
    LINKEDIN_BASE_URL = os.getenv("LINKEDIN_BASE_URL", "https://www.linkedin.com")
    INDEED_BASE_URL = os.getenv("INDEED_BASE_URL", "https://www.indeed.com")

config = Config()