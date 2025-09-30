# automation-service/main.py
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError
from typing import Dict, Any, Optional, List
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import asyncio
import logging
import os
import json
import redis.asyncio as redis
import hashlib
import re
from pathlib import Path
import time
from contextlib import asynccontextmanager
import aiofiles
from config import config

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("/app/logs/automation.log") if os.path.exists("/app/logs") else logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Ensure directories exist
os.makedirs(config.SCREENSHOT_DIR, exist_ok=True)
os.makedirs("/app/logs", exist_ok=True)

# Global variables for rate limiting and queue management
redis_client = None
application_queue = asyncio.Queue(maxsize=100)
active_applications = {}
rate_limiter = None

# Pydantic Models
class FileUpload(BaseModel):
    selector: str
    file_path: str

class ApplicationPayload(BaseModel):
    application_url: str
    answers: Dict[str, Any]
    job_id: str = Field(..., description="Unique job identifier")
    db_id: Optional[int] = Field(None, description="Database record ID")
    file_uploads: Optional[List[FileUpload]] = None
    priority: Optional[str] = Field("normal", description="Priority: high, normal, low")

class DeepScrapePayload(BaseModel):
    company_name: str
    keywords: List[str] = Field(..., description="Keywords to guide scraping")
    max_pages: Optional[int] = Field(3, description="Maximum pages to scrape")

class AnalyzePayload(BaseModel):
    application_url: str = Field(..., description="URL of the application form")

class ApplicationStatus(BaseModel):
    job_id: str
    status: str
    position: Optional[int] = None
    estimated_time: Optional[str] = None
    details: Optional[Dict[str, Any]] = None

# Rate Limiter Class
class RateLimiter:
    def __init__(self, max_per_minute: int):
        self.max_per_minute = max_per_minute
        self.requests = []
        
    async def acquire(self):
        """Wait if necessary to respect rate limit"""
        now = time.time()
        # Remove requests older than 1 minute
        self.requests = [req_time for req_time in self.requests if now - req_time < 60]
        
        if len(self.requests) >= self.max_per_minute:
            # Wait until the oldest request is more than 1 minute old
            wait_time = 60 - (now - self.requests[0]) + 1
            logger.info(f"Rate limit reached. Waiting {wait_time:.2f} seconds")
            await asyncio.sleep(wait_time)
            return await self.acquire()
        
        self.requests.append(now)
        return True

# Form Analyzer Class
class FormAnalyzer:
    """Advanced form analysis and field mapping"""
    
    FIELD_PATTERNS = {
        'full_name': ['name', 'full name', 'your name', 'applicant name', 'candidate name'],
        'first_name': ['first name', 'given name', 'firstname'],
        'last_name': ['last name', 'surname', 'lastname', 'family name'],
        'email': ['email', 'e-mail', 'email address', 'contact email', 'email id'],
        'phone': ['phone', 'mobile', 'contact number', 'telephone', 'cell', 'contact no'],
        'linkedin': ['linkedin', 'profile url', 'linkedin profile', 'linkedin url'],
        'github': ['github', 'portfolio', 'github profile', 'github url'],
        'website': ['website', 'personal site', 'portfolio url', 'web site'],
        'years_experience': ['years', 'experience', 'years of experience', 'total experience'],
        'salary': ['salary', 'compensation', 'expected salary', 'current ctc', 'expected ctc'],
        'location': ['location', 'city', 'current location', 'preferred location', 'address'],
        'resume': ['resume', 'cv', 'upload resume', 'attach cv', 'curriculum vitae'],
        'cover_letter': ['cover letter', 'letter', 'why join', 'motivation', 'covering letter'],
        'notice_period': ['notice period', 'availability', 'when can you join', 'joining date'],
        'visa_status': ['visa', 'work authorization', 'citizenship', 'work permit'],
        'education': ['education', 'degree', 'qualification', 'university', 'college']
    }
    
    @classmethod
    async def analyze_form(cls, page):
        """Extract and classify all form fields"""
        try:
            # Wait for form to be loaded
            await page.wait_for_selector('form, input, textarea, select', timeout=10000)
            
            # Get page content
            content = await page.content()
            soup = BeautifulSoup(content, 'html.parser')
            
            fields = []
            seen_fields = set()
            
            # Find all form elements
            for element in soup.find_all(['input', 'textarea', 'select']):
                # Skip hidden and submit buttons
                if element.get('type') in ['hidden', 'submit', 'button']:
                    continue
                
                field_id = element.get('id', '') or element.get('name', '')
                if field_id in seen_fields:
                    continue
                seen_fields.add(field_id)
                
                field_info = {
                    'type': element.get('type', 'text'),
                    'name': element.get('name', ''),
                    'id': element.get('id', ''),
                    'placeholder': element.get('placeholder', ''),
                    'label': cls._find_label(element, soup),
                    'required': element.get('required') is not None or 'required' in element.get('class', []),
                    'classification': cls._classify_field(element, soup),
                    'selector': cls._generate_selector(element),
                    'options': cls._get_options(element) if element.name == 'select' else None
                }
                fields.append(field_info)
            
            logger.info(f"Analyzed form: found {len(fields)} fields")
            return fields
            
        except Exception as e:
            logger.error(f"Form analysis failed: {str(e)}")
            return []
    
    @classmethod
    def _find_label(cls, element, soup):
        """Find the label associated with a form field"""
        # Check for label with 'for' attribute
        if element.get('id'):
            label = soup.find('label', {'for': element.get('id')})
            if label:
                return label.get_text(strip=True)
        
        # Check for parent label
        parent = element.parent
        if parent and parent.name == 'label':
            return parent.get_text(strip=True)
        
        # Check for nearby text
        prev = element.find_previous_sibling(['label', 'span', 'div'])
        if prev and len(prev.get_text(strip=True)) < 50:
            return prev.get_text(strip=True)
        
        return ''
    
    @classmethod
    def _classify_field(cls, element, soup):
        """Classify field based on patterns"""
        # Combine all text hints
        text_hints = ' '.join([
            element.get('name', ''),
            element.get('id', ''),
            element.get('placeholder', ''),
            cls._find_label(element, soup),
            element.get('aria-label', '')
        ]).lower()
        
        for field_type, patterns in cls.FIELD_PATTERNS.items():
            for pattern in patterns:
                if pattern in text_hints:
                    return field_type
        
        return 'unknown'
    
    @classmethod
    def _generate_selector(cls, element):
        """Generate a robust selector for the element"""
        if element.get('id'):
            return f"#{element.get('id')}"
        elif element.get('name'):
            return f"[name='{element.get('name')}']"
        elif element.get('class'):
            classes = '.'.join(element.get('class'))
            return f".{classes}"
        else:
            return element.name
    
    @classmethod
    def _get_options(cls, element):
        """Extract options from select element"""
        if element.name != 'select':
            return None
        
        options = []
        for option in element.find_all('option'):
            if option.get('value'):
                options.append({
                    'value': option.get('value'),
                    'text': option.get_text(strip=True)
                })
        return options

# Application Processor
class ApplicationProcessor:
    def __init__(self):
        self.browser = None
        self.context = None
        
    async def __aenter__(self):
        playwright = await async_playwright().start()
        self.browser = await playwright.chromium.launch(
            headless=config.PLAYWRIGHT_HEADLESS,
            args=['--no-sandbox', '--disable-setuid-sandbox']
        )
        self.context = await self.browser.new_context(
            user_agent=config.USER_AGENT,
            viewport={'width': 1920, 'height': 1080},
            locale='en-US'
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.context:
            await self.context.close()
        if self.browser:
            await self.browser.close()
    
    async def submit_application(self, payload: ApplicationPayload):
        """Submit a job application"""
        page = await self.context.new_page()
        
        try:
            logger.info(f"Starting application for job {payload.job_id}")
            
            # Navigate to application page
            await page.goto(payload.application_url, wait_until="networkidle", timeout=60000)
            
            # Take initial screenshot
            await page.screenshot(
                path=f"{config.SCREENSHOT_DIR}/initial_{payload.job_id}.png",
                full_page=True
            )
            
            # Analyze form structure
            fields = await FormAnalyzer.analyze_form(page)
            
            # Fill form fields
            filled_fields = []
            for field in fields:
                if field['classification'] in payload.answers:
                    value = payload.answers[field['classification']]
                elif field['name'] in payload.answers:
                    value = payload.answers[field['name']]
                elif field['label'] in payload.answers:
                    value = payload.answers[field['label']]
                else:
                    continue
                
                try:
                    selector = field['selector']
                    await page.wait_for_selector(selector, timeout=5000)
                    
                    if field['type'] == 'select':
                        await page.select_option(selector, value)
                    elif field['type'] in ['checkbox', 'radio']:
                        if value:
                            await page.check(selector)
                    else:
                        await page.fill(selector, str(value))
                    
                    filled_fields.append(field['classification'] or field['name'])
                    logger.info(f"Filled field: {field['classification']}")
                    
                except Exception as e:
                    logger.warning(f"Could not fill field {field['classification']}: {e}")
            
            # Handle file uploads
            if payload.file_uploads:
                for upload in payload.file_uploads:
                    try:
                        async with page.expect_file_chooser() as fc_info:
                            await page.click(upload.selector)
                        file_chooser = await fc_info.value
                        await file_chooser.set_files(upload.file_path)
                        logger.info(f"Uploaded file: {upload.file_path}")
                    except Exception as e:
                        logger.error(f"File upload failed: {e}")
            
            # Take pre-submission screenshot
            await page.screenshot(
                path=f"{config.SCREENSHOT_DIR}/filled_{payload.job_id}.png",
                full_page=True
            )
            
            # Find and click submit button
            submit_selectors = [
                'button[type="submit"]',
                'input[type="submit"]',
                'button:has-text("Submit")',
                'button:has-text("Apply")',
                'button:has-text("Send")',
                'a:has-text("Submit Application")'
            ]
            
            submit_clicked = False
            for selector in submit_selectors:
                try:
                    if await page.locator(selector).first.is_visible():
                        await page.locator(selector).first.click()
                        submit_clicked = True
                        logger.info(f"Clicked submit button: {selector}")
                        break
                except:
                    continue
            
            if not submit_clicked:
                raise Exception("Could not find submit button")
            
            # Wait for submission to complete
            await page.wait_for_load_state("networkidle", timeout=30000)
            
            # Take confirmation screenshot
            await page.screenshot(
                path=f"{config.SCREENSHOT_DIR}/confirmation_{payload.job_id}.png",
                full_page=True
            )
            
            return {
                "status": "success",
                "job_id": payload.job_id,
                "fields_filled": filled_fields,
                "screenshot": f"/screenshots/confirmation_{payload.job_id}.png"
            }
            
        except Exception as e:
            logger.error(f"Application submission failed: {str(e)}")
            
            # Take error screenshot
            try:
                await page.screenshot(
                    path=f"{config.SCREENSHOT_DIR}/error_{payload.job_id}.png",
                    full_page=True
                )
            except:
                pass
            
            raise
        
        finally:
            await page.close()
    
    async def deep_scrape(self, company_name: str, keywords: List[str], max_pages: int = 3):
        """Perform deep scraping for company information"""
        page = await self.context.new_page()
        results = []
        
        try:
            # Build search query
            search_query = f"{company_name} {' '.join(keywords)} engineering blog culture"
            search_url = f"https://www.google.com/search?q={search_query.replace(' ', '+')}"
            
            await page.goto(search_url)
            
            # Get search results
            links = await page.locator('a[href*="http"]').all()
            relevant_links = []
            
            for link in links[:10]:  # Check first 10 results
                href = await link.get_attribute('href')
                if href and company_name.lower() in href.lower():
                    relevant_links.append(href)
            
            # Scrape each relevant page
            for url in relevant_links[:max_pages]:
                try:
                    await page.goto(url, wait_until="domcontentloaded", timeout=15000)
                    
                    # Extract text content
                    content = await page.evaluate('''() => {
                        const elements = document.querySelectorAll('p, h1, h2, h3, article, section');
                        let text = '';
                        elements.forEach(el => {
                            text += el.innerText + '\\n';
                        });
                        return text;
                    }''')
                    
                    # Filter for relevant content
                    relevant_content = []
                    for paragraph in content.split('\n'):
                        if any(keyword.lower() in paragraph.lower() for keyword in keywords):
                            relevant_content.append(paragraph)
                    
                    if relevant_content:
                        results.append({
                            'url': url,
                            'content': '\n'.join(relevant_content[:10])  # Limit content
                        })
                    
                except Exception as e:
                    logger.warning(f"Failed to scrape {url}: {e}")
            
            return {
                "company": company_name,
                "keywords": keywords,
                "pages_scraped": len(results),
                "content": '\n\n'.join([r['content'] for r in results])
            }
            
        except Exception as e:
            logger.error(f"Deep scrape failed: {str(e)}")
            raise
        
        finally:
            await page.close()

# Lifecycle management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    global redis_client, rate_limiter
    
    # Startup
    logger.info("Starting automation service...")
    
    # Initialize Redis connection
    try:
        redis_client = await redis.create_redis_pool(
            f"redis://{config.REDIS_HOST}:{config.REDIS_PORT}",
            password=config.REDIS_PASSWORD,
            encoding="utf-8"
        )
        logger.info("Redis connected successfully")
    except Exception as e:
        logger.warning(f"Redis connection failed: {e} - Running without Redis")
        redis_client = None
    
    # Initialize rate limiter
    rate_limiter = RateLimiter(config.RATE_LIMIT_PER_MINUTE)
    
    # Start background workers
    asyncio.create_task(process_application_queue())
    
    yield
    
    # Shutdown
    logger.info("Shutting down automation service...")
    if redis_client:
        redis_client.close()
        await redis_client.wait_closed()

# FastAPI App
app = FastAPI(
    title="Project Ares - Automation Service",
    description="Browser automation for job applications",
    version="2.0.0",
    lifespan=lifespan
)

# Background worker for processing applications
async def process_application_queue():
    """Process applications from the queue"""
    while True:
        try:
            if not application_queue.empty():
                payload = await application_queue.get()
                
                # Respect rate limit
                await rate_limiter.acquire()
                
                # Process application
                async with ApplicationProcessor() as processor:
                    try:
                        result = await processor.submit_application(payload)
                        
                        # Update status in Redis
                        if redis_client:
                            await redis_client.setex(
                                f"job_status:{payload.job_id}",
                                3600,  # 1 hour expiry
                                json.dumps({"status": "completed", "result": result})
                            )
                        
                        # Update active applications
                        if payload.job_id in active_applications:
                            del active_applications[payload.job_id]
                        
                    except Exception as e:
                        logger.error(f"Application failed for {payload.job_id}: {e}")
                        
                        # Update error status
                        if redis_client:
                            await redis_client.setex(
                                f"job_status:{payload.job_id}",
                                3600,
                                json.dumps({"status": "failed", "error": str(e)})
                            )
            else:
                await asyncio.sleep(1)
                
        except Exception as e:
            logger.error(f"Queue processor error: {e}")
            await asyncio.sleep(5)

# API Endpoints

@app.get("/")
async def root():
    return {
        "service": "Automation Service",
        "status": "running",
        "version": "2.0.0",
        "queue_size": application_queue.qsize(),
        "active_applications": len(active_applications),
        "redis_connected": redis_client is not None
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "queue_size": application_queue.qsize(),
        "redis": "connected" if redis_client else "disconnected"
    }