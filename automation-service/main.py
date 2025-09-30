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

# Enhanced Form Analyzer Class with ML-ready submit button detection
class FormAnalyzer:
    """Advanced form analysis with ML-ready submit button detection"""
    
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
    
    # Enhanced submit button patterns with priority scoring
    SUBMIT_BUTTON_PATTERNS = [
        # Primary patterns - highest priority
        {'selector': 'button[type="submit"]', 'priority': 10},
        {'selector': 'input[type="submit"]', 'priority': 10},
        
        # Text-based patterns with high priority
        {'selector': 'button:has-text("Submit Application")', 'priority': 9},
        {'selector': 'button:has-text("Apply Now")', 'priority': 9},
        {'selector': 'button:has-text("Submit")', 'priority': 9},
        {'selector': 'button:has-text("Apply")', 'priority': 9},
        
        # Secondary text patterns
        {'selector': 'button:has-text("Send Application")', 'priority': 8},
        {'selector': 'button:has-text("Complete Application")', 'priority': 8},
        {'selector': 'button:has-text("Send")', 'priority': 8},
        {'selector': 'button:has-text("Complete")', 'priority': 7},
        {'selector': 'button:has-text("Finish")', 'priority': 7},
        
        # Tertiary patterns
        {'selector': 'button:has-text("Continue")', 'priority': 6},
        {'selector': 'button:has-text("Next")', 'priority': 5},
        {'selector': 'button:has-text("Proceed")', 'priority': 5},
        
        # Link patterns
        {'selector': 'a:has-text("Submit Application")', 'priority': 7},
        {'selector': 'a:has-text("Apply Now")', 'priority': 7},
        {'selector': 'a.submit-button', 'priority': 8},
        {'selector': 'a.btn-submit', 'priority': 8},
        {'selector': 'a.apply-button', 'priority': 8},
        
        # Class-based patterns
        {'selector': '.submit-button', 'priority': 8},
        {'selector': '.btn-submit', 'priority': 8},
        {'selector': '.btn-apply', 'priority': 8},
        {'selector': '.application-submit', 'priority': 8},
        {'selector': 'button.primary', 'priority': 6},
        {'selector': 'button.btn-primary', 'priority': 6},
        
        # ID-based patterns
        {'selector': '#submit', 'priority': 8},
        {'selector': '#submitButton', 'priority': 8},
        {'selector': '#submitBtn', 'priority': 8},
        {'selector': '#applyButton', 'priority': 8},
        {'selector': '#applyBtn', 'priority': 8},
        {'selector': '#applicationSubmit', 'priority': 8},
        
        # Data attribute patterns
        {'selector': '[data-action="submit"]', 'priority': 8},
        {'selector': '[data-action="apply"]', 'priority': 8},
        {'selector': '[data-testid="submit-button"]', 'priority': 8},
        {'selector': '[data-testid="apply-button"]', 'priority': 8},
    ]
    
    @classmethod
    async def find_submit_button(cls, page):
        """Find the most likely submit button using pattern matching and context analysis"""
        
        candidates = []
        
        for pattern in cls.SUBMIT_BUTTON_PATTERNS:
            try:
                elements = await page.locator(pattern['selector']).all()
                for element in elements:
                    if await element.is_visible():
                        # Get button context
                        text = await element.text_content() or ""
                        
                        # Calculate confidence score
                        confidence = pattern['priority']
                        
                        # Boost confidence for specific keywords
                        boost_words = ['submit', 'apply', 'send', 'complete', 'finish', 'application']
                        for word in boost_words:
                            if word in text.lower():
                                confidence += 2
                        
                        # Reduce confidence for negative indicators
                        negative_words = ['cancel', 'back', 'reset', 'clear', 'close', 'delete', 'remove']
                        for word in negative_words:
                            if word in text.lower():
                                confidence -= 5
                        
                        # Check if button is at the bottom of the form (submit buttons usually are)
                        try:
                            box = await element.bounding_box()
                            if box:
                                viewport_size = page.viewport_size
                                if viewport_size and box['y'] > viewport_size['height'] * 0.6:
                                    confidence += 1
                        except:
                            pass
                        
                        # Check for disabled state
                        is_disabled = await element.is_disabled()
                        if is_disabled:
                            confidence -= 10
                        
                        candidates.append({
                            'selector': pattern['selector'],
                            'element': element,
                            'text': text,
                            'confidence': confidence
                        })
                        
            except Exception as e:
                logger.debug(f"Pattern {pattern['selector']} failed: {e}")
        
        # Sort by confidence and return best candidate
        if candidates:
            candidates.sort(key=lambda x: x['confidence'], reverse=True)
            best = candidates[0]
            logger.info(f"Found submit button: '{best['text']}' with confidence {best['confidence']}")
            return best['element']
        
        return None
    
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

# Application Processor with enhanced submit button detection
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
        """Submit a job application with enhanced submit button detection"""
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
                value = None
                
                # Try to find matching value in answers
                if field['classification'] in payload.answers:
                    value = payload.answers[field['classification']]
                elif field['name'] in payload.answers:
                    value = payload.answers[field['name']]
                elif field['label'] in payload.answers:
                    value = payload.answers[field['label']]
                
                if value is None:
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
            
            # Find and click submit button with enhanced detection
            submit_button = await FormAnalyzer.find_submit_button(page)
            
            if submit_button:
                # Scroll button into view before clicking
                await submit_button.scroll_into_view_if_needed()
                await asyncio.sleep(0.5)  # Small delay for scroll to complete
                
                # Try to click the button
                await submit_button.click()
                logger.info("Clicked submit button successfully")
            else:
                # Fallback: Try basic selectors if ML-ready detection fails
                logger.warning("Could not find submit button with confidence scoring, trying fallback")
                
                fallback_selectors = [
                    'button[type="submit"]',
                    'input[type="submit"]',
                    'button:visible:last-of-type',  # Last visible button often is submit
                ]
                
                submit_clicked = False
                for selector in fallback_selectors:
                    try:
                        btn = page.locator(selector).first
                        if await btn.is_visible() and not await btn.is_disabled():
                            await btn.click()
                            submit_clicked = True
                            logger.info(f"Clicked button using fallback selector: {selector}")
                            break
                    except:
                        continue
                
                if not submit_clicked:
                    # Ultimate fallback: find any button with positive text
                    all_buttons = await page.locator('button:visible, input[type="submit"]:visible, a.btn:visible').all()
                    for button in all_buttons:
                        text = (await button.text_content() or "").lower()
                        # Check if text is positive (not cancel, back, etc.)
                        if any(word in text for word in ['submit', 'apply', 'send', 'complete', 'finish', 'next']):
                            if not any(word in text for word in ['cancel', 'back', 'reset', 'clear']):
                                await button.click()
                                logger.info(f"Clicked button with text: {text}")
                                submit_clicked = True
                                break
                    
                    if not submit_clicked:
                        raise Exception("Could not find submit button after all attempts")
            
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
            
            # Combine results
            combined_content = '\n\n'.join([r['content'] for r in results]) if results else "No relevant content found"
            
            return {
                "status": "success",
                "company": company_name,
                "keywords": keywords,
                "pages_scraped": len(results),
                "content": combined_content
            }
            
        except Exception as e:
            logger.error(f"Deep scrape failed: {str(e)}")
            raise
        
        finally:
            await page.close()

# [Rest of the code remains the same - lifecycle management, FastAPI app, endpoints, etc.]
# ... (keeping all the existing endpoints and other functions as they were)

# Lifecycle management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    global redis_client, rate_limiter
    
    # Startup
    logger.info("Starting automation service...")
    
    # Initialize Redis connection with modern async approach
    try:
        redis_client = redis.from_url(
            f"redis://{config.REDIS_HOST}:{config.REDIS_PORT}",
            password=config.REDIS_PASSWORD,
            encoding="utf-8",
            decode_responses=True
        )
        await redis_client.ping()
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
        await redis_client.close()

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

# [All the API endpoints remain exactly the same as in your original code]
# ============= API ENDPOINTS =============

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

# ============= MAIN API ENDPOINTS =============

@app.post("/apply/submit")
async def submit_application(payload: ApplicationPayload, background_tasks: BackgroundTasks):
    """Submit a job application - Queue for processing"""
    
    # Basic validation
    if not payload.answers:
        raise HTTPException(status_code=400, detail="Answers cannot be empty")
    
    # Add default answers if missing
    if 'email' not in payload.answers and 'full_name' not in payload.answers:
        logger.warning(f"Missing basic fields for job {payload.job_id}")
    
    # Check queue capacity
    if application_queue.full():
        raise HTTPException(
            status_code=503, 
            detail="Application queue is full. Please try again later."
        )
    
    # Add to queue
    try:
        await application_queue.put(payload)
        
        # Track active application
        active_applications[payload.job_id] = {
            "queued_at": datetime.utcnow().isoformat(),
            "priority": payload.priority,
            "url": payload.application_url
        }
        
        # Store initial status in Redis if available
        if redis_client:
            await redis_client.setex(
                f"job_status:{payload.job_id}",
                3600,
                json.dumps({
                    "status": "queued",
                    "position": application_queue.qsize(),
                    "estimated_time": f"{application_queue.qsize() * 2} minutes"
                })
            )
        
        logger.info(f"Application queued for job {payload.job_id}")
        
        return {
            "status": "queued",
            "message": f"Application for job {payload.job_id} has been queued",
            "job_id": payload.job_id,
            "position": application_queue.qsize(),
            "estimated_time": f"{application_queue.qsize() * 2} minutes"
        }
        
    except Exception as e:
        logger.error(f"Failed to queue application: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to queue application: {str(e)}")

@app.post("/scrape/deep")
async def deep_scrape_endpoint(payload: DeepScrapePayload):
    """Perform deep scraping for company information"""
    
    logger.info(f"Deep scrape requested for {payload.company_name}")
    
    # Check cache first if Redis is available
    if redis_client:
        cache_key = f"scrape:{payload.company_name}:{','.join(payload.keywords)}"
        cached = await redis_client.get(cache_key)
        if cached:
            logger.info("Returning cached scrape results")
            return json.loads(cached)
    
    # Perform scraping
    async with ApplicationProcessor() as processor:
        try:
            result = await processor.deep_scrape(
                payload.company_name,
                payload.keywords,
                payload.max_pages or 3
            )
            
            # Cache result if Redis is available
            if redis_client and result.get("status") == "success":
                await redis_client.setex(
                    cache_key, 
                    86400,  # Cache for 24 hours
                    json.dumps(result)
                )
            
            return result
            
        except Exception as e:
            logger.error(f"Deep scrape failed: {str(e)}")
            raise HTTPException(
                status_code=500, 
                detail=f"Scraping failed: {str(e)}"
            )

@app.post("/apply/analyze")
async def analyze_application_form(payload: AnalyzePayload):
    """Analyze an application form to understand its structure"""
    
    logger.info(f"Analyzing form at {payload.application_url}")
    
    async with ApplicationProcessor() as processor:
        page = await processor.context.new_page()
        
        try:
            # Navigate to the application URL
            await page.goto(payload.application_url, wait_until="networkidle", timeout=30000)
            
            # Analyze the form
            fields = await FormAnalyzer.analyze_form(page)
            
            # Find the submit button
            submit_button = await FormAnalyzer.find_submit_button(page)
            submit_button_info = None
            if submit_button:
                submit_button_text = await submit_button.text_content()
                submit_button_info = {
                    "found": True,
                    "text": submit_button_text,
                    "selector": "detected_dynamically"
                }
            else:
                submit_button_info = {
                    "found": False,
                    "fallback_selector": 'button[type="submit"]'
                }
            
            # Take a screenshot for reference
            screenshot_name = f"form_analysis_{hashlib.md5(payload.application_url.encode()).hexdigest()}.png"
            screenshot_path = f"{config.SCREENSHOT_DIR}/{screenshot_name}"
            await page.screenshot(path=screenshot_path, full_page=True)
            
            # Categorize fields
            field_categories = {}
            for field in fields:
                category = field['classification']
                if category not in field_categories:
                    field_categories[category] = []
                field_categories[category].append({
                    'name': field['name'],
                    'type': field['type'],
                    'required': field['required'],
                    'label': field['label']
                })
            
            # Identify required fields
            required_fields = [f for f in fields if f.get('required')]
            
            # Build form schema
            form_schema = {
                "fields": fields,
                "submit_button": submit_button_info
            }
            
            return {
                "status": "success",
                "url": payload.application_url,
                "fields_count": len(fields),
                "required_fields_count": len(required_fields),
                "field_categories": field_categories,
                "schema": form_schema,
                "screenshot": f"/screenshots/{screenshot_name}",
                "analyzed_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Form analysis failed: {str(e)}")
            raise HTTPException(
                status_code=500, 
                detail=f"Form analysis failed: {str(e)}"
            )
        
        finally:
            await page.close()

# ============= ADDITIONAL ENDPOINTS =============

@app.get("/apply/status/{job_id}")
async def get_application_status(job_id: str):
    """Get the status of a queued or processed application"""
    
    # Check Redis for status
    if redis_client:
        status_data = await redis_client.get(f"job_status:{job_id}")
        if status_data:
            return json.loads(status_data)
    
    # Check if in active applications
    if job_id in active_applications:
        return {
            "status": "processing",
            "details": active_applications[job_id]
        }
    
    # Not found
    return {
        "status": "not_found",
        "job_id": job_id,
        "message": "Application not found in queue or recent history"
    }

@app.get("/metrics")
async def get_metrics():
    """Get service metrics and statistics"""
    
    metrics = {
        "queue_size": application_queue.qsize(),
        "queue_capacity": application_queue.maxsize,
        "active_applications": len(active_applications),
        "rate_limit": config.RATE_LIMIT_PER_MINUTE,
        "redis_connected": redis_client is not None
    }
    
    # Add screenshot count
    try:
        screenshots = os.listdir(config.SCREENSHOT_DIR)
        metrics["screenshots_count"] = len(screenshots)
    except:
        metrics["screenshots_count"] = 0
    
    # Add Redis metrics if available
    if redis_client:
        try:
            db_size = await redis_client.dbsize()
            metrics["redis_keys"] = db_size
        except:
            pass
    
    return metrics

@app.delete("/cache/clear")
async def clear_cache():
    """Clear all cached data in Redis"""
    
    if not redis_client:
        return {
            "status": "no_cache",
            "message": "Redis is not connected"
        }
    
    try:
        await redis_client.flushdb()
        return {
            "status": "success",
            "message": "Cache cleared successfully"
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clear cache: {str(e)}"
        )

@app.delete("/screenshots/cleanup")
async def cleanup_screenshots(days_old: int = 7):
    """Clean up old screenshots"""
    
    try:
        cutoff_time = datetime.now() - timedelta(days=days_old)
        deleted_count = 0
        
        for filename in os.listdir(config.SCREENSHOT_DIR):
            filepath = os.path.join(config.SCREENSHOT_DIR, filename)
            file_time = datetime.fromtimestamp(os.path.getmtime(filepath))
            
            if file_time < cutoff_time:
                os.remove(filepath)
                deleted_count += 1
        
        return {
            "status": "success",
            "deleted_files": deleted_count,
            "message": f"Deleted {deleted_count} screenshots older than {days_old} days"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Cleanup failed: {str(e)}"
        )

# Error handler
@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "An internal error occurred",
            "error": str(exc),
            "path": request.url.path
        }
    )

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=config.SERVICE_PORT,
        reload=False,
        log_level=config.LOG_LEVEL.lower()
    )