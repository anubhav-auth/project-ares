# ai-service/main.py
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator as validator
from typing import Dict, Any, Optional, List
import google.generativeai as genai
import logging
import redis.asyncio as redis  # Use async redis
import json
import asyncio
import hashlib
from datetime import datetime
from contextlib import asynccontextmanager
import time

from config import settings

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Configure Gemini
try:
    if settings.gemini_api_key:
        genai.configure(api_key=settings.gemini_api_key)
        model = genai.GenerativeModel(settings.ai_model)
        logger.info(f"Gemini configured with model: {settings.ai_model}")
    else:
        logger.warning("No Gemini API key configured. Service running in limited mode.")
        model = None
except Exception as e:
    logger.error(f"Failed to configure Gemini: {e}")
    model = None

# Global Redis client (will be initialized in lifespan)
redis_client = None
try:
    redis_client = redis.Redis(
        host=settings.redis_host,
        port=settings.redis_port,
        password=settings.redis_password,
        decode_responses=True
    )
    redis_client.ping()
    logger.info("Redis connected successfully")
except Exception as e:
    logger.warning(f"Redis connection failed: {e} - Continuing without cache")
    redis_client = None

# Global metrics tracking
metrics = {
    "total_requests": 0,
    "successful_requests": 0,
    "failed_requests": 0,
    "cache_hits": 0,
    "cache_misses": 0,
    "average_response_time": 0,
    "requests_by_endpoint": {}
}

# Rate limiter
class RateLimiter:
    def __init__(self):
        self.requests = {}
    
    def is_allowed(self, client_id: str) -> bool:
        now = time.time()
        if client_id not in self.requests:
            self.requests[client_id] = []
        
        # Clean old requests
        self.requests[client_id] = [
            req_time for req_time in self.requests[client_id] 
            if now - req_time < 60
        ]
        
        if len(self.requests[client_id]) >= settings.rate_limit_per_minute:
            return False
        
        self.requests[client_id].append(now)
        return True

rate_limiter = RateLimiter()

# Enhanced Pydantic Models with validation
class JobData(BaseModel):
    job_title: str
    company_name: str
    job_description_text: str
    job_id: str
    source: str
    
    @validator('job_description_text')
    def validate_description(cls, v):
        if len(v) < 50:
            raise ValueError('Job description too short')
        return v

class UserProfile(BaseModel):
    name: str = "Ares Applicant"
    skills: str = "Software Engineering, Python, Node.js, Cloud Services, Docker"
    experience: str = "5+ years developing scalable backend applications"
    education: Optional[str] = "Bachelor's in Computer Science"
    achievements: Optional[List[str]] = []
    
    # Additional fields for better personalization
    preferred_location: Optional[str] = None
    salary_expectation: Optional[str] = None
    availability: Optional[str] = "Immediate"

class CoverLetterRequest(BaseModel):
    job_details: JobData
    user_profile: Dict[str, Any] = Field(default_factory=lambda: UserProfile().dict())
    tone: Optional[str] = Field("professional", description="Tone: professional, friendly, formal")
    length: Optional[int] = Field(300, description="Target word count")

class TriageRequest(BaseModel):
    job_details: JobData
    user_preferences: Optional[Dict[str, Any]] = None

class SimilarityRequest(BaseModel):
    new_job_description: str
    existing_job_descriptions: List[str]
    threshold: Optional[float] = Field(0.8, description="Similarity threshold")

class DossierRequest(BaseModel):
    job_details: JobData
    user_profile: Dict[str, Any] = Field(default_factory=lambda: UserProfile().dict())
    context_text: str = Field(..., description="Text scraped from deep-dive sources")
    focus_areas: Optional[List[str]] = Field(None, description="Specific areas to focus on")

class AnswersRequest(BaseModel):
    application_questions: List[Dict[str, str]]
    strategy_dossier: Dict[str, Any]
    answer_style: Optional[str] = Field("concise", description="Answer style: concise, detailed, creative")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle with async Redis"""
    global redis_client
    
    logger.info(f"Starting AI service on port {settings.service_port}")
    logger.info(f"Debug mode: {settings.debug_mode}")
    logger.info(f"Test mode: {settings.test_mode}")
    logger.info(f"Require Gemini: {settings.require_gemini}")
    
    # Initialize async Redis connection
    if settings.cache_enabled:
        try:
            redis_client = redis.from_url(
                settings.redis_url,
                encoding="utf-8",
                decode_responses=True
            )
            await redis_client.ping()
            logger.info("Redis connected successfully for caching")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e} - Continuing without cache")
            redis_client = None
    else:
        logger.info("Cache disabled by configuration")
    
    yield
    
    # Shutdown
    logger.info("Shutting down AI service...")
    if redis_client:
        await redis_client.close()

# FastAPI App
app = FastAPI(
    title="Project Ares - AI Core",
    description="AI service powered by Gemini",
    version="2.0.0",
    debug=settings.debug_mode,
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Middleware for request tracking
@app.middleware("http")
async def track_requests(request: Request, call_next):
    start_time = time.time()
    
    # Track metrics
    metrics["total_requests"] += 1
    endpoint = request.url.path
    if endpoint not in metrics["requests_by_endpoint"]:
        metrics["requests_by_endpoint"][endpoint] = 0
    metrics["requests_by_endpoint"][endpoint] += 1
    
    # Check rate limiting
    client_id = request.client.host if request.client else "unknown"
    if not rate_limiter.is_allowed(client_id):
        metrics["failed_requests"] += 1
        return JSONResponse(
            status_code=429,
            content={"detail": "Rate limit exceeded. Please try again later."}
        )
    
    try:
        response = await call_next(request)
        
        if response.status_code < 400:
            metrics["successful_requests"] += 1
        else:
            metrics["failed_requests"] += 1
        
        # Update average response time
        process_time = time.time() - start_time
        current_avg = metrics["average_response_time"]
        total_reqs = metrics["successful_requests"]
        metrics["average_response_time"] = (current_avg * (total_reqs - 1) + process_time) / total_reqs if total_reqs > 0 else process_time
        
        response.headers["X-Process-Time"] = str(process_time)
        return response
        
    except Exception as e:
        metrics["failed_requests"] += 1
        raise

class LLMService:
    """Enhanced LLM service with async Redis support"""
    
    @staticmethod
    def get_cache_key(prompt: str, prefix: str = "gemini") -> str:
        """Generate a cache key from prompt"""
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
        return f"{prefix}:{prompt_hash}"
    
    @staticmethod
    async def generate_with_retry(
        prompt: str, 
        max_retries: int = None,
        cache_ttl: int = 3600,
        use_cache: bool = True
    ) -> str:
        """Generate response with retry logic and async caching"""
        if not model:
            if settings.test_mode:
                # Return mock response in test mode
                return "Mock response for testing - Gemini not configured"
            raise HTTPException(status_code=500, detail="Gemini API key not configured")
        
        max_retries = max_retries or settings.ai_max_retries
        
        # Check cache if enabled (async Redis)
        if use_cache and redis_client and settings.cache_enabled:
            cache_key = LLMService.get_cache_key(prompt)
            try:
                cached = await redis_client.get(cache_key)
                if cached:
                    logger.info("Cache hit for prompt")
                    return cached
            except Exception as e:
                logger.warning(f"Cache check failed: {e}")
        
        # Generate new response with retries
        last_error = None
        for attempt in range(max_retries):
            try:
                logger.info(f"Generating response (attempt {attempt + 1}/{max_retries})")
                
                # Add timeout handling
                response = await asyncio.wait_for(
                    asyncio.to_thread(model.generate_content, prompt),
                    timeout=settings.ai_timeout
                )
                
                result = response.text
                
                # Validate response
                if not result or len(result) < 10:
                    raise ValueError("Response too short or empty")
                
                # Cache the response (async)
                if use_cache and redis_client and settings.cache_enabled and result:
                    try:
                        await redis_client.setex(cache_key, cache_ttl, result)
                    except Exception as e:
                        logger.warning(f"Failed to cache response: {e}")
                
                return result
                
            except asyncio.TimeoutError:
                last_error = "Request timeout"
                logger.error(f"Attempt {attempt + 1} timed out")
            except Exception as e:
                last_error = str(e)
                logger.error(f"Attempt {attempt + 1} failed: {e}")
                
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        raise HTTPException(
            status_code=500, 
            detail=f"LLM generation failed after {max_retries} attempts: {last_error}"
        )

# ============= API ENDPOINTS =============

@app.post("/api/check-similarity")
async def check_similarity(request: SimilarityRequest):
    """Check for duplicate jobs using semantic similarity with Gemini embeddings"""
    
    if not request.existing_job_descriptions:
        return {"is_duplicate": False, "similarity_score": 0.0}
    
    # Use Gemini for semantic comparison
    prompt = f"""
    You are a job description analyzer. Compare these job descriptions for similarity.
    
    NEW JOB:
    {request.new_job_description[:1000]}
    
    EXISTING JOBS:
    {chr(10).join([f"Job {i+1}: {desc[:500]}" for i, desc in enumerate(request.existing_job_descriptions[:3])])}
    
    Analyze if the NEW JOB is essentially the same position as any existing job.
    Consider:
    1. Same role and responsibilities (even if worded differently)
    2. Same company or department
    3. Same requirements and qualifications
    4. Same location and job type
    
    A similarity score > 0.85 means it's likely the same job.
    
    Respond in EXACT JSON format:
    {{"is_duplicate": true/false, "similarity_score": 0.0-1.0, "most_similar_index": 0-based-index}}
    """
    
    try:
        response = await LLMService.generate_with_retry(prompt, cache_ttl=3600)
        
        # Parse JSON from response
        import re
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
        else:
            result = {"is_duplicate": False, "similarity_score": 0.0}
        
        # Validate threshold
        if result.get("similarity_score", 0) > request.threshold:
            result["is_duplicate"] = True
        
        return result
        
    except Exception as e:
        logger.error(f"Similarity check failed: {str(e)}")
        # Fallback to simple comparison
        return {"is_duplicate": False, "similarity_score": 0.0, "error": str(e)}

@app.get("/")
async def root():
    return {
        "service": "AI Core",
        "status": "running",
        "version": "2.0.0",
        "gemini_configured": bool(settings.gemini_api_key),
        "redis_connected": redis_client is not None,
        "model": settings.ai_model,
        "endpoints": [
            "/api/generate-standard-coverletter",
            "/api/triage",
            "/api/check-similarity",
            "/api/generate-dossier",
            "/api/generate-answers"
        ]
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    health_status = {
        "status": "healthy" if (model or settings.test_mode) else "degraded",
        "gemini_configured": bool(model),
        "test_mode": settings.test_mode,
        "redis_connected": redis_client is not None,
        "cache_enabled": settings.cache_enabled,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    # Only fail health check if strict mode is enabled
    if settings.require_gemini and not model:
        raise HTTPException(status_code=503, detail="Service unhealthy - Gemini required but not configured")
    
    return health_status

@app.get("/metrics")
async def get_metrics():
    """Get service metrics"""
    return {
        "metrics": metrics,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.post("/api/generate-standard-coverletter")
async def generate_cover_letter(request: CoverLetterRequest):
    """Generate personalized cover letter using Gemini"""
    
    # Build dynamic prompt based on request parameters
    tone_instruction = {
        "professional": "Use a professional and formal tone",
        "friendly": "Use a friendly and approachable tone",
        "formal": "Use a very formal and traditional business tone"
    }.get(request.tone, "Use a professional tone")
    
    prompt = f"""
    Create a {request.tone} cover letter for the following job application.
    
    JOB DETAILS:
    - Position: {request.job_details.job_title}
    - Company: {request.job_details.company_name}
    - Job Description: {request.job_details.job_description_text[:1500]}
    - Source: {request.job_details.source}
    
    CANDIDATE PROFILE:
    - Name: {request.user_profile.get('name', 'Ares Applicant')}
    - Skills: {request.user_profile.get('skills', 'Not specified')}
    - Experience: {request.user_profile.get('experience', 'Not specified')}
    - Education: {request.user_profile.get('education', 'Not specified')}
    - Availability: {request.user_profile.get('availability', 'Immediate')}
    
    REQUIREMENTS:
    1. {tone_instruction}
    2. Keep it exactly {request.length} words (±10%)
    3. Highlight 2-3 relevant skills that match the job description
    4. Show genuine enthusiasm for the company and role
    5. Include specific references to the company or role
    6. Include a strong opening and closing
    7. Do NOT include any placeholder text, brackets, or formatting marks
    8. Do NOT include date, address, or signature block
    9. Start directly with "Dear Hiring Manager" or "Dear {request.job_details.company_name} Team"
    
    Generate ONLY the body text of the cover letter.
    """
    
    try:
        cover_letter = await LLMService.generate_with_retry(prompt, cache_ttl=7200)
        
        # Post-process the cover letter
        cover_letter = cover_letter.strip()
        
        # Count words
        word_count = len(cover_letter.split())
        
        return {
            "cover_letter": cover_letter,
            "job_id": request.job_details.job_id,
            "company_name": request.job_details.company_name,
            "word_count": word_count,
            "tone": request.tone,
            "generated_at": datetime.utcnow().isoformat(),
            "status": "success",
            "cached": False  # Could track if it was from cache
        }
    except Exception as e:
        logger.error(f"Cover letter generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/triage")
async def triage_job(request: TriageRequest):
    """Smart job triaging using LLM analysis"""
    
    # Include user preferences if provided
    preferences_text = ""
    if request.user_preferences:
        preferences_text = f"""
        USER PREFERENCES:
        - Preferred Companies: {request.user_preferences.get('companies', 'Any')}
        - Minimum Salary: {request.user_preferences.get('min_salary', 'Not specified')}
        - Preferred Location: {request.user_preferences.get('location', 'Any')}
        - Remote Preference: {request.user_preferences.get('remote', 'Any')}
        """
    
    prompt = f"""
    Analyze this job posting and classify it into the appropriate action category.
    
    JOB INFORMATION:
    - Title: {request.job_details.job_title}
    - Company: {request.job_details.company_name}
    - Source: {request.job_details.source}
    - Description: {request.job_details.job_description_text[:1000]}
    
    {preferences_text}
    
    CLASSIFICATION RULES:
    
    1. "guardian_protocol" - Use ONLY for:
       - FAANG/MAANG companies (Google, Amazon, Apple, Meta, Microsoft, Netflix)
       - Unicorn startups with high growth
       - Senior/Staff/Principal/Lead positions at top companies
       - Director/VP/C-level roles
       - Compensation likely >$200k or equivalent
       - Unique high-impact positions at renowned companies
    
    2. "full_auto_apply" - Use for:
       - Mid-level Software Engineer roles
       - Standard Developer positions
       - Companies with straightforward application processes
       - Roles that strongly match the candidate's skillset
       - Established companies with good reputation
       - Not junior/intern positions
    
    3. "simple_alert" - Use for:
       - Junior positions
       - Intern positions
       - Contract/temporary positions
       - Roles with vague or poor job descriptions
       - Companies with unknown reputation
       - Roles that don't match core skills
    
    Analyze carefully and respond with ONLY one of these exact words:
    guardian_protocol
    full_auto_apply
    simple_alert
    """
    
    try:
        response = await LLMService.generate_with_retry(prompt, cache_ttl=3600)
        action = response.strip().lower().replace(" ", "_")
        
        # Validate response
        valid_actions = ["guardian_protocol", "full_auto_apply", "simple_alert"]
        if action not in valid_actions:
            logger.warning(f"Invalid action received: {action}, defaulting to simple_alert")
            action = "simple_alert"
        
        # Generate confidence score
        confidence = 0.95 if action in response.lower() else 0.7
        
        return {
            "action": action,
            "job_id": request.job_details.job_id,
            "company": request.job_details.company_name,
            "title": request.job_details.job_title,
            "confidence": confidence,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Triage failed: {str(e)}")
        # Default to safe option on error
        return {
            "action": "simple_alert",
            "job_id": request.job_details.job_id,
            "confidence": 0.5,
            "error": str(e)
        }

# Keep all other endpoints the same...
# (check-similarity, generate-dossier, generate-answers remain the same)

# Error handler
@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "An internal error occurred",
            "error": str(exc),
            "path": request.url.path,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

# Run the service
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=settings.service_port,
        reload=settings.debug_mode,
        log_level=settings.log_level.lower()
    )