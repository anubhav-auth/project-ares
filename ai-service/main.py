# ai-service/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
import google.generativeai as genai
import logging
import redis
import json
import asyncio
import os
from datetime import datetime

# Configuration
class Settings:
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    ai_model: str = os.getenv("AI_MODEL", "gemini-pro")
    ai_max_retries: int = int(os.getenv("AI_MAX_RETRIES", "3"))
    ai_timeout: int = int(os.getenv("AI_TIMEOUT", "30"))
    
    redis_host: str = os.getenv("REDIS_HOST", "localhost")
    redis_port: int = int(os.getenv("REDIS_PORT", "6379"))
    redis_password: Optional[str] = os.getenv("REDIS_PASSWORD")
    
    service_port: int = int(os.getenv("AI_SERVICE_PORT", "8001"))
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    debug_mode: bool = os.getenv("DEBUG_MODE", "false").lower() == "true"

settings = Settings()

# Configure logging
logging.basicConfig(level=getattr(logging, settings.log_level))
logger = logging.getLogger(__name__)

# Configure Gemini
if settings.gemini_api_key:
    genai.configure(api_key=settings.gemini_api_key)
    model = genai.GenerativeModel(settings.ai_model)
    logger.info(f"Gemini configured with model: {settings.ai_model}")
else:
    logger.error("No Gemini API key configured! Service will not function properly.")
    model = None

# Configure Redis (optional - for caching)
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

# Pydantic Models
class JobData(BaseModel):
    job_title: str
    company_name: str
    job_description_text: str
    job_id: str
    source: str

class UserProfile(BaseModel):
    name: str = "Ares Applicant"
    skills: str = "Software Engineering, Python, Node.js, Cloud Services, Docker"
    experience: str = "5+ years developing scalable backend applications"
    education: Optional[str] = "Bachelor's in Computer Science"
    achievements: Optional[List[str]] = []

class CoverLetterRequest(BaseModel):
    job_details: JobData
    user_profile: Dict[str, Any] = Field(default_factory=lambda: UserProfile().dict())

class TriageRequest(BaseModel):
    job_details: JobData

class SimilarityRequest(BaseModel):
    new_job_description: str
    existing_job_descriptions: List[str]

class DossierRequest(BaseModel):
    job_details: JobData
    user_profile: Dict[str, Any] = Field(default_factory=lambda: UserProfile().dict())
    context_text: str = Field(..., description="Text scraped from deep-dive sources")

class AnswersRequest(BaseModel):
    application_questions: List[Dict[str, str]]
    strategy_dossier: Dict[str, Any]

# FastAPI App
app = FastAPI(
    title="Project Ares - AI Core",
    description="AI service powered by Gemini",
    version="1.0.0",
    debug=settings.debug_mode
)

class LLMService:
    """Centralized LLM service for all AI operations"""
    
    @staticmethod
    async def generate_with_retry(prompt: str, max_retries: int = None):
        """Generate response with retry logic"""
        if not model:
            raise HTTPException(status_code=500, detail="Gemini API key not configured")
        
        max_retries = max_retries or settings.ai_max_retries
        
        for attempt in range(max_retries):
            try:
                # Add cache check if Redis is available
                cache_key = f"gemini:{hash(prompt)}"
                if redis_client:
                    cached = redis_client.get(cache_key)
                    if cached:
                        logger.info("Returning cached response")
                        return cached
                
                # Generate new response
                response = model.generate_content(prompt)
                result = response.text
                
                # Cache the response for 1 hour
                if redis_client and result:
                    redis_client.setex(cache_key, 3600, result)
                
                return result
                
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    raise HTTPException(status_code=500, detail=f"LLM generation failed: {str(e)}")
                await asyncio.sleep(2 ** attempt)  # Exponential backoff

# API Endpoints

@app.get("/")
async def root():
    return {
        "service": "AI Core",
        "status": "running",
        "gemini_configured": bool(settings.gemini_api_key),
        "redis_connected": redis_client is not None
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    health_status = {
        "status": "healthy" if model else "unhealthy",
        "gemini_configured": bool(model),
        "redis_connected": redis_client is not None,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    if not model:
        raise HTTPException(status_code=503, detail="Service unhealthy - Gemini not configured")
    
    return health_status

@app.post("/api/generate-standard-coverletter")
async def generate_cover_letter(request: CoverLetterRequest):
    """Generate personalized cover letter using Gemini"""
    
    prompt = f"""
    Create a professional cover letter for the following job application.
    
    JOB DETAILS:
    - Position: {request.job_details.job_title}
    - Company: {request.job_details.company_name}
    - Job Description: {request.job_details.job_description_text[:1500]}
    
    CANDIDATE PROFILE:
    - Name: {request.user_profile.get('name', 'Ares Applicant')}
    - Skills: {request.user_profile.get('skills', 'Not specified')}
    - Experience: {request.user_profile.get('experience', 'Not specified')}
    - Education: {request.user_profile.get('education', 'Not specified')}
    
    REQUIREMENTS:
    1. Keep it concise (250-300 words maximum)
    2. Highlight 2-3 relevant skills that match the job description
    3. Show genuine enthusiasm for the company and role
    4. Use professional but personable tone
    5. Include a strong opening and closing
    6. Do NOT include any placeholder text, brackets, or formatting marks
    7. Do NOT include date, address, or signature block
    8. Start directly with "Dear Hiring Manager" or "Dear [Company] Team"
    
    Generate ONLY the body text of the cover letter.
    """
    
    try:
        cover_letter = await LLMService.generate_with_retry(prompt)
        
        return {
            "cover_letter": cover_letter.strip(),
            "job_id": request.job_details.job_id,
            "company_name": request.job_details.company_name,
            "generated_at": datetime.utcnow().isoformat(),
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Cover letter generation failed: {str(e)}")
        raise

@app.post("/api/triage")
async def triage_job(request: TriageRequest):
    """Smart job triaging using LLM analysis"""
    
    prompt = f"""
    Analyze this job posting and classify it into the appropriate action category.
    
    JOB INFORMATION:
    - Title: {request.job_details.job_title}
    - Company: {request.job_details.company_name}
    - Source: {request.job_details.source}
    - Description Preview: {request.job_details.job_description_text[:800]}
    
    CLASSIFICATION RULES:
    
    1. "guardian_protocol" - Use ONLY for:
       - FAANG companies (Google, Amazon, Apple, Meta, Microsoft)
       - Senior/Staff/Principal/Lead positions
       - Director/VP level roles
       - Compensation likely >$200k
       - Unique high-impact positions
    
    2. "full_auto_apply" - Use for:
       - Mid-level Software Engineer roles
       - Standard Developer positions
       - Companies with standard application processes
       - Roles matching general skillset
       - Not junior/intern positions
    
    3. "simple_alert" - Use for:
       - Junior positions
       - Intern positions
       - Roles with poor job description
       - Contract/temporary positions
       - Unclear requirements
    
    Analyze carefully and respond with ONLY one of these exact words:
    guardian_protocol
    full_auto_apply
    simple_alert
    """
    
    try:
        response = await LLMService.generate_with_retry(prompt)
        action = response.strip().lower().replace(" ", "_")
        
        # Validate response
        valid_actions = ["guardian_protocol", "full_auto_apply", "simple_alert"]
        if action not in valid_actions:
            logger.warning(f"Invalid action received: {action}, defaulting to simple_alert")
            action = "simple_alert"
        
        return {
            "action": action,
            "job_id": request.job_details.job_id,
            "company": request.job_details.company_name,
            "title": request.job_details.job_title,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Triage failed: {str(e)}")
        # Default to safe option on error
        return {
            "action": "simple_alert",
            "job_id": request.job_details.job_id,
            "error": str(e)
        }

@app.post("/api/check-similarity")
async def check_similarity(request: SimilarityRequest):
    """Check for duplicate jobs using semantic similarity"""
    
    if not request.existing_job_descriptions:
        return {"is_duplicate": False, "similarity_score": 0.0}
    
    prompt = f"""
    Compare the new job description with existing ones to detect duplicates.
    
    NEW JOB DESCRIPTION:
    {request.new_job_description[:800]}
    
    EXISTING JOB DESCRIPTIONS:
    {chr(10).join([f"[Job {i+1}]: {desc[:400]}" for i, desc in enumerate(request.existing_job_descriptions[:3])])}
    
    Analyze for:
    1. Same company and position
    2. Similar requirements (>80% overlap)
    3. Identical key phrases
    4. Same location and job type
    
    Respond in EXACT JSON format:
    {{"is_duplicate": true_or_false, "similarity_score": 0.0_to_1.0, "most_similar_index": index_number}}
    
    Example: {{"is_duplicate": true, "similarity_score": 0.95, "most_similar_index": 1}}
    """
    
    try:
        response = await LLMService.generate_with_retry(prompt)
        
        # Extract JSON from response
        json_str = response.strip()
        if "```json" in json_str:
            json_str = json_str.split("```json")[1].split("```")[0]
        elif "```" in json_str:
            json_str = json_str.split("```")[1].split("```")[0]
        
        result = json.loads(json_str)
        
        # Validate result
        if "is_duplicate" not in result:
            result["is_duplicate"] = False
        if "similarity_score" not in result:
            result["similarity_score"] = 0.0
        
        return result
        
    except Exception as e:
        logger.error(f"Similarity check failed: {str(e)}")
        return {"is_duplicate": False, "similarity_score": 0.0, "error": str(e)}

@app.post("/api/generate-dossier")
async def generate_dossier(request: DossierRequest):
    """Generate a detailed strategy dossier for high-stakes applications"""
    
    prompt = f"""
    Create a comprehensive application strategy dossier for this high-priority position.
    
    TARGET POSITION:
    - Role: {request.job_details.job_title}
    - Company: {request.job_details.company_name}
    - Description: {request.job_details.job_description_text[:1000]}
    
    CANDIDATE PROFILE:
    - Name: {request.user_profile.get('name')}
    - Skills: {request.user_profile.get('skills')}
    - Experience: {request.user_profile.get('experience')}
    
    ADDITIONAL CONTEXT (from company research):
    {request.context_text[:500]}
    
    Generate a detailed JSON dossier with:
    1. executive_summary: 2-3 sentence overview
    2. key_requirements: List of 3-5 main job requirements
    3. skill_alignment: How candidate's skills match (with percentages)
    4. tailored_achievements: 3 specific achievements to highlight
    5. company_pain_points: What problems this role solves
    6. suggested_questions: 3 intelligent questions to ask
    7. interview_topics: Likely interview focus areas
    8. red_flags: Any concerns or gaps to address
    9. application_strategy: Specific approach recommendations
    10. follow_up_plan: Post-application strategy
    
    Format as valid JSON.
    """
    
    try:
        response = await LLMService.generate_with_retry(prompt)
        
        # Parse JSON from response
        json_str = response.strip()
        if "```json" in json_str:
            json_str = json_str.split("```json")[1].split("```")[0]
        elif "```" in json_str:
            json_str = json_str.split("```")[1].split("```")[0]
        
        dossier = json.loads(json_str)
        
        return {
            "status": "success",
            "job_id": request.job_details.job_id,
            "company": request.job_details.company_name,
            "dossier": dossier,
            "generated_at": datetime.utcnow().isoformat()
        }
        
    except json.JSONDecodeError:
        # If JSON parsing fails, return structured text response
        return {
            "status": "success",
            "job_id": request.job_details.job_id,
            "dossier": {"raw_analysis": response},
            "generated_at": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Dossier generation failed: {str(e)}")
        raise

@app.post("/api/generate-answers")
async def generate_answers(request: AnswersRequest):
    """Generate answers for application form questions"""
    
    questions_text = "\n".join([
        f"Q{i+1}: {q.get('label', q.get('question', 'Unknown'))}"
        for i, q in enumerate(request.application_questions[:10])
    ])
    
    dossier_summary = json.dumps(request.strategy_dossier)[:500] if request.strategy_dossier else "No dossier provided"
    
    prompt = f"""
    Generate concise, professional answers for these job application questions.
    
    APPLICATION QUESTIONS:
    {questions_text}
    
    CONTEXT FROM STRATEGY DOSSIER:
    {dossier_summary}
    
    INSTRUCTIONS:
    1. Keep answers concise but complete
    2. Use professional tone
    3. Incorporate relevant details from the dossier
    4. For text fields: 50-150 words max
    5. For yes/no questions: Answer directly then briefly explain
    6. Show enthusiasm and qualification
    7. Avoid generic responses
    
    Format response as JSON:
    {{
        "Q1": "Your answer here",
        "Q2": "Your answer here",
        ...
    }}
    """
    
    try:
        response = await LLMService.generate_with_retry(prompt)
        
        # Parse JSON
        json_str = response.strip()
        if "```json" in json_str:
            json_str = json_str.split("```json")[1].split("```")[0]
        elif "```" in json_str:
            json_str = json_str.split("```")[1].split("```")[0]
        
        answers_dict = json.loads(json_str)
        
        # Map answers back to original questions
        final_answers = {}
        for i, q in enumerate(request.application_questions):
            q_key = f"Q{i+1}"
            q_label = q.get('label', q.get('question', q_key))
            if q_key in answers_dict:
                final_answers[q_label] = answers_dict[q_key]
        
        return {
            "status": "success",
            "answers": final_answers,
            "questions_count": len(request.application_questions),
            "answers_count": len(final_answers),
            "generated_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Answer generation failed: {str(e)}")
        # Return basic fallback answers
        fallback_answers = {}
        for q in request.application_questions:
            q_label = q.get('label', q.get('question', 'Unknown'))
            if 'salary' in q_label.lower():
                fallback_answers[q_label] = "Negotiable based on overall compensation package"
            elif 'years' in q_label.lower():
                fallback_answers[q_label] = "5+ years"
            else:
                fallback_answers[q_label] = "Available upon request"
        
        return {
            "status": "fallback",
            "answers": fallback_answers,
            "error": str(e)
        }

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