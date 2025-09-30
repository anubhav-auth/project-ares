# anubhav-auth/project-ares/project-ares-357a8caaaa955a900e061f9383cd190e7570d899/automation-service/main.py
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError
import asyncio
import logging
import os
from typing import Dict, Any, Optional, List

# --- Configuration ---
LOGGING_LEVEL = logging.INFO
SCREENSHOT_DIR = "/app/screenshots"
os.makedirs(SCREENSHOT_DIR, exist_ok=True)

# --- Logging Setup ---
logging.basicConfig(
    level=LOGGING_LEVEL,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Pydantic Models ---
class FileUpload(BaseModel):
    selector: str
    file_path: str

class ApplicationPayload(BaseModel):
    application_url: str
    answers: Dict[str, Any]
    job_id: str = Field(..., description="A unique identifier for the job application.")
    file_uploads: Optional[List[FileUpload]] = Field(None, description="List of files to upload.")


app = FastAPI(
    title="Project Ares - Browser Automation Service",
    description="Handles the automated submission of job applications.",
)


async def perform_application_submission(payload: ApplicationPayload):
    """
    This function contains the core browser automation logic and is run in the background.
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        )
        page = await context.new_page()

        try:
            logger.info(f"Navigating to application URL: {payload.application_url} for job ID: {payload.job_id}")
            await page.goto(payload.application_url, wait_until="networkidle", timeout=60000)

            # --- Intelligent Form Filling ---
            for key, value in payload.answers.items():
                try:
                    # Use a robust selector strategy to find the input field.
                    # This looks for labels, placeholders, and common attribute names.
                    selector = (
                        f'input[name*="{key}" i], input[id*="{key}" i], '
                        f'textarea[name*="{key}" i], textarea[id*="{key}" i], '
                        f'input[placeholder*="{key}" i], textarea[placeholder*="{key}" i]'
                    )
                    logger.info(f"Attempting to fill field '{key}' with selector: {selector}")
                    await page.wait_for_selector(selector, timeout=10000)
                    await page.fill(selector, str(value))
                    logger.info(f"Successfully filled field '{key}'.")
                except PlaywrightTimeoutError:
                    logger.warning(f"Could not find a matching input field for '{key}'. Skipping.")

            # --- File Uploads ---
            if payload.file_uploads:
                for upload in payload.file_uploads:
                    try:
                        logger.info(f"Attempting to upload file '{upload.file_path}' to selector '{upload.selector}'")
                        async with page.expect_file_chooser() as fc_info:
                            await page.click(upload.selector, timeout=10000)
                        file_chooser = await fc_info.value
                        await file_chooser.set_files(upload.file_path)
                        logger.info(f"Successfully uploaded file '{upload.file_path}'.")
                    except (PlaywrightTimeoutError, FileNotFoundError) as e:
                        logger.error(f"Failed to upload file '{upload.file_path}': {e}")
                        # Depending on the criticality, you might want to raise an exception here.

            # --- Robust Submit Button Identification ---
            submit_selectors = [
                'button[type="submit"]',
                'button:has-text("Submit")',
                'button:has-text("Apply")',
                'input[type="submit"]'
            ]
            
            submit_button = None
            for selector in submit_selectors:
                try:
                    btn = page.locator(selector).first
                    if await btn.is_visible():
                        submit_button = btn
                        logger.info(f"Found submit button with selector: {selector}")
                        break
                except PlaywrightTimeoutError:
                    continue

            if not submit_button:
                raise RuntimeError("Could not identify a valid submit button on the page.")

            await submit_button.click()
            await page.wait_for_load_state("networkidle", timeout=30000)

            # --- Confirmation and Screenshot ---
            screenshot_path = os.path.join(SCREENSHOT_DIR, f"confirmation_{payload.job_id}.png")
            await page.screenshot(path=screenshot_path, full_page=True)
            logger.info(f"Application for job ID '{payload.job_id}' submitted successfully. Screenshot saved to {screenshot_path}")

        except Exception as e:
            error_screenshot_path = os.path.join(SCREENSHOT_DIR, f"error_{payload.job_id}.png")
            await page.screenshot(path=error_screenshot_path, full_page=True)
            logger.error(f"An error occurred during application submission for job ID '{payload.job_id}': {e}. Screenshot saved to {error_screenshot_path}")
            # Here you would add logic to notify your analytics service of the failure.

        finally:
            await browser.close()


@app.post("/apply/submit")
async def submit_application(payload: ApplicationPayload, background_tasks: BackgroundTasks):
    """
    Receives an application payload and adds the submission process to a background task queue.
    This allows the API to respond immediately while the browser automation runs.
    """
    background_tasks.add_task(perform_application_submission, payload)
    return {"status": "success", "message": f"Application submission for job ID '{payload.job_id}' has been queued."}


@app.get("/")
def read_root():
    return {"message": "Browser Automation Service is running"}