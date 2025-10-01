FROM mcr.microsoft.com/playwright/python:v1.40.0-jammy

WORKDIR /app

# Create necessary directories
RUN mkdir -p /app/screenshots /app/logs

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Install additional browsers if needed
RUN playwright install chromium firefox webkit

# Copy application code
COPY config.py .
COPY main.py .

# Create non-root user for security
RUN chown -R pwuser:pwuser /app

# Switch to the existing non-root user
USER pwuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["python", "main.py"]