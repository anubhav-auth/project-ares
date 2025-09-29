# automation-service/Dockerfile
FROM mcr.microsoft.com/playwright/python:v1.28.0-focal

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]