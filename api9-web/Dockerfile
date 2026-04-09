FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY static/ ./static/

EXPOSE 8080

ENV API8_URL=http://api8-pipeline:8008
ENV HTTP_TIMEOUT=600

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
