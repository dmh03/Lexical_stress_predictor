FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

EXPOSE 8008

# Service URLs — override in docker-compose if needed
ENV API1_URL=http://api1-prepare-corpus:8001
ENV API2_URL=http://api2-mfa-align:8002
ENV API3_URL=http://api3-wavlm-features:8003
ENV API4_URL=http://api4-syllable-average:8004
ENV API6_URL=http://api6-build-padded-npz:8006
ENV API7_URL=http://api7-postnet-infer:8007
ENV HTTP_TIMEOUT=300
ENV PAD_LEN=9

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8008"]
