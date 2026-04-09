# Use PyTorch CUDA base for GPU support; falls back to CPU automatically
FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

# Pre-download the WavLM model weights at build time so the first
# request does not block while pulling ~1.2 GB from Hugging Face.
# Set HF_HOME so the cache lands in an image layer.
ENV HF_HOME=/app/hf_cache
RUN python - <<'PY'
from transformers import AutoFeatureExtractor, AutoModel
AutoFeatureExtractor.from_pretrained("microsoft/wavlm-large")
AutoModel.from_pretrained("microsoft/wavlm-large")
print("WavLM model cached.")
PY

VOLUME ["/data/wavlm"]

EXPOSE 8003

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8003"]
