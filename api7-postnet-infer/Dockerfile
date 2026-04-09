# TF 2.x + CUDA base — matches the HPC environment (TF 2.11, CUDA 11.8)
# 2.15.0 is the first TF GPU image that ships Python 3.11.
# Falls back to CPU automatically if no GPU is present.
FROM tensorflow/tensorflow:2.15.0-gpu

WORKDIR /app

# Install additional Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model source + the postnet41.py original for reference
COPY app.py .
COPY postnet41.py ./postnet41_original.py

# Weight file is bind-mounted at runtime from the host (see docker-compose.yml).
# You can also bake weights into the image by uncommenting:
#   COPY best_model_weights_WLM41.h5 /data/weights/best_model_weights_WLM41.h5

VOLUME ["/data/weights"]
VOLUME ["/data/npz"]

EXPOSE 8007

ENV WEIGHTS_PATH=/data/weights/best_model_weights_WLM41.h5
ENV SEQUENCE_LENGTH=9

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8007"]
