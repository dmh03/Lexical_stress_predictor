# MFA is available via the montreal-forced-aligner conda package.
# We use the official MFA Docker image as the base so mfa is pre-installed.
FROM mmcauliffe/montreal-forced-aligner:latest

WORKDIR /app

# Install Python web server on top of the MFA environment
RUN pip install --no-cache-dir \
    fastapi==0.111.0 \
    uvicorn[standard]==0.29.0 \
    python-multipart==0.0.9

COPY app.py .

# Pre-download the default MFA models at build time so the first request
# does not need to pull them from the network.
RUN mfa model download dictionary english_us_arpa || true && \
    mfa model download acoustic english_us_arpa || true

VOLUME ["/data/mfa"]

EXPOSE 8002

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8002"]
