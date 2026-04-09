"""
API 9 — Web Frontend
====================
Serves the static HTML/CSS/JS stress-prediction UI and proxies
POST /api/run to API 8 (the pipeline orchestrator).

This eliminates browser CORS issues and means users only need to
open one URL: http://localhost:8080
"""

import os
from pathlib import Path

import httpx
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

API8_URL     = os.getenv("API8_URL", "http://api8-pipeline:8008")
HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "600"))
STATIC_DIR   = Path(__file__).parent / "static"

app = FastAPI(title="Stress Prediction UI", docs_url=None, redoc_url=None)

# Serve static files (CSS, JS, assets) if any are added later
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
async def index():
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/health")
def health():
    return {"status": "ok", "service": "api9-web"}


@app.post("/api/run")
async def proxy_run(
    request: Request,
    wav_file: UploadFile = File(...),
    lab_file: UploadFile = File(...),
    utt_id:   str        = Form(default=""),
):
    """
    Proxy the multipart upload to API 8 and stream the JSON response back.
    API 8 always runs both PreNet and PostNet in parallel — no model selector needed.
    """
    files: dict = {
        "wav_file": (wav_file.filename, await wav_file.read(), wav_file.content_type or "audio/wav"),
        "lab_file": (lab_file.filename, await lab_file.read(), lab_file.content_type or "text/plain"),
    }

    data: dict = {}
    if utt_id.strip():
        data["utt_id"] = utt_id.strip()

    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        resp = await client.post(f"{API8_URL}/run", files=files, data=data)

    return JSONResponse(content=resp.json(), status_code=resp.status_code)
