"""
API 3 — WavLM Feature Extraction  (mirrors myst_wavlm_features.pbs)
====================================================================
Accepts a single 16 kHz .wav file, runs it through microsoft/wavlm-large,
and returns the frame-level hidden-state tensor as a .npy file.

The model produces a (T × 1024) float16 array where T is the number of
20 ms frames in the utterance.  This matches exactly what the original PBS
script produced for every MyST utterance.

Endpoints
---------
POST /extract
    Body : multipart/form-data
        wav_file : .wav audio file (must be 16 kHz mono)
        utt_id   : (optional) utterance ID; defaults to wav filename stem

    Returns : application/octet-stream  — the .npy file (T × 1024, float16)
    Header  : X-Utt-Id, X-Shape, X-Dtype

POST /extract/json
    Same inputs as POST /extract but returns a JSON response with metadata
    and the .npy saved to disk instead of streamed.

    Returns : JSON
        {
          "utt_id":   "myst_002004_...",
          "npy_path": "/data/wavlm/myst_002004_....npy",
          "shape":    [187, 1024],
          "dtype":    "float16",
          "n_frames": 187
        }

GET /features/{utt_id}
    Download the pre-computed .npy for a given utterance.

GET /utterances
    List all utterance IDs whose features have been extracted.

GET /health
    Liveness probe.
"""

import io
import os
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response, FileResponse
from transformers import AutoFeatureExtractor, AutoModel

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
FEATS_ROOT = Path(os.getenv("FEATS_ROOT", "/data/wavlm"))
MODEL_ID = os.getenv("WAVLM_MODEL_ID", "microsoft/wavlm-large")
SAVE_DTYPE = "float16"
EXPECTED_SR = 16_000

app = FastAPI(
    title="API 3 — WavLM Feature Extraction",
    description=(
        "Extracts frame-level WavLM-Large embeddings (T×1024 float16) from a "
        "16 kHz .wav file.  Mirrors myst_wavlm_features.pbs."
    ),
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# Model — loaded once at startup
# ---------------------------------------------------------------------------
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_feature_extractor: AutoFeatureExtractor | None = None
_model: AutoModel | None = None


@app.on_event("startup")
def _load_model():
    global _feature_extractor, _model
    print(f"[startup] Loading {MODEL_ID} on {_device} …")
    _feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID)
    _model = AutoModel.from_pretrained(MODEL_ID).to(_device)
    _model.eval()
    print("[startup] Model ready.")


# ---------------------------------------------------------------------------
# Core extraction helper
# ---------------------------------------------------------------------------

def _extract_features(audio_bytes: bytes, utt_id: str) -> np.ndarray:
    """
    Read audio bytes, validate sample rate, run WavLM inference.

    Returns (T, 1024) float16 array.
    """
    with io.BytesIO(audio_bytes) as buf:
        x, sr = sf.read(buf, dtype="float32", always_2d=False)

    if sr != EXPECTED_SR:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Audio sample rate is {sr} Hz; expected {EXPECTED_SR} Hz. "
                "Please run the audio through API 1 (prepare-corpus) first."
            ),
        )

    if x.ndim == 2:
        x = x.mean(axis=1)  # stereo → mono

    inputs = _feature_extractor(
        x, sampling_rate=sr, return_tensors="pt", padding=True
    )
    inputs = {k: v.to(_device) for k, v in inputs.items()}

    with torch.inference_mode():
        out = _model(**inputs)
        feats = out.last_hidden_state.squeeze(0).detach().cpu().numpy()

    return feats.astype(np.float16, copy=False)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "api3-wavlm-features",
        "model": MODEL_ID,
        "device": str(_device),
        "model_loaded": _model is not None,
    }


@app.post("/extract")
async def extract(
    wav_file: UploadFile = File(..., description="16 kHz .wav audio file"),
    utt_id: str = Form(default="", description="Optional utterance ID"),
):
    """
    Extract WavLM frame features and **stream** the .npy file back directly.

    The response body is a raw NumPy .npy binary (use ``np.load(io.BytesIO(response.content))``).
    """
    if not wav_file.filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="wav_file must be a .wav file.")

    stem = (utt_id.strip() or Path(wav_file.filename).stem).replace(" ", "_")
    audio_bytes = await wav_file.read()

    feats = _extract_features(audio_bytes, stem)

    # Serialise to bytes
    buf = io.BytesIO()
    np.save(buf, feats)
    buf.seek(0)
    npy_bytes = buf.read()

    return Response(
        content=npy_bytes,
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": f'attachment; filename="{stem}.npy"',
            "X-Utt-Id": stem,
            "X-Shape": str(list(feats.shape)),
            "X-Dtype": str(feats.dtype),
        },
    )


@app.post("/extract/json")
async def extract_json(
    wav_file: UploadFile = File(..., description="16 kHz .wav audio file"),
    utt_id: str = Form(default="", description="Optional utterance ID"),
):
    """
    Extract WavLM frame features, save the .npy to disk, and return metadata as JSON.

    The .npy is stored at ``FEATS_ROOT/{utt_id}.npy`` and can later be
    retrieved via GET /features/{utt_id}.
    """
    if not wav_file.filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="wav_file must be a .wav file.")

    stem = (utt_id.strip() or Path(wav_file.filename).stem).replace(" ", "_")
    audio_bytes = await wav_file.read()

    feats = _extract_features(audio_bytes, stem)

    FEATS_ROOT.mkdir(parents=True, exist_ok=True)
    npy_path = FEATS_ROOT / f"{stem}.npy"
    np.save(str(npy_path), feats)

    return JSONResponse(
        content={
            "utt_id": stem,
            "npy_path": str(npy_path),
            "shape": list(feats.shape),
            "dtype": str(feats.dtype),
            "n_frames": feats.shape[0],
        }
    )


@app.get("/features/{utt_id}")
def get_features(utt_id: str):
    """Download the pre-computed .npy for a given utterance."""
    npy_path = FEATS_ROOT / f"{utt_id}.npy"
    if not npy_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No features found for utterance '{utt_id}'.",
        )
    return FileResponse(
        path=str(npy_path),
        media_type="application/octet-stream",
        filename=f"{utt_id}.npy",
    )


@app.get("/utterances")
def list_utterances():
    """Return all utterance IDs whose WavLM features have been computed."""
    if not FEATS_ROOT.exists():
        return {"utterances": [], "count": 0}
    ids = sorted(p.stem for p in FEATS_ROOT.glob("*.npy"))
    return {"utterances": ids, "count": len(ids)}
