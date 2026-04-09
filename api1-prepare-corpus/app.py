"""
API 1 — Prepare Corpus  (mirrors myst_make_mystfa.pbs)
=======================================================
Accepts a .wav + .lab file pair for a single utterance, resamples the audio
to 16 kHz if necessary, and stores the pair in a corpus directory laid out as:

    /data/corpus/{utt_id}/{utt_id}.wav
    /data/corpus/{utt_id}/{utt_id}.lab

This mirrors what myst_make_mystfa.pbs produced from the MyST manifest but
works on individual utterances submitted via HTTP.

Endpoints
---------
POST /prepare
    Body : multipart/form-data
        wav_file  : audio file  (.wav, any sample rate)
        lab_file  : label file  (.lab, UTF-8 plain text transcript)
        utt_id    : (optional) custom utterance ID; defaults to wav filename stem

    Returns : JSON
        {
          "utt_id":   "myst_002004_...",
          "wav_path": "/data/corpus/myst_002004_.../myst_002004_....wav",
          "lab_path": "/data/corpus/myst_002004_.../myst_002004_....lab",
          "original_sr": 8000,
          "resampled":   true
        }

GET /utterances
    Returns the list of utterance IDs that are ready in the corpus.

GET /health
    Liveness probe.
"""

import os
import wave
import audioop
import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CORPUS_ROOT = Path(os.getenv("CORPUS_ROOT", "/data/corpus"))
TARGET_SR = 16_000

app = FastAPI(
    title="API 1 — Prepare Corpus",
    description=(
        "Resamples a .wav utterance to 16 kHz and saves it alongside its .lab "
        "transcript into a flat corpus directory ready for MFA alignment. "
        "Mirrors the logic of myst_make_mystfa.pbs."
    ),
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resample_to_16k(src_wav: Path, dst_wav: Path) -> int:
    """
    Copy src_wav to dst_wav, resampling to TARGET_SR if needed.
    Returns the original sample rate.
    """
    with wave.open(str(src_wav), "rb") as wf:
        channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        nframes = wf.getnframes()
        frames = wf.readframes(nframes)

    original_sr = framerate
    if framerate != TARGET_SR:
        frames, _ = audioop.ratecv(
            frames, sampwidth, channels, framerate, TARGET_SR, None
        )

    dst_wav.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(dst_wav), "wb") as out:
        out.setnchannels(channels)
        out.setsampwidth(sampwidth)
        out.setframerate(TARGET_SR)
        out.writeframes(frames)

    return original_sr


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "service": "api1-prepare-corpus"}


@app.post("/prepare")
async def prepare(
    wav_file: UploadFile = File(..., description=".wav audio file (any sample rate)"),
    lab_file: UploadFile = File(..., description=".lab transcript file (UTF-8 plain text)"),
    utt_id: str = Form(
        default="",
        description="Optional utterance ID. Defaults to the wav filename stem.",
    ),
):
    """
    Resample a single .wav to 16 kHz and pair it with its .lab transcript.

    The corpus layout produced is::

        CORPUS_ROOT/{utt_id}/{utt_id}.wav   ← 16 kHz PCM
        CORPUS_ROOT/{utt_id}/{utt_id}.lab   ← plain-text transcript

    This is the exact layout expected by MFA (API 2).
    """
    # Validate file extensions
    if not wav_file.filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="wav_file must be a .wav file.")
    if not lab_file.filename.lower().endswith(".lab"):
        raise HTTPException(status_code=400, detail="lab_file must be a .lab file.")

    # Determine utterance ID
    stem = utt_id.strip() or Path(wav_file.filename).stem
    # Sanitise: no slashes or spaces
    stem = stem.replace("/", "_").replace("\\", "_").replace(" ", "_")

    utt_dir = CORPUS_ROOT / stem
    utt_dir.mkdir(parents=True, exist_ok=True)

    # Save uploaded files to a temp location first
    with tempfile.TemporaryDirectory() as tmp:
        tmp_wav = Path(tmp) / "upload.wav"
        tmp_lab = Path(tmp) / "upload.lab"

        tmp_wav.write_bytes(await wav_file.read())
        tmp_lab.write_bytes(await lab_file.read())

        # Validate the wav is readable
        try:
            original_sr = _resample_to_16k(tmp_wav, utt_dir / f"{stem}.wav")
        except Exception as exc:
            raise HTTPException(
                status_code=422,
                detail=f"Failed to process WAV file: {exc}",
            ) from exc

        # Copy lab file
        dst_lab = utt_dir / f"{stem}.lab"
        shutil.copy2(str(tmp_lab), str(dst_lab))

    resampled = original_sr != TARGET_SR

    return JSONResponse(
        content={
            "utt_id": stem,
            "wav_path": str(utt_dir / f"{stem}.wav"),
            "lab_path": str(dst_lab),
            "original_sr": original_sr,
            "resampled": resampled,
            "message": (
                f"Resampled {original_sr} Hz → {TARGET_SR} Hz."
                if resampled
                else f"Audio was already {TARGET_SR} Hz — copied without resampling."
            ),
        }
    )


@app.get("/utterances")
def list_utterances():
    """Return all utterance IDs that are stored in the corpus directory."""
    if not CORPUS_ROOT.exists():
        return {"utterances": [], "count": 0}

    utt_ids = sorted(
        d.name
        for d in CORPUS_ROOT.iterdir()
        if d.is_dir()
        and (d / f"{d.name}.wav").exists()
        and (d / f"{d.name}.lab").exists()
    )
    return {"utterances": utt_ids, "count": len(utt_ids)}
