"""
API 8 — Full Pipeline Orchestrator
====================================
Accepts a .wav + .lab file pair and runs the complete stress-prediction
pipeline end-to-end by calling the upstream APIs in order:

    API 1  → Prepare Corpus     (resample + store)
    API 2  → MFA Align          (phone-level timings)
    API 3  → WavLM Features     (frame-level 1024-dim embeddings)
    API 4  → Syllable Average   (per-syllable 1024-dim vectors + labels)
    API 6  → Build Padded NPZ   (TDNN-ready fixed-length sequences)
    API 7  → PostNet Inference  (per-syllable stress probabilities)

The orchestrator then assembles the final output: for every word that has
**more than one syllable** in CMUdict it reports whether the model predicted
the stress pattern correctly.

Output format
-------------
Each multi-syllable word is reported as:

    {
      "word":       "beyond",
      "syllables":  ["bih", "yaan"],
      "predicted":  [0, 1],            WPP per-syllable predictions
      "label":      [0, 1],            ground-truth from label file
      "result":     "correct"          "correct" | "incorrect"
    }

Words with a single syllable are omitted entirely.

Endpoints
---------
POST /run
    Body : multipart/form-data
        wav_file    : .wav audio (any sample rate — API 1 will resample to 16 kHz)
        lab_file    : .lab plain-text transcript
        utt_id      : (optional) utterance ID; defaults to wav filename stem

    Returns : JSON
        {
          "utt_id":           "ks0000b0",
          "words_evaluated":  1,
          "results": [
            {
              "word":       "beyond",
              "syllables":  ["B IH0 Y", "AA1 N D"],
              "predicted":  [0, 1],
              "label":      [0, 1],
              "result":     "correct"
            }
          ],
          "pipeline_timings": { "api1_ms": 120, ... }
        }

GET /health
    Liveness probe — also checks reachability of all upstream APIs.
"""

import asyncio
import io
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional  # kept for future use

import httpx
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

# ---------------------------------------------------------------------------
# Configuration — service URLs (docker-compose service names as hostnames)
# ---------------------------------------------------------------------------
API1_URL = os.getenv("API1_URL", "http://api1-prepare-corpus:8001")
API2_URL = os.getenv("API2_URL", "http://api2-mfa-align:8002")
API3_URL = os.getenv("API3_URL", "http://api3-wavlm-features:8003")
API4_URL = os.getenv("API4_URL", "http://api4-syllable-average:8004")
API6_URL = os.getenv("API6_URL", "http://api6-build-padded-npz:8006")
API7_URL = os.getenv("API7_URL", "http://api7-postnet-infer:8007")

# Timeouts — MFA and WavLM can be slow
HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "300"))   # seconds per step
PAD_LEN      = int(os.getenv("PAD_LEN", "9"))

app = FastAPI(
    title="API 8 — Full Pipeline Orchestrator",
    description=(
        "Chains APIs 1→2→3→4→6→7 in a single call. "
        "Returns stress-prediction results (correct/incorrect) "
        "for every multi-syllable word in the utterance."
    ),
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _t() -> float:
    return time.perf_counter()


def _ms(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)


async def _check_upstream(client: httpx.AsyncClient, name: str, url: str) -> dict:
    try:
        r = await client.get(f"{url}/health", timeout=5.0)
        return {"service": name, "url": url, "status": r.status_code, "ok": r.status_code == 200}
    except Exception as exc:
        return {"service": name, "url": url, "status": None, "ok": False, "error": str(exc)}


def _parse_lab(lab_bytes: bytes) -> list[str]:
    """Return upper-cased transcript words from a .lab file."""
    text = lab_bytes.decode("utf-8").strip()
    text = re.sub(r"[^\w\s']", " ", text)
    return [w.upper() for w in text.split() if w]


# ---------------------------------------------------------------------------
# Pipeline steps — each calls one upstream API
# ---------------------------------------------------------------------------

async def _step1_prepare(
    client: httpx.AsyncClient,
    wav_bytes: bytes,
    lab_bytes: bytes,
    wav_name: str,
    lab_name: str,
    utt_id: str,
) -> dict:
    """POST /prepare to API 1."""
    r = await client.post(
        f"{API1_URL}/prepare",
        files={
            "wav_file": (wav_name, wav_bytes, "audio/wav"),
            "lab_file": (lab_name, lab_bytes, "text/plain"),
        },
        data={"utt_id": utt_id},
        timeout=HTTP_TIMEOUT,
    )
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"API 1 failed: {r.text}")
    return r.json()


async def _step2_align(
    client: httpx.AsyncClient,
    wav_bytes: bytes,
    lab_bytes: bytes,
    wav_name: str,
    lab_name: str,
    utt_id: str,
) -> bytes:
    """POST /align to API 2; returns the MFA alignment CSV bytes."""
    r = await client.post(
        f"{API2_URL}/align",
        files={
            "wav_file": (wav_name, wav_bytes, "audio/wav"),
            "lab_file": (lab_name, lab_bytes, "text/plain"),
        },
        data={"utt_id": utt_id},
        timeout=HTTP_TIMEOUT,
    )
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"API 2 (MFA) failed: {r.text}")

    meta = r.json()
    stem = meta["utt_id"]

    # Fetch the CSV from API 2's /alignments endpoint
    r2 = await client.get(f"{API2_URL}/alignments/{stem}", timeout=HTTP_TIMEOUT)
    if r2.status_code != 200:
        raise HTTPException(status_code=502, detail=f"API 2 CSV fetch failed: {r2.text}")
    return r2.content  # raw CSV bytes


async def _step3_wavlm(
    client: httpx.AsyncClient,
    wav_bytes: bytes,
    wav_name: str,
    utt_id: str,
) -> bytes:
    """POST /extract to API 3; returns raw .npy bytes."""
    r = await client.post(
        f"{API3_URL}/extract",
        files={"wav_file": (wav_name, wav_bytes, "audio/wav")},
        data={"utt_id": utt_id},
        timeout=HTTP_TIMEOUT,
    )
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"API 3 (WavLM) failed: {r.text}")
    return r.content  # raw .npy bytes


async def _step4_syllable(
    client: httpx.AsyncClient,
    wav_bytes: bytes,
    lab_bytes: bytes,
    wav_name: str,
    lab_name: str,
    mfa_csv_bytes: bytes,
    wavlm_npy_bytes: bytes,
    utt_id: str,
) -> bytes:
    """POST /average to API 4; returns raw .npy bytes."""
    files: dict = {
        "wav_file":  (wav_name,      wav_bytes,       "audio/wav"),
        "lab_file":  (lab_name,      lab_bytes,       "text/plain"),
        "mfa_csv":   ("align.csv",   mfa_csv_bytes,   "text/csv"),
        "wavlm_npy": ("feats.npy",   wavlm_npy_bytes, "application/octet-stream"),
    }

    r = await client.post(
        f"{API4_URL}/average",
        files=files,
        data={"utt_id": utt_id},
        timeout=HTTP_TIMEOUT,
    )
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"API 4 (syllable average) failed: {r.text}")
    return r.content  # raw .npy bytes


async def _step6_pad(
    client: httpx.AsyncClient,
    syl_npy_bytes: bytes,
    utt_id: str,
) -> bytes:
    """
    Upload the syllable .npy to API 6 then call /build/train-pad restricted
    to this single utterance — returns a padded .npz.

    Note: for a single inference utterance we use the train-pad path (which
    reads from the syllable store) because we don't have a myst_test.npz.
    """
    # Upload the syllable npy
    r_up = await client.post(
        f"{API6_URL}/upload-syllable",
        files={"npy_file": (f"{utt_id}.npy", syl_npy_bytes, "application/octet-stream")},
        data={"utt_id": utt_id},
        timeout=HTTP_TIMEOUT,
    )
    if r_up.status_code != 200:
        raise HTTPException(status_code=502, detail=f"API 6 upload-syllable failed: {r_up.text}")

    # Build padded NPZ
    r_pad = await client.post(
        f"{API6_URL}/build/train-pad",
        data={"pad_len": PAD_LEN, "utt_id": utt_id},
        timeout=HTTP_TIMEOUT,
    )
    if r_pad.status_code != 200:
        raise HTTPException(status_code=502, detail=f"API 6 build/train-pad failed: {r_pad.text}")
    return r_pad.content  # raw .npz bytes


async def _step7_predict(
    client: httpx.AsyncClient,
    padded_npz_bytes: bytes,
) -> dict:
    """POST /predict to API 7; returns inference JSON."""
    r = await client.post(
        f"{API7_URL}/predict",
        files={"padded_npz": ("padded.npz", padded_npz_bytes, "application/octet-stream")},
        data={"apply_wpp": "true", "sequence_length": str(PAD_LEN)},
        timeout=HTTP_TIMEOUT,
    )
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"API 7 (PostNet) failed: {r.text}")
    return r.json()


async def _step7_predict_prenet(
    client: httpx.AsyncClient,
    syl_npy_bytes: bytes,
) -> dict:
    """POST /predict/prenet to API 7 with raw unpadded syllable .npy."""
    r = await client.post(
        f"{API7_URL}/predict/prenet",
        files={"syl_npy": ("syllables.npy", syl_npy_bytes, "application/octet-stream")},
        data={"apply_wpp": "true"},
        timeout=HTTP_TIMEOUT,
    )
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"API 7 (PreNet) failed: {r.text}")
    return r.json()


# ---------------------------------------------------------------------------
# Result assembly
# ---------------------------------------------------------------------------

def _assemble_results(
    syl_npy_bytes: bytes,
    infer_json: dict,
    transcript_words: list[str],
) -> list[dict]:
    """
    Match API 7 per-syllable WPP predictions back to transcript words.
    Returns one entry per multi-syllable word only.

      result = "correct"   if predicted stress matches label for every syllable
               "incorrect" otherwise
    """
    # Load syllable payload from API 4 output
    data          = np.load(io.BytesIO(syl_npy_bytes), allow_pickle=True).item()
    token_index   = np.asarray(data["token_index"]).flatten().astype(int)
    labels        = np.asarray(data["label"]).flatten().astype(int)
    syl_phonemes  = data["syllable_phonemes"]   # list[str], one per syllable

    # WPP predictions from API 7 — aligned to the same real syllables (padding stripped)
    pred_wpp = np.asarray(infer_json["predictions_wpp"], dtype=int)

    if len(pred_wpp) != len(token_index):
        # Length mismatch — return raw predictions without word grouping
        return [{"error": "syllable count mismatch between API 4 and API 7"}]

    # Group syllables by token_index (word position in transcript)
    word_groups: dict[int, list[int]] = defaultdict(list)
    for syl_i, tok in enumerate(token_index):
        word_groups[int(tok)].append(syl_i)

    results: list[dict] = []
    for tok in sorted(word_groups.keys()):
        indices = word_groups[tok]
        n_syl   = len(indices)

        # Only report multi-syllable words
        if n_syl <= 1:
            continue

        word_text  = transcript_words[tok] if tok < len(transcript_words) else f"word_{tok}"
        word_preds = pred_wpp[indices].tolist()
        word_labs  = labels[indices].tolist()
        word_phons = [syl_phonemes[i] for i in indices]

        # "correct" means the WPP-predicted stressed syllable matches the label
        # Label: 2 = primary stress, 1 = secondary stress, 0 = unstressed
        # Prediction: 1 = stressed, 0 = unstressed
        # Map labels to binary: anything > 0 is stressed
        binary_labs = [1 if l > 0 else 0 for l in word_labs]
        is_correct  = word_preds == binary_labs

        results.append({
            "word":      word_text.lower(),
            "syllables": word_phons,
            "predicted": word_preds,
            "label":     binary_labs,
            "result":    "correct" if is_correct else "incorrect",
        })

    return results


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    """Check this service and all upstream APIs."""
    async with httpx.AsyncClient() as client:
        checks = await asyncio.gather(
            _check_upstream(client, "api1-prepare-corpus", API1_URL),
            _check_upstream(client, "api2-mfa-align",      API2_URL),
            _check_upstream(client, "api3-wavlm-features", API3_URL),
            _check_upstream(client, "api4-syllable-average", API4_URL),
            _check_upstream(client, "api6-build-padded-npz", API6_URL),
            _check_upstream(client, "api7-postnet-infer",  API7_URL),
        )
    all_ok = all(c["ok"] for c in checks)
    return JSONResponse(
        status_code=200 if all_ok else 207,
        content={
            "status":   "ok" if all_ok else "degraded",
            "service":  "api8-pipeline",
            "upstream": checks,
        },
    )


@app.post("/run")
async def run_pipeline(
    wav_file: UploadFile = File(..., description=".wav audio file (any sample rate)"),
    lab_file: UploadFile = File(..., description=".lab plain-text transcript"),
    utt_id:   str = Form(default="", description="Optional utterance ID (defaults to wav filename stem)"),
):
    """
    Run the full pipeline and return results for **both** PreNet and PostNet in one call.

    Steps 1–4 run once (shared). Then PreNet and PostNet inference run in parallel.

    Returns:
        {
          "utt_id": "...",
          "transcript": "...",
          "prenet":  { "words_evaluated": N, "results": [...] },
          "postnet": { "words_evaluated": N, "results": [...] },
          "pipeline_timings": { ... }
        }
    """
    # ── Read uploaded files ────────────────────────────────────────────────
    wav_bytes = await wav_file.read()
    lab_bytes = await lab_file.read()

    wav_name  = wav_file.filename or "audio.wav"
    lab_name  = lab_file.filename or "transcript.lab"

    stem = (utt_id.strip() or Path(wav_name).stem).replace("/", "_").replace(" ", "_")

    transcript_words = _parse_lab(lab_bytes)
    if not transcript_words:
        raise HTTPException(status_code=422, detail="lab_file is empty or contains no words.")

    timings: dict[str, int] = {}

    async with httpx.AsyncClient() as client:
        # Step 1 — Prepare corpus
        t0 = _t()
        await _step1_prepare(client, wav_bytes, lab_bytes, wav_name, lab_name, stem)
        timings["api1_prepare_ms"] = _ms(t0)

        # Steps 2 & 3 — MFA + WavLM in parallel
        t0 = _t()
        mfa_csv_bytes, wavlm_npy_bytes = await asyncio.gather(
            _step2_align(client, wav_bytes, lab_bytes, wav_name, lab_name, stem),
            _step3_wavlm(client, wav_bytes, wav_name, stem),
        )
        timings["api2_mfa_ms"]   = _ms(t0)
        timings["api3_wavlm_ms"] = _ms(t0)

        # Step 4 — Syllable averaging (shared by both models)
        t0 = _t()
        syl_npy_bytes = await _step4_syllable(
            client, wav_bytes, lab_bytes, wav_name, lab_name,
            mfa_csv_bytes, wavlm_npy_bytes, stem,
        )
        timings["api4_syllable_ms"] = _ms(t0)

        # Short-circuit if all words are single-syllable
        _syl_check = np.load(io.BytesIO(syl_npy_bytes), allow_pickle=True).item()
        if len(_syl_check.get("token_index", [])) == 0:
            empty = {"words_evaluated": 0, "results": [],
                     "note": "All words are single-syllable; nothing to evaluate."}
            return JSONResponse(content={
                "utt_id":     stem,
                "transcript": " ".join(w.lower() for w in transcript_words),
                "prenet":     empty,
                "postnet":    empty,
                "pipeline_timings": timings,
            })

        # Steps 6 & 7 — PreNet and PostNet inference run in parallel
        # PreNet: raw syl_npy → API 7 flat DNN  (no padding needed)
        # PostNet: syl_npy → API 6 pad → API 7 TDNN
        async def _run_prenet():
            t = _t()
            result = await _step7_predict_prenet(client, syl_npy_bytes)
            timings["api7_prenet_ms"] = _ms(t)
            return result

        async def _run_postnet():
            t = _t()
            padded = await _step6_pad(client, syl_npy_bytes, stem)
            timings["api6_pad_ms"] = _ms(t)
            t = _t()
            result = await _step7_predict(client, padded)
            timings["api7_postnet_ms"] = _ms(t)
            return result

        prenet_json, postnet_json = await asyncio.gather(
            _run_prenet(),
            _run_postnet(),
        )

    # ── Assemble both result sets ──────────────────────────────────────────
    prenet_results  = _assemble_results(syl_npy_bytes, prenet_json,  transcript_words)
    postnet_results = _assemble_results(syl_npy_bytes, postnet_json, transcript_words)

    return JSONResponse(content={
        "utt_id":     stem,
        "transcript": " ".join(w.lower() for w in transcript_words),
        "prenet":  {"words_evaluated": len(prenet_results),  "results": prenet_results},
        "postnet": {"words_evaluated": len(postnet_results), "results": postnet_results},
        "pipeline_timings": timings,
    })
