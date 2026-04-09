"""
API 4 — Syllable Feature Averaging  (mirrors average_syllable_features_grouped.pbs)
====================================================================================
For a single utterance, combines:
  • MFA phone-timing CSV  (from API 2)
  • WavLM frame-level .npy  (from API 3)
  • CMUdict — looked up live via NLTK (no CSV upload needed)

to produce one per-utterance syllable .npy payload with keys:
  feature           — (N_syl, 1024) float32
  syllable_phonemes — list of space-joined phone strings per syllable
  token_index       — (N_syl,) int array — which word each syllable belongs to
  label             — (N_syl,) int8 — 0 = unstressed, 1 = secondary stress,
                                       2 = primary stress

Endpoints
---------
POST /average
    Body : multipart/form-data
        wav_file    : original .wav (used only to derive the utterance ID)
        lab_file    : .lab transcript
        mfa_csv     : phone-timing CSV from API 2
        wavlm_npy   : WavLM frame .npy from API 3
        utt_id      : (optional) utterance ID
        frame_shift : (optional, default 0.02) seconds per WavLM frame

    Returns : application/octet-stream — the syllable .npy file

POST /average/json
    Same inputs, saves the .npy to /data/syls/{utt_id}.npy, returns JSON metadata.

GET /syllables/{utt_id}
    Download a pre-computed syllable .npy.

GET /utterances
    List all processed utterance IDs.

GET /health
    Liveness probe.
"""

import csv
import io
import os
import re
from pathlib import Path

import nltk
from nltk.corpus import cmudict as _nltk_cmu_corpus
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SYLS_ROOT           = Path(os.getenv("SYLS_ROOT", "/data/syls"))
FRAME_SHIFT_DEFAULT = 0.02   # seconds — matches WavLM-Large 20 ms hop

app = FastAPI(
    title="API 4 — Syllable Feature Averaging",
    description=(
        "Averages WavLM frame features within MFA-aligned syllable boundaries. "
        "Uses NLTK CMUdict directly for syllabification. "
        "Mirrors average_syllable_features_grouped.pbs."
    ),
    version="2.0.0",
)

# In-memory CMUdict: word (lower) → list of pronunciation variants, each a list of phones
_CMUDICT: dict[str, list[list[str]]] = {}

# ---------------------------------------------------------------------------
# ARPABET vowel nuclei — used for syllabification
# ---------------------------------------------------------------------------
_VOWEL_BASES = {
    "AA", "AE", "AH", "AO", "AW", "AY",
    "EH", "ER", "EY",
    "IH", "IY",
    "OW", "OY",
    "UH", "UW",
}


def _is_vowel(phone: str) -> bool:
    """Return True if *phone* (with or without stress digit) is a vowel."""
    return phone.rstrip("012") in _VOWEL_BASES


def _syllabify(phones: list[str]) -> list[list[str]]:
    """
    Split a flat ARPABET phone list into syllables using an onset-based rule:
    each syllable begins on the consonants immediately before its vowel nucleus.

    Example:  ['B', 'IY2', 'AO1', 'N', 'D']  →  [['B','IY2'], ['AO1','N','D']]
    """
    syllables: list[list[str]] = []
    current:   list[str]       = []

    for ph in phones:
        if _is_vowel(ph):
            # Pull trailing consonants from the previous syllable into the onset
            onset: list[str] = []
            while current and not _is_vowel(current[-1]):
                onset.insert(0, current.pop())
            if current:
                syllables.append(current)
            current = onset + [ph]
        else:
            current.append(ph)

    if current:
        syllables.append(current)

    return syllables if syllables else [phones]


def _get_syllables(word: str) -> list[list[str]] | None:
    """
    Look up *word* in CMUdict and return its syllabified pronunciation
    (first variant only), or None if the word is not in the dictionary.
    """
    entry = _CMUDICT.get(word.lower())
    if entry is None:
        return None
    # entry is a list of pronunciation variants; take the first
    return _syllabify(entry[0])


# ---------------------------------------------------------------------------
# Startup — load CMUdict once into memory
# ---------------------------------------------------------------------------
@app.on_event("startup")
def _startup():
    global _CMUDICT
    SYLS_ROOT.mkdir(parents=True, exist_ok=True)
    nltk.data.path.append("/root/nltk_data")  # default download path in container
    raw = _nltk_cmu_corpus.dict()              # { 'word': [['P','R','OW0',...], ...], ... }
    _CMUDICT = raw
    print(f"[API 4] CMUdict loaded: {len(_CMUDICT):,} entries", flush=True)


# ---------------------------------------------------------------------------
# Core averaging logic
# ---------------------------------------------------------------------------

def _read_mfa_csv(csv_bytes: bytes) -> list[dict]:
    """Return only the phone rows from an MFA alignment CSV."""
    rows = []
    reader = csv.DictReader(io.StringIO(csv_bytes.decode("utf-8")))
    for row in reader:
        if row.get("Type", "").strip().lower() == "phones":
            rows.append({
                "begin": float(row["Begin"]),
                "end":   float(row["End"]),
                "phone": row["Label"].strip(),
            })
    return rows


def _read_lab_text(lab_bytes: bytes) -> list[str]:
    """Return the transcript words (lower-cased) from a .lab file."""
    text = lab_bytes.decode("utf-8").strip()
    # Normalise punctuation
    text = re.sub(r"[^\w\s']", " ", text)
    return [w.lower() for w in text.split() if w]


def _phone_to_frame(time_sec: float, frame_shift: float) -> int:
    return int(time_sec / frame_shift)


def _stress_from_phones(phones: list[str]) -> int:
    """Return stress label: 0 = unstressed, 1 = stressed, 2 = primary stress."""
    for ph in phones:
        if ph.endswith("1"):
            return 2   # primary stress
        if ph.endswith("2"):
            return 1   # secondary stress
    return 0           # unstressed


def _average_syllable_features(
    mfa_phones: list[dict],
    wavlm_feats: np.ndarray,
    transcript_words: list[str],
    frame_shift: float,
) -> dict:
    """
    Core logic — mirrors average_syllable_features_grouped.py.

    For each word in the transcript, looks up CMUdict live, syllabifies the
    pronunciation, and averages the WavLM frames that fall within each
    syllable's MFA-aligned phone boundaries.

    Returns a dict with: feature, syllable_phonemes, token_index, label
    """
    T        = wavlm_feats.shape[0]
    feat_dim = wavlm_feats.shape[1] if wavlm_feats.ndim == 2 else 1024

    all_features:  list[np.ndarray] = []
    all_phonemes:  list[str]        = []
    all_token_idx: list[int]        = []
    all_labels:    list[int]        = []

    # phone_cursor advances through mfa_phones as we consume phones per syllable
    phone_cursor = 0

    for word_idx, word in enumerate(transcript_words):
        syllables = _get_syllables(word)
        if syllables is None:
            # Word not in CMUdict — skip
            continue

        if len(syllables) < 2:
            # Single-syllable word — skip but still advance the phone cursor
            total_phones = sum(len(s) for s in syllables)
            phone_cursor += total_phones
            continue

        for syl_phones in syllables:
            n_ph = len(syl_phones)
            syl_mfa = mfa_phones[phone_cursor: phone_cursor + n_ph]
            phone_cursor += n_ph

            if not syl_mfa:
                continue

            begin_frame = _phone_to_frame(syl_mfa[0]["begin"], frame_shift)
            end_frame   = _phone_to_frame(syl_mfa[-1]["end"],  frame_shift)

            # Clip to valid frame range
            begin_frame = max(0, min(begin_frame, T - 1))
            end_frame   = max(begin_frame + 1, min(end_frame, T))

            syl_feats = wavlm_feats[begin_frame:end_frame]  # (k, 1024)
            if syl_feats.shape[0] == 0:
                continue

            avg_feat = syl_feats.mean(axis=0).astype(np.float32)
            label    = _stress_from_phones(syl_phones)

            all_features.append(avg_feat)
            all_phonemes.append(" ".join(syl_phones))
            all_token_idx.append(word_idx)
            all_labels.append(label)

    if not all_features:
        # All words are single-syllable or not found in CMUdict — return empty payload
        return {
            "feature":           np.zeros((0, feat_dim), dtype=np.float32),
            "syllable_phonemes": [],
            "token_index":       np.array([], dtype=np.int32),
            "label":             np.array([], dtype=np.int8),
        }

    return {
        "feature":           np.stack(all_features, axis=0),
        "syllable_phonemes": all_phonemes,
        "token_index":       np.array(all_token_idx, dtype=np.int32),
        "label":             np.array(all_labels,    dtype=np.int8),
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status":          "ok",
        "service":         "api4-syllable-average",
        "cmudict_entries": len(_CMUDICT),
        "cmudict_loaded":  len(_CMUDICT) > 0,
    }


async def _run_average(
    wav_file:    UploadFile,
    lab_file:    UploadFile,
    mfa_csv:     UploadFile,
    wavlm_npy:   UploadFile,
    utt_id:      str,
    frame_shift: float,
) -> tuple[str, dict]:
    """Shared core logic for both /average endpoints."""
    stem = (utt_id.strip() or Path(wav_file.filename).stem).replace(" ", "_")

    lab_bytes = await lab_file.read()
    mfa_bytes = await mfa_csv.read()
    npy_bytes = await wavlm_npy.read()

    transcript_words = _read_lab_text(lab_bytes)
    mfa_phones       = _read_mfa_csv(mfa_bytes)

    wavlm_feats = np.load(io.BytesIO(npy_bytes)).astype(np.float32)

    payload = _average_syllable_features(
        mfa_phones, wavlm_feats, transcript_words, frame_shift
    )

    return stem, payload


@app.post("/average")
async def average(
    wav_file:    UploadFile = File(..., description=".wav file (for utterance ID)"),
    lab_file:    UploadFile = File(..., description=".lab transcript"),
    mfa_csv:     UploadFile = File(..., description="Phone-timing CSV from API 2"),
    wavlm_npy:   UploadFile = File(..., description="WavLM .npy from API 3"),
    utt_id:      str        = Form(default="",                  description="Optional utterance ID"),
    frame_shift: float      = Form(default=FRAME_SHIFT_DEFAULT, description="Seconds per WavLM frame"),
):
    """Average WavLM frames into syllable vectors and **stream** the .npy back."""
    stem, payload = await _run_average(
        wav_file, lab_file, mfa_csv, wavlm_npy, utt_id, frame_shift
    )
    buf = io.BytesIO()
    np.save(buf, payload)
    buf.seek(0)
    return Response(
        content=buf.read(),
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": f'attachment; filename="{stem}.npy"',
            "X-Utt-Id": stem,
            "X-N-Syllables": str(payload["feature"].shape[0]),
        },
    )


@app.post("/average/json")
async def average_json(
    wav_file:    UploadFile = File(..., description=".wav file (for utterance ID)"),
    lab_file:    UploadFile = File(..., description=".lab transcript"),
    mfa_csv:     UploadFile = File(..., description="Phone-timing CSV from API 2"),
    wavlm_npy:   UploadFile = File(..., description="WavLM .npy from API 3"),
    utt_id:      str        = Form(default="",                  description="Optional utterance ID"),
    frame_shift: float      = Form(default=FRAME_SHIFT_DEFAULT, description="Seconds per WavLM frame"),
):
    """Average WavLM frames into syllable vectors, save .npy to disk, return JSON metadata."""
    stem, payload = await _run_average(
        wav_file, lab_file, mfa_csv, wavlm_npy, utt_id, frame_shift
    )
    SYLS_ROOT.mkdir(parents=True, exist_ok=True)
    npy_path = SYLS_ROOT / f"{stem}.npy"
    np.save(str(npy_path), payload)

    feat = payload["feature"]
    return JSONResponse(content={
        "utt_id":            stem,
        "npy_path":          str(npy_path),
        "n_syllables":       int(feat.shape[0]),
        "feature_dim":       int(feat.shape[1]) if feat.ndim == 2 and feat.shape[0] > 0 else 1024,
        "token_index":       payload["token_index"].tolist(),
        "labels":            payload["label"].tolist(),
        "syllable_phonemes": payload["syllable_phonemes"],
    })


@app.get("/syllables/{utt_id}")
def get_syllables(utt_id: str):
    """Download a pre-computed syllable .npy for a given utterance."""
    npy_path = SYLS_ROOT / f"{utt_id}.npy"
    if not npy_path.exists():
        raise HTTPException(status_code=404, detail=f"No syllable file found for '{utt_id}'.")
    return FileResponse(
        path=str(npy_path),
        media_type="application/octet-stream",
        filename=f"{utt_id}.npy",
    )


@app.get("/utterances")
def list_utterances():
    """Return all utterance IDs whose syllable features have been computed."""
    if not SYLS_ROOT.exists():
        return {"utterances": [], "count": 0}
    ids = sorted(p.stem for p in SYLS_ROOT.glob("*.npy"))
    return {"utterances": ids, "count": len(ids)}
