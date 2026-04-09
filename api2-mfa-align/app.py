"""
API 2 — MFA Align  (mirrors mfa_myst_align_full.pbs)
=====================================================
Runs Montreal Forced Aligner on the corpus prepared by API 1 and produces
per-utterance phone-timing CSV files in the layout:

    /data/mfa/{stem}/{stem}.csv

Endpoints
---------
POST /align
    Body : multipart/form-data
        wav_file      : .wav audio file (16 kHz, already resampled)
        lab_file      : .lab transcript file (UTF-8 plain text)
        utt_id        : (optional) utterance ID, defaults to wav filename stem
        dict_model    : (optional, default "english_us_arpa") MFA dictionary model
        acoustic_model: (optional, default "english_us_arpa") MFA acoustic model

    Returns : JSON
        {
          "utt_id":    "myst_002004_...",
          "csv_path":  "/data/mfa/myst_002004_.../myst_002004_....csv",
          "csv_rows":  45,
          "phone_rows": 32
        }

GET /alignments/{utt_id}
    Download the alignment CSV for a given utterance as text/csv.

GET /utterances
    List all utterance IDs that have been aligned.

GET /health
    Liveness probe.

Notes
-----
MFA is installed inside the container via conda.  The endpoint writes the
.wav + .lab pair to a temp corpus dir, runs `mfa align`, then flattens the
speaker subdirectory layout that MFA emits into the required {stem}/{stem}.csv
structure, mirroring the flattening step in the original PBS script.
"""

import os
import shutil
import subprocess
import tempfile
import csv
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MFA_OUT_ROOT = Path(os.getenv("MFA_OUT_ROOT", "/data/mfa"))
MFA_TMP_ROOT = Path(os.getenv("MFA_TMP_ROOT", "/tmp/mfa_tmp"))
DICT_MODEL_DEFAULT = "english_us_arpa"
ACOUSTIC_MODEL_DEFAULT = "english_us_arpa"

app = FastAPI(
    title="API 2 — MFA Align",
    description=(
        "Runs Montreal Forced Aligner on a single .wav + .lab utterance pair "
        "and returns a phone-timing CSV. Mirrors mfa_myst_align_full.pbs."
    ),
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_mfa_align(
    corpus_dir: Path,
    out_dir: Path,
    tmp_dir: Path,
    dict_model: str,
    acoustic_model: str,
) -> subprocess.CompletedProcess:
    """Invoke `mfa align` as a subprocess and return the CompletedProcess."""
    cmd = [
        "mfa", "align",
        str(corpus_dir),
        dict_model,
        acoustic_model,
        str(out_dir),
        "-j", "4",
        "-t", str(tmp_dir),
        "--output_format", "csv",
        "--clean",
        "--final_clean",
        "--overwrite",
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=900,  # 15 minute max per utterance
    )
    return result


def _flatten_mfa_output(out_dir: Path) -> list[Path]:
    """
    MFA writes {speaker}/{stem}.csv inside out_dir.
    The out_dir is already named after the utterance, so we want the CSV
    directly at out_dir/{stem}.csv (one level up from the speaker subdir).
    Returns a list of final CSV paths.
    """
    final_paths: list[Path] = []

    for csv_path in list(out_dir.rglob("*.csv")):
        if csv_path.name == "alignment_analysis.csv":
            continue
        stem = csv_path.stem
        # Target is directly inside out_dir (not in a subdirectory)
        target_file = out_dir / csv_path.name
        if csv_path == target_file:
            final_paths.append(target_file)
            continue
        shutil.move(str(csv_path), str(target_file))
        final_paths.append(target_file)

    # Remove empty speaker subdirectories left behind
    for d in sorted(out_dir.iterdir()):
        if d.is_dir():
            try:
                d.rmdir()
            except OSError:
                pass

    return final_paths


def _count_phone_rows(csv_path: Path) -> int:
    """Count rows with Type == 'phones' in an MFA alignment CSV."""
    count = 0
    try:
        with csv_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("Type", "").strip().lower() == "phones":
                    count += 1
    except Exception:
        pass
    return count


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "service": "api2-mfa-align"}


@app.post("/align")
async def align(
    wav_file: UploadFile = File(..., description=".wav audio file (16 kHz)"),
    lab_file: UploadFile = File(..., description=".lab transcript file"),
    utt_id: str = Form(default="", description="Optional utterance ID"),
    dict_model: str = Form(default=DICT_MODEL_DEFAULT, description="MFA dictionary model"),
    acoustic_model: str = Form(default=ACOUSTIC_MODEL_DEFAULT, description="MFA acoustic model"),
):
    """
    Align a single .wav + .lab pair with MFA and return phone timings as CSV.

    The output CSV has columns: ``Begin, End, Label, Type, Speaker``  
    Rows with ``Type == phones`` carry the phone-level boundaries used by API 4.
    """
    if not wav_file.filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="wav_file must be a .wav file.")
    if not lab_file.filename.lower().endswith(".lab"):
        raise HTTPException(status_code=400, detail="lab_file must be a .lab file.")

    stem = utt_id.strip() or Path(wav_file.filename).stem
    stem = stem.replace("/", "_").replace("\\", "_").replace(" ", "_")

    with tempfile.TemporaryDirectory(prefix="mfa_corpus_") as tmp_corpus:
        corpus_dir = Path(tmp_corpus) / stem
        corpus_dir.mkdir()

        (corpus_dir / f"{stem}.wav").write_bytes(await wav_file.read())
        (corpus_dir / f"{stem}.lab").write_bytes(await lab_file.read())

        utt_out_dir = MFA_OUT_ROOT / stem
        utt_out_dir.mkdir(parents=True, exist_ok=True)

        tmp_dir = MFA_TMP_ROOT / stem
        tmp_dir.mkdir(parents=True, exist_ok=True)

        result = _run_mfa_align(
            corpus_dir=Path(tmp_corpus),
            out_dir=utt_out_dir,
            tmp_dir=tmp_dir,
            dict_model=dict_model,
            acoustic_model=acoustic_model,
        )

        if result.returncode != 0:
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "MFA alignment failed.",
                    "stdout": result.stdout[-3000:],
                    "stderr": result.stderr[-3000:],
                },
            )

        final_csvs = _flatten_mfa_output(utt_out_dir)
        target_csv = utt_out_dir / f"{stem}.csv"

        if not target_csv.exists():
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "MFA alignment produced no CSV for this utterance.",
                    "stdout": result.stdout[-2000:],
                    "stderr": result.stderr[-2000:],
                },
            )

        total_rows = sum(1 for _ in target_csv.open())  # includes header
        phone_rows = _count_phone_rows(target_csv)

    return JSONResponse(
        content={
            "utt_id": stem,
            "csv_path": str(target_csv),
            "total_rows": total_rows,
            "phone_rows": phone_rows,
            "dict_model": dict_model,
            "acoustic_model": acoustic_model,
        }
    )


@app.get("/alignments/{utt_id}")
def get_alignment(utt_id: str):
    """Download the alignment CSV for a given utterance."""
    csv_path = MFA_OUT_ROOT / utt_id / f"{utt_id}.csv"
    if not csv_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No alignment CSV found for utterance '{utt_id}'.",
        )
    return FileResponse(
        path=str(csv_path),
        media_type="text/csv",
        filename=f"{utt_id}.csv",
    )


@app.get("/utterances")
def list_utterances():
    """Return all utterance IDs that have been aligned."""
    if not MFA_OUT_ROOT.exists():
        return {"utterances": [], "count": 0}

    utt_ids = sorted(
        d.name
        for d in MFA_OUT_ROOT.iterdir()
        if d.is_dir() and (d / f"{d.name}.csv").exists()
    )
    return {"utterances": utt_ids, "count": len(utt_ids)}
