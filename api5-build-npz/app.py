"""
API 5 — Build NPZ Datasets  (mirrors build_myst_train_npz.pbs + build_myst_test_npz.pbs)
==========================================================================================
Assembles flat NumPy compressed archives (.npz) from the per-utterance syllable .npy
files produced by API 4.  Exposes two separate build endpoints — one for training data
and one for test data — mirroring the original split logic (85 % train / 15 % test).

NPZ array schema (same as original PBS outputs):
  feature : (N, 1024) float32 — one row per syllable
  label   : (N,)       int8   — 0 = unstressed, 1 = secondary, 2 = primary stress
  w       : (N,)       int32  — monotonically increasing word ID starting at 2

Endpoints
---------
POST /build/train
    Body : multipart/form-data
        cmu_csv    : cmudict_no_one_syl.csv  (optional if pre-loaded via /upload-cmu)
        ratio      : (optional, default 0.85) fraction of unique file stems used for training
        utt_id     : (optional) include only this utterance (for single-utterance mode)

    Returns : application/octet-stream — myst_train.npz

POST /build/test
    Same as /build/train but uses the complementary (1 - ratio) portion of stems.
    Returns : application/octet-stream — myst_test.npz

POST /build/train/json
POST /build/test/json
    Same as above but save to disk and return JSON metadata.

POST /upload-cmu
    Cache the CMUdict CSV (same as API 4 — included here so this service is self-contained).

POST /upload-syllable
    Upload an individual syllable .npy (from API 4) to add it to the local store
    without relying on a shared volume.

GET /datasets
    List NPZ files that have been built and saved to disk.

GET /datasets/{name}
    Download a pre-built NPZ by filename (e.g. myst_train.npz).

GET /health
    Liveness probe.
"""

import csv
import io
import os
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SYLS_ROOT   = Path(os.getenv("SYLS_ROOT",   "/data/syls"))
NPZ_ROOT    = Path(os.getenv("NPZ_ROOT",    "/data/npz"))
CMU_CACHE   = Path(os.getenv("CMU_CACHE",   "/data/cmu/cmudict_no_one_syl.csv"))

app = FastAPI(
    title="API 5 — Build NPZ Datasets",
    description=(
        "Assembles train/test .npz archives from syllable .npy files. "
        "Mirrors build_myst_train_npz.pbs and build_myst_test_npz.pbs."
    ),
    version="1.0.0",
)

_cmu_stems: list[str] = []   # ordered unique file stems from CMUdict
_cmu_loaded = False


# ---------------------------------------------------------------------------
# CMUdict loading (stems only — to reproduce the 85/15 split)
# ---------------------------------------------------------------------------

def _load_cmu_stems(csv_path: Path) -> list[str]:
    """
    Return unique, ordered audio-file stems from the CMUdict CSV.
    The original scripts use the CMUdict CSV row order to derive the split.
    """
    seen: set[str] = set()
    stems: list[str] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            stem = (row.get("file", "") or row.get("filename", "") or "").strip()
            if stem and stem not in seen:
                seen.add(stem)
                stems.append(stem)
    return stems


@app.on_event("startup")
def _startup():
    global _cmu_stems, _cmu_loaded
    SYLS_ROOT.mkdir(parents=True, exist_ok=True)
    NPZ_ROOT.mkdir(parents=True, exist_ok=True)
    CMU_CACHE.parent.mkdir(parents=True, exist_ok=True)
    if CMU_CACHE.exists():
        _cmu_stems = _load_cmu_stems(CMU_CACHE)
        _cmu_loaded = True


# ---------------------------------------------------------------------------
# Core NPZ builder
# ---------------------------------------------------------------------------

def _build_npz(stem_filter: set[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Iterate over syllable .npy files, keep those whose stem is in stem_filter,
    concatenate features/labels, and assign monotonic word IDs.

    Returns (feature, label, w) arrays.
    """
    all_feat:  list[np.ndarray] = []
    all_label: list[np.ndarray] = []
    all_w:     list[np.ndarray] = []
    word_id = 2  # postnet expects IDs starting at 2

    npy_files = sorted(SYLS_ROOT.glob("*.npy"))
    if not npy_files:
        raise HTTPException(
            status_code=404,
            detail=(
                "No syllable .npy files found in /data/syls. "
                "Run API 4 first or upload via POST /upload-syllable."
            ),
        )

    for npy_path in npy_files:
        file_stem = npy_path.stem
        # Match on exact stem or on partial stem prefix (handles sub-corpus IDs)
        if stem_filter and file_stem not in stem_filter:
            continue

        try:
            data = np.load(str(npy_path), allow_pickle=True).item()
        except Exception:
            continue

        feat        = data.get("feature")
        label       = data.get("label")
        token_index = data.get("token_index")

        if feat is None or label is None or token_index is None:
            continue

        feat = feat.astype(np.float32)
        label = label.astype(np.int8)

        # Assign a unique word ID per token_index group within this utterance
        prev_tok = -1
        w_arr = np.empty(len(token_index), dtype=np.int32)
        for i, tok in enumerate(token_index):
            if tok != prev_tok:
                word_id += 1
                prev_tok = tok
            w_arr[i] = word_id

        all_feat.append(feat)
        all_label.append(label)
        all_w.append(w_arr)

    if not all_feat:
        raise HTTPException(
            status_code=404,
            detail="No matching syllable files found for the requested split.",
        )

    return (
        np.vstack(all_feat),
        np.concatenate(all_label),
        np.concatenate(all_w),
    )


def _split_stems(ratio: float) -> tuple[set[str], set[str]]:
    """Return (train_stems, test_stems) based on the loaded CMU stem list."""
    if _cmu_stems:
        n_train = int(len(_cmu_stems) * ratio)
        train = set(_cmu_stems[:n_train])
        test  = set(_cmu_stems[n_train:])
    else:
        # No CMU stem list — use all files for the requested split
        train = set()
        test  = set()
    return train, test


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "api5-build-npz",
        "cmu_loaded": _cmu_loaded,
        "cmu_stems": len(_cmu_stems),
    }


@app.post("/upload-cmu")
async def upload_cmu(cmu_file: UploadFile = File(...)):
    """Upload and cache the CMUdict CSV to enable the 85/15 split."""
    global _cmu_stems, _cmu_loaded
    data = await cmu_file.read()
    CMU_CACHE.parent.mkdir(parents=True, exist_ok=True)
    CMU_CACHE.write_bytes(data)
    _cmu_stems = _load_cmu_stems(CMU_CACHE)
    _cmu_loaded = True
    return {"message": "CMUdict stems loaded.", "stem_count": len(_cmu_stems)}


@app.post("/upload-syllable")
async def upload_syllable(
    npy_file: UploadFile = File(..., description="Syllable .npy from API 4"),
    utt_id:   str = Form(default="", description="Optional utterance ID"),
):
    """Upload a syllable .npy to the local syllable store."""
    stem = (utt_id.strip() or Path(npy_file.filename).stem).replace(" ", "_")
    SYLS_ROOT.mkdir(parents=True, exist_ok=True)
    dst = SYLS_ROOT / f"{stem}.npy"
    dst.write_bytes(await npy_file.read())
    return {"message": f"Saved {stem}.npy", "path": str(dst)}


@app.post("/build/train")
async def build_train(
    cmu_csv: Optional[UploadFile] = File(default=None),
    ratio:   float = Form(default=0.85),
    utt_id:  str   = Form(default=""),
):
    """Build myst_train.npz from the first ``ratio`` fraction of utterances."""
    global _cmu_stems, _cmu_loaded
    if cmu_csv is not None:
        data = await cmu_csv.read()
        CMU_CACHE.write_bytes(data)
        _cmu_stems = _load_cmu_stems(CMU_CACHE)
        _cmu_loaded = True

    if utt_id.strip():
        stem_filter = {utt_id.strip()}
    else:
        train_stems, _ = _split_stems(ratio)
        stem_filter = train_stems  # empty set → use all files

    feat, label, w = _build_npz(stem_filter)

    buf = io.BytesIO()
    np.savez_compressed(buf, feature=feat, label=label, w=w)
    buf.seek(0)
    return Response(
        content=buf.read(),
        media_type="application/octet-stream",
        headers={"Content-Disposition": 'attachment; filename="myst_train.npz"'},
    )


@app.post("/build/test")
async def build_test(
    cmu_csv: Optional[UploadFile] = File(default=None),
    ratio:   float = Form(default=0.85),
    utt_id:  str   = Form(default=""),
):
    """Build myst_test.npz from the last ``1 - ratio`` fraction of utterances."""
    global _cmu_stems, _cmu_loaded
    if cmu_csv is not None:
        data = await cmu_csv.read()
        CMU_CACHE.write_bytes(data)
        _cmu_stems = _load_cmu_stems(CMU_CACHE)
        _cmu_loaded = True

    if utt_id.strip():
        stem_filter = {utt_id.strip()}
    else:
        _, test_stems = _split_stems(ratio)
        stem_filter = test_stems

    feat, label, w = _build_npz(stem_filter)

    buf = io.BytesIO()
    np.savez_compressed(buf, feature=feat, label=label, w=w)
    buf.seek(0)
    return Response(
        content=buf.read(),
        media_type="application/octet-stream",
        headers={"Content-Disposition": 'attachment; filename="myst_test.npz"'},
    )


@app.post("/build/train/json")
async def build_train_json(
    cmu_csv: Optional[UploadFile] = File(default=None),
    ratio:   float = Form(default=0.85),
    utt_id:  str   = Form(default=""),
):
    """Build myst_train.npz, save to disk, return JSON metadata."""
    global _cmu_stems, _cmu_loaded
    if cmu_csv is not None:
        data = await cmu_csv.read()
        CMU_CACHE.write_bytes(data)
        _cmu_stems = _load_cmu_stems(CMU_CACHE)
        _cmu_loaded = True

    if utt_id.strip():
        stem_filter = {utt_id.strip()}
    else:
        train_stems, _ = _split_stems(ratio)
        stem_filter = train_stems

    feat, label, w = _build_npz(stem_filter)
    out_path = NPZ_ROOT / "myst_train.npz"
    np.savez_compressed(str(out_path), feature=feat, label=label, w=w)
    return JSONResponse(content={
        "path":          str(out_path),
        "n_syllables":   int(feat.shape[0]),
        "feature_shape": list(feat.shape),
        "label_shape":   list(label.shape),
        "w_shape":       list(w.shape),
    })


@app.post("/build/test/json")
async def build_test_json(
    cmu_csv: Optional[UploadFile] = File(default=None),
    ratio:   float = Form(default=0.85),
    utt_id:  str   = Form(default=""),
):
    """Build myst_test.npz, save to disk, return JSON metadata."""
    global _cmu_stems, _cmu_loaded
    if cmu_csv is not None:
        data = await cmu_csv.read()
        CMU_CACHE.write_bytes(data)
        _cmu_stems = _load_cmu_stems(CMU_CACHE)
        _cmu_loaded = True

    if utt_id.strip():
        stem_filter = {utt_id.strip()}
    else:
        _, test_stems = _split_stems(ratio)
        stem_filter = test_stems

    feat, label, w = _build_npz(stem_filter)
    out_path = NPZ_ROOT / "myst_test.npz"
    np.savez_compressed(str(out_path), feature=feat, label=label, w=w)
    return JSONResponse(content={
        "path":          str(out_path),
        "n_syllables":   int(feat.shape[0]),
        "feature_shape": list(feat.shape),
        "label_shape":   list(label.shape),
        "w_shape":       list(w.shape),
    })


@app.get("/datasets")
def list_datasets():
    """List all .npz files saved on disk."""
    if not NPZ_ROOT.exists():
        return {"datasets": [], "count": 0}
    files = sorted(p.name for p in NPZ_ROOT.glob("*.npz"))
    return {"datasets": files, "count": len(files)}


@app.get("/datasets/{name}")
def get_dataset(name: str):
    """Download a pre-built .npz by filename."""
    path = NPZ_ROOT / name
    if not path.exists() or path.suffix != ".npz":
        raise HTTPException(status_code=404, detail=f"Dataset '{name}' not found.")
    return FileResponse(
        path=str(path),
        media_type="application/octet-stream",
        filename=name,
    )
