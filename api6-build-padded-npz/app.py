"""
API 6 — Build Padded NPZ  (mirrors build_myst_train_npz_pad9.py + build_myst_test_npz_pad9.py)
================================================================================================
Pads every word in a dataset to a fixed number of syllable slots (default 9)
by appending -1-filled rows, producing the fixed-length sequences required by
the TDNN postnet.

Two modes are exposed:
  • /build/train-pad  — reads syllable .npy files directly (like build_myst_train_npz_pad9.py)
  • /build/test-pad   — reads an already-built myst_test.npz (like build_myst_test_npz_pad9.py)

Both return the padded .npz either streamed or saved to disk (*/json variant).

Output NPZ schema
-----------------
train-pad output:
  feature  : (N_words * pad_len, 1024) float32
  label    : (N_words * pad_len,)      int8    (-1 for padding rows)

test-pad output:
  feature  : (N_words * pad_len, 1024) float32
  label    : (N_words * pad_len,)      int8    (-1 for padding rows)
  w        : (N_orig_syllables,)       int32   (copied as-is from input)
  test_ind : (N_pad_rows,)             int64   (indices of padding rows in output)
  word_len : scalar array [pad_len]

Endpoints
---------
POST /build/train-pad
    Body : multipart/form-data
        pad_len    : (optional, default 9)
        utt_id     : (optional) restrict to a single utterance
    Returns : application/octet-stream — padded train .npz

POST /build/test-pad
    Body : multipart/form-data
        test_npz   : myst_test.npz from API 5  (or pre-saved version)
        pad_len    : (optional, default 9)
    Returns : application/octet-stream — padded test .npz

POST /build/train-pad/json
POST /build/test-pad/json
    Same but save to /data/npz and return JSON metadata.

POST /upload-syllable
    Upload a syllable .npy to the local store.

POST /upload-test-npz
    Upload a pre-built myst_test.npz.

GET /datasets/{name}
    Download a pre-built padded .npz.

GET /health
    Liveness probe.
"""

import io
import os
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SYLS_ROOT = Path(os.getenv("SYLS_ROOT", "/data/syls"))
NPZ_ROOT  = Path(os.getenv("NPZ_ROOT",  "/data/npz"))
PAD_LEN_DEFAULT = 9

app = FastAPI(
    title="API 6 — Build Padded NPZ",
    description=(
        "Zero-pads every word to a fixed syllable count (default 9) for TDNN postnet. "
        "Mirrors build_myst_train_npz_pad9.py and build_myst_test_npz_pad9.py."
    ),
    version="1.0.0",
)


@app.on_event("startup")
def _startup():
    SYLS_ROOT.mkdir(parents=True, exist_ok=True)
    NPZ_ROOT.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers — train-pad (reads syllable .npy files)
# ---------------------------------------------------------------------------

def _build_train_pad(stem_filter: Optional[str], pad_len: int) -> dict[str, np.ndarray]:
    """
    Mirrors build_myst_train_npz_pad9.py:
      - loads grouped .npy files
      - groups syllables by token_index
      - pads each word to pad_len slots with -1 features/-1 labels
    Also records test_ind (padding row indices) and w (word ID per real syllable)
    so that API 7 can strip padding correctly.
    Returns dict with keys: feature, label, w, test_ind, word_len
    """
    npy_files = sorted(SYLS_ROOT.glob("*.npy"))
    if not npy_files:
        raise HTTPException(
            status_code=404,
            detail="No syllable .npy files found. Run API 4 or upload via /upload-syllable.",
        )

    padded_features: list[np.ndarray] = []
    padded_labels:   list[np.ndarray] = []
    word_ids_real:   list[int] = []   # word id per REAL syllable
    test_ind:        list[int] = []   # absolute row indices of padding slots
    processed = 0
    skipped   = 0
    global_word_id = 0   # monotonically increasing word counter across utterances

    for npy_path in npy_files:
        if stem_filter and npy_path.stem != stem_filter:
            continue

        try:
            data = np.load(str(npy_path), allow_pickle=True).item()
        except Exception:
            skipped += 1
            continue

        features    = data.get("feature")
        token_index = data.get("token_index")
        labels      = data.get("label")

        if features is None or token_index is None:
            skipped += 1
            continue

        n_syllables = features.shape[0]
        if n_syllables != len(token_index):
            skipped += 1
            continue

        if labels is None:
            labels = np.zeros(n_syllables, dtype=np.int8)

        # Group syllables by token_index (each unique token = one word)
        words: dict[int, list[int]] = defaultdict(list)
        for i, tid in enumerate(token_index):
            words[int(tid)].append(i)

        current_row = len(padded_features) * pad_len  # absolute row offset

        for tid in sorted(words.keys()):
            indices  = words[tid]
            n_syl    = len(indices)
            wf       = features[indices].astype(np.float32)   # (n_syl, 1024)
            wl       = labels[indices].astype(np.int8)         # (n_syl,)

            # Record real syllable word IDs
            for _ in indices:
                word_ids_real.append(global_word_id)

            real_count = min(n_syl, pad_len)

            if n_syl < pad_len:
                pad_count = pad_len - n_syl
                # Record padding row indices
                for p in range(n_syl, pad_len):
                    test_ind.append(current_row + p)
                wf = np.vstack([wf, np.full((pad_count, wf.shape[1]), -1.0, dtype=np.float32)])
                wl = np.concatenate([wl, np.full(pad_count, -1, dtype=np.int8)])
            elif n_syl > pad_len:
                wf = wf[:pad_len]
                wl = wl[:pad_len]
                # No padding rows when word is truncated

            padded_features.append(wf)
            padded_labels.append(wl)
            current_row += pad_len
            global_word_id += 1

        processed += 1

    if not padded_features:
        raise HTTPException(status_code=404, detail="No data after padding.")

    all_features = np.vstack(padded_features)                 # (N_words * pad_len, 1024)
    all_labels   = np.concatenate(padded_labels)              # (N_words * pad_len,)
    w_arr        = np.asarray(word_ids_real, dtype=np.int32)  # (N_real_syl,)
    ti_arr       = np.asarray(test_ind, dtype=np.int64)       # (N_pad_rows,)
    wl_arr       = np.array([pad_len], dtype=np.int32)        # scalar

    return {
        "feature":  all_features,
        "label":    all_labels,
        "w":        w_arr,
        "test_ind": ti_arr,
        "word_len": wl_arr,
    }


# ---------------------------------------------------------------------------
# Helpers — test-pad (reads an existing myst_test.npz)
# ---------------------------------------------------------------------------

def _build_test_pad(
    npz_bytes: bytes, pad_len: int
) -> dict[str, np.ndarray]:
    """
    Mirrors build_myst_test_npz_pad9.py:
      - loads myst_test.npz (feature, label, w)
      - pads each word group to pad_len
      - records padding row indices in test_ind
    Returns dict with keys: feature, label, w, test_ind, word_len
    """
    z    = np.load(io.BytesIO(npz_bytes), allow_pickle=True)
    feat = z["feature"].astype(np.float32)
    lab  = z["label"].astype(np.int8)
    w    = z["w"].astype(np.int32)

    unique_w, first_occ = np.unique(w, return_index=True)
    order    = np.argsort(first_occ)
    unique_w = unique_w[order]
    n_words  = len(unique_w)

    N_out    = n_words * pad_len
    out_feat = np.full((N_out, feat.shape[1]), -1, dtype=np.float32)
    out_lab  = np.full((N_out,), -1, dtype=np.int8)
    test_ind: list[int] = []

    pos = 0
    for wid in unique_w:
        syl_mask = np.where(w == wid)[0]
        n_syl    = len(syl_mask)
        if n_syl > pad_len:
            syl_mask = syl_mask[:pad_len]
            n_syl    = pad_len
        out_feat[pos:pos + n_syl] = feat[syl_mask]
        out_lab [pos:pos + n_syl] = lab [syl_mask]
        pos += n_syl
        n_pad = pad_len - n_syl
        if n_pad > 0:
            test_ind.extend(range(pos, pos + n_pad))
            pos += n_pad

    test_ind_arr = np.array(test_ind, dtype=np.int64)

    # Validation
    if pos != N_out:
        raise HTTPException(
            status_code=500,
            detail=f"Padding row count mismatch: {pos} != {N_out}",
        )

    return {
        "feature":  out_feat,
        "label":    out_lab,
        "w":        w,
        "test_ind": test_ind_arr,
        "word_len": np.array([pad_len]),
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "service": "api6-build-padded-npz"}


@app.post("/upload-syllable")
async def upload_syllable(
    npy_file: UploadFile = File(...),
    utt_id:   str        = Form(default=""),
):
    """Upload a syllable .npy produced by API 4."""
    stem = (utt_id.strip() or Path(npy_file.filename).stem).replace(" ", "_")
    dst  = SYLS_ROOT / f"{stem}.npy"
    dst.write_bytes(await npy_file.read())
    return {"message": f"Saved {stem}.npy", "path": str(dst)}


@app.post("/upload-test-npz")
async def upload_test_npz(npz_file: UploadFile = File(...)):
    """Upload a pre-built myst_test.npz to use as input for /build/test-pad."""
    dst = NPZ_ROOT / "myst_test.npz"
    dst.write_bytes(await npz_file.read())
    return {"message": "myst_test.npz saved.", "path": str(dst)}


@app.post("/build/train-pad")
async def build_train_pad(
    pad_len: int = Form(default=PAD_LEN_DEFAULT, description="Syllables per word (default 9)"),
    utt_id:  str = Form(default="", description="Restrict to a single utterance ID"),
):
    """Pad syllable .npy files from /data/syls into a train padded .npz (streamed)."""
    stem_filter = utt_id.strip() or None
    result = _build_train_pad(stem_filter, pad_len)
    buf = io.BytesIO()
    np.savez_compressed(buf, **result)
    buf.seek(0)
    return Response(
        content=buf.read(),
        media_type="application/octet-stream",
        headers={"Content-Disposition": 'attachment; filename="myst_train_pad.npz"'},
    )


@app.post("/build/test-pad")
async def build_test_pad(
    test_npz: Optional[UploadFile] = File(
        default=None,
        description="myst_test.npz from API 5 (optional if already uploaded)",
    ),
    pad_len: int = Form(default=PAD_LEN_DEFAULT),
):
    """Pad a myst_test.npz into a TDNN-ready padded .npz (streamed)."""
    if test_npz is not None:
        npz_bytes = await test_npz.read()
    else:
        cached = NPZ_ROOT / "myst_test.npz"
        if not cached.exists():
            raise HTTPException(
                status_code=400,
                detail="No myst_test.npz found. Upload via POST /upload-test-npz or provide test_npz.",
            )
        npz_bytes = cached.read_bytes()

    result = _build_test_pad(npz_bytes, pad_len)
    buf = io.BytesIO()
    np.savez_compressed(buf, **result)
    buf.seek(0)
    return Response(
        content=buf.read(),
        media_type="application/octet-stream",
        headers={"Content-Disposition": 'attachment; filename="myst_test_pad.npz"'},
    )


@app.post("/build/train-pad/json")
async def build_train_pad_json(
    pad_len: int = Form(default=PAD_LEN_DEFAULT),
    utt_id:  str = Form(default=""),
):
    """Pad syllable .npy files, save to disk, return JSON metadata."""
    stem_filter = utt_id.strip() or None
    result = _build_train_pad(stem_filter, pad_len)
    out_path = NPZ_ROOT / "myst_train_pad.npz"
    np.savez_compressed(str(out_path), **result)
    feature = result["feature"]
    return JSONResponse(content={
        "path":          str(out_path),
        "total_rows":    int(feature.shape[0]),
        "pad_len":       pad_len,
        "pad_rows":      int(result["test_ind"].shape[0]),
        "real_syllables": int(result["w"].shape[0]),
        "feature_shape": list(feature.shape),
        "label_shape":   list(result["label"].shape),
    })


@app.post("/build/test-pad/json")
async def build_test_pad_json(
    test_npz: Optional[UploadFile] = File(default=None),
    pad_len:  int = Form(default=PAD_LEN_DEFAULT),
):
    """Pad a myst_test.npz, save to disk, return JSON metadata."""
    if test_npz is not None:
        npz_bytes = await test_npz.read()
    else:
        cached = NPZ_ROOT / "myst_test.npz"
        if not cached.exists():
            raise HTTPException(status_code=400, detail="No myst_test.npz found.")
        npz_bytes = cached.read_bytes()

    result   = _build_test_pad(npz_bytes, pad_len)
    out_path = NPZ_ROOT / "myst_test_pad.npz"
    np.savez_compressed(str(out_path), **result)
    return JSONResponse(content={
        "path":          str(out_path),
        "total_rows":    int(result["feature"].shape[0]),
        "pad_len":       pad_len,
        "pad_rows":      int(result["test_ind"].shape[0]),
        "real_syllables": int(result["w"].shape[0]),
        "feature_shape": list(result["feature"].shape),
    })


@app.get("/datasets")
def list_datasets():
    """List all padded .npz files saved on disk."""
    if not NPZ_ROOT.exists():
        return {"datasets": [], "count": 0}
    files = sorted(p.name for p in NPZ_ROOT.glob("*.npz"))
    return {"datasets": files, "count": len(files)}


@app.get("/datasets/{name}")
def get_dataset(name: str):
    """Download a pre-built padded .npz by filename."""
    path = NPZ_ROOT / name
    if not path.exists() or path.suffix != ".npz":
        raise HTTPException(status_code=404, detail=f"Dataset '{name}' not found.")
    return FileResponse(
        path=str(path),
        media_type="application/octet-stream",
        filename=name,
    )
