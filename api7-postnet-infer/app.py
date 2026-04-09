"""
API 7 — PostNet TDNN Inference  (postnet41.py — instance / inference mode only)
================================================================================
Loads a pre-trained TDNN (postnet41) weight file once at startup and exposes
HTTP endpoints to run stress prediction on padded-NPZ inputs produced by API 6.

This service NEVER trains — it is pure inference.

The model architecture exactly mirrors postnet41.py:
  Masking(mask=-1)
  → TimeDistributed Dense(1024, relu) + BatchNorm + Dropout(0.2)
  → TimeDistributed Dense(1, sigmoid)

Inputs accepted:
  • A padded .npz file (output of API 6)  — the primary mode
  • A plain DNN-style .npz file (feature, label, w) — for syllable-level inference

Post-processing (make_partitions2) can optionally be applied so the highest-
scoring syllable per word is selected as stressed, matching the WPP evaluation
used in the original paper.

Endpoints
---------
POST /predict
    Body : multipart/form-data
        padded_npz      : myst_*_pad.npz from API 6  (feature, label, w, test_ind, word_len)
        apply_wpp       : (optional, default true) apply within-word post-processing
        sequence_length : (optional, default 9) must match the pad_len used in API 6

    Returns : JSON
        {
          "n_syllables":   42000,
          "n_words":       6000,
          "predictions_wopp": [0,1,0,...],       raw per-syllable predictions
          "predictions_wpp":  [0,1,0,...],       after within-word normalisation
          "accuracy_wopp": 0.843,               only if labels are present
          "accuracy_wpp":  0.861,
          "f1_wopp":       0.821,
          "f1_wpp":        0.839
        }

POST /predict/utterance
    Convenience endpoint: accepts a single word's syllable features as a JSON
    array of shape (n_syl, 1024), pads it to sequence_length internally, and
    returns the stress prediction for each syllable.

    Body : JSON
        {
          "features": [[...1024 floats...], ...],   shape (n_syl, 1024)
          "word_id":  42,                            optional, for bookkeeping
          "sequence_length": 9                       optional, default from env
        }

    Returns : JSON
        {
          "word_id": 42,
          "n_syllables": 3,
          "raw_scores":    [0.12, 0.87, 0.31],
          "predicted":     [0, 1, 0],
          "predicted_wpp": [0, 1, 0]
        }

POST /upload-weights
    Upload a new .h5 weight file to replace the currently loaded weights.
    The model is rebuilt and weights are reloaded immediately.

GET /model/info
    Returns model architecture summary and weight file path.

GET /health
    Liveness probe + model status.
"""

import io
import os
import threading
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
WEIGHTS_PATH     = Path(os.getenv("WEIGHTS_PATH", "/data/weights/best_model_weights_WLM41.h5"))
SEQUENCE_LENGTH  = int(os.getenv("SEQUENCE_LENGTH", "9"))   # must match API 6 pad_len
ORIGINAL_DIM     = 1024
_model_lock      = threading.Lock()

# Prenet (flat DNN) — separate model instance, same weight file
_prenet_model    = None
_prenet_loaded   = False

app = FastAPI(
    title="API 7 — PostNet TDNN Inference",
    description=(
        "Inference-only wrapper around the trained PostNet41 TDNN. "
        "Accepts a padded .npz (from API 6) and returns per-syllable stress predictions. "
        "Mirrors postnet41.py — instance mode only, no training."
    ),
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# Lazy TF import — deferred so the app starts fast even on CPU-only machines
# ---------------------------------------------------------------------------
_tf   = None
_keras = None
_model = None
_weights_loaded = False


def _get_tf():
    global _tf, _keras
    if _tf is None:
        import tensorflow as tf
        from tensorflow import keras
        # Allow GPU memory growth so we share GPU with other services
        for g in tf.config.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(g, True)
        _tf   = tf
        _keras = keras
    return _tf, _keras


def _build_model(seq_len: int = SEQUENCE_LENGTH):
    """Recreate the exact TDNN architecture matching best_model_weights_WLM41.h5.

    Saved layer shapes (from h5py inspection):
      dense_16: (1024, 1024)  → Dense(1024)
      dense_17: (1024, 512)   → Dense(512)
      dense_18: (512,  256)   → Dense(256)
      dense_19: (256,  128)   → Dense(128)
      stress:   (128,  1)     → Dense(1, sigmoid)
    Each hidden block is: TimeDistributed(Dense) + BatchNorm + Dropout(0.2)
    """
    tf, keras = _get_tf()
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (
        TimeDistributed, Dense, Dropout, BatchNormalization, Masking
    )
    m = Sequential([
        Masking(mask_value=-1.0, input_shape=(seq_len, ORIGINAL_DIM)),
        # Block 1 — 1024
        TimeDistributed(Dense(1024, activation="relu", kernel_initializer="he_normal")),
        TimeDistributed(BatchNormalization()),
        TimeDistributed(Dropout(0.2)),
        # Block 2 — 512
        TimeDistributed(Dense(512, activation="relu", kernel_initializer="he_normal")),
        TimeDistributed(BatchNormalization()),
        TimeDistributed(Dropout(0.2)),
        # Block 3 — 256
        TimeDistributed(Dense(256, activation="relu", kernel_initializer="he_normal")),
        TimeDistributed(BatchNormalization()),
        TimeDistributed(Dropout(0.2)),
        # Block 4 — 128
        TimeDistributed(Dense(128, activation="relu", kernel_initializer="he_normal")),
        TimeDistributed(BatchNormalization()),
        TimeDistributed(Dropout(0.2)),
        # Output
        TimeDistributed(Dense(1, activation="sigmoid", name="stress")),
    ])
    from tensorflow.keras.optimizers import Adam
    m.compile(optimizer=Adam(learning_rate=0.005),
              loss="binary_crossentropy",
              metrics=["accuracy"])
    return m


def _build_prenet_model():
    """Flat DNN matching prenet41.py — no TimeDistributed, no Masking, no padding.

    Architecture (uses dense_16 and stress layers from the shared weight file):
      Dense(1024, relu, input_shape=(1024,)) + BatchNorm + Dropout(0.2)
      Dense(1, sigmoid, name='stress')
    """
    tf, keras = _get_tf()
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    m = Sequential([
        Dense(1024, activation="relu", kernel_initializer="he_normal",
              input_shape=(ORIGINAL_DIM,)),
        BatchNormalization(),
        Dropout(0.2),
        Dense(1, activation="sigmoid", name="stress"),
    ])
    m.compile(optimizer=Adam(learning_rate=1e-4),
              loss="binary_crossentropy",
              metrics=["accuracy"])
    return m


def _load_model_from_weights(weights_path: Path, seq_len: int = SEQUENCE_LENGTH):
    """Build model and load weights from disk."""
    global _model, _weights_loaded
    model = _build_model(seq_len)
    model.load_weights(str(weights_path))
    _model = model
    _weights_loaded = True
    return model


@app.on_event("startup")
def _startup():
    """Load weights at startup if the weight file exists."""
    global _prenet_model, _prenet_loaded
    WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    if WEIGHTS_PATH.exists():
        try:
            _load_model_from_weights(WEIGHTS_PATH, SEQUENCE_LENGTH)
            print(f"[startup] Loaded postnet weights from {WEIGHTS_PATH}")
        except Exception as exc:
            print(f"[startup] WARNING: could not load postnet weights: {exc}")
        try:
            pm = _build_prenet_model()
            pm.load_weights(str(WEIGHTS_PATH), by_name=True, skip_mismatch=True)
            _prenet_model = pm
            _prenet_loaded = True
            print(f"[startup] Loaded prenet weights from {WEIGHTS_PATH}")
        except Exception as exc:
            print(f"[startup] WARNING: could not load prenet weights: {exc}")
    else:
        print(
            f"[startup] Weight file not found at {WEIGHTS_PATH}. "
            "Upload weights via POST /upload-weights before predicting."
        )


# ---------------------------------------------------------------------------
# Helpers — postnet41 post-processing (make_partitions2)
# ---------------------------------------------------------------------------

def _make_partitions2(word_ids: np.ndarray, scores: np.ndarray) -> list[int]:
    """
    Within-word normalisation: divide each syllable score by the word max,
    then assign label=1 to the syllable with normalised score==1.
    Mirrors make_partitions2() from postnet41.py.
    """
    word_ids = np.asarray(word_ids).flatten()
    scores   = np.asarray(scores).flatten()

    result: list[int] = []
    i = 0
    n = len(word_ids)

    while i < n:
        wid   = word_ids[i]
        j     = i
        # Collect all syllables in this word
        while j < n and word_ids[j] == wid:
            j += 1
        word_scores = scores[i:j]
        wmax = word_scores.max()
        if wmax == 0:
            result.extend([0] * (j - i))
        else:
            normed = word_scores / wmax
            result.extend([1 if s == 1.0 else 0 for s in normed])
        i = j

    return result


def _calculate_accuracy(pred: list, true: np.ndarray) -> float:
    true = np.asarray(true).flatten()
    pred = np.asarray(pred).flatten()
    return float(np.mean(pred == true))


def _f1_binary(pred: list, true: np.ndarray) -> float:
    true = np.asarray(true).flatten().astype(np.int32)
    pred = np.asarray(pred).flatten().astype(np.int32)
    tp = float(np.sum((true == 1) & (pred == 1)))
    fp = float(np.sum((true == 0) & (pred == 1)))
    fn = float(np.sum((true == 1) & (pred == 0)))
    prec   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 2 * prec * recall / (prec + recall) if (prec + recall) > 0 else 0.0


# ---------------------------------------------------------------------------
# Core inference
# ---------------------------------------------------------------------------

def _infer_padded_npz(
    npz_bytes: bytes,
    sequence_length: int,
    apply_wpp: bool,
) -> dict:
    """
    Run the TDNN on a padded .npz (API 6 output).

    Expected NPZ keys (matching API 6 output):
      feature   : (N_words * seq_len, 1024)  float32
      label     : (N_words * seq_len,)        int8      (-1 for padding)
      w         : (N_real_syl,)               int32     (word IDs for real syllables)
      test_ind  : (N_pad_rows,)               int64     (indices of padding rows)
      word_len  : [seq_len]                             (scalar — must match sequence_length)
    """
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail="Model weights not loaded. POST /upload-weights first.",
        )

    z = np.load(io.BytesIO(npz_bytes), allow_pickle=True)

    feature  = z["feature"].astype(np.float32)  # (N_words*seq_len, 1024)
    test_ind = np.asarray(z["test_ind"]).flatten().astype(int) if "test_ind" in z else np.array([], dtype=int)
    w_real   = np.asarray(z["w"]).flatten().astype(np.int32)   if "w"        in z else None
    label    = np.asarray(z["label"]).flatten()                if "label"    in z else None

    # Validate word_len if present
    if "word_len" in z:
        stored_len = int(np.asarray(z["word_len"]).flatten()[0])
        if stored_len != sequence_length:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"NPZ word_len={stored_len} does not match requested "
                    f"sequence_length={sequence_length}. "
                    "Re-build the padded NPZ with API 6 using the correct pad_len."
                ),
            )

    n_total = feature.shape[0]
    if n_total % sequence_length != 0:
        # Trim to a whole number of words
        trim = (n_total // sequence_length) * sequence_length
        feature = feature[:trim]

    n_words = feature.shape[0] // sequence_length

    # Reshape to (n_words, seq_len, 1024)
    x_reshaped = feature.reshape((n_words, sequence_length, ORIGINAL_DIM))

    with _model_lock:
        pred_output = _model.predict(x_reshaped, batch_size=256, verbose=0)

    # pred_output shape: (n_words, seq_len, 1)
    flat_scores = pred_output.reshape(-1, 1)   # (n_words*seq_len, 1)

    # Remove padding rows → only real syllable predictions
    all_idx = np.arange(len(flat_scores))
    real_mask = np.ones(len(flat_scores), dtype=bool)
    if test_ind.size > 0:
        # Clip test_ind to valid range
        valid_ti = test_ind[test_ind < len(flat_scores)]
        real_mask[valid_ti] = False

    real_scores  = flat_scores[real_mask].flatten()   # (N_real_syl,)
    pred_wopp    = (real_scores > 0.5).astype(int).tolist()

    # WPP — within-word normalisation
    pred_wpp: list[int] = []
    if apply_wpp and w_real is not None and len(w_real) == len(real_scores):
        pred_wpp = _make_partitions2(w_real, real_scores)
    else:
        pred_wpp = pred_wopp[:]

    # Metrics — only compute if real labels exist and align
    result: dict = {
        "n_syllables":      int(len(real_scores)),
        "n_words":          n_words,
        "predictions_wopp": pred_wopp,
        "predictions_wpp":  pred_wpp,
        "sequence_length":  sequence_length,
    }

    if label is not None:
        real_labels_all = label[real_mask] if len(label) == len(flat_scores) else label[:len(real_scores)]
        # Only evaluate on non-(-1) labels
        eval_mask = real_labels_all != -1
        if eval_mask.any():
            yt  = real_labels_all[eval_mask]
            pw  = np.asarray(pred_wopp)[eval_mask]
            pwp = np.asarray(pred_wpp)[eval_mask]
            result["accuracy_wopp"] = round(_calculate_accuracy(pw,  yt), 4)
            result["accuracy_wpp"]  = round(_calculate_accuracy(pwp, yt), 4)
            result["f1_wopp"]       = round(_f1_binary(pw,  yt), 4)
            result["f1_wpp"]        = round(_f1_binary(pwp, yt), 4)

    return result


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status":          "ok",
        "service":         "api7-postnet-infer",
        "model_loaded":    _weights_loaded,
        "prenet_loaded":   _prenet_loaded,
        "weights_path":    str(WEIGHTS_PATH),
        "sequence_length": SEQUENCE_LENGTH,
    }


@app.get("/model/info")
def model_info():
    """Return the model summary and weight file path."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")
    buf = io.StringIO()
    _model.summary(print_fn=lambda x: buf.write(x + "\n"))
    return {
        "weights_path":    str(WEIGHTS_PATH),
        "sequence_length": SEQUENCE_LENGTH,
        "input_dim":       ORIGINAL_DIM,
        "summary":         buf.getvalue(),
    }


@app.post("/predict/prenet")
async def predict_prenet(
    syl_npy: UploadFile = File(
        ...,
        description="Raw syllable .npy from API 4 (dict with 'feature', 'token_index', 'label', 'syllable_phonemes')"
    ),
    apply_wpp: bool = Form(
        default=True,
        description="Apply within-word score normalisation (WPP)",
    ),
):
    """
    Run flat-DNN (prenet41) stress inference on raw unpadded syllable features.
    Accepts the .npy payload directly from API 4 — no padding required.
    """
    if _prenet_model is None:
        raise HTTPException(status_code=503, detail="Prenet weights not loaded.")

    raw = await syl_npy.read()
    data = np.load(io.BytesIO(raw), allow_pickle=True).item()

    feature     = np.asarray(data["feature"],      dtype=np.float32)   # (N_syl, 1024)
    token_index = np.asarray(data["token_index"]).flatten().astype(int) # (N_syl,)
    label       = np.asarray(data["label"]).flatten().astype(int)       # (N_syl,)

    if feature.shape[0] == 0:
        return JSONResponse(content={
            "n_syllables": 0, "n_words": 0,
            "predictions_wopp": [], "predictions_wpp": [],
        })

    with _model_lock:
        raw_scores = _prenet_model.predict(feature, batch_size=256, verbose=0).flatten()  # (N_syl,)

    pred_wopp = (raw_scores > 0.5).astype(int).tolist()

    # WPP — within-word normalisation using token_index as word IDs
    pred_wpp: list[int]
    if apply_wpp:
        pred_wpp = _make_partitions2(token_index, raw_scores)
    else:
        pred_wpp = pred_wopp[:]

    n_words = len(np.unique(token_index))
    result: dict = {
        "n_syllables":      int(feature.shape[0]),
        "n_words":          n_words,
        "predictions_wopp": pred_wopp,
        "predictions_wpp":  pred_wpp,
    }

    # Metrics if labels present and valid
    eval_mask = label != -1
    if eval_mask.any():
        yt  = label[eval_mask]
        pw  = np.asarray(pred_wopp)[eval_mask]
        pwp = np.asarray(pred_wpp)[eval_mask]
        result["accuracy_wopp"] = round(_calculate_accuracy(pw,  yt), 4)
        result["accuracy_wpp"]  = round(_calculate_accuracy(pwp, yt), 4)
        result["f1_wopp"]       = round(_f1_binary(pw,  yt), 4)
        result["f1_wpp"]        = round(_f1_binary(pwp, yt), 4)

    return JSONResponse(content=result)


@app.post("/upload-weights")
async def upload_weights(
    weights_file: UploadFile = File(..., description=".h5 Keras weight file from prenet41.py"),
    sequence_length: int = Form(default=SEQUENCE_LENGTH, description="Pad length used in API 6"),
):
    """
    Upload a trained weight file (``best_model_weights_WLM41.h5``) and reload the model.

    This lets you swap weights without restarting the container.
    """
    if not weights_file.filename.lower().endswith(".h5"):
        raise HTTPException(status_code=400, detail="weights_file must be a .h5 file.")

    WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    WEIGHTS_PATH.write_bytes(await weights_file.read())

    try:
        with _model_lock:
            _load_model_from_weights(WEIGHTS_PATH, sequence_length)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Failed to load weights: {exc}") from exc

    return {"message": "Weights loaded successfully.", "path": str(WEIGHTS_PATH)}


@app.post("/predict")
async def predict(
    padded_npz: UploadFile = File(
        ...,
        description="Padded .npz from API 6 (feature, label, w, test_ind, word_len)"
    ),
    apply_wpp: bool = Form(
        default=True,
        description="Apply within-word score normalisation (WPP) — mirrors postnet41 post-processing",
    ),
    sequence_length: int = Form(
        default=SEQUENCE_LENGTH,
        description="Syllables per word slot — must match the pad_len used in API 6",
    ),
):
    """
    Run TDNN stress inference on a padded .npz.

    Returns per-syllable predictions (WoPP and WPP) and accuracy/F1 metrics
    if ground-truth labels are present in the file.
    """
    if not padded_npz.filename.lower().endswith(".npz"):
        raise HTTPException(status_code=400, detail="padded_npz must be a .npz file.")

    npz_bytes = await padded_npz.read()
    result = _infer_padded_npz(npz_bytes, sequence_length, apply_wpp)
    return JSONResponse(content=result)


class UtteranceRequest(BaseModel):
    features: list[list[float]]  # shape (n_syl, 1024)
    word_id: Optional[int] = None
    sequence_length: Optional[int] = None


@app.post("/predict/utterance")
async def predict_utterance(req: UtteranceRequest):
    """
    Predict stress for a **single word's** syllables.

    Send the syllable feature vectors (one per syllable, 1024-dim each).
    The endpoint pads the word to ``sequence_length`` internally, runs inference,
    and strips the padding back before returning predictions.

    This is useful when calling the API utterance-by-utterance rather than
    building a full padded NPZ first.
    """
    if _model is None:
        raise HTTPException(status_code=503, detail="Model weights not loaded.")

    seq_len = req.sequence_length or SEQUENCE_LENGTH
    features = np.array(req.features, dtype=np.float32)  # (n_syl, 1024)

    if features.ndim != 2 or features.shape[1] != ORIGINAL_DIM:
        raise HTTPException(
            status_code=422,
            detail=f"features must be shape (n_syl, {ORIGINAL_DIM}). Got {list(features.shape)}.",
        )

    n_syl = features.shape[0]
    if n_syl > seq_len:
        raise HTTPException(
            status_code=422,
            detail=f"n_syl ({n_syl}) exceeds sequence_length ({seq_len}). Increase sequence_length.",
        )

    # Pad to (1, seq_len, 1024)
    padded = np.full((1, seq_len, ORIGINAL_DIM), -1.0, dtype=np.float32)
    padded[0, :n_syl, :] = features

    with _model_lock:
        pred = _model.predict(padded, verbose=0)  # (1, seq_len, 1)

    scores_all  = pred[0, :, 0]         # (seq_len,)
    real_scores = scores_all[:n_syl]    # strip padding

    raw_binary = (real_scores > 0.5).astype(int).tolist()

    # WPP for a single word: just pick the argmax
    if n_syl == 1:
        wpp = [1]
    else:
        wmax = real_scores.max()
        if wmax == 0:
            wpp = [0] * n_syl
        else:
            normed = real_scores / wmax
            wpp = [1 if s == 1.0 else 0 for s in normed]

    return {
        "word_id":       req.word_id,
        "n_syllables":   n_syl,
        "raw_scores":    [round(float(s), 4) for s in real_scores],
        "predicted":     raw_binary,
        "predicted_wpp": wpp,
    }
