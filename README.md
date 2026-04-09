# Lexical Stress Detection — Docker API Pipeline

This folder contains **six independent FastAPI microservices** that reproduce
the entire MyST data-preparation pipeline as HTTP APIs.  
Each service maps 1-to-1 to an original PBS job script.

---

## Architecture Overview

```
[.wav + .lab input]
       │
       ▼
┌──────────────────────────────┐  :8001
│  API 1 — prepare-corpus      │  POST /prepare
│  (myst_make_mystfa.pbs)      │  → 16 kHz WAV + .lab stored in corpus/
└──────────────┬───────────────┘
               │
       ┌───────┴────────┐
       │                │
       ▼                ▼
┌─────────────┐  :8002  ┌──────────────────────────┐  :8003
│  API 2      │         │  API 3                    │
│  mfa-align  │         │  wavlm-features           │
│  (MFA)      │         │  (microsoft/wavlm-large)  │
│  → .csv     │         │  → (T×1024) .npy          │
└──────┬──────┘         └───────────┬──────────────┘
       │                            │
       └──────────┬─────────────────┘
                  │
                  ▼
┌──────────────────────────────────┐  :8004
│  API 4 — syllable-average        │
│  (average_syllable_features.pbs) │
│  → per-utterance syllable .npy   │
└────────────────┬─────────────────┘
                 │
        ┌────────┴────────┐
        │                 │
        ▼                 ▼
┌──────────────┐  :8005  ┌─────────────────────────┐  :8006
│  API 5       │         │  API 6                   │
│  build-npz   │         │  build-padded-npz        │
│  train/test  │         │  (pad9 for TDNN postnet) │
│  .npz        │         │  → padded .npz           │
└──────────────┘         └──────────────────────────┘
```

| Service | Port | PBS Source | Output |
|---------|------|------------|--------|
| api1-prepare-corpus | 8001 | `myst_make_mystfa.pbs` | 16 kHz WAV + `.lab` |
| api2-mfa-align | 8002 | `mfa_myst_align_full.pbs` | Phone CSV |
| api3-wavlm-features | 8003 | `myst_wavlm_features.pbs` | `(T,1024)` float16 `.npy` |
| api4-syllable-average | 8004 | `average_syllable_features_grouped.pbs` | Syllable `.npy` |
| api5-build-npz | 8005 | `build_myst_train_npz.pbs` / `build_myst_test_npz.pbs` | `train.npz` / `test.npz` |
| api6-build-padded-npz | 8006 | `build_myst_train_npz_pad9.py` / `build_myst_test_npz_pad9.py` | `train_pad.npz` / `test_pad.npz` |
| api7-postnet-infer | 8007 | `postnet41.py` (inference only) | Per-syllable stress predictions (WoPP + WPP) |

---

## Quick Start

### 1. Build and start all services

```bash
cd docker-apis
docker compose up --build
```

### 2. Interactive API docs

Every service exposes Swagger UI at `/docs`:

| API | Swagger UI URL |
|-----|----------------|
| API 1 | http://localhost:8001/docs |
| API 2 | http://localhost:8002/docs |
| API 3 | http://localhost:8003/docs |
| API 4 | http://localhost:8004/docs |
| API 5 | http://localhost:8005/docs |
| API 6 | http://localhost:8006/docs |
| API 7 | http://localhost:8007/docs |

---

## End-to-End Walkthrough (single utterance)

The recommended inputs are a **`.wav` file** (any sample rate) and a **`.lab` file**
(plain-text transcript of the utterance).

### Step 1 — Prepare the corpus

```bash
curl -X POST http://localhost:8001/prepare \
  -F "wav_file=@my_utterance.wav" \
  -F "lab_file=@my_utterance.lab"
```

**Response:**
```json
{
  "utt_id":     "my_utterance",
  "wav_path":   "/data/corpus/my_utterance/my_utterance.wav",
  "lab_path":   "/data/corpus/my_utterance/my_utterance.lab",
  "original_sr": 44100,
  "resampled":   true
}
```

---

### Step 2 — MFA phone alignment

```bash
curl -X POST http://localhost:8002/align \
  -F "wav_file=@my_utterance.wav" \
  -F "lab_file=@my_utterance.lab" \
  --output my_utterance.csv
```

The response is a CSV with columns `Begin, End, Label, Type, Speaker`.

Download a stored alignment:
```bash
curl http://localhost:8002/alignments/my_utterance --output my_utterance.csv
```

---

### Step 3 — WavLM feature extraction

Stream the `.npy` directly:
```bash
curl -X POST http://localhost:8003/extract \
  -F "wav_file=@my_utterance.wav" \
  --output my_utterance.npy
```

Or save to disk and get metadata:
```bash
curl -X POST http://localhost:8003/extract/json \
  -F "wav_file=@my_utterance.wav"
```

**Response:**
```json
{
  "utt_id":   "my_utterance",
  "npy_path": "/data/wavlm/my_utterance.npy",
  "shape":    [187, 1024],
  "dtype":    "float16",
  "n_frames": 187
}
```

---

### Step 4 — Syllable feature averaging

Upload the CMUdict once:
```bash
curl -X POST http://localhost:8004/upload-cmu \
  -F "cmu_file=@cmudict_no_one_syl.csv"
```

Then average for a single utterance:
```bash
curl -X POST http://localhost:8004/average/json \
  -F "wav_file=@my_utterance.wav" \
  -F "lab_file=@my_utterance.lab" \
  -F "mfa_csv=@my_utterance.csv" \
  -F "wavlm_npy=@my_utterance.npy"
```

**Response:**
```json
{
  "utt_id":      "my_utterance",
  "npy_path":    "/data/syls/my_utterance.npy",
  "n_syllables": 6,
  "feature_dim": 1024,
  "labels":      [0, 2, 0, 1, 2, 0],
  "syllable_phonemes": ["L IH0 V", "IH0 NG", "S IH1 S", ...]
}
```

---

### Step 5 — Build NPZ datasets (train / test split)

Upload the CMUdict to API 5 (needed for the 85/15 split):
```bash
curl -X POST http://localhost:8005/upload-cmu \
  -F "cmu_file=@cmudict_no_one_syl.csv"
```

Build training archive:
```bash
curl -X POST http://localhost:8005/build/train/json \
  -F "ratio=0.85"
```

Build test archive:
```bash
curl -X POST http://localhost:8005/build/test/json \
  -F "ratio=0.85"
```

Download a pre-built archive:
```bash
curl http://localhost:8005/datasets/myst_train.npz --output myst_train.npz
```

---

### Step 6 — Build padded NPZ (TDNN postnet)

Pad train syllable files (reads from shared `/data/syls` volume):
```bash
curl -X POST http://localhost:8006/build/train-pad/json \
  -F "pad_len=9"
```

Pad test NPZ (upload the test.npz produced by API 5):
```bash
curl -X POST http://localhost:8006/build/test-pad/json \
  -F "test_npz=@myst_test.npz" \
  -F "pad_len=9"
```

**Response:**
```json
{
  "path":          "/data/npz/myst_test_pad.npz",
  "total_rows":    54000,
  "pad_len":       9,
  "pad_rows":      12000,
  "real_syllables": 42000,
  "feature_shape": [54000, 1024]
}
```

---

### Step 7 — PostNet TDNN inference (stress prediction)

First, upload your trained weight file (produced by `prenet41.py` training):
```bash
curl -X POST http://localhost:8007/upload-weights \
  -F "weights_file=@best_model_weights_WLM41.h5" \
  -F "sequence_length=9"
```

Run stress inference on the padded test NPZ from Step 6:
```bash
curl -X POST http://localhost:8007/predict \
  -F "padded_npz=@myst_test_pad.npz" \
  -F "apply_wpp=true" \
  -F "sequence_length=9"
```

**Response:**
```json
{
  "n_syllables":      42000,
  "n_words":          6000,
  "sequence_length":  9,
  "predictions_wopp": [0, 1, 0, 1, ...],
  "predictions_wpp":  [0, 1, 0, 1, ...],
  "accuracy_wopp":    0.843,
  "accuracy_wpp":     0.861,
  "f1_wopp":          0.821,
  "f1_wpp":           0.839
}
```

Or predict for a **single word** by sending its syllable feature vectors directly:
```bash
curl -X POST http://localhost:8007/predict/utterance \
  -H "Content-Type: application/json" \
  -d '{
    "features": [[0.1, 0.2, ...], [0.3, 0.4, ...], [0.5, 0.6, ...]],
    "word_id": 42,
    "sequence_length": 9
  }'
```

**Response:**
```json
{
  "word_id":       42,
  "n_syllables":   3,
  "raw_scores":    [0.12, 0.87, 0.31],
  "predicted":     [0, 1, 0],
  "predicted_wpp": [0, 1, 0]
}
```

Check model info (architecture + weight path):
```bash
curl http://localhost:8007/model/info
```

---

## API Reference

### API 1 — `prepare-corpus` (port 8001)

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/prepare` | Resample .wav to 16 kHz and pair with .lab |
| `GET`  | `/utterances` | List all prepared utterances |
| `GET`  | `/health` | Liveness probe |

**`POST /prepare` inputs:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `wav_file` | file | ✅ | .wav audio (any sample rate) |
| `lab_file` | file | ✅ | .lab transcript (UTF-8 plain text) |
| `utt_id`   | string | ❌ | Custom utterance ID (defaults to wav stem) |

---

### API 2 — `mfa-align` (port 8002)

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/align` | Align a .wav + .lab pair with MFA |
| `GET`  | `/alignments/{utt_id}` | Download alignment CSV |
| `GET`  | `/utterances` | List all aligned utterances |
| `GET`  | `/health` | Liveness probe |

**`POST /align` inputs:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `wav_file` | file | ✅ | 16 kHz .wav (from API 1) |
| `lab_file` | file | ✅ | .lab transcript |
| `utt_id`   | string | ❌ | Utterance ID |
| `dict_model` | string | ❌ | MFA dictionary (default: `english_us_arpa`) |
| `acoustic_model` | string | ❌ | MFA acoustic model (default: `english_us_arpa`) |

---

### API 3 — `wavlm-features` (port 8003)

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/extract` | Extract features, stream .npy |
| `POST` | `/extract/json` | Extract features, save to disk, return JSON |
| `GET`  | `/features/{utt_id}` | Download saved .npy |
| `GET`  | `/utterances` | List all extracted utterances |
| `GET`  | `/health` | Liveness probe + model status |

**`POST /extract` or `/extract/json` inputs:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `wav_file` | file | ✅ | 16 kHz .wav (must be 16 kHz — use API 1 first) |
| `utt_id`   | string | ❌ | Utterance ID |

---

### API 4 — `syllable-average` (port 8004)

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/upload-cmu` | Upload and cache CMUdict CSV |
| `POST` | `/average` | Average frames → syllables, stream .npy |
| `POST` | `/average/json` | Average frames → syllables, save + JSON |
| `GET`  | `/syllables/{utt_id}` | Download saved syllable .npy |
| `GET`  | `/utterances` | List all processed utterances |
| `GET`  | `/health` | Liveness probe + CMUdict status |

**`POST /average` or `/average/json` inputs:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `wav_file` | file | ✅ | .wav (for utterance ID) |
| `lab_file` | file | ✅ | .lab transcript |
| `mfa_csv`  | file | ✅ | Phone-timing CSV from API 2 |
| `wavlm_npy` | file | ✅ | WavLM .npy from API 3 |
| `cmu_csv`  | file | ❌ | CMUdict CSV (optional if pre-loaded via /upload-cmu) |
| `utt_id`   | string | ❌ | Utterance ID |
| `frame_shift` | float | ❌ | Seconds per frame (default: 0.02) |

**Syllable `.npy` payload structure (loaded with `allow_pickle=True`):**
```python
payload = np.load("my_utterance.npy", allow_pickle=True).item()
payload["feature"]           # (N_syl, 1024) float32
payload["syllable_phonemes"] # e.g. ["L IH0 V", "IH0 NG"]
payload["token_index"]       # (N_syl,) int32 — word index per syllable
payload["label"]             # (N_syl,) int8  — 0/1/2 stress label
```

---

### API 5 — `build-npz` (port 8005)

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/upload-cmu` | Upload CMUdict (for split) |
| `POST` | `/upload-syllable` | Upload a syllable .npy |
| `POST` | `/build/train` | Build train.npz (streamed) |
| `POST` | `/build/test` | Build test.npz (streamed) |
| `POST` | `/build/train/json` | Build train.npz, save, return JSON |
| `POST` | `/build/test/json` | Build test.npz, save, return JSON |
| `GET`  | `/datasets` | List saved .npz files |
| `GET`  | `/datasets/{name}` | Download a .npz |
| `GET`  | `/health` | Liveness probe |

---

### API 6 — `build-padded-npz` (port 8006)

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/upload-syllable` | Upload a syllable .npy |
| `POST` | `/upload-test-npz` | Upload myst_test.npz |
| `POST` | `/build/train-pad` | Pad train syllables (streamed) |
| `POST` | `/build/test-pad` | Pad test.npz (streamed) |
| `POST` | `/build/train-pad/json` | Pad train, save, JSON |
| `POST` | `/build/test-pad/json` | Pad test, save, JSON |
| `GET`  | `/datasets` | List saved padded .npz files |
| `GET`  | `/datasets/{name}` | Download a padded .npz |
| `GET`  | `/health` | Liveness probe |

**Padded `.npz` schema:**
```python
d = np.load("myst_test_pad.npz")
d["feature"]   # (N_words * pad_len, 1024) float32  — padding rows are all -1
d["label"]     # (N_words * pad_len,)      int8     — padding rows are -1
d["w"]         # (N_orig_syllables,)       int32    — word IDs (test only)
d["test_ind"]  # (N_pad_rows,)             int64    — indices of padding rows (test only)
d["word_len"]  # scalar [9]                         — pad length (test only)
```

---

### API 7 — `postnet-infer` (port 8007)

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/upload-weights` | Upload a trained `.h5` weight file |
| `POST` | `/predict` | Run TDNN inference on a padded .npz |
| `POST` | `/predict/utterance` | Run inference for a single word's syllable vectors |
| `GET`  | `/model/info` | Model architecture summary and weight path |
| `GET`  | `/health` | Liveness probe + model loaded status |

**`POST /predict` inputs:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `padded_npz` | file | ✅ | Padded `.npz` from API 6 (`feature`, `label`, `w`, `test_ind`, `word_len`) |
| `apply_wpp` | bool | ❌ | Apply within-word score normalisation (default: `true`) |
| `sequence_length` | int | ❌ | Syllables per word slot — must match API 6 `pad_len` (default: `9`) |

**`POST /predict/utterance` body (JSON):**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `features` | array | ✅ | `(n_syl, 1024)` float array — one row per syllable |
| `word_id` | int | ❌ | Optional identifier returned as-is in the response |
| `sequence_length` | int | ❌ | Pad length to use (default: env `SEQUENCE_LENGTH`) |

**Model architecture (exact mirror of `postnet41.py`):**
```
Masking(mask=-1, input=(seq_len, 1024))
→ TimeDistributed Dense(1024, relu) + BatchNorm + Dropout(0.2)
→ TimeDistributed Dense(1, sigmoid)
```
- Weights loaded from `best_model_weights_WLM41.h5` (produced by `prenet41.py`)
- `WoPP` = raw per-syllable threshold at 0.5
- `WPP` = within-word normalise → argmax → label stressed syllable

---

## Environment Variables

Each service can be configured via environment variables (set in `docker-compose.yml` or `docker run -e`):

| Service | Variable | Default | Description |
|---------|----------|---------|-------------|
| API 1 | `CORPUS_ROOT` | `/data/corpus` | Where to store prepared corpus |
| API 2 | `MFA_OUT_ROOT` | `/data/mfa` | Where to store alignment CSVs |
| API 2 | `MFA_TMP_ROOT` | `/tmp/mfa_tmp` | Temporary MFA working dir |
| API 3 | `FEATS_ROOT` | `/data/wavlm` | Where to store WavLM .npy files |
| API 3 | `WAVLM_MODEL_ID` | `microsoft/wavlm-large` | Hugging Face model ID |
| API 4 | `SYLS_ROOT` | `/data/syls` | Where to store syllable .npy files |
| API 4 | `CMU_CACHE` | `/data/cmu/cmudict_no_one_syl.csv` | Path to cached CMUdict |
| API 5 | `SYLS_ROOT` | `/data/syls` | Read-only source of syllable .npy files |
| API 5 | `NPZ_ROOT` | `/data/npz` | Where to save .npz archives |
| API 5 | `CMU_CACHE` | `/data/cmu/cmudict_no_one_syl.csv` | Path to cached CMUdict |
| API 6 | `SYLS_ROOT` | `/data/syls` | Read-only source of syllable .npy files |
| API 6 | `NPZ_ROOT` | `/data/npz` | Where to save padded .npz archives |
| API 7 | `WEIGHTS_PATH` | `/data/weights/best_model_weights_WLM41.h5` | Path to trained `.h5` weight file |
| API 7 | `SEQUENCE_LENGTH` | `9` | Pad length per word — must match API 6 `pad_len` |

---

## GPU Support (API 3 and API 7)

API 3 automatically uses GPU if available.  To enable it with Docker Compose, uncomment the `deploy` block in `docker-compose.yml`:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

Requires the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

---

## Notes

- The `.lab` file should contain the plain-text transcript of the utterance with no punctuation other than apostrophes, e.g.: `living systems dealing`
- All APIs implement **idempotency** — re-submitting the same utterance overwrites the previous result.
- The CMUdict CSV (`cmudict_no_one_syl.csv`) must be uploaded to API 4 and API 5 before processing.  It is cached in the shared `/data/cmu` volume.
- API 2 requires MFA to be installed in the container.  The provided Dockerfile uses the official `mmcauliffe/montreal-forced-aligner` base image.
- API 3 downloads `microsoft/wavlm-large` (~1.2 GB) at build time; the first container start will be slow if the weights are not cached.
