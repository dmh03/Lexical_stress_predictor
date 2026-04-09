# How to Run — Lexical Stress Predictor

This guide walks you through running the full pipeline from scratch on any machine.

---

## What You Need Before Starting

| Requirement | Version | Notes |
|---|---|---|
| [Docker Desktop](https://www.docker.com/products/docker-desktop/) | Latest | Must be running |
| [Git](https://git-scm.com/) | Any | To clone the repo |
| Disk space | ~15 GB | For Docker images (WavLM model is large) |
| RAM | 8 GB minimum | 16 GB recommended |
| Internet connection | Required | First build downloads WavLM from HuggingFace |

---

## Step 1 — Clone the Repository

Open a terminal and run:

```bash
git clone https://github.com/dmh03/Lexical_Stress_predictor.git
cd Lexical_Stress_predictor
```

---

## Step 2 — Confirm the Weights File is Present

The model weights file should already be included in the repo at:

```
weights/best_model_weights_WLM41.h5
```

Verify it exists:

```bash
ls weights/
# Expected: best_model_weights_WLM41.h5
```

If it is missing for any reason, contact the repo owner to obtain the file and place it in the `weights/` folder.

---

## Step 3 — Start Docker Desktop

Make sure Docker Desktop is open and running before proceeding. You should see the Docker icon in your system tray/menu bar.

---

## Step 4 — Build and Start All Services

From inside the `Lexical_Stress_predictor` folder, run:

```bash
docker compose up --build
```

> ⚠️ **This will take 10–30 minutes on the first run.** Docker needs to:
> - Build 9 container images
> - Download the WavLM-Large model from HuggingFace (~1.2 GB)
> - Download the MFA acoustic model and dictionary
>
> Subsequent runs will be much faster (images are cached).

You will see a lot of log output scrolling by. Wait until you see lines like:

```
api9-web          | INFO:     Uvicorn running on http://0.0.0.0:8080
api8-pipeline     | INFO:     Uvicorn running on http://0.0.0.0:8008
api3-wavlm-features | [startup] Model ready.
api4-syllable-average | [API 4] CMUdict loaded: 123,455 entries
```

That means everything is up and ready.

---

## Step 5 — Open the Website

Open your browser and go to:

```
http://localhost:8080
```

You should see the **Lexical Stress Predictor** web interface.

---

## Step 6 — Run a Prediction

You need two files for each utterance:

| File | Format | Description |
|---|---|---|
| `.wav` | 16 kHz mono WAV (any sample rate works — it will be resampled) | The speech recording |
| `.lab` | Plain text file | The transcript, e.g. `beyond` |

**Example `.lab` file content:**
```
beyond
```

On the website:
1. Drag and drop (or click to select) your `.wav` file into the **WAV** drop zone
2. Drag and drop your `.lab` file into the **LAB** drop zone
3. Click **▶ Start**
4. Wait for the pipeline to complete (~1–4 minutes depending on your machine)
5. Results appear showing each multi-syllable word with:
   - The syllables
   - Which syllable the model predicted as stressed
   - Whether the prediction was correct

Use the **PreNet / PostNet** toggle to switch between the two model outputs.

---

## Step 7 — Stop the Services

When you are done, press `Ctrl+C` in the terminal running `docker compose up`, then run:

```bash
docker compose down
```

To start again later (without rebuilding):

```bash
docker compose up
```

---

## Troubleshooting

### "Internal Server Error" or pipeline fails
- Check logs: `docker compose logs api2-mfa-align --tail=30`
- MFA alignment can take 3–4 minutes — be patient, it has a 15-minute timeout
- Make sure your `.lab` file contains only simple English words (no punctuation)

### Website won't load at localhost:8080
- Make sure Docker Desktop is running
- Run `docker compose ps` to check all containers show `Up`
- Try `docker compose up` again if any container exited

### "Port already in use" error
- Something else on your machine is using port 8080 or 8001–8008
- Stop the conflicting service, or change the port mappings in `docker-compose.yml`

### First run is very slow
- Normal — WavLM-Large (~1.2 GB) is downloaded on first use by API 3
- After the first run it is cached inside the container image

---

## Pipeline Architecture (for reference)

```
.wav + .lab
     │
     ▼
API 1 — Resample to 16 kHz
     │
     ├──────────────────────┐
     ▼                      ▼
API 2 — MFA Align     API 3 — WavLM Features
(phone timings)       (1024-dim frame vectors)
     │                      │
     └──────────┬───────────┘
                ▼
         API 4 — Syllable Averaging
         (CMUdict syllabification + feature averaging)
                │
        ┌───────┴────────┐
        ▼                ▼
  API 6 — Pad       (raw syllable .npy)
  (TDNN input)           │
        │                ▼
        ▼          API 7 PreNet
  API 7 PostNet    (flat DNN)
  (TDNN)                 │
        │                │
        └───────┬─────────┘
                ▼
         API 8 — Orchestrator
         (assembles results)
                │
                ▼
         API 9 — Web UI
         http://localhost:8080
```
