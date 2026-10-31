# ASFR Rice Disease Classification Web App

Repository package version: `0.1.0`

## Stage 2 package contents

- `model_artifacts/best_model.keras`: supplied ASFR checkpoint.
- `model_artifacts/class_names.json`: nine-class mapping from the supplied final evaluation metadata.
- `model_artifacts/model_metadata.json`: model version, artifact SHA-256, input/output contract, preprocessing notes, and research limitations.
- `backend/`: FastAPI scaffold and metadata integrity tests.
- `frontend/`: Next.js scaffold.
- `docs/`: stage records and future deployment notes.

## Verification status

Checkpoint SHA-256: `14bc371dbcae04ccb311c28dc5842a9d6248942d88da24faf2970c7db5134ee3`

Model loading, forward-pass inference, prediction parity, dependency installation, Docker build, and frontend build have **not** been verified in this environment. The backend dependency pins are a proposed compatibility baseline, not a claim of successful installation. Run the commands below in a clean environment before treating this package as independently runnable.

## Run package metadata tests

From repository root:

```bash
python -m pytest backend/tests -q
```

## Backend environment

Python 3.11 is the target container base. From repository root:

```bash
python -m venv .venv
# Activate the environment, then:
pip install -r backend/requirements.txt
uvicorn backend.app.main:app --reload
```

For the Dockerfile, build from the repository root:

```bash
docker build -f backend/Dockerfile -t asfr-rice-backend:0.1.0 .
```

## Frontend

```bash
cd frontend
npm install
npm run dev
```

## Research limitations

The saved benchmark evaluation does not establish field robustness. The supplied audit notes residual source-provenance/near-duplicate risk and single-seed limitations. Do not use the application as an autonomous agricultural diagnostic system without independent field validation.
