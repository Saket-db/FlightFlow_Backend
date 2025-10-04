# Hack_Actual

Simple README for the project.

## What this is
A small full-stack prototype that includes a Python backend (ML models) and a TypeScript + Vite frontend. The repository contains trained models, data, and scripts used for flight delay analysis and predictions.

## Repository layout (important files/folders)
- `app.py` - top-level Python entrypoint (may start the backend).
- `backend/` - backend application code and its own `requirements.txt`.
  - `backend/main.py` - backend server entrypoint.
- `frontend/` - Vite + TypeScript frontend app.
  - `package.json` - frontend dependencies and scripts.
- `models/` - serialized ML models (e.g. `delay_classifier.pkl`, `delay_p50.pkl`).
- `data/` - raw and processed data used by the models.
- `src/` - analysis, feature engineering, and model training scripts.

## Quick start (Windows PowerShell)
1. Create and activate a Python virtual environment (recommended):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install Python dependencies (top-level or backend):

```powershell
pip install -r requirements.txt
# or, if you want only backend dependencies
pip install -r backend\requirements.txt
```

3. Run the backend (either the top-level app or the backend module):

```powershell
python app.py
# or
python backend\main.py
```

4. Run the frontend (from the `frontend` folder):

```powershell
cd frontend
npm install
npm run dev
```

Open the dev server address printed by Vite (usually http://localhost:5173).

## Notes
- Models are stored in `models/` and `data/processed/` contains cleaned CSVs used for training/evaluation.
- If you change Python dependencies, update `requirements.txt` and consider using a lock file.
- No automated tests are included by default.

## Contributing
Feel free to open issues or submit pull requests. Keep changes small and document any new dependencies.

## License
No license specified. Add a `LICENSE` file if you want to make the project's licensing explicit.


