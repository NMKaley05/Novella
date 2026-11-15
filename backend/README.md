# NOVELLA API

This FastAPI service exposes the same telemetry used by the Streamlit dashboard. It reads the processed CSV configured by `NOVELLA_DATA_PATH`; without that variable it serves clearly labelled synthetic demo data.

## Run on Windows

From the repository root in PowerShell:

```powershell
py -m pip install -r backend/requirements.txt
$env:NOVELLA_DATA_PATH = 'C:\path\to\novella_lusaka_processed_data.csv'
py -m uvicorn backend.main:app --reload
```

Open `http://127.0.0.1:8000/docs` for interactive API documentation. Omit `NOVELLA_DATA_PATH` to use demo data.

## Endpoints

- `GET /health` — service health.
- `GET /api/v1/meta` — data mode, coverage, zone locations, and schema.
- `GET /api/v1/observations` — filter by `observation_date`, `zone_id`, or `severity` (`Normal`, `Moderate`, `Critical`).
- `GET /api/v1/summary` — KPIs and observations for a date (defaults to latest).
- `GET /api/v1/anomalies` — observations whose absolute z-score meets `min_abs_z` (defaults to 2).
- `GET /api/v1/recovery/{zone_id}` — date series and baseline for one zone; accepts optional `start` and `end` dates.

The CSV must include `zone_id`, `date`, `radiance_nW`, `baseline_mean`, `baseline_std`, `pct_change`, `z_score`, `status`, and `investigation_note`. Context fields such as rainfall, cloud cover, fire count, population density, critical assets, and Isolation Forest prediction are optional and returned when present.
