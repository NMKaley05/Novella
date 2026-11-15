# NOVELLA API

This FastAPI service serves processed Lusaka telemetry to the Streamlit dashboard. From the repository root, it reads `novella_lusaka_processed_data.csv` by default. Set `NOVELLA_DATA_PATH` to use a different processed CSV. If the CSV contains only one observation date, NOVELLA adds its clearly labelled simulated demo history before that date so the date selector and trend charts remain usable.

## Run on Windows

From the repository root in PowerShell:

```powershell
py -m pip install -r backend/requirements.txt
py -m uvicorn backend.main:app --reload --port 8000
```

To select a different CSV before starting the API:

```powershell
$env:NOVELLA_DATA_PATH = 'C:\path\to\novella_lusaka_processed_data.csv'
py -m uvicorn backend.main:app --reload --port 8000
```

Open `http://127.0.0.1:8000/docs` for interactive API documentation.

## Endpoints

- `GET /health` — service health.
- `GET /api/v1/meta` — data mode, zone locations, and schema.
- `GET /api/v1/observations` — filter by `observation_date`, `zone_id`, or `status`.
- `GET /api/v1/summary` — KPIs and observations for a date (defaults to latest).
- `GET /api/v1/anomalies` — observations whose absolute z-score meets `min_abs_z` (defaults to 2).
- `GET /api/v1/recovery/{zone_id}` — date series and baseline for one zone; accepts optional `start` and `end` dates.

The CSV must include `zone_id`, `date`, `radiance_nW`, `baseline_mean`, `baseline_std`, `pct_change`, `z_score`, `status`, and `investigation_note`. Context fields such as rainfall, cloud cover, fire count, population density, and critical assets are also supported.
