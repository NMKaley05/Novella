# NOVELLA — Lusaka Environmental Intelligence

NOVELLA is a Streamlit dashboard and FastAPI service for exploring satellite and environmental observations across monitored locations in Lusaka.

## Dashboard sections

- **Home** — overview of monitored locations, status, and flagged anomalies.
- **Explore Map** — satellite map with monitored locations and status markers.
- **Trends** — satellite map with a selected location’s observation history.
- **Data & Analysis** — analyzed location data beside the satellite map.
- **Insights** — locations flagged for unusual changes.
- **About** — project information.

## Requirements

Use Python 3.12. If it is not installed, install it with:

```powershell
py install 3.12
```

From the project folder, create a virtual environment and install dependencies:

```powershell
py -3.12 -m venv .venv312
.\.venv312\Scripts\python.exe -m pip install --upgrade pip
.\.venv312\Scripts\python.exe -m pip install -r requirements.txt
```

## Run the application

Open two terminals in the project folder.

In the first terminal, start the API:

```powershell
.\.venv312\Scripts\python.exe -m uvicorn main:app --reload --port 8000
```

In the second terminal, start the dashboard:

```powershell
.\.venv312\Scripts\python.exe -m streamlit run app.py
```

Open the local address shown by Streamlit, usually `http://localhost:8501`.

## Data

The API loads `novella_lusaka_processed_data.csv` from the project folder by default. To use another CSV, set `NOVELLA_DATA_PATH` in PowerShell before starting the API:

```powershell
$env:NOVELLA_DATA_PATH = "C:\path\to\your\data.csv"
```

The supplied CSV contains one observation date. To keep the date selector and trend charts usable, the app includes simulated demonstration history before that date. The dashboard labels this history as simulated; it should not be treated as measured historical data.

## API endpoints

- `GET /health` — service health.
- `GET /api/v1/meta` — data mode, zone locations, and schema.
- `GET /api/v1/observations` — observations, with optional date, zone, and status filters.
- `GET /api/v1/summary` — summary for a date.
- `GET /api/v1/anomalies` — flagged observations.
- `GET /api/v1/recovery/{zone_id}` — observation history for one location.

Interactive API documentation is available at `http://127.0.0.1:8000/docs` while the API is running.
