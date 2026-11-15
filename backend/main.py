"""NOVELLA telemetry API, compatible with the Streamlit dashboard CSV."""
from datetime import date
import os
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="NOVELLA API", version="0.2.0", description="Lusaka nighttime-light anomaly and recovery telemetry")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("NOVELLA_CORS_ORIGINS", "*").split(","),
    allow_methods=["GET"], allow_headers=["*"],
)

DEFAULT_ZONES = pd.DataFrame([
    ("LSK-01", "Lusaka CBD / Cairo Rd", -15.4167, 28.2833, 52),
    ("LSK-02", "Kabulonga / Woodlands", -15.4350, 28.3320, 24),
    ("LSK-03", "Manda Hill / Arcades Hub", -15.4192, 28.3164, 38),
    ("LSK-04", "Makeni Industrial Area", -15.4680, 28.2480, 41),
    ("LSK-05", "Heavy Industrial Area", -15.4450, 28.2900, 58),
    ("LSK-06", "Chilenje", -15.4470, 28.3350, 33),
    ("LSK-07", "Matero", -15.3870, 28.2550, 29),
    ("LSK-08", "Kanyama", -15.4400, 28.2450, 21),
    ("LSK-09", "Chelstone", -15.3930, 28.3680, 26),
    ("LSK-10", "Libala South", -15.4650, 28.3250, 30),
    ("LSK-11", "Roma / Northmead", -15.4010, 28.3050, 36),
    ("LSK-12", "KKIA Airport Zone", -15.3308, 28.4526, 45),
], columns=["zone_id", "zone_name", "lat", "lon", "demo_base"])

CORE = ["radiance_nW", "baseline_mean", "baseline_std", "pct_change", "z_score"]
CONTEXT = ["cloud_cover_pct", "rainfall_mm", "fire_count_5km", "pop_density_km2", "osm_critical_assets_2km", "iforest_pred"]
REQUIRED = ["zone_id", "date", *CORE, "status", "investigation_note"]
DATA_PATH = os.getenv("NOVELLA_DATA_PATH")


def severity_for(row):
    if str(row.status).strip().lower().startswith("normal"):
        return "Normal"
    return "Critical" if abs(float(row.z_score)) >= 3 else "Moderate"


def demo_data():
    """Create clearly labelled deterministic development data."""
    rows = []
    dips = {(1, 23): .58, (1, 22): .66, (1, 21): .80, (3, 23): .45, (2, 23): .86,
            (4, 18): .70, (4, 19): .82, (7, 23): .88, (8, 23): .60, (9, 22): .85}
    for i, zone in DEFAULT_ZONES.iterrows():
        rng = np.random.default_rng(1000 + i)
        values = zone.demo_base * (1 + (rng.random(24) - .5) * .12)
        base, sd = float(values[:12].mean()), float(values[:12].std(ddof=1))
        for month, value in enumerate(values):
            radiance = round(float(value) * dips.get((i, month), 1), 2)
            zscore = (radiance - base) / sd if sd else 0
            pct = (radiance - base) / base * 100 if base else 0
            status = "Normal Baseline" if abs(zscore) < 2 else "Unusual Decrease" if zscore < 0 else "Unusual Increase"
            rain = round(float(rng.random() * (220 if month % 12 < 5 or month % 12 > 9 else 8)), 1)
            fires = int(rng.integers(0, 7 if radiance < value else 3))
            note = "Within expected variance." if status == "Normal Baseline" else f"Radiance {pct:.1f}% vs baseline. Rain {rain} mm and {fires} fires within 5 km; verify weather and seasonal effects."
            rows.append({"zone_id": zone.zone_id, "zone_name": zone.zone_name, "lat": zone.lat, "lon": zone.lon,
                         "date": (pd.Timestamp("2024-11-01") + pd.DateOffset(months=month)).date(),
                         "radiance_nW": radiance, "cloud_cover_pct": int(rng.integers(0, 70)),
                         "rainfall_mm": rain, "fire_count_5km": fires, "pop_density_km2": np.nan,
                         "osm_critical_assets_2km": np.nan, "baseline_mean": round(base, 2),
                         "baseline_std": round(sd, 2), "pct_change": round(pct, 1), "z_score": round(zscore, 2),
                         "iforest_pred": np.nan, "status": status, "investigation_note": note,
                         "data_source": "Synthetic demo data"})
    return pd.DataFrame(rows)


def load_data():
    if not DATA_PATH:
        frame, mode = demo_data(), "simulated"
    else:
        path = Path(DATA_PATH)
        if not path.is_file():
            raise RuntimeError(f"NOVELLA_DATA_PATH is not a file: {path}")
        frame, mode = pd.read_csv(path), "real"
    missing = [column for column in REQUIRED if column not in frame.columns]
    if missing:
        raise RuntimeError("Telemetry CSV missing required columns: " + ", ".join(missing))
    frame = frame.copy()
    frame["zone_id"] = frame.zone_id.astype(str).str.strip().str[:32]
    frame["date"] = pd.to_datetime(frame.date, errors="coerce").dt.date
    for column in CORE + CONTEXT:
        if column not in frame:
            frame[column] = np.nan
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["status"] = frame.status.astype(str).str.strip().str[:80]
    frame["investigation_note"] = frame.investigation_note.fillna("").astype(str).str[:2000]
    defaults = DEFAULT_ZONES.set_index("zone_id")
    if "zone_name" not in frame:
        frame["zone_name"] = frame.zone_id.map(defaults.zone_name)
    frame["zone_name"] = frame.zone_name.fillna(frame.zone_id).astype(str).str[:120]
    for column in ("lat", "lon"):
        if column not in frame:
            frame[column] = np.nan
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["lat"] = frame["lat"].fillna(frame.zone_id.map(defaults.lat))
    frame["lon"] = frame["lon"].fillna(frame.zone_id.map(defaults.lon))
    frame = frame.dropna(subset=["date", *CORE, "lat", "lon"])
    frame = frame[frame.lat.between(-90, 90) & frame.lon.between(-180, 180) & frame.zone_id.ne("") & frame.status.ne("")]
    frame["severity"] = frame.apply(severity_for, axis=1)
    if "data_source" not in frame:
        frame["data_source"] = ""
    frame["data_source"] = frame.data_source.fillna("").astype(str).str[:200]
    return frame.sort_values(["date", "zone_id"]).reset_index(drop=True), mode


def json_rows(frame):
    return frame.astype(object).where(pd.notna(frame), None).to_dict(orient="records")


def get_data():
    try:
        return load_data()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/health")
def health():
    return {"status": "ok", "service": "novella-api"}


@app.get("/api/v1/meta")
def metadata():
    frame, mode = get_data()
    zones = frame.drop_duplicates("zone_id")[ ["zone_id", "zone_name", "lat", "lon"] ]
    return {"mode": mode, "record_count": len(frame), "zone_count": len(zones),
            "date_min": frame.date.min() if len(frame) else None, "date_max": frame.date.max() if len(frame) else None,
            "zones": json_rows(zones), "required_columns": REQUIRED, "optional_columns": CONTEXT}


@app.get("/api/v1/observations")
def observations(observation_date: date | None = None, zone_id: str | None = None,
                 severity: str | None = Query(None, pattern="^(Normal|Moderate|Critical)$"),
                 limit: int = Query(5000, ge=1, le=20000)):
    frame, mode = get_data()
    if observation_date:
        frame = frame[frame.date == observation_date]
    if zone_id:
        frame = frame[frame.zone_id == zone_id]
    if severity:
        frame = frame[frame.severity == severity]
    return {"mode": mode, "count": min(len(frame), limit), "observations": json_rows(frame.head(limit))}


@app.get("/api/v1/summary")
def summary(observation_date: date | None = None):
    frame, mode = get_data()
    day = observation_date or (frame.date.max() if len(frame) else None)
    snap = frame[frame.date == day] if day else frame.iloc[0:0]
    flagged = snap[snap.severity != "Normal"]
    critical = int((flagged.severity == "Critical").sum())
    return {"mode": mode, "observation_date": day, "active_zones": int(snap.zone_id.nunique()),
            "health": "Degraded" if critical else "Watch" if len(flagged) else "Nominal",
            "flagged_anomalies": len(flagged), "critical_anomalies": critical,
            "mean_radiance_change_pct": round(float(snap.pct_change.mean()), 2) if len(snap) else None,
            "observations": json_rows(snap)}


@app.get("/api/v1/anomalies")
def anomalies(observation_date: date | None = None, min_abs_z: float = Query(2, ge=0, le=20)):
    frame, _ = get_data()
    day = observation_date or (frame.date.max() if len(frame) else None)
    selected = frame[(frame.date == day) & (frame.z_score.abs() >= min_abs_z)] if day else frame.iloc[0:0]
    selected = selected.sort_values("z_score", key=lambda values: values.abs(), ascending=False)
    return {"observation_date": day, "count": len(selected), "anomalies": json_rows(selected)}


@app.get("/api/v1/recovery/{zone_id}")
def recovery(zone_id: str, start: date | None = None, end: date | None = None):
    frame, _ = get_data()
    if zone_id not in set(frame.zone_id):
        raise HTTPException(status_code=404, detail="Unknown zone_id")
    selected = frame[frame.zone_id == zone_id]
    if start:
        selected = selected[selected.date >= start]
    if end:
        selected = selected[selected.date <= end]
    selected = selected.sort_values("date")
    last = selected.iloc[-1] if len(selected) else None
    return {"zone_id": zone_id, "observation_count": len(selected),
            "baseline_mean": float(selected.baseline_mean.iloc[0]) if len(selected) else None,
            "latest_status": last.status if last is not None else None,
            "latest_date": last.date if last is not None else None, "observations": json_rows(selected)}
