"""NOVELLA telemetry and anomaly analysis API."""
from datetime import date
import os
from pathlib import Path
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="NOVELLA API",
    version="0.2.0",
    description="Lusaka anomaly detection and recovery telemetry across 12 zones",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("NOVELLA_CORS_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)

ZONES = pd.DataFrame(
    [
        ("LSK-01", "Lusaka CBD / Cairo Rd", -15.4167, 28.2833, 52, 9200, 31),
        ("LSK-02", "Kabulonga / Woodlands", -15.4350, 28.3320, 24, 4100, 12),
        ("LSK-03", "Manda Hill / Arcades Hub", -15.4192, 28.3164, 38, 6800, 19),
        ("LSK-04", "Makeni Industrial Area", -15.4680, 28.2480, 41, 2300, 8),
        ("LSK-05", "Heavy Industrial Area", -15.4450, 28.2900, 58, 1500, 6),
        ("LSK-06", "Chilenje / Libala Corridor", -15.4485, 28.3180, 33, 8400, 14),
        ("LSK-07", "Matero / George Compound", -15.3740, 28.2520, 36, 11200, 15),
        ("LSK-08", "Chelston / Airport Corridor", -15.3780, 28.3820, 29, 4900, 11),
        ("LSK-09", "Roma / Olympia Park", -15.3880, 28.3080, 31, 5100, 16),
        ("LSK-10", "Kanyama / Chinika West", -15.4260, 28.2420, 35, 12800, 13),
        ("LSK-11", "Avondale / Ibex Hill", -15.3960, 28.3620, 26, 3800, 9),
        ("LSK-12", "Lilayi / Kafue Rd South", -15.5020, 28.2760, 22, 2900, 7),
    ],
    columns=[
        "zone_id",
        "zone_name",
        "lat",
        "lon",
        "base",
        "pop_density_km2",
        "osm_critical_assets_2km",
    ],
)

NUMERIC = [
    "radiance_nW",
    "cloud_cover_pct",
    "rainfall_mm",
    "fire_count_5km",
    "pop_density_km2",
    "osm_critical_assets_2km",
    "baseline_mean",
    "baseline_std",
    "pct_change",
    "z_score",
]
REQUIRED = ["zone_id", "date", *NUMERIC, "status", "investigation_note"]
STATUSES = {
    "Normal Baseline",
    "Moderate Deviation",
    "Critical Anomaly",
    "Unusual Decrease",
    "Unusual Increase",
}
DEFAULT_DATA_PATH = Path(__file__).resolve().parent.parent / "novella_lusaka_processed_data.csv"
DATA_PATH = Path(os.getenv("NOVELLA_DATA_PATH", str(DEFAULT_DATA_PATH)))


def classify(z: float) -> str:
    return (
        "Critical Anomaly"
        if abs(z) >= 3
        else "Moderate Deviation"
        if abs(z) >= 2
        else "Normal Baseline"
    )


def demo_data() -> pd.DataFrame:
    rows = []
    # Simulated radiance dip events across multiple Lusaka zones (zone_index, month_index)
    dips = {
        (1, 23): 0.58,
        (1, 22): 0.66,
        (1, 21): 0.80,
        (2, 23): 0.86,
        (3, 23): 0.45,
        (4, 18): 0.70,
        (4, 19): 0.82,
        (5, 22): 0.62,
        (5, 23): 0.78,
        (6, 23): 0.52,
        (8, 20): 0.74,
        (9, 23): 0.61,
        (11, 21): 0.68,
    }
    for i, z in ZONES.iterrows():
        rng = np.random.default_rng(1000 + i)
        vals = z.base * (1 + (rng.random(24) - 0.5) * 0.12)
        mean = float(vals[:12].mean())
        sd = float(vals[:12].std(ddof=1))
        for m, v in enumerate(vals):
            rad = round(float(v) * dips.get((i, m), 1.0), 2)
            rain = round(float(rng.random() * (220 if m % 12 < 5 or m % 12 > 9 else 8)), 1)
            fires = int(rng.integers(0, 7 if rad < v else 3))
            zz = (rad - mean) / sd if sd else 0.0
            pct = (rad - mean) / mean * 100 if mean else 0.0
            status = classify(zz)
            note = (
                "Within expected variance."
                if status == "Normal Baseline"
                else f"Radiance {pct:.1f}% vs baseline. Rain {rain} mm, {fires} fires within 5 km. Verify outage vs. cloud or seasonal effects."
            )
            rows.append(
                dict(
                    zone_id=z.zone_id,
                    zone_name=z.zone_name,
                    lat=float(z.lat),
                    lon=float(z.lon),
                    date=(pd.Timestamp("2024-11-01") + pd.DateOffset(months=m)).date(),
                    radiance_nW=rad,
                    cloud_cover_pct=int(rng.integers(0, 70)),
                    rainfall_mm=rain,
                    fire_count_5km=fires,
                    pop_density_km2=int(z.pop_density_km2),
                    osm_critical_assets_2km=int(z.osm_critical_assets_2km),
                    baseline_mean=round(mean, 2),
                    baseline_std=round(sd, 2),
                    pct_change=round(pct, 1),
                    z_score=round(zz, 2),
                    status=status,
                    investigation_note=note,
                )
            )
    return pd.DataFrame(rows)


def load_data() -> tuple[pd.DataFrame, str]:
    if not DATA_PATH.is_file():
        if os.getenv("NOVELLA_DATA_PATH"):
            raise RuntimeError(f"NOVELLA_DATA_PATH is not a file: {DATA_PATH}")
        demo = demo_data()
        demo["data_source"] = "Simulated demo history"
        return demo, "simulated"
    source = pd.read_csv(DATA_PATH)
    missing = [c for c in REQUIRED if c not in source]
    if missing:
        raise RuntimeError("CSV missing required columns: " + ", ".join(missing))
    metadata_columns = ["zone_id", "zone_name", "lat", "lon"]
    if not all(column in source.columns for column in metadata_columns):
        metadata = ZONES[["zone_id", "zone_name", "lat", "lon"]].copy()
    else:
        metadata = source[metadata_columns].drop_duplicates("zone_id")
    df = source[REQUIRED].copy()
    df["date"] = pd.to_datetime(df.date, errors="coerce").dt.date
    for c in NUMERIC:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # Population density is optional in processed source files; missing values
    # should not discard otherwise valid radiance observations.
    required_numeric = [column for column in NUMERIC if column != "pop_density_km2"]
    df = df.dropna(subset=["date", *required_numeric])
    df = df[df.zone_id.isin(ZONES.zone_id) & df.status.isin(STATUSES)]
    df = df.merge(metadata, on="zone_id", how="left")
    df["data_source"] = "Processed CSV"

    # Keep the original multi-date demo timeline when the supplied CSV is a
    # single snapshot, so the date selector remains useful. The source is
    # recorded per row and shown in the dashboard to distinguish simulated
    # history from the processed CSV observation.
    if df["date"].nunique() == 1:
        snapshot_date = df["date"].min()
        history = demo_data()
        history = history[history["date"] < snapshot_date].copy()
        history = history.drop(columns=["zone_name", "lat", "lon"], errors="ignore").merge(
            metadata, on="zone_id", how="left"
        )
        history["data_source"] = "Simulated demo history"
        if not history.empty:
            df = pd.concat([history, df], ignore_index=True)
            mode = "combined"
        else:
            mode = "real"
    else:
        mode = "real"
    return df.sort_values(["date", "zone_id"]).reset_index(drop=True), mode


def rows(df: pd.DataFrame) -> list[dict]:
    return df.astype(object).where(pd.notna(df), None).to_dict(orient="records")


def get_data() -> tuple[pd.DataFrame, str]:
    try:
        return load_data()
    except Exception as e:
        raise HTTPException(503, str(e)) from e


@app.get("/health")
def health():
    return {"status": "ok", "service": "novella-api", "total_zones": len(ZONES)}


@app.get("/api/v1/meta")
def meta():
    _, mode = get_data()
    return {
        "mode": mode,
        "zones": rows(ZONES.drop(columns="base")),
        "required_columns": REQUIRED,
        "statuses": sorted(STATUSES),
    }


@app.get("/api/v1/observations")
def observations(
    observation_date: date | None = None,
    zone_id: str | None = None,
    status: str | None = None,
    limit: int = Query(10000, ge=1, le=50000),
):
    df, _ = get_data()
    if observation_date:
        df = df[df.date == observation_date]
    if zone_id:
        if zone_id not in set(ZONES.zone_id):
            raise HTTPException(404, "Unknown zone_id")
        df = df[df.zone_id == zone_id]
    if status:
        if status not in STATUSES:
            raise HTTPException(422, "Invalid status")
        df = df[df.status == status]
    return {"count": min(len(df), limit), "observations": rows(df.head(limit))}


@app.get("/api/v1/summary")
def summary(observation_date: date | None = None):
    df, mode = get_data()
    if df.empty:
        return {
            "mode": mode,
            "observation_date": observation_date,
            "active_zones": 0,
            "health": "Unknown",
            "flagged_anomalies": 0,
            "observations": [],
        }
    day = observation_date or df.date.max()
    snap = df[df.date == day]
    critical = int((snap.status == "Critical Anomaly").sum())
    flagged = int((snap.status != "Normal Baseline").sum())
    return {
        "mode": mode,
        "observation_date": day,
        "active_zones": int(snap.zone_id.nunique()),
        "health": "Degraded" if critical else "Watch" if flagged else "Nominal",
        "flagged_anomalies": flagged,
        "critical_anomalies": critical,
        "observations": rows(snap),
    }


@app.get("/api/v1/anomalies")
def anomalies(
    observation_date: date | None = None,
    min_abs_z: float = Query(2.0, ge=0, le=20),
):
    df, _ = get_data()
    day = observation_date or df.date.max()
    df = df[(df.date == day) & (df.z_score.abs() >= min_abs_z)]
    df = df.sort_values("z_score", key=lambda s: s.abs(), ascending=False)
    return {"observation_date": day, "count": len(df), "anomalies": rows(df)}


@app.get("/api/v1/recovery/{zone_id}")
def recovery(zone_id: str, start: date | None = None, end: date | None = None):
    if zone_id not in set(ZONES.zone_id):
        raise HTTPException(404, "Unknown zone_id")
    df, _ = get_data()
    df = df[df.zone_id == zone_id]
    if start:
        df = df[df.date >= start]
    if end:
        df = df[df.date <= end]
    df = df.sort_values("date")
    last = df.iloc[-1] if len(df) else None
    return {
        "zone_id": zone_id,
        "observation_count": len(df),
        "baseline_mean": float(df.baseline_mean.iloc[0]) if len(df) else None,
        "latest_status": last.status if last is not None else None,
        "latest_date": last.date if last is not None else None,
        "observations": rows(df),
    }
