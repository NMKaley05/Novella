"""NOVELLA: Lusaka anomaly detection & recovery tracking (Streamlit).

Run:  pip install -r requirements.txt && streamlit run app.py
API:  start FastAPI backend first (uvicorn main:app --reload --port 8000).
      Set NOVELLA_API_URL if hosted elsewhere (defaults to http://127.0.0.1:8000).
"""
import html
import os
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pydeck as pdk
import requests
import streamlit as st

# ---------- Design tokens ----------
BG, NAVY, BORDER = "#0B1424", "#14243A", "rgba(140,155,176,0.20)"
TEXT, MUTED = "#F3F6FA", "#8C9BB0"
TEAL, SKY, AMBER, RED, GREEN = "#20C9C3", "#75B8FF", "#F0B85A", "#E56B76", "#2ca02c"
STATUS_COLOR = {
    "Normal Baseline": GREEN,
    "Moderate Deviation": AMBER,
    "Critical Anomaly": RED,
    "Unusual Decrease": RED,
    "Unusual Increase": AMBER,
}

ZONES = pd.DataFrame([
    ("LSK-01", "Lusaka CBD / Cairo Rd", -15.4167, 28.2833),
    ("LSK-02", "Kabulonga / Woodlands", -15.4350, 28.3320),
    ("LSK-03", "Manda Hill / Arcades Hub", -15.4192, 28.3164),
    ("LSK-04", "Makeni Industrial Area", -15.4680, 28.2480),
    ("LSK-05", "Heavy Industrial Area", -15.4450, 28.2900),
    ("LSK-06", "KK International Airport", -15.3308, 28.4526),
    ("LSK-07", "Kafue Road Corridor", -15.4950, 28.2650),
    ("LSK-08", "Chilenje", -15.4530, 28.3110),
    ("LSK-09", "Matero", -15.3850, 28.2550),
    ("LSK-10", "Kanyama", -15.4320, 28.2510),
    ("LSK-11", "Chelstone", -15.3890, 28.3850),
    ("LSK-12", "Chaisa", -15.3780, 28.2950),
], columns=["zone_id", "zone_name", "lat", "lon"])

NUMERIC = [
    "radiance_nW", "cloud_cover_pct", "rainfall_mm", "fire_count_5km",
    "pop_density_km2", "osm_critical_assets_2km", "baseline_mean",
    "baseline_std", "pct_change", "z_score",
]
REQUIRED = ["zone_id", "date", *NUMERIC, "status", "investigation_note"]
API_BASE_URL = os.environ.get("NOVELLA_API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="NOVELLA | Lusaka anomaly monitor", layout="wide")
st.markdown(f"""<style>
*, *::before, *::after {{ animation: none !important; transition: none !important; }}
.block-container {{ padding-top: 1rem; max-width: 100%; }}
header[data-testid="stHeader"] {{ background: {BG}; }}
div[data-testid="stMetric"] {{ background:{NAVY}; border:1px solid {BORDER}; padding:.6rem .9rem; }}
div[data-testid="stMetricLabel"] p {{ color:{MUTED}; }}
.card {{ background:{NAVY}; border:1px solid {BORDER}; padding:.6rem .8rem; margin-bottom:.5rem; }}
.card:hover {{ border-color:{SKY}; }}
.muted {{ color:{MUTED}; font-size:.8rem; }}
</style>""", unsafe_allow_html=True)


# ---------- Data ----------
def sanitize_text(s: str) -> str:
    """Keep letters, digits, spaces and - _ . ; cap at 48 chars."""
    return re.sub(r"[^\w\s.\-]", "", s, flags=re.UNICODE)[:48].strip()


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    resp = requests.get(f"{API_BASE_URL}/api/v1/observations", params={"limit": 20000}, timeout=10)
    resp.raise_for_status()
    records = resp.json().get("observations", [])
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"API response is missing columns: {', '.join(missing)}")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for c in NUMERIC:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["pop_density_km2"] = df["pop_density_km2"].fillna(0)
    df["investigation_note"] = df["investigation_note"].astype(str)
    before = len(df)
    df = df.dropna(subset=["date", *NUMERIC])
    df = df[df.zone_id.isin(ZONES.zone_id) & df.status.isin(STATUS_COLOR)]
    if len(df) < before:
        st.session_state["dropped"] = before - len(df)
    return df.sort_values(["zone_id", "date"]).reset_index(drop=True)


try:
    with st.spinner("Loading telemetry from NOVELLA API..."):
        data = load_data()
except Exception as exc:  # inline, recoverable error
    st.error(f"Telemetry failed to load from {API_BASE_URL}: {exc}")
    st.caption("Ensure the FastAPI backend is running (`uvicorn main:app --reload --port 8000`).")
    if st.button("Retry"):
        load_data.clear()
        st.rerun()
    st.stop()

if data.empty:
    st.info("No valid telemetry records returned by the API. Check NOVELLA_DATA_PATH and the column schema.")
    st.stop()
if st.session_state.get("dropped"):
    st.warning(f"Dropped {st.session_state['dropped']} malformed records.")

# ---------- Ticker ----------
dates = sorted(data["date"].dt.date.unique())
t1, t2, t3, t4 = st.columns([1, 1, 1, 1.4])
with t4:
    if len(dates) > 1:
        sel_date = st.select_slider("Observation date", options=dates, value=dates[-1], format_func=lambda d: d.isoformat())
    else:
        sel_date = dates[0]
        st.metric("Observation date", sel_date.isoformat())

snap = data[data["date"].dt.date == sel_date].merge(ZONES, on="zone_id")
flagged = snap[snap.status != "Normal Baseline"]
critical = int(flagged.status.isin(["Critical Anomaly", "Unusual Decrease"]).sum())
health = "Degraded" if critical else "Watch" if len(flagged) else "Nominal"
t1.metric("Active zones", snap.zone_id.nunique())
t2.metric("System health", health)
t3.metric("Flagged anomalies", len(flagged))

# ---------- Workspace ----------
left, right = st.columns([7, 3], gap="small")

with left:
    if snap.empty:
        st.info("No observations for this date. Pick another date.")
    else:
        m = snap.copy()
        m["color"] = m.status.map(lambda s: [int(STATUS_COLOR[s][i:i + 2], 16) for i in (1, 3, 5)] + [170])
        m["radius"] = 250 + m["pct_change"].abs().clip(upper=60) * 14
        # pydeck tooltips render HTML, so every dynamic string is escaped.
        m["tip"] = [f"<b>{html.escape(r.zone_name)}</b><br/>{html.escape(r.status)}, z = {r.z_score}"
                    f"<br/>{html.escape(r.investigation_note)}" for r in m.itertuples()]
        layer = pdk.Layer("ScatterplotLayer", m, get_position="[lon, lat]", get_radius="radius",
                          get_fill_color="color", get_line_color=[243, 246, 250], line_width_min_pixels=1,
                          stroked=True, pickable=True, radius_min_pixels=8)
        st.pydeck_chart(pdk.Deck(
            layers=[layer], map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
            initial_view_state=pdk.ViewState(latitude=-15.438, longitude=28.295, zoom=11.3),
            tooltip={"html": "{tip}", "style": {"backgroundColor": NAVY, "color": TEXT, "maxWidth": "320px"}}),
            height=640)
        st.caption("Marker size scales with |% change| from baseline. Colour shows status: "
                   "green normal, amber moderate, red critical.")

with right:
    st.subheader("Recovery timeline", anchor=False)
    zone_id = st.selectbox("Zone", ZONES.zone_id, format_func=lambda z: f"{z} {ZONES.set_index('zone_id').zone_name[z]}")
    s = data[data.zone_id == zone_id]
    if len(s) < 2:
        st.info("Not enough observations to plot a timeline.")
    else:
        mu, sd = s.baseline_mean.iloc[0], s.baseline_std.iloc[0]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[s.date.min(), s.date.max(), s.date.max(), s.date.min()],
                                 y=[mu + 2 * sd, mu + 2 * sd, mu - 2 * sd, mu - 2 * sd], fill="toself",
                                 fillcolor="rgba(117,184,255,0.14)", line=dict(width=0), name="Baseline ±2σ", hoverinfo="skip"))
        fig.add_hline(y=mu, line=dict(color=SKY, dash="dash", width=1))
        fig.add_trace(go.Scatter(x=s.date, y=s.radiance_nW, mode="lines", line=dict(color=TEAL, width=2), name="Observed radiance"))
        f = s[s.status != "Normal Baseline"]
        fig.add_trace(go.Scatter(x=f.date, y=f.radiance_nW, mode="markers", name="Flagged",
                                 marker=dict(symbol="diamond", size=11, color=f.status.map(STATUS_COLOR), line=dict(color=BG, width=1)),
                                 customdata=f.z_score, hovertemplate="%{x|%Y-%m}: %{y} nW, z=%{customdata}<extra></extra>"))
        fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font=dict(color=MUTED), legend=dict(orientation="h", y=-0.2), yaxis_title="nW",
                          xaxis=dict(gridcolor="rgba(140,155,176,.15)"), yaxis=dict(gridcolor="rgba(140,155,176,.15)"))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.subheader("Flagged locations", anchor=False)
    q = sanitize_text(st.text_input("Search zone", max_chars=48, placeholder="e.g. Kabulonga or LSK-04"))
    status_f = st.selectbox("Status", ["All", "Moderate Deviation", "Critical Anomaly", "Unusual Decrease", "Unusual Increase"])
    rows = flagged
    if status_f != "All":
        rows = rows[rows.status == status_f]
    if q:
        rows = rows[rows.zone_id.str.contains(q, case=False, regex=False) | rows.zone_name.str.contains(q, case=False, regex=False)]
    if flagged.empty:
        st.info("No anomalies on this date. All zones are within baseline.")
    elif rows.empty:
        st.info("No flagged zones match. Clear the search or status filter.")
    else:
        for r in rows.sort_values("z_score", key=abs, ascending=False).itertuples():
            st.markdown(
                f"<div class='card'><div style='display:flex;justify-content:space-between;gap:8px'>"
                f"<b>{html.escape(r.zone_id)} <span class='muted'>{html.escape(r.zone_name)}</span></b>"
                f"<span style='color:{STATUS_COLOR[r.status]}'>{html.escape(r.status)}</span></div>"
                f"<div class='muted'>z {r.z_score} · {r.pct_change}% · rain {r.rainfall_mm} mm · fires {r.fire_count_5km} · "
                f"cloud {r.cloud_cover_pct}%</div>"
                f"<div style='font-size:.85rem;margin-top:4px'>{html.escape(r.investigation_note)}</div></div>",
                unsafe_allow_html=True)