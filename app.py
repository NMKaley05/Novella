"""NOVELLA: Lusaka anomaly detection & recovery tracking (Streamlit).

Run:  pip install -r requirements.txt && streamlit run app.py
API:  start FastAPI backend first (uvicorn main:app --reload --port 8000).
      Set NOVELLA_API_URL if hosted elsewhere (defaults to http://127.0.0.1:8000).
"""
import base64
import html
import json
import os
import re
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import streamlit.components.v1 as components

# ---------- Design tokens ----------
BG, NAVY, BORDER = "#030b19", "#07162a", "rgba(70,103,144,0.42)"
TEXT, MUTED = "#F3F6FA", "#9aacc7"
TEAL, SKY, AMBER, RED, GREEN = "#20C9C3", "#75B8FF", "#F0B85A", "#E56B76", "#52dda0"
PINK = "#f04eb5"

earth_path = Path(__file__).with_name("earth-background.jpg")
EARTH_IMAGE = base64.b64encode(earth_path.read_bytes()).decode("ascii") if earth_path.is_file() else ""
HERO_BG_IMAGE = (
    f'linear-gradient(90deg,rgba(2,8,20,.86) 0%,rgba(3,10,23,.58) 44%,rgba(3,10,23,.12) 100%),'
    f'linear-gradient(0deg,rgba(3,9,20,.27),rgba(3,9,20,.08)),'
    f'url("data:image/jpeg;base64,{EARTH_IMAGE}")'
    if EARTH_IMAGE
    else "linear-gradient(120deg, #071832 0%, #0d274d 50%, #1a153a 100%)"
)

STATUS_COLOR = {
    "Normal Baseline": GREEN,
    "Moderate Deviation": AMBER,
    "Critical Anomaly": RED,
    "Unusual Decrease": RED,
    "Unusual Increase": AMBER,
}

ZONES = pd.DataFrame(
    [
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
    ],
    columns=["zone_id", "zone_name", "lat", "lon"],
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
API_BASE_URL = os.environ.get("NOVELLA_API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="NOVELLA | Lusaka anomaly monitor", page_icon="✦", layout="wide")
st.markdown(
    f"""<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');
*, *::before, *::after {{ font-family:'DM Sans',sans-serif; box-sizing:border-box; }}
.stApp {{background:radial-gradient(ellipse at 76% 0%,#112543 0%,#050d1c 42%,#020713 100%);color:{TEXT};}}
.block-container {{ padding:2.8rem 1.5rem 2rem !important; max-width:1700px; overflow-x:hidden; }}
header[data-testid="stHeader"] {{ background:transparent; }}
.brand {{display:flex;align-items:center;gap:14px;padding:7px 0 16px;border-bottom:1px solid #19314e;margin-bottom:20px;}}
.brand-mark {{width:48px;height:48px;border-radius:50%;display:grid;place-items:center;background:radial-gradient(circle at 35% 30%,#2d67b9,#071a3b 70%);border:1px solid #3b6cab;font-size:24px;}}
.brand-name {{font-weight:800;letter-spacing:5px;font-size:25px;line-height:1.1;}}
.brand-sub {{color:#c1cce0;font-size:10px;letter-spacing:2.5px;margin-top:5px;}}
.nav {{margin-left:auto;display:flex;gap:8px;flex-wrap:wrap;}}
.nav span {{border:1px solid #263d5e;background:#0c1a30;border-radius:24px;padding:9px 13px;font-size:12px;color:#e1e9f8;}}
.nav .active {{background:linear-gradient(100deg,#e951b0,#8e4a9d);border-color:#f45fbd;}}
.hero {{position:relative;isolation:isolate;overflow:hidden;min-height:176px;padding:20px 22px 20px;border-radius:14px;margin:0 0 8px;}}
.hero::before {{content:"";position:absolute;z-index:-2;inset:0;background-image:{HERO_BG_IMAGE};background-position:center,center,center 53%;background-size:cover;}}
.hero::after {{content:"";position:absolute;z-index:-1;inset:auto 0 0;height:32px;background:linear-gradient(0deg,rgba(2,8,18,.38),transparent);}}
.hero h1,.hero p {{position:relative;z-index:1;}}
.hero h1 {{font-size:clamp(25px,3vw,39px);line-height:1.16;letter-spacing:-.8px;margin:0 0 10px;color:#f4f7ff;max-width:690px;}}
.hero h1 em {{color:{PINK};font-style:normal;}}
.hero p {{color:#c0ccdf;font-size:15px;margin:0;border-left:3px solid {PINK};padding:5px 0 5px 13px;}}
.panel {{background:linear-gradient(145deg,rgba(8,24,45,.97),rgba(3,13,28,.97));border:1px solid #203d60;border-radius:16px;padding:16px 17px;box-shadow:0 12px 35px rgba(0,0,0,.25);}}
.panel-title {{display:flex;align-items:center;gap:10px;font-weight:700;font-size:17px;margin:0 0 13px;}}
.panel-title b {{color:{PINK};font-size:22px;}}
.metric {{border-bottom:1px solid rgba(95,126,164,.18);padding:11px 0;}}
.metric:last-child {{border-bottom:0;padding-bottom:0;}}
.metric-label {{color:#c1cde0;font-size:13px;}}
.metric-value {{font-weight:700;font-size:25px;margin-top:3px;}}
.metric-note {{font-size:11px;color:#91a5c3;margin-top:3px;}}
.sat-note {{color:#cfdbec;font-size:10px;background:#071326dd;border:1px solid #57718d;border-radius:20px;padding:5px 9px;}}
.card {{background:{NAVY};border:1px solid {BORDER};border-radius:12px;padding:.7rem .85rem;margin-bottom:.5rem;}}
.card:hover {{border-color:{SKY};}}
.muted {{color:{MUTED};font-size:.8rem;}}
div[data-testid="stMetric"] {{background:transparent;border:0;padding:.1rem 0;}}
div[data-testid="stMetricLabel"] p {{color:{MUTED};}}
</style>""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="brand"><div class="brand-mark">✦</div>
  <div><div class="brand-name">NOVELLA</div><div class="brand-sub">ENVIRONMENTAL INTELLIGENCE FOR LUSAKA</div></div>
  <div class="nav"><span class="active">⌂ &nbsp;Home</span><span>⌖ &nbsp;Explore Map</span><span>▥ &nbsp;Trends</span><span>▦ &nbsp;Data &amp; Analysis</span><span>✧ &nbsp;Insights</span><span>ⓘ &nbsp;About</span><span>◎ &nbsp;Lusaka</span></div>
</div>
<div class="hero"><h1>Understanding how our cities are changing,<br>one <em>satellite image</em> at a time.</h1>
<p>Explore Lusaka's environmental changes through satellite imagery and Earth observation data.</p></div>
""",
    unsafe_allow_html=True,
)


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
except Exception as exc:
    st.error(f"Telemetry failed to load from {API_BASE_URL}: {exc}")
    st.caption("Ensure the FastAPI backend is running (`python -m uvicorn main:app --reload --port 8000`).")
    if st.button("Retry"):
        load_data.clear()
        st.rerun()
    st.stop()

if data.empty:
    st.info("No valid telemetry records returned by the API. Check NOVELLA_DATA_PATH and the column schema.")
    st.stop()
if st.session_state.get("dropped"):
    st.warning(f"Dropped {st.session_state['dropped']} malformed records.")

# ---------- Dashboard ----------
dates = sorted(data["date"].dt.date.unique())
date_col, spacer = st.columns([1.2, 4])
with date_col:
    sel_date = (
        st.select_slider("Observation date", options=dates, value=dates[-1], format_func=lambda d: d.isoformat())
        if len(dates) > 1
        else dates[0]
    )

# Drop existing zone_name/lat/lon before merging ZONES to prevent lat_x/lat_y suffixes
snap_base = data[data["date"].dt.date == sel_date].drop(columns=["zone_name", "lat", "lon"], errors="ignore")
snap = snap_base.merge(ZONES, on="zone_id")

flagged = snap[snap.status != "Normal Baseline"]
critical = int(flagged.status.isin(["Critical Anomaly", "Unusual Decrease"]).sum())
health = "Degraded" if critical else "Watch" if len(flagged) else "Nominal"

left, center, right = st.columns([1.0, 2.25, 1.35], gap="small")
with left:
    st.markdown('<div class="panel"><div class="panel-title"><b>▥</b> Lusaka Overview</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="metric"><div class="metric-label">Active zones</div><div class="metric-value">{snap.zone_id.nunique()}</div><div class="metric-note">Monitoring locations</div></div>',
        unsafe_allow_html=True,
    )
    health_color = RED if health == "Degraded" else AMBER if health == "Watch" else GREEN
    st.markdown(
        f'<div class="metric"><div class="metric-label">System health</div><div class="metric-value" style="color:{health_color}">{html.escape(health)}</div><div class="metric-note">Status for {sel_date}</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="metric"><div class="metric-label">Flagged anomalies</div><div class="metric-value">{len(flagged)}</div><div class="metric-note">{critical} critical · {len(flagged)-critical} moderate</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="metric"><div class="metric-label">Satellite context</div><div class="metric-note">Imagery basemap with monitored zone markers.</div></div></div>',
        unsafe_allow_html=True,
    )

with center:
    st.markdown(
        '<div class="panel-title"><b>⌖</b> Lusaka Satellite Map <span class="sat-note">SATELLITE IMAGERY</span></div>',
        unsafe_allow_html=True,
    )
    map_rows = [
        {
            "lat": float(r.lat),
            "lon": float(r.lon),
            "name": str(r.zone_name),
            "zone": str(r.zone_id),
            "status": str(r.status),
            "change": float(r.pct_change),
            "z": float(r.z_score),
            "color": STATUS_COLOR[r.status],
        }
        for r in snap.itertuples()
    ]
    map_json = json.dumps(map_rows).replace("</", "<\\/")
    map_html = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"><style>
    html,body,#map{{height:100%;width:100%;margin:0;background:#081525}}#map{{border-radius:12px;overflow:hidden}}
    .leaflet-control-attribution{{font-size:9px!important}}.leaflet-popup-content-wrapper,.leaflet-popup-tip{{background:#07162a;color:#f4f7ff}}
    .leaflet-bar a{{background:#07162a;color:white;border-color:#344d6e}}
    .legend{{background:#07162aee;color:#e5edf8;padding:9px 11px;border:1px solid #45617f;border-radius:9px;font:11px 'Arial',sans-serif;line-height:1.7}}
    .dot{{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:5px}}
    </style></head><body><div id="map"></div><script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script><script>
    const zones={map_json};const map=L.map('map',{{zoomControl:false}}).setView([-15.425,28.315],11);
    L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}',{{maxZoom:19,attribution:'Tiles © Esri — Source: Esri, Maxar, Earthstar Geographics, and the GIS User Community'}}).addTo(map);
    L.control.zoom({{position:'topright'}}).addTo(map);
    const esc=s=>String(s).replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));
    zones.forEach(z=>{{const marker=L.circleMarker([z.lat,z.lon],{{radius:8+Math.min(Math.abs(z.change),30)/6,color:'#fff',weight:2,fillColor:z.color,fillOpacity:.92}}).addTo(map);
      marker.bindPopup('<b>'+esc(z.name)+'</b> <span style="color:#9aacc7">('+esc(z.zone)+')</span><br>'+esc(z.status)+' · Δ '+esc(z.change)+'% · z '+esc(z.z));}});
    const legend=L.control({{position:'bottomleft'}});legend.onAdd=()=>{{const d=L.DomUtil.create('div','legend');d.innerHTML='<b>Zone status</b><br><span class="dot" style="background:{GREEN}"></span>Normal &nbsp; <span class="dot" style="background:{AMBER}"></span>Moderate<br><span class="dot" style="background:{RED}"></span>Critical';return d;}};legend.addTo(map);
    L.control.scale({{imperial:false,position:'bottomright'}}).addTo(map);
    </script></body></html>"""
    components.html(map_html, height=410, scrolling=False)
    st.caption("Marker size scales with deviation from baseline. Colour shows zone status. Imagery © Esri and its data providers.")

with right:
    st.markdown('<div class="panel"><div class="panel-title"><b>⌁</b> Environmental Trends</div>', unsafe_allow_html=True)
    zone_names = ZONES.set_index("zone_id")["zone_name"].to_dict()
    zone_id = st.selectbox("Zone", ZONES.zone_id, format_func=lambda z: f"{zone_names[z]} ({z})")
    s = data[data.zone_id == zone_id]
    if len(s) < 2:
        st.info("Not enough observations to plot a timeline.")
    else:
        mu, sd = s.baseline_mean.iloc[0], s.baseline_std.iloc[0]
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=[s.date.min(), s.date.max(), s.date.max(), s.date.min()],
                y=[mu + 2 * sd, mu + 2 * sd, mu - 2 * sd, mu - 2 * sd],
                fill="toself",
                fillcolor="rgba(117,184,255,.12)",
                line=dict(width=0),
                name="Baseline ±2σ",
                hoverinfo="skip",
            )
        )
        fig.add_hline(y=mu, line=dict(color=SKY, dash="dash", width=1))
        fig.add_trace(
            go.Scatter(
                x=s.date,
                y=s.radiance_nW,
                mode="lines+markers",
                line=dict(color=PINK, width=2.5),
                marker=dict(size=5, color=PINK),
                name="Observed radiance",
            )
        )
        f = s[s.status != "Normal Baseline"]
        fig.add_trace(
            go.Scatter(
                x=f.date,
                y=f.radiance_nW,
                mode="markers",
                name="Flagged",
                marker=dict(
                    symbol="diamond",
                    size=10,
                    color=f.status.map(STATUS_COLOR),
                    line=dict(color=BG, width=1),
                ),
                customdata=f.z_score,
                hovertemplate="%{x|%Y-%m}: %{y} nW, z=%{customdata}<extra></extra>",
            )
        )
        fig.update_layout(
            height=260,
            margin=dict(l=0, r=0, t=8, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=MUTED, size=10),
            legend=dict(orientation="h", y=-0.22),
            yaxis_title="nW",
            xaxis=dict(gridcolor="rgba(140,155,176,.15)"),
            yaxis=dict(gridcolor="rgba(140,155,176,.15)"),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.markdown('<div class="panel-title" style="font-size:15px;margin-top:8px"><b>⌖</b> Flagged Locations</div>', unsafe_allow_html=True)
    q = sanitize_text(st.text_input("Search zone", max_chars=48, placeholder="e.g. Kabulonga or Chilenje"))
    status_f = st.selectbox("Status", ["All", "Moderate Deviation", "Critical Anomaly", "Unusual Decrease", "Unusual Increase"])
    filtered = flagged
    if status_f != "All":
        filtered = filtered[filtered.status == status_f]
    if q:
        filtered = filtered[
            filtered.zone_id.str.contains(q, case=False, regex=False)
            | filtered.zone_name.str.contains(q, case=False, regex=False)
        ]
    if flagged.empty:
        st.info("No anomalies on this date. All zones are within baseline.")
    elif filtered.empty:
        st.info("No flagged zones match. Clear the search or status filter.")
    else:
        for r in filtered.sort_values("z_score", key=abs, ascending=False).itertuples():
            st.markdown(
                f"<div class='card'><b>{html.escape(r.zone_name)} <span class='muted'>({html.escape(r.zone_id)})</span></b> "
                f"<span style='color:{STATUS_COLOR[r.status]}'>{html.escape(r.status)}</span>"
                f"<div class='muted'>z {r.z_score} · {r.pct_change}% · rain {r.rainfall_mm} mm · fires {r.fire_count_5km} · cloud {r.cloud_cover_pct}%</div>"
                f"<div style='font-size:.82rem;margin-top:4px'>{html.escape(r.investigation_note)}</div></div>",
                unsafe_allow_html=True,
            )
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
about, trend, insight_col = st.columns([1.15, 1.35, 1], gap="small")
with about:
    mode_label = "Configured data" if os.getenv("NOVELLA_DATA_PATH") else "Demonstration data"
    st.markdown(
        f'<div class="panel"><div class="panel-title"><b>✧</b> About NOVELLA</div>'
        f'<p style="color:#cfdbec;font-size:13px;line-height:1.55">NOVELLA monitors nighttime radiance changes across Lusaka zones to help investigate potential disruptions and environmental signals.</p>'
        f'<span class="muted">Data source mode: {html.escape(mode_label)}</span></div>',
        unsafe_allow_html=True,
    )
with trend:
    st.markdown(
        '<div class="panel"><div class="panel-title"><b>▧</b> Recovery Timeline</div>'
        '<p style="color:#cfdbec;font-size:13px;line-height:1.55">Select a zone in Environmental Trends to inspect its radiance history against the expected baseline range.</p>'
        '<span class="muted">Shaded region: baseline ±2 standard deviations.</span></div>',
        unsafe_allow_html=True,
    )
with insight_col:
    st.markdown('<div class="panel"><div class="panel-title"><b>✦</b> Key Insights</div>', unsafe_allow_html=True)
    if flagged.empty:
        st.markdown('<div class="card">✓ &nbsp; All monitored zones are within baseline on this date.</div>', unsafe_allow_html=True)
    else:
        for r in flagged.sort_values("z_score", key=abs, ascending=False).head(3).itertuples():
            st.markdown(
                f'<div class="card"><span style="color:{STATUS_COLOR[r.status]}">●</span> &nbsp;{html.escape(r.zone_name)}'
                f'<br><span class="muted">{html.escape(r.status)} · {r.pct_change:+.1f}% vs baseline</span></div>',
                unsafe_allow_html=True,
            )
    st.markdown("</div>", unsafe_allow_html=True)