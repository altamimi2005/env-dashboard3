# env_dashboard_qatar.py
# Environmental Monitoring Dashboard — with optional OpenAI NLQ
# - Long-format CSVs like: device, measure, units, value, probe_serial, time
# - Station from device serial (baked mapping)
# - Parameter from "measure"
# - Robust time parsing (full datetime or "18:15.0")
# - Tabs: By Parameter, By Station, Compare, Stats, Map, Useful Stats, AI Q&A

import io
import os
import re
import json
from datetime import datetime, timedelta, date
from typing import List
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# Optional map stack
try:
    import folium
    from folium.plugins import HeatMap
    from streamlit_folium import st_folium
except Exception:
    folium = None
    HeatMap = None
    st_folium = None

# ───────────────────────── OpenAI (optional) ─────────────────────────
OPENAI_ENABLED = bool(os.getenv("OPENAI_API_KEY"))
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")  # or gpt-4o-mini

def _openai_chat(prompt: str) -> str:
    """Small helper around OpenAI Chat Completions (new SDK)."""
    from openai import OpenAI
    client = OpenAI()
    try:
        rsp = client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        return rsp.choices[0].message.content.strip()
    except Exception as e:
        return json.dumps({"error": f"{e}"})


# ───────────────────────────── App UI ─────────────────────────────
st.set_page_config(page_title="Environmental Dashboard", layout="wide")
st.title("🌍 Environmental Monitoring Dashboard")

REQUIRED_COLS = ["station_name", "parameter", "timestamp", "value"]

# ───────────────────── Qatar Station Coordinates ─────────────────────
FIXED_STATIONS = {
    "RC-B2":          (25.322, 51.425),
    "Al Batna":       (25.104, 51.174),
    "Abu Samra":      (24.746, 50.823),
    "Al Kharsaah":    (25.239, 51.015),
    "Al Karaanah":    (25.007, 51.036),
    "Turayna":        (24.736, 51.212),
    "Al Wakrah":      (25.193, 51.619),
    "Ghasham Farm":   (24.854, 51.271),
    "Al Jumayliyah":  (25.614, 51.080),
    "Al Khor":        (25.659, 51.464),
    "Dukhan":         (25.406, 50.757),
    "Al Shehaimiyah": (25.857, 50.962),
    "Al Shahaniya":   (25.388, 51.114),
    "Al Ghuwayriyah": (25.841, 51.270),
    "Sudanthile":     (24.632, 51.055),
    "Education City": (25.312, 51.437),
}

# ───────────── Device Serial → Station Name (your table) ─────────────
SERIAL_TO_STATION = {
    "428344E819623C0B":"RC-B2", "47861CE819623CCF":"RC-B2",
    "7B541CE819623C74":"Al Batna", "346844E819623C65":"Al Batna",
    "4E4D1CE819623C54":"Abu Samra", "30251CE819623C47":"Abu Samra",
    "593C1CE819623C6E":"Al Kharsaah", "484F1CE819623CCA":"Al Kharsaah",
    "2A101CE819623CEF":"Al Karaanah", "25061CE819623C7E":"Al Karaanah",
    "5D411CE819623C0A":"Turayna", "1D571CE819623C27":"Turayna",
    "15311CE819623C76":"Al Wakrah", "093144E819623CD9":"Al Wakrah",
    "3D7944E819623CE7":"Ghasham Farm", "1D6B1CE819623C69":"Ghasham Farm",
    "54721CE819623CE1":"Al Jumayliyah", "641C1CE819623C67":"Al Jumayliyah",
    "2E211CE819623C16":"Al Khor", "38811CE819623CDE":"Al Khor",
    "1F0F44E819623CDA":"Dukhan", "485F1CE819623C54":"Dukhan",
    "3A401CE819623CD4":"Al Shehaimiyah", "376A1CE819623CB2":"Al Shehaimiyah",
    "191044E819623C18":"Al Shahaniya", "57581CE819623CA1":"Al Shahaniya",
    "102D44E819623C4A":"Al Ghuwayriyah", "18541CE819623CF6":"Al Ghuwayriyah",
    "2D6244E819623C9E":"Sudanthile", "51631CE819623CFC":"Sudanthile",
}

# ─────────────────────────── Helpers ───────────────────────────
def tidy_axes(fig, x_title, y_title):
    fig.update_layout(
        xaxis_title=x_title, yaxis_title=y_title, font=dict(size=15),
        plot_bgcolor="white", margin=dict(l=10, r=10, t=60, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,0.08)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.08)")
    return fig

def fix_mojibake(s):
    if not isinstance(s, str): return s
    return s.replace("â€“", "–").replace("Âº", "°")

def normalize_serial(x: str) -> str:
    if x is None: return ""
    s = str(x).strip()
    m = re.search(r'#?([0-9A-Fa-f]{16})', s)
    if m: return m.group(1).upper()
    return re.sub(r'[^0-9A-Fa-f]', '', s).upper()

def parse_time_any(series: pd.Series) -> pd.Series:
    if series is None: return pd.Series(pd.NaT, index=[])
    s = series.astype(str).str.strip()
    try:
        fast = pd.to_datetime(s, errors="coerce", infer_datetime_format=True)
        if fast.notna().mean() >= 0.6: return fast
    except Exception: pass
    time_re = re.compile(r"^(\d{1,2}):(\d{2})(?::(\d{2}))?(?:\.(\d+))?$")
    def _to_dt(txt: str):
        m = time_re.match(txt)
        if not m:
            try: return pd.to_datetime(txt, errors="coerce")
            except Exception: return pd.NaT
        h = int(m.group(1)); mnt = int(m.group(2))
        sec = int(m.group(3)) if m.group(3) else 0
        try: return datetime.combine(date.today(), datetime(2000,1,1,h,mnt,sec).time())
        except Exception: return pd.NaT
    return pd.to_datetime(s.map(_to_dt), errors="coerce")

@st.cache_data(show_spinner=False)
def _read_data_csv_robust(file_bytes: bytes) -> pd.DataFrame:
    import chardet
    try: enc = chardet.detect(file_bytes).get("encoding") or "utf-8"
    except Exception: enc = "utf-8"
    txt = file_bytes.decode(enc, errors="replace")
    for sep in [",", ";", "\t", "|"]:
        try:
            df = pd.read_csv(io.StringIO(txt), sep=sep)
            if df.shape[1] > 1: return df
        except Exception: continue
    return pd.read_csv(io.StringIO(txt))

def pick_header(cols, *candidates):
    low = {str(c).strip().lower(): c for c in cols}
    for group in candidates:
        for name in (group if isinstance(group, (list, tuple, set)) else [group]):
            key = str(name).strip().lower()
            if key in low: return low[key]
    return None

def canonical_param(label: str) -> str:
    if not isinstance(label, str): return str(label)
    s = fix_mojibake(label).strip()
    s = re.sub(r"\s+", " ", s)
    low = s.lower()
    low = re.sub(r"\bpm\s*2[.,]?\s*5\b", "pm2.5", low)
    low = re.sub(r"\bpm\s*10\b", "pm10", low)
    low = re.sub(r"\bpm\s*1\b", "pm1", low)
    low = low.replace("wind vane", "wind direction").replace("battery level","battery")
    pretty = low.title().replace("Pm","PM").replace("Pm2.5","PM2.5").replace("Pm10","PM10").replace("Soiling Ratio","SoilingRatio")
    return pretty

def resample_df(ddf: pd.DataFrame, rule: str) -> pd.DataFrame:
    if ddf is None or ddf.empty or "timestamp" not in ddf.columns:
        return pd.DataFrame(columns=REQUIRED_COLS)
    if rule == "none": return ddf
    d = ddf.copy()
    d["timestamp"] = pd.to_datetime(d["timestamp"], errors="coerce")
    d = d.dropna(subset=["timestamp"])
    keys = [c for c in ["station_name","parameter"] if c in d.columns]
    return d.groupby(keys + [pd.Grouper(key="timestamp", freq=rule)], as_index=False)["value"].mean()

# ───────────────────────── Loader ─────────────────────────
@st.cache_data(show_spinner=False)
def load_measure_files(files_payload: List[bytes], clean_text: bool) -> pd.DataFrame:
    frames = []
    for fb in files_payload:
        df_raw = _read_data_csv_robust(fb)
        df_raw.columns = [str(c).replace("\ufeff", "").strip() for c in df_raw.columns]
        cols = list(df_raw.columns)
        if clean_text:
            for c in cols:
                if df_raw[c].dtype == object:
                    df_raw[c] = df_raw[c].map(fix_mojibake)
        device_col  = pick_header(cols, ("device","device #","serial"))
        measure_col = pick_header(cols, ("measure","parameter","label"))
        value_col   = pick_header(cols, ("value","reading"))
        time_col    = pick_header(cols, ("time","timestamp"))
        units_col   = pick_header(cols, ("units","unit"))
        out = pd.DataFrame()
        out["parameter"] = df_raw[measure_col].astype(str).map(canonical_param)
        out["value"] = pd.to_numeric(df_raw[value_col], errors="coerce")
        out["units"] = df_raw[units_col].astype(str) if units_col else ""
        out["timestamp"] = parse_time_any(df_raw[time_col])
        serials = df_raw[device_col].map(normalize_serial)
        out["station_name"] = serials.map(lambda s: SERIAL_TO_STATION.get(s, s))
        out = out.dropna(subset=["timestamp"])
        frames.append(out[["station_name","parameter","timestamp","value","units"]])
    if not frames:
        return pd.DataFrame(columns=REQUIRED_COLS)
    df = pd.concat(frames, ignore_index=True)
    return df

# ───────────────────────── Sidebar ─────────────────────────
with st.sidebar:
    st.header("📥 Upload CSVs")
    data_files = st.file_uploader(
        "Upload CSVs", type=["csv"], accept_multiple_files=True, key="upload_csv"
    )
    fix_text = st.checkbox("Fix encoding", value=True, key="fix_enc")
    show_preview = st.checkbox("Show normalized preview", value=False, key="show_prev")
    ai_switch = st.checkbox("Use OpenAI (if key set)", value=False, key="ai_switch")

if not data_files:
    st.info("Upload one or more CSVs to begin.")
    st.stop()

files_bytes = [f.getvalue() for f in data_files]
df = load_measure_files(files_bytes, fix_text)
if show_preview:
    st.dataframe(df.head(300), use_container_width=True)

if df.empty:
    st.stop()

# ───────────────────────── Filters ─────────────────────────
tmin, tmax = pd.to_datetime(df["timestamp"]).min(), pd.to_datetime(df["timestamp"]).max()
c1, c2, c3, c4 = st.columns(4)
with c1: d_from = st.date_input("From", tmin.date(), key="from_date")
with c2: d_to   = st.date_input("To", tmax.date(), key="to_date")
with c3: t_from = st.time_input("Start Time", tmin.time(), key="from_time")
with c4: t_to   = st.time_input("End Time", tmax.time(), key="to_time")

start_dt = datetime.combine(d_from, t_from)
end_dt   = datetime.combine(d_to, t_to)
downsample = st.selectbox("Downsample", ["none","5min","15min","1H","1D"], index=2, key="downsample")

mask = (df["timestamp"] >= start_dt) & (df["timestamp"] <= end_dt)
df_rs = resample_df(df.loc[mask], downsample)

# ───────────────────────── KPIs ─────────────────────────
st.markdown("### 📊 KPIs")

# Let the user pick which parameter the KPIs should reflect (prevents mixed units)
all_params = sorted(pd.Series(df_rs["parameter"]).dropna().unique().tolist())
def _pick_default_kpi(pars):
    pri = ["soilingratio", "pm2.5", "pm10", "temperature"]
    low = [p.lower() for p in pars]
    for want in pri:
        if want in low:
            return pars[low.index(want)]
    return pars[0] if pars else None

kpi_param_default = _pick_default_kpi(all_params) if all_params else None
kpi_param = st.selectbox(
    "KPI parameter",
    all_params,
    index=all_params.index(kpi_param_default) if kpi_param_default in all_params else 0,
    key="kpi_param_sel",
)

# Subset for the KPI parameter only (avoid mixing units)
kpi_df = df_rs[df_rs["parameter"] == kpi_param].copy()

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.metric(f"Rows ({kpi_param})", f"{len(kpi_df):,}")

# Avg of latest value per station (for this parameter)
with k2:
    if not kpi_df.empty:
        latest_by_station = (
            kpi_df.sort_values("timestamp")
            .groupby("station_name")
            .tail(1)["value"]
            .mean()
        )
        st.metric("Avg (latest per station)", f"{latest_by_station:.3f}" if pd.notna(latest_by_station) else "—")
    else:
        st.metric("Avg (latest per station)", "—")

with k3:
    vmax = kpi_df["value"].max() if not kpi_df.empty else np.nan
    st.metric(f"Max ({kpi_param})", f"{vmax:.3f}" if pd.notna(vmax) else "—")

with k4:
    if not kpi_df.empty:
        latest_any = kpi_df.sort_values("timestamp").tail(1)["value"].iloc[0]
        st.metric(f"Latest overall ({kpi_param})", f"{latest_any:.3f}" if pd.notna(latest_any) else "—")
    else:
        st.metric(f"Latest overall ({kpi_param})", "—")

# ───────────────────────── Tabs ─────────────────────────
tab_param, tab_station, tab_compare, tab_stats, tab_map, tab_ustats, tab_ai = st.tabs(
    ["📊 By Parameter","🏭 By Station","🧭 Compare","📈 Stats & Insights","🗺 Soiling Ratio Map","📚 Useful Stats","🤖 AI Q&A"]
)

# 📊 By Parameter
with tab_param:
    par_opt = sorted(df_rs["parameter"].dropna().unique())
    chosen_param = st.selectbox("Parameter", par_opt, index=0, key="param_main")
    chart_type = st.selectbox("Chart", ["line","scatter","bar","box"], index=0, key="chart_main")
    view_mode = st.radio("View", ["Time series","Latest per station"], horizontal=True, key="view_mode_main")
    dpf = df_rs[df_rs["parameter"] == chosen_param]
    if view_mode == "Time series":
        if chart_type == "line": fig = px.line(dpf, x="timestamp", y="value", color="station_name")
        elif chart_type == "scatter": fig = px.scatter(dpf, x="timestamp", y="value", color="station_name")
        elif chart_type == "bar": fig = px.bar(dpf, x="timestamp", y="value", color="station_name")
        else: fig = px.box(dpf, x="station_name", y="value", color="station_name")
    else:
        latest = dpf.sort_values("timestamp").groupby("station_name").tail(1)
        fig = px.bar(latest, x="station_name", y="value", color="station_name")
    tidy_axes(fig, "Time", chosen_param)
    st.plotly_chart(fig, use_container_width=True)

# 🏭 By Station
with tab_station:
    st_opt = sorted(df_rs["station_name"].dropna().unique())
    chosen_station = st.selectbox("Station", st_opt, index=0, key="station_tab")
    dps = df_rs[df_rs["station_name"] == chosen_station]
    fig = px.line(dps, x="timestamp", y="value", color="parameter", title=chosen_station)
    tidy_axes(fig, "Time", "Value")
    st.plotly_chart(fig, use_container_width=True)

# 🧭 Compare
with tab_compare:
    mode = st.radio("Compare mode", ["By Station","By Parameter"], horizontal=True, key="compare_mode")
    if mode == "By Station":
        c_station = st.selectbox("Choose Station", sorted(df_rs["station_name"].unique()), key="cmp_station")
        sel_params = st.multiselect("Parameters", sorted(df_rs["parameter"].unique()), key="cmp_params")
        layout = st.radio("Layout", ["Overlay","Separate"], horizontal=True, key="cmp_layout")
        for p in sel_params:
            sub = df_rs[(df_rs["station_name"] == c_station) & (df_rs["parameter"] == p)]
            fig = px.line(sub, x="timestamp", y="value", title=f"{c_station} – {p}")
            tidy_axes(fig, "Time", p)
            st.plotly_chart(fig, use_container_width=True)
    else:
        # BY PARAMETER (properly indented under the else block)
        par_opts3 = sorted(df_rs["parameter"].unique().tolist())
        st_opt3 = sorted(df_rs["station_name"].unique().tolist())
        c_param = st.selectbox("Choose Parameter", par_opts3, index=0, key="compare_param")

        sel_stations = st.multiselect("Stations", st_opt3, default=st_opt3[:1], key="compare_stations")
        layout2 = st.radio("Layout", ["Overlay", "Separate"], horizontal=True, key="compare_layout2")

        if not sel_stations:
            st.info("Pick at least one station.")
        else:
            if layout2 == "Overlay":
                sub = df_rs[(df_rs["station_name"].isin(sel_stations)) & (df_rs["parameter"] == c_param)].dropna(subset=["value"])
                if sub.empty:
                    st.info("No data points to plot.")
                else:
                    fig = px.line(sub, x="timestamp", y="value", color="station_name", title=f"{c_param} — overlay")
                    tidy_axes(fig, "Time", c_param)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                # Separate charts, one per station
                for s_id in sel_stations:
                    sub = df_rs[(df_rs["station_name"] == s_id) & (df_rs["parameter"] == c_param)].dropna(subset=["value"])
                    if not sub.empty:
                        fig = px.line(sub, x="timestamp", y="value", title=f"{c_param} – {s_id}")
                        tidy_axes(fig, "Time", c_param)
                        st.plotly_chart(fig, use_container_width=True)

# 📈 Stats & Insights
with tab_stats:
    par_list = sorted(df_rs["parameter"].unique())
    sel_param2 = st.selectbox("Parameter", par_list, index=0, key="param_stats")
    sel_station2 = st.selectbox("Station", sorted(df_rs["station_name"].unique()), index=0, key="station_stats")
    win = st.slider("Rolling window", 3, 200, 24, key="roll_win")
    dstat = df_rs[(df_rs["parameter"] == sel_param2) & (df_rs["station_name"] == sel_station2)].sort_values("timestamp")
    if not dstat.empty:
        dstat["mean"] = dstat["value"].rolling(win, min_periods=1).mean()
        plot_df = dstat.melt(id_vars=["timestamp"], value_vars=["value","mean"], var_name="series", value_name="reading")
        fig = px.line(plot_df, x="timestamp", y="reading", color="series", title=f"{sel_param2} at {sel_station2}")
        tidy_axes(fig, "Time", sel_param2)
        st.plotly_chart(fig, use_container_width=True)
    wide = df_rs.pivot_table(index="timestamp", columns="parameter", values="value", aggfunc="mean").dropna(how="all")
    if not wide.empty and wide.shape[1] >= 2:
        st.subheader("🔗 Correlation heatmap")
        fig_hm = px.imshow(wide.corr(), text_auto=True, color_continuous_scale="RdBu", origin="lower")
        st.plotly_chart(fig_hm, use_container_width=True)

# 🗺 Map
with tab_map:
    latest = df_rs[df_rs["parameter"].str.lower() == "soilingratio"].sort_values("timestamp").groupby("station_name").tail(1)
    rows = []
    for _, r in latest.iterrows():
        if r["station_name"] in FIXED_STATIONS and pd.notna(r["value"]):
            lat, lon = FIXED_STATIONS[r["station_name"]]
            rows.append({"station_name": r["station_name"], "lat": lat, "lon": lon, "soiling": r["value"]})
    map_df = pd.DataFrame(rows)
    if map_df.empty:
        st.info("No SoilingRatio data.")
    else:
        figm = px.scatter_mapbox(
            map_df, lat="lat", lon="lon", color="soiling", size="soiling",
            color_continuous_scale="YlOrRd", zoom=6, hover_name="station_name",
            mapbox_style="open-street-map"
        )
        st.plotly_chart(figm, use_container_width=True)

# 📚 Useful Stats
with tab_ustats:
    d = df_rs.copy()
    d["day"] = pd.to_datetime(d["timestamp"]).dt.floor("D")
    daily = d.groupby(["station_name","parameter","day"], as_index=False)["value"].agg(["count","mean","max","min"]).reset_index()
    st.dataframe(daily.head(300), use_container_width=True)

# 🤖 AI Q&A
with tab_ai:
    st.subheader("🤖 Ask a question about your data")
    st.caption("Examples: 'average PM2.5 at Al Batna last week', 'max Temperature this month', 'count of Battery by station yesterday'.")
    q = st.text_input("Your question", key="ai_q_input")
    run_ai = st.button("Answer with AI", key="ai_btn")

    def _local_guess_time(text: str):
        tl = text.lower()
        now = datetime.now()
        if "last week" in tl:
            return now - timedelta(days=7), now
        if "yesterday" in tl:
            start = datetime(now.year, now.month, now.day) - timedelta(days=1)
            end = start + timedelta(days=1) - timedelta(seconds=1)
            return start, end
        if "this month" in tl:
            return datetime(now.year, now.month, 1), now
        if "last month" in tl:
            first_this = datetime(now.year, now.month, 1)
            last_month_end = first_this - timedelta(seconds=1)
            last_month_start = datetime(last_month_end.year, last_month_end.month, 1)
            return last_month_start, last_month_end
        m = re.search(r"in\s+(\d{4})", tl)
        if m:
            y = int(m.group(1))
            return datetime(y, 1, 1), datetime(y, 12, 31, 23, 59, 59)
        return None, None

    def _local_parse_plan(question: str) -> dict:
        tl = question.lower()
        metric = "mean"
        for k, m in [("max","max"), ("minimum","min"), ("min","min"), ("median","median"),
                     ("count","count"), ("average","mean"), ("avg","mean"), ("mean","mean")]:
            if k in tl:
                metric = m
                break
        params = sorted(set(map(str, df["parameter"].dropna().unique())))
        param = None
        for p in params:
            if p.lower() in tl:
                param = p
                break
        if not param:
            alias = {"pm2.5":"PM2.5","pm 2.5":"PM2.5","pm10":"PM10","pm 10":"PM10","temp":"Temperature"}
            for k, v in alias.items():
                if k in tl and any(v.lower() == pp.lower() for pp in params):
                    param = next(pp for pp in params if pp.lower() == v.lower())
                    break
        stations = sorted(set(map(str, df["station_name"].dropna().unique())))
        station = None
        tl_spaced = f" {tl} "
        for s in stations:
            if f" {s.lower()} " in tl_spaced:
                station = s
                break
        s, e = _local_guess_time(question)
        return {
            "parameter": param,
            "station": station,
            "time": {"start": s.isoformat() if s else None, "end": e.isoformat() if e else None},
            "metric": metric,
            "__q__": question,
        }

    def _execute_plan(plan: dict) -> str:
        sub = df.copy()
        ts = plan.get("time") or {}
        if ts.get("start"):
            try: sub = sub[sub["timestamp"] >= pd.to_datetime(ts["start"])]
            except Exception: pass
        if ts.get("end"):
            try: sub = sub[sub["timestamp"] <= pd.to_datetime(ts["end"])]
            except Exception: pass
        s2, e2 = _local_guess_time(plan.get("__q__", ""))
        if s2: sub = sub[sub["timestamp"] >= s2]
        if e2: sub = sub[sub["timestamp"] <= e2]
        if plan.get("parameter"):
            p = canonical_param(plan["parameter"])
            sub = sub[sub["parameter"].str.lower() == p.lower()]
        if plan.get("station"):
            stn = plan["station"]
            sub = sub[sub["station_name"].str.lower() == str(stn).lower()]
        sub = sub.dropna(subset=["value"])
        if sub.empty:
            return "No matching data."
        metric = (plan.get("metric") or "mean").lower()
        if metric == "max":
            v = sub["value"].max(); when = sub.loc[sub["value"].idxmax(), "timestamp"]
            return f"Max = **{v:.4f}** on **{when}**."
        elif metric == "min":
            v = sub["value"].min(); when = sub.loc[sub["value"].idxmin(), "timestamp"]
            return f"Min = **{v:.4f}** on **{when}**."
        elif metric == "median":
            v = sub["value"].median()
            return f"Median = **{v:.4f}**."
        elif metric == "count":
            v = int(sub["value"].count())
            return f"Count = **{v}** points."
        else:
            v = sub["value"].mean()
            return f"Average = **{v:.4f}**."

    if run_ai and q:
        plan = None
        used_openai = False
        if OPENAI_ENABLED:
            prompt = f"""
You are given a dataframe with columns:
- station_name (string)
- parameter (string)
- timestamp (datetime)
- value (float)

Task: Convert the user's question into a STRICT JSON plan with keys:
{{
  "parameter": <string or null>,
  "station":   <string or null>,
  "time": {{"start": <ISO or null>, "end": <ISO or null>}},
  "metric": "mean" | "max" | "min" | "median" | "count"
}}
Only output valid JSON (no prose). If the question doesn't name a station or parameter, use null.
User question: {q!r}
"""
            raw = _openai_chat(prompt)
            if isinstance(raw, str) and "error" in raw.lower():
                plan = _local_parse_plan(q)
            else:
                try:
                    plan = json.loads(raw)
                    used_openai = True
                except Exception:
                    plan = _local_parse_plan(q)
        else:
            plan = _local_parse_plan(q)

        plan["__q__"] = q
        msg = _execute_plan(plan)
        if OPENAI_ENABLED and not used_openai:
            st.caption("AI call unavailable (package/keys/quota). Used local parser instead.")
        st.markdown(msg)
