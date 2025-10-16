import io
import re
import os
import base64
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.dates import AutoDateLocator, ConciseDateFormatter

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage

# ================= PAGE CONFIG & THEME =================
st.set_page_config(page_title="Day-wise Analysis Dashboard", page_icon="📈", layout="wide")

st.markdown("""
<style>
/* Page bg */
[data-testid="stAppViewContainer"] {
  background: linear-gradient(135deg, #0ea5e9 0%, #e2e8f0 35%, #ffffff 100%);
}

/* Hide df index */
thead tr th:first-child {display:none} tbody th {display:none}

/* Card */
.card {
  background: rgba(255,255,255,0.75);
  border-radius: 16px;
  padding: 18px 20px;
  box-shadow: 0 10px 30px rgba(2, 6, 23, 0.12);
  border: 1px solid rgba(148,163,184,0.25);
  margin-bottom: 12px;
}

/* Metrics */
.metric { font-size:14px; letter-spacing:.4px; color:#475569; }
.metric strong { font-size:22px; color:#0f172a; }
h1, h2, h3 { color:#0f172a !important; }
.small-note{ color:#64748b; font-size:12px; }

/* Stat cards */
.stat-card {
  background: rgba(255,255,255,0.9);
  border: 1px solid rgba(148,163,184,0.35);
  border-radius: 16px;
  padding: 16px;
  box-shadow: 0 8px 24px rgba(2,6,23,0.10);
  margin-top: 8px;
}
.stat-title { font-size: 18px; font-weight: 700; margin: 0 0 10px 0; color:#0f172a; }
.subtle { color:#475569; font-size: 13px; margin-bottom: 12px; }
.chips { display:flex; flex-wrap:wrap; gap:8px; }
.chip { border:1px solid rgba(148,163,184,0.35); background:#ffffffcc; padding:8px 10px; border-radius:12px; display:flex; flex-direction:column; min-width:120px; }
.chip .label { font-size:11px; color:#64748b; letter-spacing:.2px; }
.chip .value { font-size:14px; font-weight:700; color:#0f172a; }
.group-head { font-size:13px; font-weight:700; color:#0f172a; margin:12px 0 6px; }

/* Total chips */
.total-chip {
  border: 1px solid rgba(148,163,184,0.35);
  background: linear-gradient(180deg, #ffffffcc, #f8fafc);
  padding: 14px 16px; border-radius: 14px;
  box-shadow: 0 6px 16px rgba(2,6,23,.08);
}
.total-chip .label { font-size:12px; color:#64748b; }
.total-chip .value { font-size:20px; font-weight:800; color:#0f172a; margin-top:4px; }

/* Header bits */
.badge{
  display:inline-block; margin-top:4px; padding:4px 10px;
  border-radius:999px; font-size:12px; letter-spacing:.3px;
  background:#0ea5e91a; color:#0f172a; border:1px solid #0ea5e966;
  backdrop-filter: blur(6px);
}
.glass{
  background: rgba(255,255,255,0.6);
  border:1px solid rgba(148,163,184,0.25);
  border-radius:18px;
  padding:16px 18px;
  box-shadow: 0 12px 36px rgba(2,6,23,.12);
  margin-top:10px;
}

/* Flex header for perfect alignment */
.header {
  display:flex; align-items:center; gap:14px;
  margin: 4px 0 10px 0;
}
.header .logo {
  width:84px; height:auto; border-radius:8px; object-fit:contain;
}
.header .title-group h1 {
  margin:0; line-height:1.1; font-size:34px; font-weight:800;
}
.header .subtitle { margin-top:4px; }
@media (max-width: 680px){
  .header { flex-direction:column; align-items:flex-start; gap:8px; }
  .header .title-group h1 { font-size:26px; }
}
</style>
""", unsafe_allow_html=True)

# ===================== HEADER =====================
logo_path = os.path.join("assets", "logo.png")

def _img_b64(p):
    if os.path.exists(p):
        with open(p, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    return None

logo_b64 = _img_b64(logo_path)

st.markdown(
    f"""
    <div class="header">
      {f'<img class="logo" src="data:image/png;base64,{logo_b64}" />' if logo_b64 else ''}
      <div class="title-group">
        <h1>Day-wise Analysis Dashboard</h1>
        <div class="badge subtitle">Upload your CSV/Excel, pick a date range, explore daily/weekly/monthly insights, and export a PDF report </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Start a soft glass wrapper for main content
st.markdown("<div class='glass'>", unsafe_allow_html=True)

# =================== SIDEBAR: "Data" WITH DEMO TOGGLE ONLY ===================
DEMO_PATHS = ["data_ana.csv"]  

def load_demo_df_or_error():
    last_err = None
    for p in DEMO_PATHS:
        if os.path.exists(p):
            try:
                return pd.read_csv(p), f"Loaded demo file: {p}"
            except Exception as e:
                last_err = f"{p}: {e}"
    st.error(
        "Demo file not found/readable.\n"
        f"{'Last error: ' + str(last_err) if last_err else ''}"
    )
    st.stop()

with st.sidebar:
    st.markdown("##  Data")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    use_demo = st.toggle(
        "Use bundled demo CSV",
        value=False,
        help="Loads /mnt/data/months.csv (no synthetic data)."
    )
    st.markdown('</div>', unsafe_allow_html=True)

    df = None
    if use_demo:
        df, msg = load_demo_df_or_error()
        st.success(msg)
    else:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        uploaded = st.file_uploader("Upload CSV/Excel", type=["csv", "xlsx", "xls"])
        st.markdown('</div>', unsafe_allow_html=True)
        if uploaded is not None:
            name = uploaded.name.lower()
            df = pd.read_csv(uploaded) if name.endswith(".csv") else pd.read_excel(uploaded)

if df is None:
    st.info("Toggle the demo in the sidebar or upload a file.")
    st.stop()

# ================= DATA NORMALIZATION =================
# Very early normalization for 'Date'
if "Date" in df.columns:
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values(by="Date").reset_index(drop=True)
else:
    alt = next((c for c in df.columns if str(c).strip().lower() == "date"), None)
    if alt:
        df.rename(columns={alt: "Date"}, inplace=True)
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
        df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"]).sort_values(by="Date").reset_index(drop=True)

# Clean column names
df.columns = [str(c).strip() for c in df.columns]

# ---------- DROP UNWANTED PRODUCTION COLUMNS ----------
to_drop_patterns = [
    r'^Bran Production \( ?Per Bag ?/?Kg ?\) #(?:2|3|4|5|6)$',
    r'^Broken Production \( ?Per Bag ?/?Kg ?\) #(?:2|3|4|5|7|8|9|10|11|12)$',
    r'^Broken Production \( ?Per Bag ?Kg ?\) #6$',
    r'^Husk Production \( ?Per ?day/?Kg ?\) #(?:2|3|4)$',
]
drop_cols = []
for col in list(df.columns):
    name = str(col).strip()
    for pat in to_drop_patterns:
        if re.match(pat, name, flags=re.IGNORECASE):
            drop_cols.append(col); break
if drop_cols:
    df = df.drop(columns=drop_cols)
    with st.expander(f"Removed {len(drop_cols)} unwanted production column(s)"):
        st.write(sorted(map(str, drop_cols)))

# =============== DATE COLUMN PICKER (fallback-friendly) ===============
likely_date_cols = [c for c in df.columns if "date" in c.lower()]
datetime_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.datetime64)]
date_col_default = "Date" if "Date" in df.columns else (
    datetime_cols[0] if datetime_cols else (likely_date_cols[0] if likely_date_cols else df.columns[0])
)
date_col = st.selectbox("Select the date column", options=df.columns.tolist(),
                        index=(df.columns.tolist().index(date_col_default) if date_col_default in df.columns else 0))

df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
df = df.dropna(subset=[date_col]).copy()

# =============== DATE RANGE PICKER ===============
df["__date__"] = df[date_col].dt.date
min_d, max_d = df["__date__"].min(), df["__date__"].max()

st.markdown('<div class="card">', unsafe_allow_html=True)
colA, colB = st.columns(2)
with colA:
    start_date = st.date_input("Start date", value=min_d, min_value=min_d, max_value=max_d)
with colB:
    end_date = st.date_input("End date", value=max_d, min_value=min_d, max_value=max_d)
st.markdown('</div>', unsafe_allow_html=True)

if start_date > end_date:
    st.error("Start date cannot be after End date."); st.stop()

mask = (df["__date__"] >= start_date) & (df["__date__"] <= end_date)
dff = df.loc[mask].copy()
if dff.empty:
    st.warning("No rows in the selected date range."); st.stop()

# =============== NUMERIC COLUMNS ===============
numeric_cols = dff.select_dtypes(include=[np.number]).columns.tolist()
selected_numeric = st.multiselect("Select numeric column(s) to aggregate (sum & mean)",
                                  options=numeric_cols, default=numeric_cols[: min(3, len(numeric_cols))])

# =============== ROLLUPS ===============
@st.cache_data(show_spinner=False)
def build_rollup(frame: pd.DataFrame, freq: str, date_key: str, cols: tuple) -> pd.DataFrame:
    g = frame.groupby(pd.Grouper(key=date_key, freq=freq))
    roll = g.size().rename("row_count").to_frame()
    for c in cols:
        roll[f"sum_{c}"] = g[c].sum(min_count=1)
        roll[f"mean_{c}"] = g[c].mean()
    roll = roll.reset_index().rename(columns={date_key: "date"})
    return roll.sort_values("date")

dff[date_col] = pd.to_datetime(dff[date_col], errors="coerce")
daily   = build_rollup(dff, "D",     date_col, tuple(selected_numeric))
weekly  = build_rollup(dff, "W-MON", date_col, tuple(selected_numeric))
monthly = build_rollup(dff, "MS",    date_col, tuple(selected_numeric))
for _df in (daily, weekly, monthly):
    _df["date"] = pd.to_datetime(_df["date"], errors="coerce")

# =============== SUMMARY CARDS ===============
total_rows = int(len(dff))
total_days = int(daily.shape[0])
avg_rows_per_day = float(daily["row_count"].mean()) if total_days else 0.0
peak_day = daily.loc[daily["row_count"].idxmax()] if not daily.empty else None
if daily["row_count"].std(ddof=0) > 0:
    z = (daily["row_count"] - daily["row_count"].mean()) / daily["row_count"].std(ddof=0)
    daily["anomaly_flag"] = np.where(np.abs(z) >= 2, "⚠️", "")
else:
    daily["anomaly_flag"] = ""

st.markdown('<div class="card">', unsafe_allow_html=True)
m1, m2, m3, m4 = st.columns(4)
m1.markdown(f'<div class="metric">Total rows<br><strong>{total_rows:,}</strong></div>', unsafe_allow_html=True)
m2.markdown(f'<div class="metric">Days in range<br><strong>{total_days}</strong></div>', unsafe_allow_html=True)
m3.markdown(f'<div class="metric">Avg rows/day<br><strong>{avg_rows_per_day:.2f}</strong></div>', unsafe_allow_html=True)
if peak_day is not None:
    m4.markdown(f'<div class="metric">Peak day<br><strong>{peak_day["date"].date()} • {int(peak_day["row_count"]):,}</strong></div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# =============== Indian number formatting helpers ===============
def _choose_auto_decimals(v: float) -> int:
    av = abs(v)
    if av >= 1_000_000: return 0
    if av >= 1_000: return 0
    if av >= 100: return 1
    if av >= 1: return 2
    if av >= 0.01: return 3
    return 4

def _indian_group_format(abs_value: float, decimals: int) -> str:
    s = f"{abs_value:.{decimals}f}"
    if "." in s: int_part, frac_part = s.split(".")
    else: int_part, frac_part = s, ""
    if len(int_part) > 3:
        head, tail = int_part[:-3], int_part[-3:]
        groups = []
        while len(head) > 2:
            groups.insert(0, head[-2:]); head = head[:-2]
        if head: groups.insert(0, head)
        int_part = ",".join(groups) + "," + tail
    return int_part + (("." + frac_part) if decimals > 0 else "")

def _fmt_num(v, decimals=None, compact=True):
    if pd.isna(v): return "—"
    v = float(v); av = abs(v); dec = _choose_auto_decimals(v) if decimals is None else decimals
    sign = "-" if v < 0 else ""
    if compact and av >= 1_000:
        if av >= 1e7:  return f"{sign}{av/1e7:.{dec}f} Cr"
        if av >= 1e5:  return f"{sign}{av/1e5:.{dec}f} Lakh"
        return f"{sign}{av/1e3:.{dec}f} Thousand"
    return f"{sign}{_indian_group_format(av, dec)}"

def _apply_indian_y_ticks(fig: go.Figure, values: pd.Series, n_ticks: int = 6) -> go.Figure:
    vals = pd.to_numeric(values, errors="coerce").dropna()
    if vals.empty: return fig
    vmin, vmax = float(vals.min()), float(vals.max())
    if vmin == vmax: vmin = 0.0 if vmax >= 0 else vmax * 0.9
    ticks = np.linspace(vmin, vmax, num=max(2, n_ticks))
    labels = [_fmt_num(t, decimals=2, compact=True) for t in ticks]
    fig.update_yaxes(tickmode="array", tickvals=list(ticks), ticktext=labels)
    return fig

# =============== Plotly chart helpers ===============
def combined_sum_mean_line(df_roll: pd.DataFrame, col: str, title: str, tickfmt: str = "%b %d, %Y") -> go.Figure:
    ysum, ymean = f"sum_{col}", f"mean_{col}"
    long = df_roll[["date", ysum, ymean]].melt("date", var_name="metric", value_name="value")
    long["date"] = pd.to_datetime(long["date"], errors="coerce")
    fig = px.line(long, x="date", y="value", color="metric", markers=True, title=title,
                  labels={"date": "Date", "value": "Value", "metric": "Metric"})
    fig.update_xaxes(type="date", tickformat=tickfmt)
    return _apply_indian_y_ticks(fig, long["value"])

def line_rows(df_roll: pd.DataFrame, title: str, tickfmt: str = "%b %d, %Y") -> go.Figure:
    df_plot = df_roll.copy(); df_plot["date"] = pd.to_datetime(df_plot["date"], errors="coerce")
    fig = px.line(df_plot, x="date", y="row_count", markers=True, title=title,
                  labels={"date": "Date", "row_count": "Count"})
    fig.update_xaxes(type="date", tickformat=tickfmt)
    return _apply_indian_y_ticks(fig, df_plot["row_count"])

# =============== TABS ===============
st.subheader(" Trends by Granularity")
tab_d, tab_w, tab_m = st.tabs(["Daily", "Weekly", "Monthly"])
with tab_d:
    st.markdown("**Daily activity and metrics**")
    st.plotly_chart(line_rows(daily, "Rows per Day", tickfmt="%d %b"), use_container_width=True)
    for col in selected_numeric:
        st.plotly_chart(combined_sum_mean_line(daily, col, f"{col}: Daily Sum & Mean", tickfmt="%d %b"), use_container_width=True)
    st.subheader(" Day-wise Table"); st.dataframe(daily, use_container_width=True)
with tab_w:
    st.markdown("**Weekly activity and metrics**")
    st.plotly_chart(line_rows(weekly, "Rows per Week", tickfmt="%d %b %Y"), use_container_width=True)
    for col in selected_numeric:
        st.plotly_chart(combined_sum_mean_line(weekly, col, f"{col}: Weekly Sum & Mean", tickfmt="%d %b %Y"), use_container_width=True)
    with st.expander("Weekly Table"): st.dataframe(weekly, use_container_width=True)
with tab_m:
    st.markdown("**Monthly activity and metrics**")
    st.plotly_chart(line_rows(monthly, "Rows per Month", tickfmt="%b %Y"), use_container_width=True)
    for col in selected_numeric:
        st.plotly_chart(combined_sum_mean_line(monthly, col, f"{col}: Monthly Sum & Mean", tickfmt="%b %Y"), use_container_width=True)
    with st.expander("Monthly Table"): st.dataframe(monthly, use_container_width=True)

# =============== Advanced stats (daily sum only) ===============
st.subheader(" Advanced Statistical Summary & Trends")
st.markdown('<div class="card">', unsafe_allow_html=True)
c1, c2, c3 = st.columns([1,1,2])
fmt_choice = c1.selectbox("Number format", ["Auto", "0 decimals", "1 decimal", "2 decimals", "3 decimals"], index=0)
compact_kmb = c2.checkbox("Compact (Thousand/Lakh/Cr)", value=True)
show_legend = c3.checkbox("Show short legend", value=True)
st.markdown('</div>', unsafe_allow_html=True)

dec_map = {"Auto": None, "0 decimals": 0, "1 decimal": 1, "2 decimals": 2, "3 decimals": 3}
decimals_setting = dec_map[fmt_choice]

def _describe_plus(series: pd.Series):
    s = series.dropna()
    if s.empty: return pd.Series(dtype=float)
    out = s.describe(percentiles=[.05, .25, .5, .75, .95])
    out["skew"] = s.skew(); out["kurtosis"] = s.kurtosis()
    return out

friendly = {"count":"Count","mean":"Mean","std":"Std dev","min":"Min","5%":"P5","25%":"Q1 (25%)","50%":"Median","75%":"Q3 (75%)","95%":"P95","max":"Max","skew":"Skewness","kurtosis":"Kurtosis"}

def _fmt_metric(key: str, val):
    if decimals_setting is None and key in {"skew","kurtosis"}: return _fmt_num(val, decimals=2, compact=False)
    if decimals_setting is None and key in {"5%","25%","50%","75%","95%"}: return _fmt_num(val, decimals=2, compact=compact_kmb)
    return _fmt_num(val, decimals=decimals_setting, compact=compact_kmb)

trend_notes, stat_buckets = [], []
if selected_numeric:
    for col in selected_numeric:
        sum_col = "sum_" + col
        stats_sum = _describe_plus(daily[sum_col]) if sum_col in daily.columns else pd.Series(dtype=float)
        trend_text = "No data"
        if sum_col in daily.columns:
            x = np.arange(len(daily), dtype=float); y = daily[sum_col].astype(float).values
            mask = np.isfinite(y)
            if mask.sum() >= 2:
                slope, _ = np.polyfit(x[mask], y[mask], 1)
                xr = pd.Series(x[mask]).rank().values; yr = pd.Series(y[mask]).rank().values
                rho = np.corrcoef(xr, yr)[0,1]
            else:
                slope, rho = 0.0, np.nan
            direction = "increasing" if slope > 0 else ("decreasing" if slope < 0 else "flat")
            slope_txt = _fmt_num(slope, decimals=3, compact=False); rho_txt = "—" if pd.isna(rho) else f"{rho:.3f}"
            trend_text = f"**{col} (daily sum)** trend: {direction}; slope={slope_txt}, Spearman ρ={rho_txt}"
            trend_notes.append(f"• {trend_text}")
        stat_buckets.append((col, stats_sum, trend_text))

if stat_buckets:
    if show_legend:
        st.markdown("**Legend** — P5/P95: 5th/95th percentile; Q1/Q3: 25th/75th percentile; Skewness: asymmetry; Kurtosis: tail weight (~3 normal).")
    for i in range(0, len(stat_buckets), 2):
        cols2 = st.columns(2)
        for j, spot in enumerate(stat_buckets[i:i+2]):
            with cols2[j]:
                col, stats_sum, trend_text = spot
                st.markdown('<div class="stat-card">', unsafe_allow_html=True)
                st.markdown(f'<div class="stat-title"> {col}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="subtle">{trend_text}</div>', unsafe_allow_html=True)
                if not stats_sum.empty:
                    st.markdown('<div class="group-head">Daily Sum</div>', unsafe_allow_html=True)
                    st.markdown('<div class="chips">', unsafe_allow_html=True)
                    for key in ["count","mean","std","min","5%","25%","50%","75%","95%","max","skew","kurtosis"]:
                        if key in stats_sum.index:
                            st.markdown(f'<div class="chip"><div class="label">{friendly.get(key,key)}</div><div class="value">{_fmt_metric(key, stats_sum[key])}</div></div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
else:
    st.info("Select one or more numeric columns to view advanced stats.")

# =============== Range Totals (Sum) ===============
st.subheader(" Range Totals — Sum per Column")
default_totals = selected_numeric if selected_numeric else numeric_cols[:min(8, len(numeric_cols))]
cols_to_sum = st.multiselect("Choose columns (sum over selected range)", options=numeric_cols, default=default_totals)
totals = pd.Series(dtype=float)
if cols_to_sum:
    totals = dff[cols_to_sum].sum(numeric_only=True)
    for i in range(0, len(cols_to_sum), 4):
        row = st.columns(4)
        for j, cname in enumerate(cols_to_sum[i:i+4]):
            val = totals.get(cname, np.nan); pretty = _fmt_num(val, decimals=2, compact=True)
            with row[j]:
                st.markdown(f'<div class="total-chip"><div class="label">Σ {cname}</div><div class="value">{pretty}</div></div>', unsafe_allow_html=True)

# =============== Range Averages (Mean) ===============
st.subheader(" Range Averages — Mean per Column")
st.markdown('<div class="card">', unsafe_allow_html=True)
c1, c2 = st.columns(2)
with c1:
    start_avg = st.date_input("Average start date", value=start_date, min_value=min_d, max_value=max_d, key="avg_start")
with c2:
    end_avg = st.date_input("Average end date", value=end_date, min_value=min_d, max_value=max_d, key="avg_end")
st.markdown('</div>', unsafe_allow_html=True)

means = pd.Series(dtype=float); cols_to_avg = []
if start_avg > end_avg:
    st.error("Average start date cannot be after average end date.")
else:
    mask_avg = (df["__date__"] >= start_avg) & (df["__date__"] <= end_avg)
    dff_avg = df.loc[mask_avg].copy()
    if dff_avg.empty:
        st.info("No rows in the chosen average date range.")
    else:
        default_means = selected_numeric if selected_numeric else numeric_cols[:min(8, len(numeric_cols))]
        cols_to_avg = st.multiselect("Choose columns (mean over chosen range)", options=numeric_cols, default=default_means, key="avg_cols")
        if cols_to_avg:
            means = dff_avg[cols_to_avg].mean(numeric_only=True)
            for i in range(0, len(cols_to_avg), 4):
                row = st.columns(4)
                for j, cname in enumerate(cols_to_avg[i:i+4]):
                    val = means.get(cname, np.nan); pretty = _fmt_num(val, decimals=2, compact=True)
                    with row[j]:
                        st.markdown(f'<div class="total-chip"><div class="label">μ {cname}</div><div class="value">{pretty}</div></div>', unsafe_allow_html=True)

# =============== Single-Day Snapshot ===============
st.subheader(" Single-Day Snapshot")
st.markdown('<div class="card">', unsafe_allow_html=True)
day_pick = st.date_input("Pick a day", value=end_date, min_value=start_date, max_value=end_date, key="single_day")
show_all_values = st.checkbox("Show all row values (may be long)", value=False)
st.markdown('</div>', unsafe_allow_html=True)

day_df = dff.loc[dff["__date__"] == day_pick].copy()
if day_df.empty:
    st.info("No rows on the chosen day.")
else:
    c_top = st.columns(4)
    with c_top[0]:
        st.markdown(f'<div class="total-chip"><div class="label">Row count ({day_pick})</div><div class="value">{len(day_df):,}</div></div>', unsafe_allow_html=True)
    if selected_numeric:
        max_values = None if show_all_values else 12
        for i in range(0, len(selected_numeric), 2):
            row = st.columns(2)
            for j, cname in enumerate(selected_numeric[i:i+2]):
                vals = day_df[cname].dropna().tolist()
                pretty_vals = [_fmt_num(v, decimals=2, compact=True) for v in vals]
                display_vals = pretty_vals if (max_values is None or len(pretty_vals) <= max_values) else (pretty_vals[:max_values] + [f"... +{len(pretty_vals)-max_values} more"])
                with row[j]:
                    st.markdown('<div class="stat-card">', unsafe_allow_html=True)
                    st.markdown(f'<div class="stat-title"> {cname}</div>', unsafe_allow_html=True)
                    if not display_vals:
                        st.markdown('<div class="subtle">No values</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="chips">', unsafe_allow_html=True)
                        for v in display_vals:
                            st.markdown(f'<div class="chip"><div class="label">Value</div><div class="value">{v}</div></div>', unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)

# =============== Optional: Correlations toggle ===============
st.subheader(" Correlation Heatmap & Scatterplot Matrix")
enable_corr = st.checkbox("Enable correlation visuals", value=False)
corr_pairs = []
if enable_corr:
    corr_cols_all = [f"mean_{c}" for c in selected_numeric if f"mean_{c}" in daily.columns]
    if len(corr_cols_all) >= 2:
        corr_matrix = daily[corr_cols_all].corr()
        fig_corr = px.imshow(corr_matrix, text_auto=True, aspect="auto", title="Correlation heatmap (daily means)")
        st.plotly_chart(fig_corr, use_container_width=True)
        for i in range(len(corr_cols_all)):
            for j in range(i+1, len(corr_cols_all)):
                corr_pairs.append((corr_cols_all[i], corr_cols_all[j], corr_matrix.iloc[i, j]))
        corr_pairs = sorted(corr_pairs, key=lambda t: abs(t[2]), reverse=True)[:5]
        if corr_pairs:
            st.markdown("**Top correlated pairs:** " + ", ".join([f"`{a}`–`{b}` = {r:.3f}" for a, b, r in corr_pairs]))

        st.markdown('<div class="card">', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            max_dims = st.slider("Max columns in scatter-matrix", min_value=2, max_value=min(10, len(corr_cols_all)), value=min(5, len(corr_cols_all)))
        with c2:
            sample_cap = st.slider("Max points to plot (sampling; 0 = no sampling)", min_value=0, max_value=int(min(5000, len(daily))), value=min(1000, len(daily)))
        st.markdown('</div>', unsafe_allow_html=True)

        variances = daily[corr_cols_all].var().sort_values(ascending=False)
        chosen_dims = variances.index.tolist()[:max_dims]

        daily_sm = daily.copy()
        if sample_cap and len(daily_sm) > sample_cap:
            daily_sm = daily_sm.sample(n=sample_cap, random_state=42)

        fig_scatter_matrix = px.scatter_matrix(daily_sm, dimensions=chosen_dims, title=f"Scatterplot matrix (daily means, {len(chosen_dims)} dims, {len(daily_sm)} points)")
        fig_scatter_matrix.update_traces(marker=dict(size=4, opacity=0.5))
        try: fig_scatter_matrix.update_traces(diagonal_visible=False)
        except Exception: pass
        st.plotly_chart(fig_scatter_matrix, use_container_width=True)
    else:
        st.info("Select at least two numeric columns to see correlations.")

# =============== Analysis summary bullets ===============
st.subheader(" Analysis Summary")
bullets = [
    f"• Date range analyzed: **{start_date} → {end_date}** ({total_days} day(s))",
    f"• Total rows in range: **{total_rows:,}**",
    f"• Average rows per day: **{avg_rows_per_day:.2f}**",
]
if peak_day is not None:
    bullets.append(f"• Peak activity on **{peak_day['date'].date()}** with **{int(peak_day['row_count']):,}** rows")
if "anomaly_flag" in daily.columns and daily["anomaly_flag"].eq("⚠️").any():
    flagged = daily.loc[daily["anomaly_flag"] == "⚠️", "date"].astype(str).tolist()
    bullets.append(f"• Potential anomalies (z-score ≥ 2) on: **{', '.join(flagged)}**")
else:
    bullets.append("• No strong anomalies detected in daily row counts.")
if 'trend_notes' in locals() and trend_notes:
    bullets.append("• Trend insights:"); bullets.extend(trend_notes[:5])
if 'corr_pairs' in locals() and corr_pairs:
    bullets.append("• Top correlations (daily means): " + ", ".join([f"{a} vs {b}: r={r:.3f}" for a, b, r in corr_pairs]))
st.markdown("\n".join(bullets))

# =============== Matplotlib PNG helpers for PDF ===============
def ts_png(df: pd.DataFrame, ycols, labels, title) -> bytes:
    fig, ax = plt.subplots(figsize=(8.2, 4.2), dpi=150)
    df2 = df.copy(); df2["date"] = pd.to_datetime(df2["date"], errors="coerce")
    for y, lab in zip(ycols, labels):
        ax.plot(df2["date"], df2[y], marker="o", linewidth=1.5, markersize=3.5, label=lab)
    ax.set_title(title); ax.set_xlabel("Date"); ax.set_ylabel("Value")
    loc = AutoDateLocator(); ax.xaxis.set_major_locator(loc); ax.xaxis.set_major_formatter(ConciseDateFormatter(loc))
    ax.grid(True, alpha=0.25); 
    if len(ycols) > 1: ax.legend(frameon=False)
    fig.tight_layout(); buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig); return buf.getvalue()

def build_chart_pngs_for_pdf(selected_cols):
    pngs = []
    if not daily.empty:   pngs.append(ts_png(daily[["date","row_count"]], ["row_count"], ["Count"], "Rows per Day"))
    if not weekly.empty:  pngs.append(ts_png(weekly[["date","row_count"]], ["row_count"], ["Count"], "Rows per Week"))
    if not monthly.empty: pngs.append(ts_png(monthly[["date","row_count"]], ["row_count"], ["Count"], "Rows per Month"))
    for col in selected_cols:
        for frame, label in [(daily, "Daily"), (weekly, "Weekly"), (monthly, "Monthly")]:
            cols_here, labels = [], []
            if f"sum_{col}" in frame.columns:  cols_here.append(f"sum_{col}");  labels.append(f"sum_{col}")
            if f"mean_{col}" in frame.columns: cols_here.append(f"mean_{col}"); labels.append(f"mean_{col}")
            if cols_here:
                pngs.append(ts_png(frame[["date"] + cols_here], cols_here, labels, f"{col}: {label} Sum & Mean"))
    return pngs

# =============== PDF utilities ===============
def _series_to_rows(s: pd.Series, decimals=None, compact=True):
    return [[str(k), _fmt_num(v, decimals=decimals, compact=compact)] for k, v in s.items()]

def build_pdf_report(title, summary_lines, sum_title, totals_rows, avg_title, means_rows, day_title, day_rows, chart_pngs):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="TitleBig", fontSize=20, leading=24, spaceAfter=12))
    styles.add(ParagraphStyle(name="Small", fontSize=9, leading=12))
    styles.add(ParagraphStyle(name="H2", fontSize=14, leading=18, spaceBefore=10, spaceAfter=6))

    story = []
    story.append(Paragraph(title, styles["TitleBig"]))
    story.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["Small"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("<b>Summary</b>", styles["H2"]))
    for line in summary_lines:
        story.append(Paragraph(line.replace("•", "•&nbsp;"), styles["Normal"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph(f"<b>{sum_title}</b>", styles["H2"]))
    if totals_rows:
        tbl = Table([["Column", "Sum"]] + totals_rows, repeatRows=1)
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#0ea5e9")), ("TEXTCOLOR", (0,0), (-1,0), colors.white),
            ("ALIGN", (0,0), (-1,-1), "CENTER"), ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE", (0,0), (-1,0), 10), ("FONTSIZE", (0,1), (-1,-1), 9),
            ("BOTTOMPADDING", (0,0), (-1,0), 8), ("BACKGROUND", (0,1), (-1,-1), colors.whitesmoke),
            ("GRID", (0,0), (-1,-1), 0.3, colors.grey),
        ])); story.append(tbl); story.append(Spacer(1, 12))
    else:
        story.append(Paragraph("No totals to display.", styles["Normal"]))

    story.append(Paragraph(f"<b>{avg_title}</b>", styles["H2"]))
    if means_rows:
        tbl = Table([["Column", "Mean"]] + means_rows, repeatRows=1)
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#0ea5e9")), ("TEXTCOLOR", (0,0), (-1,0), colors.white),
            ("ALIGN", (0,0), (-1,-1), "CENTER"), ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE", (0,0), (-1,0), 10), ("FONTSIZE", (0,1), (-1,-1), 9),
            ("BOTTOMPADDING", (0,0), (-1,0), 8), ("BACKGROUND", (0,1), (-1,-1), colors.whitesmoke),
            ("GRID", (0,0), (-1,-1), 0.3, colors.grey),
        ])); story.append(tbl); story.append(Spacer(1, 12))
    else:
        story.append(Paragraph("No averages to display.", styles["Normal"]))

    story.append(Paragraph(f"<b>{day_title}</b>", styles["H2"]))
    if day_rows:
        tbl = Table(day_rows, repeatRows=1, colWidths=[180, 330])
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#0ea5e9")), ("TEXTCOLOR", (0,0), (-1,0), colors.white),
            ("ALIGN", (0,0), (-1,-1), "LEFT"), ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE", (0,0), (-1,0), 10), ("FONTSIZE", (0,1), (-1,-1), 9),
            ("BOTTOMPADDING", (0,0), (-1,0), 8), ("BACKGROUND", (0,1), (-1,-1), colors.whitesmoke),
            ("GRID", (0,0), (-1,-1), 0.3, colors.grey),
        ])); story.append(tbl); story.append(Spacer(1, 16))
    else:
        story.append(Paragraph("No snapshot available for the selected day.", styles["Normal"]))

    if chart_pngs:
        story.append(Paragraph("<b>Charts</b>", styles["H2"]))
        for png_bytes in chart_pngs:
            img = RLImage(io.BytesIO(png_bytes), width=6.3*inch, height=3.3*inch)
            story.append(img); story.append(Spacer(1, 12))

    doc.build(story); pdf_bytes = buffer.getvalue(); buffer.close(); return pdf_bytes

# =============== Prepare PDF sections & download ===============
def _values_str(series, max_items=25, decimals=None, compact=True):
    vals = series.dropna().tolist()
    pv = [_fmt_num(v, decimals=decimals, compact=compact) for v in vals]
    if len(pv) > max_items: return ", ".join(pv[:max_items]) + f", ... (+{len(pv)-max_items} more)"
    return ", ".join(pv) if pv else "—"

totals_rows = _series_to_rows(totals, decimals=2, compact=True) if 'totals' in locals() and not totals.empty else []
sum_title = f"Range Totals (Sum) — {start_date} to {end_date}"

means_rows = _series_to_rows(means, decimals=2, compact=True) if 'means' in locals() and not means.empty else []
avg_title = f"Range Averages (Mean) — {start_avg if 'start_avg' in locals() else start_date} to {end_avg if 'end_avg' in locals() else end_date}"

day_rows = [["Field", "Values / Count"]]
day_rows.append(["Row count", f"{len(dff.loc[dff['__date__']==day_pick]):,}"])
if 'day_df' in locals() and not day_df.empty and selected_numeric:
    for c in selected_numeric: day_rows.append([c, _values_str(day_df[c], decimals=2, compact=True)])
day_title = f"Single-Day Snapshot — {day_pick if 'day_pick' in locals() else ''}"

if "pdf_bytes" not in st.session_state: st.session_state["pdf_bytes"] = None

st.subheader(" Report")
if st.button("Generate PDF"):
    chart_pngs = build_chart_pngs_for_pdf(selected_numeric)
    st.session_state["pdf_bytes"] = build_pdf_report(
        title="Day/Week/Month Analysis Report",
        summary_lines=bullets,
        sum_title=sum_title, totals_rows=totals_rows,
        avg_title=avg_title, means_rows=means_rows,
        day_title=day_title, day_rows=day_rows,
        chart_pngs=chart_pngs
    )

if st.session_state["pdf_bytes"]:
    st.download_button(" Download PDF Report", data=st.session_state["pdf_bytes"],
                       file_name=f"analysis_report_{start_date}_to_{end_date}.pdf", mime="application/pdf")

# Close the glass wrapper
st.markdown("</div>", unsafe_allow_html=True)