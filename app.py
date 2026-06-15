"""
app.py — Streamlit dashboard: Chinese Companies in MDB Procurement, LAC
"""
from __future__ import annotations
import traceback as _tb

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data_loader import get_data, get_cleaning_report
from metrics import (
    hhi, hhi_label,
    cn_by_source, cn_by_sector, cn_by_method, cn_by_country,
    cn_by_year, cn_hhi_by_year, cn_country_year_pivot,
    top_n_labels, annual_share, stacked_area_data, rank_trajectory,
    spread_by_label, sector_mix, yoy_growth, cagr,
    procurement_profile, market_hhi,
)

# ── Palette ───────────────────────────────────────────────────────────────────
NAVY         = "#10395e"
BLUE_P       = "#0a66c2"
BLUE_M       = "#3b82c4"
BLUE_L       = "#9ec5e8"
BLUE_PALE    = "#dbeafe"
GRID         = "#e6ebf1"
GREEN        = "#1a9e5f"
RED          = "#d23b3b"
TEXT_MUTED   = "#6b7c93"

CHINA_COLOR  = BLUE_P
CHINA_FILL   = "rgba(10, 102, 194, 0.13)"
HK_COLOR     = BLUE_M
REST_COLOR   = "#c0d3e6"

COMP_BLUES   = ["#3b82c4","#5a9fd4","#7db8e2","#9ec5e8","#a8c4de","#6b9dbf","#4d88ae","#2d6fa0"]
SECTOR_PAL   = [BLUE_P,"#1a9e5f",BLUE_M,"#e8a020","#9b59b6",RED,"#5a9fd4","#20a0a0","#f39c12",BLUE_L]

METHOD_COLORS = {
    "Open/Competitive":     GREEN,
    "Limited/Shopping":     BLUE_M,
    "Direct/Single-Source": RED,
    "Consultant Selection": BLUE_L,
    "Unknown":              "#c8d3df",
}

# ── Global CSS ────────────────────────────────────────────────────────────────
_CSS = f"""
<style>
/* ── app chrome ── */
.stApp {{ background: #f4f7fb; }}
section[data-testid="stMain"] > div {{ background: #f4f7fb; }}
.block-container {{ padding-top: 0 !important; max-width: 1280px !important;
                    padding-left: 1.5rem !important; padding-right: 1.5rem !important; }}
header[data-testid="stHeader"] {{ background: transparent !important; }}
div[data-testid="stToolbar"] {{ display: none; }}

/* ── sidebar ── */
[data-testid="stSidebar"] {{
    background: #ffffff !important;
    border-right: 1px solid {GRID};
}}
[data-testid="stSidebar"] label {{ color: {TEXT_MUTED} !important; font-size: 0.78rem !important; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; }}
[data-testid="stSidebarContent"] h2 {{ color: {NAVY}; font-size: 0.9rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; }}

/* ── tabs ── */
[data-testid="stTabs"] button[role="tab"] {{
    color: {TEXT_MUTED}; font-size: 0.79rem; font-weight: 500;
    border-bottom: 2px solid transparent; padding: 6px 14px;
}}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {{
    color: {BLUE_P}; border-bottom: 2px solid {BLUE_P}; font-weight: 700;
}}
[data-testid="stTabs"] {{border-bottom: 1px solid {GRID};}}

/* ── expander ── */
[data-testid="stExpander"] {{
    border: 1px solid {GRID} !important; border-radius: 8px !important;
    background: #ffffff !important; margin-bottom: 6px;
}}
[data-testid="stExpander"] summary {{ color: {NAVY} !important; font-weight: 600; }}

/* ── info / warning / error boxes ── */
[data-testid="stInfo"] {{
    background: {BLUE_PALE} !important; border-left: 3px solid {BLUE_P} !important;
    border-radius: 0 6px 6px 0; color: {NAVY} !important;
}}
[data-testid="stWarning"] {{
    border-left: 3px solid #e8a020 !important; border-radius: 0 6px 6px 0;
}}

/* ── metrics (fallback when used directly) ── */
[data-testid="stMetric"] {{
    background: #ffffff; border: 1px solid {GRID}; border-radius: 10px;
    padding: 14px 16px; box-shadow: 0 1px 6px rgba(16,57,94,0.07);
}}
[data-testid="stMetricLabel"] span {{ color: {TEXT_MUTED} !important; font-size: 0.7rem !important; text-transform: uppercase; letter-spacing: 0.07em; font-weight: 600; }}
[data-testid="stMetricValue"] {{ color: {NAVY} !important; font-size: 1.4rem !important; font-weight: 700; }}

/* ── dataframe ── */
[data-testid="stDataFrame"] {{ border: 1px solid {GRID}; border-radius: 8px; overflow: hidden; }}

/* ── download button ── */
[data-testid="stDownloadButton"] button {{
    background: {BLUE_P} !important; color: #fff !important;
    border: none !important; border-radius: 6px; font-size: 0.78rem; font-weight: 600;
    padding: 6px 14px;
}}

/* ── divider ── */
hr {{ border-color: {GRID} !important; margin: 12px 0; }}

/* ── caption ── */
[data-testid="stCaptionContainer"] p {{ color: {TEXT_MUTED}; font-size: 0.76rem; }}

/* ── multiselect tags ── */
[data-testid="stMultiSelect"] span[data-baseweb="tag"] {{
    background: {BLUE_PALE} !important; color: {NAVY} !important;
    border: 1px solid {BLUE_L} !important; border-radius: 4px;
    font-size: 0.75rem;
}}

/* ── kpi card class ── */
.kpi-card {{
    background: #ffffff; border: 1px solid {GRID}; border-radius: 10px;
    padding: 14px 16px 10px; box-shadow: 0 1px 8px rgba(16,57,94,0.07);
    margin-bottom: 8px; min-height: 108px; display: flex; flex-direction: column; gap: 3px;
}}
.kpi-label {{ color: {TEXT_MUTED}; font-size: 0.67rem; text-transform: uppercase; letter-spacing: 0.08em; font-weight: 700; }}
.kpi-value {{ color: {NAVY}; font-size: 1.3rem; font-weight: 800; line-height: 1.1; margin-top: 3px; }}
.kpi-delta-pos {{ color: {GREEN}; font-size: 0.72rem; font-weight: 700; }}
.kpi-delta-neg {{ color: {RED}; font-size: 0.72rem; font-weight: 700; }}
</style>
"""

# ── Helpers ───────────────────────────────────────────────────────────────────

def fmt_usd(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    if abs(v) >= 1e9: return f"${v/1e9:.2f} B"
    if abs(v) >= 1e6: return f"${v/1e6:.1f} M"
    if abs(v) >= 1e3: return f"${v/1e3:.0f} K"
    return f"${v:,.0f}"


def _show_error(e: Exception, label: str = "") -> None:
    prefix = f"[{label}] " if label else ""
    st.error(f"{prefix}{type(e).__name__}: {e}")
    _tb.print_exc()


def _spark_svg(values, color=BLUE_P, fill="rgba(10,102,194,0.13)", w=84, h=34):
    vals = [float(v) for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if len(vals) < 2:
        return ""
    mn, mx = min(vals), max(vals)
    rng = mx - mn if mx != mn else 1
    n = len(vals)
    pts = [(i * w / (n - 1), h - 2 - (v - mn) / rng * (h - 6)) for i, v in enumerate(vals)]
    path = " ".join(f"{'M' if i == 0 else 'L'}{x:.1f},{y:.1f}" for i, (x, y) in enumerate(pts))
    fp   = f"{path} L{pts[-1][0]:.1f},{h} L0,{h} Z"
    return (f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" '
            f'style="display:block;margin-top:6px">'
            f'<path d="{fp}" fill="{fill}"/>'
            f'<path d="{path}" stroke="{color}" stroke-width="1.8" fill="none" '
            f'stroke-linecap="round" stroke-linejoin="round"/></svg>')


def _kpi_card(col, label, value, delta=None, spark_values=None):
    spark = _spark_svg(spark_values) if spark_values else ""
    if delta is not None:
        arrow = "▲" if delta >= 0 else "▼"
        cls   = "kpi-delta-pos" if delta >= 0 else "kpi-delta-neg"
        d_html = f'<div class="{cls}">{arrow} {abs(delta):.1f}% YoY</div>'
    else:
        d_html = ""
    with col:
        st.markdown(
            f'<div class="kpi-card">'
            f'<div class="kpi-label">{label}</div>'
            f'<div class="kpi-value">{value}</div>'
            f'{d_html}{spark}</div>',
            unsafe_allow_html=True,
        )


def _section_header(num: int, title: str, subtitle: str = "") -> None:
    sub = (f'<div style="color:{BLUE_L};font-size:0.77rem;margin-top:3px;font-weight:400">'
           f'{subtitle}</div>') if subtitle else ""
    badge = (f'<div style="color:{BLUE_L};font-size:1.6rem;font-weight:900;'
             f'opacity:0.5;margin-right:14px;line-height:1">§{num}</div>')
    st.markdown(
        f'<div style="background:linear-gradient(90deg,{NAVY} 0%,#1a527d 100%);'
        f'border-radius:8px;padding:14px 22px;margin:22px 0 10px;display:flex;align-items:center">'
        f'{badge}'
        f'<div><div style="color:#fff;font-size:1.05rem;font-weight:700;letter-spacing:0.01em">'
        f'{title}</div>{sub}</div></div>',
        unsafe_allow_html=True,
    )


def _hhi_primer() -> None:
    st.markdown(f"""
<div style="background:{BLUE_PALE};border-left:3px solid {BLUE_P};border-radius:0 8px 8px 0;
            padding:14px 18px;margin:10px 0;font-size:0.83rem;color:{NAVY}">
<strong>What is the HHI?</strong><br>
The <em>Herfindahl-Hirschman Index</em> is a number between <strong>0 and 1</strong> that measures
how concentrated or spread out something is.<br><br>
<strong>How it is calculated:</strong> Take every group's share of the total, square each share,
then add them all up. Example: if Bolivia = 25% of China's contracts → 0.25² = 0.0625. Sum across all countries.<br><br>
<table style="border-collapse:collapse;width:100%;font-size:0.8rem">
<tr style="background:{NAVY};color:#fff">
  <th style="padding:6px 10px;text-align:left">HHI</th>
  <th style="padding:6px 10px;text-align:left">Meaning</th>
</tr>
<tr style="background:#fff"><td style="padding:5px 10px">&lt; 0.01</td><td style="padding:5px 10px">Highly spread out — many groups, none dominant</td></tr>
<tr style="background:{BLUE_PALE}"><td style="padding:5px 10px">0.01 – 0.15</td><td style="padding:5px 10px">Unconcentrated — reasonably balanced</td></tr>
<tr style="background:#fff"><td style="padding:5px 10px">0.15 – 0.25</td><td style="padding:5px 10px">Moderately concentrated — a few groups stand out</td></tr>
<tr style="background:{BLUE_PALE}"><td style="padding:5px 10px">&gt; 0.25</td><td style="padding:5px 10px">Highly concentrated — one or two groups dominate</td></tr>
</table>
</div>
""", unsafe_allow_html=True)


def _theme(fig, title="", height=370, secondary_y=False):
    """Apply corporate BI Plotly theme to any figure."""
    ax = dict(
        gridcolor=GRID, linecolor="#d0dae6", zerolinecolor=GRID,
        tickcolor=TEXT_MUTED, tickfont=dict(color=TEXT_MUTED, size=11),
        title_font=dict(color=TEXT_MUTED, size=11), zeroline=False,
    )
    upd = dict(
        plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
        font=dict(family="Inter,'Helvetica Neue',Arial,sans-serif", color="#1b2a3a", size=12),
        title=dict(
            text=f"<b>{title}</b>" if title else None,
            font=dict(color=NAVY, size=13), x=0, xanchor="left", y=0.97, pad=dict(l=2),
        ),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, x=0,
            bgcolor="rgba(0,0,0,0)", font=dict(size=11), bordercolor=GRID,
        ),
        margin=dict(t=52 if title else 20, b=22, l=6, r=6),
        height=height,
        xaxis=ax, yaxis=ax,
        hoverlabel=dict(bgcolor=NAVY, font_color="#ffffff", font_size=12, bordercolor=BLUE_P),
        colorway=[BLUE_P, BLUE_M, BLUE_L, GREEN, "#e8a020", RED, "#9b59b6"],
    )
    if secondary_y:
        upd["yaxis2"] = dict(**ax)
    fig.update_layout(**upd)
    return fig


def colour_map(labels):
    cmap, ci = {}, 0
    for lbl in labels:
        if lbl == "China":           cmap[lbl] = CHINA_COLOR
        elif lbl == "Rest":          cmap[lbl] = REST_COLOR
        elif lbl == "Hong Kong SAR": cmap[lbl] = HK_COLOR
        else:
            cmap[lbl] = COMP_BLUES[ci % len(COMP_BLUES)]; ci += 1
    return cmap


def _html_table(df: pd.DataFrame, title: str = "") -> str:
    """BI-styled HTML table with navy header and zebra rows."""
    hdr = "".join(
        f'<th style="background:{NAVY};color:#fff;padding:9px 14px;font-size:0.74rem;'
        f'font-weight:700;letter-spacing:0.04em;text-align:left;white-space:nowrap">{c}</th>'
        for c in df.columns
    )
    rows = ""
    for i, (_, row) in enumerate(df.iterrows()):
        bg = "#f4f7fb" if i % 2 else "#ffffff"
        cells = "".join(
            f'<td style="padding:7px 14px;font-size:0.82rem;color:{NAVY};'
            f'border-bottom:1px solid {GRID};background:{bg}">{v}</td>'
            for v in row
        )
        rows += f"<tr>{cells}</tr>"
    ttl = (f'<div style="font-size:0.83rem;font-weight:700;color:{NAVY};margin-bottom:8px">'
           f'{title}</div>') if title else ""
    return (f'{ttl}<div style="border:1px solid {GRID};border-radius:8px;overflow:hidden;margin:8px 0">'
            f'<table style="width:100%;border-collapse:collapse;font-family:Inter,Arial,sans-serif">'
            f'<thead><tr>{hdr}</tr></thead><tbody>{rows}</tbody></table></div>')


def bar_h(data, y, x, title="", color=BLUE_P, height=340):
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=data[y].tolist(), x=data[x].tolist(),
        orientation="h", marker_color=color, marker_line_width=0,
    ))
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    return _theme(fig, title, height)


def bar_v(data, x, y, title="", color=BLUE_P, height=340):
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=data[x].tolist(), y=data[y].tolist(),
        marker_color=color, marker_line_width=0,
    ))
    return _theme(fig, title, height)


# ── Page config + CSS + data ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Chinese Companies in LAC MDB Procurement",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(_CSS, unsafe_allow_html=True)
df_all = get_data()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Filters")
    st.caption("Applied to all sections.")

    yr_min = int(df_all["year_awarded"].min())
    yr_max = int(df_all["year_awarded"].max())
    year_range = st.slider("Year range", yr_min, yr_max, (yr_min, yr_max))

    all_sources   = sorted(df_all["data_source"].dropna().unique())
    sel_sources   = st.multiselect("MDB source", all_sources, default=all_sources)

    all_sectors   = sorted(df_all["project_sector"].dropna().unique())
    sel_sectors   = st.multiselect("Sector", all_sectors, default=all_sectors)

    all_countries = sorted(df_all["borrower country"].dropna().unique())
    sel_countries = st.multiselect("Borrower country", all_countries, default=all_countries)

    st.divider()
    st.markdown("**Section 4 options**")
    sel_comp_groups = st.multiselect(
        "Comparator groups", ["BRICS", "G7", "Others"],
        default=["BRICS", "G7", "Others"],
    )
    top_n = st.slider("Top-N comparators", 3, 15, 8)

# ── Apply filters ─────────────────────────────────────────────────────────────
fmask = (
    df_all["year_awarded"].between(year_range[0], year_range[1])
    & df_all["data_source"].isin(sel_sources)
    & df_all["project_sector"].isin(sel_sectors)
    & df_all["borrower country"].isin(sel_countries)
)
df    = df_all[fmask].copy()
df_cn = df[df["is_chinese"]].copy()
df4   = df[df["is_chinese"] | df["contractor_country_group"].isin(sel_comp_groups)].copy()

# ── Dashboard header ──────────────────────────────────────────────────────────
st.markdown(f"""
<div style="background:linear-gradient(100deg,{NAVY} 0%,#1a527d 100%);
            padding:22px 28px 18px;margin:-1.5rem -1.5rem 0;
            border-bottom:3px solid {BLUE_P}">
  <div style="color:#ffffff;font-size:1.45rem;font-weight:800;letter-spacing:-0.01em;line-height:1.2">
    Chinese Companies in MDB Public Procurement — Latin America
  </div>
  <div style="color:{BLUE_L};font-size:0.82rem;margin-top:5px;font-weight:400">
    World Bank · IDB · CDB &nbsp;|&nbsp; 2000–2026 &nbsp;|&nbsp;
    Descriptive analysis of trends and characteristics
  </div>
</div>
<div style="height:18px"></div>
""", unsafe_allow_html=True)

# ── KPI row with sparklines ───────────────────────────────────────────────────
n_cn  = len(df_cn)
val   = df_cn["contract_value_usd"].sum()
yr_sp = (f"{int(df_cn['year_awarded'].min())}–{int(df_cn['year_awarded'].max())}"
         if n_cn > 0 else "—")

yr_df_kpi = cn_by_year(df_cn) if n_cn > 0 else pd.DataFrame()

def _delta(series):
    s = series.dropna()
    if len(s) < 2 or s.iloc[-2] == 0:
        return None
    return (s.iloc[-1] - s.iloc[-2]) / abs(s.iloc[-2]) * 100

if len(yr_df_kpi) > 0:
    spark_cnt  = yr_df_kpi["contracts"].tolist()
    spark_val  = yr_df_kpi["total_value"].tolist()
    spark_avg  = yr_df_kpi["avg_value"].tolist()
    delta_cnt  = _delta(yr_df_kpi["contracts"])
    delta_val  = _delta(yr_df_kpi["total_value"])
    delta_avg  = _delta(yr_df_kpi["avg_value"])
    # borrower countries per year
    ctry_yr = df_cn.groupby("year_awarded")["borrower country"].nunique()
    spark_ctry = ctry_yr.reindex(yr_df_kpi["year_awarded"]).tolist()
    delta_ctry = _delta(ctry_yr)
    # sectors per year
    sec_yr = df_cn.groupby("year_awarded")["project_sector"].nunique()
    spark_sec  = sec_yr.reindex(yr_df_kpi["year_awarded"]).tolist()
    delta_sec  = _delta(sec_yr)
    # direct-award share per year
    da_yr = df_cn.groupby("year_awarded")["is_direct_award"].mean() * 100
    spark_da   = da_yr.reindex(yr_df_kpi["year_awarded"]).tolist()
    delta_da   = _delta(da_yr)
    avg_val = df_cn["contract_value_usd"].mean()
    da_share = df_cn["is_direct_award"].mean() * 100 if n_cn > 0 else float("nan")
else:
    spark_cnt = spark_val = spark_avg = spark_ctry = spark_sec = spark_da = []
    delta_cnt = delta_val = delta_avg = delta_ctry = delta_sec = delta_da = None
    avg_val = float("nan"); da_share = float("nan")

k = st.columns(6)
_kpi_card(k[0], "Chinese contracts",      f"{n_cn:,}",         delta_cnt,  spark_cnt)
_kpi_card(k[1], "Total value (USD)",      fmt_usd(val),        delta_val,  spark_val)
_kpi_card(k[2], "Avg contract value",     fmt_usd(avg_val),    delta_avg,  spark_avg)
_kpi_card(k[3], "Borrower countries",     str(df_cn["borrower country"].nunique()), delta_ctry, spark_ctry)
_kpi_card(k[4], "Sectors",                str(df_cn["project_sector"].nunique()),   delta_sec,  spark_sec)
_kpi_card(k[5], "Direct-award share",
          f"{da_share:.1f}%" if not np.isnan(da_share) else "—",
          delta_da, spark_da)

st.markdown('<div style="height:6px"></div>', unsafe_allow_html=True)

# ── Methodology expander ──────────────────────────────────────────────────────
with st.expander("📋 Methodology & data notes", expanded=False):
    rpt = get_cleaning_report()
    st.markdown(f"""
### Where does the data come from?

**`worldbank_idb_cdb_merged_0614.xlsx`**, sheet *"Data as of 15 June"* — a single combined
dataset of contract awards from three MDBs:

| Bank | Abbreviation | Contracts |
|---|---|---|
| Inter-American Development Bank | IDB | 158,272 |
| World Bank | WB | 78,950 |
| Caribbean Development Bank | CDB | 429 |
| **Total** | | **237,651** |

Coverage: **2000–2026** (2026 partial).

---

### How were Chinese companies identified?

`contractor_country` contains *"china"* (case-insensitive), including JV combos
(e.g. "China; Germany"). **Hong Kong SAR is excluded** — it is tracked separately
because it operates under a distinct legal and economic system.

Result: **{rpt['n_chinese']} Chinese contracts**, **{rpt['n_hk']} Hong Kong SAR contracts**.

---

### What was cleaned?

1. **Typos** — `BRICKES` → `BRICS`; mixed-case group/type labels standardised.
2. **Missing values** — 4 contracts had no dollar value recorded; excluded from value totals but included in counts.
3. **Procurement method** — 50+ raw labels harmonised into 5 buckets:
   Open/Competitive, Limited/Shopping, Direct/Single-Source, Consultant Selection, Unknown.
   World Bank uses a generic catch-all → mostly Unknown.
   **Method charts are meaningful only for IDB/CDB.**

---

### HHI thresholds

Standard US DoJ / competition-economics thresholds:
`< 0.01` highly competitive · `0.01–0.15` unconcentrated ·
`0.15–0.25` moderate · `> 0.25` highly concentrated.
    """)

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1
# ═════════════════════════════════════════════════════════════════════════════
_section_header(1, "Chinese Companies — The Raw Picture",
                "Contract counts and values by MDB source, sector, and procurement method")

try:
    if n_cn == 0:
        st.warning("No Chinese contracts match the current filters.")
    else:
        tab_src, tab_sec, tab_meth, tab_tbl = st.tabs(
            ["By MDB Source", "By Sector", "By Procurement Method", "Contract Data"]
        )

        with tab_src:
            try:
                src_df = cn_by_source(df_cn)
                c1, c2 = st.columns(2)
                c1.plotly_chart(
                    bar_v(src_df, "data_source", "contracts",
                          "Chinese contracts by MDB source (count)"),
                    use_container_width=True)
                c2.plotly_chart(
                    bar_v(src_df, "data_source", "value_usd",
                          "Chinese contracts by MDB source (USD)"),
                    use_container_width=True)
                st.info(
                    "**How to read this:** Left = number of contracts; Right = total dollar value. "
                    "If the value bar is proportionally taller, Chinese contracts through that bank "
                    "tend to be larger on average. "
                    "Note: World Bank doesn't report detailed procurement methods — all WB rows "
                    "appear as 'Unknown' in the Procurement Method tab."
                )
            except Exception as e:
                _show_error(e, "S1 Source")

        with tab_sec:
            try:
                sec_df = cn_by_sector(df_cn)
                c1, c2 = st.columns(2)
                c1.plotly_chart(
                    bar_h(sec_df, "project_sector", "contracts",
                          "Chinese contracts by sector (count)"),
                    use_container_width=True)
                c2.plotly_chart(
                    bar_h(sec_df, "project_sector", "value_usd",
                          "Chinese contracts by sector (USD)"),
                    use_container_width=True)
                top_val_s = sec_df.iloc[0]["project_sector"] if len(sec_df) > 0 else "—"
                top_cnt_s = sec_df.sort_values("contracts", ascending=False).iloc[0]["project_sector"] if len(sec_df) > 0 else "—"
                st.info(
                    f"Bars are sorted largest-to-smallest. If a sector ranks higher by value than by "
                    f"count, Chinese contracts there are individually very large. "
                    f"**By value:** {top_val_s} leads. **By count:** {top_cnt_s} leads."
                )
            except Exception as e:
                _show_error(e, "S1 Sector")

        with tab_meth:
            try:
                meth_df = cn_by_method(df_cn)
                colors_m = [METHOD_COLORS.get(m, "#95a5a6") for m in meth_df["procurement_method"]]
                c1, c2 = st.columns(2)
                fig_mc = go.Figure()
                fig_mc.add_trace(go.Bar(x=meth_df["procurement_method"].tolist(),
                    y=meth_df["contracts"].tolist(), marker_color=colors_m, marker_line_width=0))
                _theme(fig_mc, "Chinese contracts by method (count)")
                c1.plotly_chart(fig_mc, use_container_width=True)

                fig_mv = go.Figure()
                fig_mv.add_trace(go.Bar(x=meth_df["procurement_method"].tolist(),
                    y=meth_df["value_usd"].tolist(), marker_color=colors_m, marker_line_width=0))
                _theme(fig_mv, "Chinese contracts by method (USD)")
                c2.plotly_chart(fig_mv, use_container_width=True)

                st.info(
                    "'Unknown' is large because World Bank doesn't report detailed methods. "
                    "Among IDB/CDB contracts where we know the method, look at the "
                    "**Direct/Single-Source** (red) share — these were awarded without competitive bidding."
                )

                jv_n = (df_cn["if_joint_venture"] == "Joint Venture").sum()
                da_n = df_cn["is_direct_award"].sum()
                m1, m2, m3 = st.columns(3)
                m1.metric("Joint ventures",           f"{jv_n} ({100*jv_n/n_cn:.1f}%)")
                m2.metric("Direct / single-source",   f"{da_n} ({100*da_n/n_cn:.1f}%)")
                m3.metric("Total Chinese contracts",  f"{n_cn:,}")
            except Exception as e:
                _show_error(e, "S1 Method")

        with tab_tbl:
            try:
                cols_ = ["year_awarded","borrower country","data_source","project_sector",
                         "procurement_method","contract_value_usd","if_joint_venture",
                         "contractor_country","contract_name"]
                st.dataframe(df_cn[cols_].sort_values("year_awarded", ascending=False),
                             use_container_width=True, height=440)
                st.download_button("⬇ Download Chinese contracts CSV",
                    df_cn[cols_].to_csv(index=False).encode(),
                    "chinese_contracts.csv", "text/csv")
            except Exception as e:
                _show_error(e, "S1 Table")

except Exception as e:
    _show_error(e, "Section 1")

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2
# ═════════════════════════════════════════════════════════════════════════════
_section_header(2, "Chinese Companies — By Borrower Country",
                "Where in Latin America are Chinese-backed contracts landing?")

try:
    if n_cn == 0:
        st.warning("No Chinese contracts match the current filters.")
    else:
        ctry_df = cn_by_country(df_cn)

        tab_val, tab_avg, tab_tbl2, tab_hhi = st.tabs(
            ["By Total Value", "By Avg / Median Value", "Country Table", "Concentration (HHI)"]
        )

        with tab_val:
            try:
                fig_tv = go.Figure()
                sd = ctry_df.sort_values("total_value")
                fig_tv.add_trace(go.Bar(
                    y=sd["borrower country"].tolist(), x=sd["total_value"].tolist(),
                    orientation="h", marker_color=BLUE_P, marker_line_width=0))
                fig_tv.update_layout(yaxis={"categoryorder": "total ascending"})
                _theme(fig_tv, "Total value of Chinese contracts by borrower country (USD)",
                       height=max(320, len(ctry_df) * 24))
                st.plotly_chart(fig_tv, use_container_width=True)
                top_c = ctry_df.iloc[0]
                st.info(
                    f"Each bar = one borrower country (where the MDB-funded project took place). "
                    f"**{top_c['borrower country']}** received the most Chinese contract value "
                    f"({fmt_usd(top_c['total_value'])}) across {int(top_c['contracts'])} contracts."
                )
            except Exception as e:
                _show_error(e, "S2 Total Value")

        with tab_avg:
            try:
                c1, c2 = st.columns(2)
                sa = ctry_df.sort_values("avg_value").dropna(subset=["avg_value"])
                fig_av = go.Figure()
                fig_av.add_trace(go.Bar(y=sa["borrower country"].tolist(),
                    x=sa["avg_value"].tolist(), orientation="h",
                    marker_color=BLUE_P, marker_line_width=0))
                fig_av.update_layout(yaxis={"categoryorder": "total ascending"})
                _theme(fig_av, "Average contract value by country (USD)",
                       height=max(320, len(ctry_df) * 24))
                c1.plotly_chart(fig_av, use_container_width=True)

                sm = ctry_df.sort_values("median_value").dropna(subset=["median_value"])
                fig_med = go.Figure()
                fig_med.add_trace(go.Bar(y=sm["borrower country"].tolist(),
                    x=sm["median_value"].tolist(), orientation="h",
                    marker_color=BLUE_M, marker_line_width=0))
                fig_med.update_layout(yaxis={"categoryorder": "total ascending"})
                _theme(fig_med, "Median contract value by country (USD)",
                       height=max(320, len(ctry_df) * 24))
                c2.plotly_chart(fig_med, use_container_width=True)
                st.info(
                    "**Average vs Median:** Average is pulled up by a few very large contracts. "
                    "Median is the 'middle' contract — half above, half below. "
                    "If avg >> median, a handful of giant contracts inflate the average."
                )
            except Exception as e:
                _show_error(e, "S2 Avg")

        with tab_tbl2:
            try:
                disp = ctry_df.copy()
                disp["total_value"]  = disp["total_value"].map(fmt_usd)
                disp["avg_value"]    = disp["avg_value"].map(fmt_usd)
                disp["median_value"] = disp["median_value"].map(fmt_usd)
                disp.columns = ["Country","Contracts","Total Value","Avg Value","Median Value"]
                st.markdown(_html_table(disp, "Chinese contracts by borrower country"),
                            unsafe_allow_html=True)
            except Exception as e:
                _show_error(e, "S2 Table")

        with tab_hhi:
            try:
                _hhi_primer()
                st.markdown(f"<div style='height:8px'></div>", unsafe_allow_html=True)

                by_ctry_v = df_cn.groupby("borrower country")["contract_value_usd"].sum().dropna()
                h_c = hhi(by_ctry_v); lbl_c = hhi_label(h_c)
                by_sec_v  = df_cn.groupby("project_sector")["contract_value_usd"].sum().dropna()
                h_s = hhi(by_sec_v);  lbl_s = hhi_label(h_s)

                hc1, hc2 = st.columns(2)
                hc1.metric("HHI — across borrower countries", f"{h_c:.4f}", lbl_c, delta_color="off")
                hc2.metric("HHI — across project sectors",    f"{h_s:.4f}", lbl_s, delta_color="off")

                st.markdown(f"""
<div style="background:#fff;border:1px solid {GRID};border-radius:8px;padding:14px 18px;
            font-size:0.83rem;color:{NAVY};margin:10px 0">
<strong>What do these numbers mean?</strong><br><br>
<span style="color:{BLUE_P}">●</span>&nbsp;
<strong>HHI by country = {h_c:.4f} → {lbl_c}.</strong>
China's contract value is reasonably spread across borrower countries.
No single country absorbs so much that it dominates the portfolio, though some
(notably Bolivia) receive disproportionately more than others.<br><br>
<span style="color:{RED}">●</span>&nbsp;
<strong>HHI by sector = {h_s:.4f} → {lbl_s}.</strong>
China's contracts skew heavily toward a small number of sectors — primarily
Infrastructure &amp; Transport and Energy. This is consistent with China's global
reputation for infrastructure-focused development financing.
</div>
""", unsafe_allow_html=True)

                if len(by_ctry_v) > 0:
                    share_df = (by_ctry_v / by_ctry_v.sum() * 100).reset_index()
                    share_df.columns = ["borrower country","share_pct"]
                    c1, c2 = st.columns(2)
                    fig_pc = go.Figure(go.Pie(
                        labels=share_df["borrower country"].tolist(),
                        values=share_df["share_pct"].tolist(),
                        hole=0.42,
                        textposition="inside", textinfo="percent+label",
                        marker=dict(colors=px.colors.sequential.Blues_r[:len(share_df)],
                                    line=dict(color="#ffffff", width=1.5)),
                    ))
                    _theme(fig_pc, "China's value — share by borrower country", height=380)
                    fig_pc.update_layout(showlegend=False)
                    c1.plotly_chart(fig_pc, use_container_width=True)

                if len(by_sec_v) > 0:
                    share_s = (by_sec_v / by_sec_v.sum() * 100).reset_index()
                    share_s.columns = ["project_sector","share_pct"]
                    fig_ps = go.Figure(go.Pie(
                        labels=share_s["project_sector"].tolist(),
                        values=share_s["share_pct"].tolist(),
                        hole=0.42,
                        textposition="inside", textinfo="percent+label",
                        marker=dict(colors=SECTOR_PAL[:len(share_s)],
                                    line=dict(color="#ffffff", width=1.5)),
                    ))
                    _theme(fig_ps, "China's value — share by sector", height=380)
                    fig_ps.update_layout(showlegend=False)
                    c2.plotly_chart(fig_ps, use_container_width=True)
                    st.caption(
                        "Larger slices = more concentrated there. A large sector slice "
                        "combined with a high HHI score confirms sector-specific engagement."
                    )
            except Exception as e:
                _show_error(e, "S2 HHI")

except Exception as e:
    _show_error(e, "Section 2")

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3
# ═════════════════════════════════════════════════════════════════════════════
_section_header(3, "Chinese Companies — By Year",
                "How has participation changed over time? Is geographic concentration rising or falling?")

try:
    if n_cn == 0:
        st.warning("No Chinese contracts match the current filters.")
    else:
        with st.expander("Debug info", expanded=False):
            yr_dbg = cn_by_year(df_cn)
            st.write(f"df_cn: {df_cn.shape} | years: {int(df_cn['year_awarded'].min())}–{int(df_cn['year_awarded'].max())}")
            st.dataframe(yr_dbg, use_container_width=True)

        yr_df = cn_by_year(df_cn)
        if len(yr_df) == 0:
            st.warning("No year-level data for the current filters.")
        else:
            tab_trends, tab_heatmap, tab_conc = st.tabs(
                ["Annual Trends", "Year × Country Heatmap", "Concentration over Time"]
            )

            with tab_trends:
                try:
                    c1, c2 = st.columns(2)
                    fig_cnt = go.Figure()
                    fig_cnt.add_trace(go.Scatter(
                        x=yr_df["year_awarded"].tolist(), y=yr_df["contracts"].tolist(),
                        mode="lines+markers", name="Contracts",
                        line=dict(color=BLUE_P, width=2.5),
                        fill="tozeroy", fillcolor=CHINA_FILL,
                        marker=dict(color=BLUE_P, size=6),
                    ))
                    _theme(fig_cnt, "Chinese contract count per year")
                    fig_cnt.update_layout(xaxis_title="Year", yaxis_title="Contracts")
                    c1.plotly_chart(fig_cnt, use_container_width=True)

                    fig_val = go.Figure()
                    fig_val.add_trace(go.Scatter(
                        x=yr_df["year_awarded"].tolist(), y=yr_df["total_value"].tolist(),
                        mode="lines+markers", name="Value (USD)",
                        line=dict(color=BLUE_P, width=2.5),
                        fill="tozeroy", fillcolor=CHINA_FILL,
                        marker=dict(color=BLUE_P, size=6),
                    ))
                    _theme(fig_val, "Chinese contract value per year (USD)")
                    fig_val.update_layout(xaxis_title="Year", yaxis_title="Value (USD)")
                    c2.plotly_chart(fig_val, use_container_width=True)

                    st.info(
                        "Left = contract count; Right = combined dollar value. "
                        "A year with few contracts but high value = a small number of very large contracts dominated. "
                        "Look for peaks that may correspond to specific large projects or MDB lending surges."
                    )

                    avg_df = yr_df.dropna(subset=["avg_value"])
                    if len(avg_df) > 0:
                        fig_avg = go.Figure()
                        fig_avg.add_trace(go.Bar(
                            x=avg_df["year_awarded"].tolist(), y=avg_df["avg_value"].tolist(),
                            marker_color=BLUE_M, marker_line_width=0))
                        _theme(fig_avg, "Average value per Chinese contract by year (USD)", height=310)
                        fig_avg.update_layout(xaxis_title="Year", yaxis_title="Avg Value (USD)")
                        st.plotly_chart(fig_avg, use_container_width=True)
                        st.info(
                            "This bar shows the *average size* of a Chinese contract each year. "
                            "Tall bars = a few very large contracts. Short bars = many smaller contracts. "
                            "Compare to the count chart to understand whether China is winning "
                            "fewer but bigger contracts over time."
                        )
                except Exception as e:
                    _show_error(e, "S3 Trends")

            with tab_heatmap:
                try:
                    metric_choice = st.radio(
                        "Metric:", ["Total value (USD)", "Contract count"], horizontal=True)
                    metric_key = "total_value" if "value" in metric_choice else "contracts"
                    piv = cn_country_year_pivot(df_cn, metric_key)

                    if piv.empty:
                        st.info("No data.")
                    else:
                        country_order = piv.sum(axis=0).sort_values(ascending=False).index.tolist()
                        piv_s = piv[country_order]
                        z_data = piv_s.values.T
                        hover = z_data.copy()

                        if metric_key == "total_value":
                            z_plot = np.where(z_data > 0, np.log10(z_data + 1), 0)
                            cs = [[0,"#f4f7fb"],[0.2,BLUE_PALE],[0.5,BLUE_L],
                                  [0.75,BLUE_M],[1.0,NAVY]]
                            cb_t = "log₁₀(USD+1)"
                            htmpl = "Year: %{x}<br>Country: %{y}<br>Value: $%{customdata:,.0f}<extra></extra>"
                        else:
                            z_plot = z_data.astype(float)
                            cs = [[0,"#f4f7fb"],[0.3,BLUE_PALE],[0.6,BLUE_L],
                                  [0.85,BLUE_M],[1.0,BLUE_P]]
                            cb_t = "Contracts"
                            htmpl = "Year: %{x}<br>Country: %{y}<br>Contracts: %{customdata}<extra></extra>"

                        fig_h = go.Figure(go.Heatmap(
                            z=z_plot, x=piv_s.index.tolist(), y=country_order,
                            colorscale=cs, colorbar=dict(title=cb_t, thickness=12),
                            hovertemplate=htmpl, customdata=hover,
                        ))
                        _theme(fig_h, f"Chinese contracts: year × borrower country ({metric_choice})",
                               height=max(400, len(country_order) * 22 + 120))
                        fig_h.update_layout(xaxis_title="Year",
                            plot_bgcolor="#f4f7fb", paper_bgcolor="#ffffff")
                        st.plotly_chart(fig_h, use_container_width=True)
                        st.info(
                            "Darker cells = more activity. Countries sorted top-to-bottom by total Chinese value. "
                            "Look for: (1) sustained dark columns = consistent recipient; "
                            "(2) isolated dark cells = one-off large project; "
                            "(3) activity spreading to new countries over time. "
                            "Value shown on log scale — hover for exact figures."
                        )
                except Exception as e:
                    _show_error(e, "S3 Heatmap")

            with tab_conc:
                try:
                    _hhi_primer()
                    st.markdown(
                        f'<div style="font-size:0.85rem;font-weight:700;color:{NAVY};'
                        f'margin:14px 0 4px">HHI of China\'s contract value across borrower countries — per year</div>'
                        f'<div style="font-size:0.78rem;color:{TEXT_MUTED};margin-bottom:10px">'
                        f'Unit = borrower country. Question: "In this year, how concentrated was '
                        f'China\'s spending across countries?"</div>',
                        unsafe_allow_html=True)

                    hhi_yr = cn_hhi_by_year(df_cn)
                    if hhi_yr.empty or hhi_yr["hhi"].isna().all():
                        st.info("Not enough data to compute HHI per year.")
                    else:
                        fig_hhi = go.Figure()
                        fig_hhi.add_trace(go.Scatter(
                            x=hhi_yr["year_awarded"].tolist(), y=hhi_yr["hhi"].tolist(),
                            mode="lines+markers",
                            line=dict(color=BLUE_P, width=2.5),
                            marker=dict(color=BLUE_P, size=7),
                            name="HHI",
                        ))
                        for thresh, ann, col in [
                            (0.01, "< 0.01 competitive", TEXT_MUTED),
                            (0.15, "0.15 moderate",      "#e8a020"),
                            (0.25, "0.25 concentrated",  RED),
                        ]:
                            fig_hhi.add_hline(y=thresh, line_dash="dot", line_color=col,
                                              annotation_text=ann,
                                              annotation_font=dict(color=col, size=11),
                                              annotation_position="right")
                        _theme(fig_hhi, "China's geographic concentration (HHI) over time", height=380)
                        fig_hhi.update_layout(
                            xaxis_title="Year",
                            yaxis_title="HHI  (0 = spread out,  1 = all in one country)",
                            yaxis_range=[0, 1.05])
                        st.plotly_chart(fig_hhi, use_container_width=True)
                        st.info(
                            "**Going up** = that year's contracts concentrated in fewer countries. "
                            "**Going down** = contracts spread across more countries. "
                            "Years near 1.0 often had only one or two recipients (common in early years). "
                            "Below 0.15 = unconcentrated; above 0.25 = highly concentrated."
                        )
                except Exception as e:
                    _show_error(e, "S3 Concentration")

except Exception as e:
    _show_error(e, "Section 3")

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4
# ═════════════════════════════════════════════════════════════════════════════
_section_header(4, "Comparison — China vs Other Contractor Countries",
                "China (blue, always highlighted) vs top-N comparators from selected groups")

try:
    if len(df4) == 0:
        st.warning("No data for the current comparator selection.")
    else:
        top_labels = top_n_labels(df4, top_n)
        all_labels = ["China"] + top_labels
        cmap = colour_map(all_labels + ["Rest"])

        (tab_share, tab_stack, tab_dist, tab_rank,
         tab_spread, tab_sector, tab_growth, tab_proc, tab_mhhi) = st.tabs([
            "Market Share", "Value Stack", "Value Distribution",
            "Rank Trajectory", "Geographic Spread", "Sector Mix",
            "Growth", "Procurement", "Market HHI",
        ])

        with tab_share:
            try:
                sh = annual_share(df4)
                fig_sh = go.Figure()
                fig_sh.add_trace(go.Scatter(
                    x=sh["year_awarded"].tolist(), y=sh["value_share_pct"].tolist(),
                    name="China — value share %",
                    line=dict(color=BLUE_P, width=3), mode="lines+markers",
                    marker=dict(color=BLUE_P, size=6)))
                fig_sh.add_trace(go.Scatter(
                    x=sh["year_awarded"].tolist(), y=sh["count_share_pct"].tolist(),
                    name="China — count share %",
                    line=dict(color=BLUE_L, width=2, dash="dash"), mode="lines+markers",
                    marker=dict(color=BLUE_L, size=5)))
                _theme(fig_sh, "China's share of total MDB contract value and count per year (%)", height=400)
                fig_sh.update_layout(xaxis_title="Year", yaxis_title="%")
                st.plotly_chart(fig_sh, use_container_width=True)
                st.info(
                    "Solid blue = share of *total dollar value*. Dashed light blue = share of *total count*. "
                    "**Solid above dashed**: Chinese contracts are larger than market average. "
                    "**Solid below dashed**: Chinese contracts are smaller than average."
                )

                prem = sh.dropna(subset=["premium_ratio"])
                if len(prem) > 0:
                    colors_bar = [GREEN if v >= 1 else RED for v in prem["premium_ratio"]]
                    fig_pr = go.Figure()
                    fig_pr.add_trace(go.Bar(
                        x=prem["year_awarded"].tolist(), y=prem["premium_ratio"].tolist(),
                        marker_color=colors_bar, marker_line_width=0))
                    fig_pr.add_hline(y=1, line_dash="dot", line_color=TEXT_MUTED,
                                     annotation_text="1.0 = market average",
                                     annotation_font=dict(color=TEXT_MUTED, size=11))
                    _theme(fig_pr, "China's value-share ÷ count-share (size premium ratio)", height=310)
                    fig_pr.update_layout(xaxis_title="Year", yaxis_title="Ratio")
                    st.plotly_chart(fig_pr, use_container_width=True)
                    st.info(
                        "**Above 1.0**: Chinese contracts were larger than market average that year. "
                        "**Below 1.0**: smaller than average. "
                        "Very large ratios in early years often reflect one massive contract "
                        "in a year when China had very few contracts — cross-check with the trend charts."
                    )
            except Exception as e:
                _show_error(e, "S4 Market Share")

        with tab_stack:
            try:
                sa = stacked_area_data(df4, top_labels)
                cols_o = (["China"] + [l for l in top_labels if l in sa.columns]
                          + (["Rest"] if "Rest" in sa.columns else []))
                fig_st = go.Figure()
                for lbl in cols_o:
                    if lbl not in sa.columns: continue
                    fig_st.add_trace(go.Scatter(
                        x=sa["year_awarded"].tolist(), y=sa[lbl].tolist(),
                        name=lbl, stackgroup="one", mode="lines",
                        line=dict(color=cmap.get(lbl, REST_COLOR), width=2 if lbl=="China" else 1),
                        fillcolor=cmap.get(lbl, REST_COLOR)))
                _theme(fig_st, f"Annual contract value: China / top-{top_n} / Rest (USD)", height=440)
                fig_st.update_layout(xaxis_title="Year", yaxis_title="Value (USD)")
                st.plotly_chart(fig_st, use_container_width=True)
                st.info(
                    "Total height = all MDB contract value that year. China (darkest blue) sits at the bottom. "
                    "Watch whether China's band is growing, stable, or shrinking "
                    "relative to the total stack and to named comparators."
                )
            except Exception as e:
                _show_error(e, "S4 Stack")

        with tab_dist:
            try:
                dist_df = df4[df4["contractor_label"].isin(all_labels)].dropna(subset=["contract_value_usd"])
                if len(dist_df) == 0:
                    st.info("No value data.")
                else:
                    fig_box = go.Figure()
                    for lbl in all_labels:
                        vals = dist_df.loc[dist_df["contractor_label"]==lbl,"contract_value_usd"]
                        if len(vals) == 0: continue
                        fig_box.add_trace(go.Box(
                            y=vals.tolist(), name=lbl,
                            marker_color=cmap.get(lbl,"#7f7f7f"),
                            line_width=2.5 if lbl=="China" else 1.5,
                            boxpoints="outliers", fillcolor=cmap.get(lbl,"#ddd")))
                    _theme(fig_box, "Contract value distribution by contractor country (log scale)", height=440)
                    fig_box.update_layout(yaxis_type="log", yaxis_title="Contract Value (USD)")
                    st.plotly_chart(fig_box, use_container_width=True)
                    st.info(
                        "Box = middle 50% of values. Line inside = median. Dots outside whiskers = outliers. "
                        "Log scale because values span several orders of magnitude. "
                        "A higher box overall = that country wins larger contracts."
                    )

                    am = (dist_df[dist_df["contractor_label"].isin(all_labels)]
                          .groupby("contractor_label", observed=True)["contract_value_usd"]
                          .agg(avg="mean", median="median")
                          .reindex(all_labels).reset_index())
                    c1, c2 = st.columns(2)
                    fig_av = go.Figure()
                    am_s = am.sort_values("avg")
                    fig_av.add_trace(go.Bar(
                        y=am_s["contractor_label"].tolist(), x=am_s["avg"].tolist(),
                        orientation="h", marker_line_width=0,
                        marker_color=[cmap.get(l, BLUE_M) for l in am_s["contractor_label"]]))
                    fig_av.update_layout(yaxis={"categoryorder":"total ascending"})
                    _theme(fig_av, "Average contract value (USD)", height=340)
                    c1.plotly_chart(fig_av, use_container_width=True)

                    fig_me = go.Figure()
                    am_ms = am.sort_values("median")
                    fig_me.add_trace(go.Bar(
                        y=am_ms["contractor_label"].tolist(), x=am_ms["median"].tolist(),
                        orientation="h", marker_line_width=0,
                        marker_color=[cmap.get(l, BLUE_M) for l in am_ms["contractor_label"]]))
                    fig_me.update_layout(yaxis={"categoryorder":"total ascending"})
                    _theme(fig_me, "Median contract value (USD)", height=340)
                    c2.plotly_chart(fig_me, use_container_width=True)
            except Exception as e:
                _show_error(e, "S4 Distribution")

        with tab_rank:
            try:
                rnk = rank_trajectory(df4)
                if len(rnk) == 0:
                    st.info("No rank data available.")
                else:
                    fig_rk = go.Figure()
                    fig_rk.add_trace(go.Scatter(
                        x=rnk["year_awarded"].tolist(), y=rnk["rank"].tolist(),
                        mode="lines+markers+text",
                        line=dict(color=BLUE_P, width=3),
                        marker=dict(size=9, color=BLUE_P),
                        text=rnk["rank"].astype(int).astype(str).tolist(),
                        textposition="top center",
                        textfont=dict(color=NAVY, size=11),
                        name="China rank"))
                    _theme(fig_rk, "China's rank among all contractor countries by annual contract value",
                           height=400)
                    fig_rk.update_layout(
                        xaxis_title="Year",
                        yaxis=dict(autorange="reversed",
                                   title="Rank (1 = highest value that year)",
                                   gridcolor=GRID, linecolor="#d0dae6"))
                    st.plotly_chart(fig_rk, use_container_width=True)
                    st.info(
                        "Rank 1 = China had the highest total contract value that year. "
                        "Y-axis is inverted (up = better). Rising line = China gaining in the rankings. "
                        "Missing years = no Chinese contracts that year."
                    )
                    rnk_d = rnk.copy()
                    rnk_d["china_value"] = rnk_d["china_value"].map(fmt_usd)
                    rnk_d["rank"] = rnk_d["rank"].astype(int)
                    rnk_d.columns = ["Year","Rank","China Value (USD)"]
                    st.markdown(_html_table(rnk_d, "China rank by year"), unsafe_allow_html=True)
            except Exception as e:
                _show_error(e, "S4 Rank")

        with tab_spread:
            try:
                sp = spread_by_label(df4, all_labels)
                if len(sp) == 0:
                    st.info("No spread data.")
                else:
                    fig_sc = px.scatter(sp, x="n_countries", y="n_sectors",
                        text="contractor_label", size="total_value",
                        color="contractor_label", color_discrete_map=cmap,
                        title="Geographic vs sectoral spread — bubble size = total value",
                        labels={"n_countries":"# Borrower countries","n_sectors":"# Sectors"})
                    fig_sc.update_traces(textposition="top center",
                                         textfont=dict(color=NAVY, size=11))
                    _theme(fig_sc, "Geographic vs sectoral spread — bubble size = total value (USD)",
                           height=440)
                    fig_sc.update_layout(showlegend=False)
                    st.plotly_chart(fig_sc, use_container_width=True)
                    st.info(
                        "Right = active in more borrower countries. Up = active in more sectors. "
                        "Bubble size = total value. Top-right = broad reach; bottom-left = niche. "
                        "China is the darkest blue bubble."
                    )
                    c1, c2 = st.columns(2)
                    sp_c = sp.sort_values("n_countries")
                    fig_c = go.Figure()
                    fig_c.add_trace(go.Bar(
                        y=sp_c["contractor_label"].tolist(), x=sp_c["n_countries"].tolist(),
                        orientation="h", marker_line_width=0,
                        marker_color=[cmap.get(l, BLUE_M) for l in sp_c["contractor_label"]]))
                    fig_c.update_layout(yaxis={"categoryorder":"total ascending"})
                    _theme(fig_c, "# Borrower countries per contractor", height=340)
                    c1.plotly_chart(fig_c, use_container_width=True)

                    sp_s = sp.sort_values("n_sectors")
                    fig_s = go.Figure()
                    fig_s.add_trace(go.Bar(
                        y=sp_s["contractor_label"].tolist(), x=sp_s["n_sectors"].tolist(),
                        orientation="h", marker_line_width=0,
                        marker_color=[cmap.get(l, BLUE_M) for l in sp_s["contractor_label"]]))
                    fig_s.update_layout(yaxis={"categoryorder":"total ascending"})
                    _theme(fig_s, "# Sectors per contractor", height=340)
                    c2.plotly_chart(fig_s, use_container_width=True)
            except Exception as e:
                _show_error(e, "S4 Spread")

        with tab_sector:
            try:
                smix = sector_mix(df4, all_labels)
                scols = [c for c in smix.columns if c != "contractor_label"]
                if not scols:
                    st.info("No sector mix data.")
                else:
                    smix_l = smix.melt(id_vars="contractor_label", value_vars=scols,
                                        var_name="sector", value_name="share_pct")
                    lo = ["China"] + [l for l in all_labels if l != "China"]
                    fig_sm = px.bar(smix_l, x="share_pct", y="contractor_label",
                        color="sector", orientation="h",
                        labels={"share_pct":"Share (%)","contractor_label":"","sector":"Sector"},
                        color_discrete_sequence=SECTOR_PAL,
                        category_orders={"contractor_label": lo})
                    _theme(fig_sm,
                           "Sector mix per contractor — each bar = 100% of that country's contracts",
                           height=max(300, len(all_labels)*44))
                    fig_sm.update_layout(
                        barmode="stack", xaxis_range=[0,100],
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
                    st.plotly_chart(fig_sm, use_container_width=True)
                    st.info(
                        "Each row = 100% of that contractor country's contracts. "
                        "Colour segments show sector shares. "
                        "Compare China's row to others — a very different colour mix "
                        "suggests China specialises in specific project types."
                    )
            except Exception as e:
                _show_error(e, "S4 Sector Mix")

        with tab_growth:
            try:
                cagr_df = cagr(df4, all_labels)
                yoy_df  = yoy_growth(df4, all_labels)

                cp = cagr_df.dropna(subset=["cagr_pct"])
                if len(cp) > 0:
                    bar_colors = [BLUE_P if v >= 0 else RED for v in cp["cagr_pct"]]
                    fig_cg = go.Figure()
                    cp_s = cp.sort_values("cagr_pct")
                    fig_cg.add_trace(go.Bar(
                        y=cp_s["contractor_label"].tolist(), x=cp_s["cagr_pct"].tolist(),
                        orientation="h", marker_color=bar_colors, marker_line_width=0))
                    fig_cg.add_vline(x=0, line_dash="dot", line_color=TEXT_MUTED)
                    _theme(fig_cg,
                           "Compound Annual Growth Rate of contract value over the full period (%)",
                           height=340)
                    fig_cg.update_layout(yaxis={"categoryorder":"total ascending"},
                                          xaxis_title="CAGR (%)")
                    st.plotly_chart(fig_cg, use_container_width=True)
                    st.info(
                        "CAGR = average yearly growth rate over the full period. "
                        "Think of it like an interest rate: 10% CAGR = value grew ~10% per year on average. "
                        "Blue bars (right of 0) = growing; red bars (left) = shrinking on average."
                    )

                fig_yoy = go.Figure()
                for lbl in all_labels:
                    if lbl not in yoy_df.columns: continue
                    is_china = lbl == "China"
                    fig_yoy.add_trace(go.Scatter(
                        x=yoy_df["year_awarded"].tolist(), y=yoy_df[lbl].tolist(),
                        name=lbl,
                        line=dict(color=cmap.get(lbl, REST_COLOR),
                                  width=3 if is_china else 1.5,
                                  dash="solid" if is_china else "dot"),
                        mode="lines",
                        opacity=1.0 if is_china else 0.55))
                fig_yoy.add_hline(y=0, line_color=GRID, line_width=1)
                _theme(fig_yoy,
                       "Year-over-year growth of contract value (%) — China solid blue",
                       height=400)
                fig_yoy.update_layout(xaxis_title="Year", yaxis_title="YoY growth (%)")
                st.plotly_chart(fig_yoy, use_container_width=True)
                st.info(
                    "Above 0% = grew vs prior year; below 0% = shrank. "
                    "China (solid, fully opaque) is highlighted; comparators are dashed and faded. "
                    "Extreme swings for China often reflect one large project entering or leaving — "
                    "always cross-check with the count and value trend charts."
                )
            except Exception as e:
                _show_error(e, "S4 Growth")

        with tab_proc:
            try:
                proc = procurement_profile(df4, all_labels)
                mcols = [c for c in proc.columns if c.startswith("method_")]
                if mcols:
                    pl = proc[["contractor_label"]+mcols].melt(
                        id_vars="contractor_label", var_name="method", value_name="share_pct")
                    pl["method"] = pl["method"].str.replace("method_","",regex=False)
                    lo = ["China"]+[l for l in all_labels if l!="China"]
                    mc = {m: METHOD_COLORS.get(m,"#c8d3df") for m in pl["method"].unique()}
                    fig_pm = px.bar(pl, x="share_pct", y="contractor_label",
                        color="method", orientation="h",
                        color_discrete_map=mc,
                        labels={"share_pct":"Share (%)","contractor_label":""},
                        category_orders={"contractor_label":lo})
                    _theme(fig_pm,
                           "Procurement method mix per contractor — % of contracts (IDB/CDB only)",
                           height=max(300,len(all_labels)*44))
                    fig_pm.update_layout(barmode="stack", xaxis_range=[0,100])
                    st.plotly_chart(fig_pm, use_container_width=True)
                    st.info(
                        "Each row = 100% of that contractor's contracts. "
                        "'Unknown' (grey) is large because World Bank doesn't record detailed methods. "
                        "**Green = Open/Competitive** (most transparent). "
                        "**Red = Direct/Single-Source** (no competition). "
                        "Focus on IDB/CDB rows for meaningful method comparisons."
                    )

                bl = proc[["contractor_label","direct_award_pct","jv_pct"]].copy()
                bl_l = bl.melt(id_vars="contractor_label", var_name="indicator", value_name="pct")
                bl_l["indicator"] = bl_l["indicator"].map({
                    "direct_award_pct": "Direct / single-source (%)",
                    "jv_pct":           "Joint venture (%)"})
                lo = ["China"]+[l for l in all_labels if l!="China"]
                fig_bl = px.bar(bl_l, x="contractor_label", y="pct", color="indicator",
                    barmode="group",
                    labels={"pct":"%","contractor_label":""},
                    category_orders={"contractor_label":lo},
                    color_discrete_map={"Direct / single-source (%)":RED,"Joint venture (%)":BLUE_M})
                _theme(fig_bl, "Direct awards and joint-venture share by contractor (%)", height=360)
                fig_bl.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02))
                st.plotly_chart(fig_bl, use_container_width=True)

                st.markdown("#### By MDB source")
                for src in sorted(df4["data_source"].unique()):
                    sub = df4[(df4["data_source"]==src) & (df4["contractor_label"].isin(all_labels))]
                    if len(sub) == 0: continue
                    with st.expander(src):
                        pp = procurement_profile(sub, all_labels)
                        disp = pp[["contractor_label","contracts","direct_award_pct","jv_pct"]].copy()
                        disp.columns = ["Contractor","Contracts","Direct/SSS (%)","JV (%)"]
                        st.markdown(_html_table(disp), unsafe_allow_html=True)
            except Exception as e:
                _show_error(e, "S4 Procurement")

        with tab_mhhi:
            try:
                _hhi_primer()
                st.markdown(
                    f'<div style="font-size:0.84rem;font-weight:700;color:{NAVY};margin:14px 0 4px">'
                    f'Market HHI — how concentrated is the contractor market in each sector / country?</div>'
                    f'<div style="font-size:0.78rem;color:{TEXT_MUTED};margin-bottom:10px">'
                    f'Unit = contractor country. Question: "Within this sector or recipient country, '
                    f'is one nationality winning most contracts? Is that nationality China?"</div>',
                    unsafe_allow_html=True)

                hhi_sec, hhi_ctr = market_hhi(df4)
                c1, c2 = st.columns(2)

                fig_hs = go.Figure()
                fig_hs.add_trace(go.Bar(
                    x=hhi_sec["project_sector"].tolist(), y=hhi_sec["hhi"].tolist(),
                    name="Market HHI", marker_color=BLUE_M, marker_line_width=0))
                fig_hs.add_trace(go.Scatter(
                    x=hhi_sec["project_sector"].tolist(),
                    y=hhi_sec["china_share_pct"].tolist(),
                    name="China share (%)", yaxis="y2", mode="markers+lines",
                    marker=dict(color=BLUE_P, size=10, symbol="diamond"),
                    line=dict(color=BLUE_P, dash="dot", width=2)))
                _theme(fig_hs, "Market HHI and China's share by sector", height=380, secondary_y=True)
                fig_hs.update_layout(
                    yaxis_title="HHI (0–1)",
                    yaxis2=dict(title="China share (%)", overlaying="y", side="right",
                                range=[0,100], gridcolor=GRID, tickfont=dict(color=TEXT_MUTED,size=11)),
                    xaxis_tickangle=-30)
                c1.plotly_chart(fig_hs, use_container_width=True)

                fig_hc = go.Figure()
                fig_hc.add_trace(go.Bar(
                    x=hhi_ctr["borrower country"].tolist(), y=hhi_ctr["hhi"].tolist(),
                    name="Market HHI", marker_color=BLUE_M, marker_line_width=0))
                fig_hc.add_trace(go.Scatter(
                    x=hhi_ctr["borrower country"].tolist(),
                    y=hhi_ctr["china_share_pct"].tolist(),
                    name="China share (%)", yaxis="y2", mode="markers+lines",
                    marker=dict(color=BLUE_P, size=8, symbol="diamond"),
                    line=dict(color=BLUE_P, dash="dot", width=2)))
                _theme(fig_hc, "Market HHI and China's share by borrower country",
                       height=420, secondary_y=True)
                fig_hc.update_layout(
                    yaxis_title="HHI (0–1)",
                    yaxis2=dict(title="China share (%)", overlaying="y", side="right",
                                range=[0,100], gridcolor=GRID, tickfont=dict(color=TEXT_MUTED,size=11)),
                    xaxis_tickangle=-45)
                c2.plotly_chart(fig_hc, use_container_width=True)

                st.info(
                    "**Blue bars (left axis)** = overall market concentration in that sector/country. "
                    "**Blue diamond line (right axis)** = China's specific share (%). "
                    "\n\n"
                    "**High HHI + High China share** → China dominates. "
                    "**High HHI + Low China share** → Dominated by someone else. "
                    "**Low HHI + High China share** → Competitive market where China is still significant. "
                    "**Low HHI + Low China share** → Fragmented market, China plays a minor role."
                )

                hd_sec = hhi_sec[["project_sector","hhi","hhi_label","china_share_pct","n_contractors","total_value"]].copy()
                hd_sec["total_value"] = hd_sec["total_value"].map(fmt_usd)
                hd_sec.columns = ["Sector","HHI","Interpretation","China share (%)","# Countries","Total Value"]
                st.markdown(_html_table(hd_sec, "Market concentration by sector"),
                            unsafe_allow_html=True)

                hd_ctr = hhi_ctr[["borrower country","hhi","hhi_label","china_share_pct","n_contractors","total_value"]].copy()
                hd_ctr["total_value"] = hd_ctr["total_value"].map(fmt_usd)
                hd_ctr.columns = ["Country","HHI","Interpretation","China share (%)","# Countries","Total Value"]
                st.markdown(_html_table(hd_ctr, "Market concentration by borrower country"),
                            unsafe_allow_html=True)
            except Exception as e:
                _show_error(e, "S4 Market HHI")

except Exception as e:
    _show_error(e, "Section 4")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="background:{NAVY};border-radius:8px;padding:14px 22px;margin-top:28px;
            color:{BLUE_L};font-size:0.75rem;text-align:center">
  World Bank Capstone · <em>Chinese Companies' Participation in MDB Public Procurement
  in Latin America</em> · IDB · World Bank · CDB · 2000–2026 · Descriptive analysis
</div>
<div style="height:20px"></div>
""", unsafe_allow_html=True)
