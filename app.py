"""
app.py — Chinese Companies in MDB Procurement, LAC
5-section Streamlit dashboard with BI light theme.
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
    sector_mix, yoy_growth, cagr, procurement_profile, market_hhi,
)

# ── Palette ───────────────────────────────────────────────────────────────────
NAVY      = "#10395e"
BLUE_P    = "#0a66c2"
BLUE_M    = "#3b82c4"
BLUE_L    = "#9ec5e8"
BLUE_PALE = "#dbeafe"
GRID      = "#e6ebf1"
GREEN     = "#1a9e5f"
RED       = "#d23b3b"
MUTED     = "#6b7c93"

CHINA_COLOR = BLUE_P
CHINA_FILL  = "rgba(10,102,194,0.13)"
HK_COLOR    = BLUE_M
REST_COLOR  = "#c0d3e6"
COMP_BLUES  = ["#3b82c4","#5a9fd4","#7db8e2","#9ec5e8","#a8c4de","#6b9dbf","#4d88ae","#2d6fa0"]
SECTOR_PAL  = [BLUE_P,"#1a9e5f",BLUE_M,"#e8a020","#9b59b6",RED,"#5a9fd4","#20a0a0","#f39c12",BLUE_L]
METHOD_COLORS = {
    "Open/Competitive":     GREEN,
    "Limited/Shopping":     BLUE_M,
    "Direct/Single-Source": RED,
    "Consultant Selection": BLUE_L,
    "Unknown":              "#c8d3df",
}

_MISSING_NOTE = ("⚠️ Some contracts lack a USD value due to unavailable currency conversion rates; "
                 "included in counts but excluded from value totals.")

# ── CSS ───────────────────────────────────────────────────────────────────────
_CSS = f"""
<style>
.stApp {{ background:#f4f7fb; }}
section[data-testid="stMain"] > div {{ background:#f4f7fb; }}
.block-container {{ padding-top:0 !important; max-width:1280px !important;
                    padding-left:1.5rem !important; padding-right:1.5rem !important; }}
header[data-testid="stHeader"] {{ background:transparent !important; }}
div[data-testid="stToolbar"] {{ display:none; }}
[data-testid="stSidebar"] {{ background:#ffffff !important; border-right:1px solid {GRID}; }}
[data-testid="stSidebar"] label {{ color:{MUTED} !important; font-size:0.78rem !important;
    font-weight:600; text-transform:uppercase; letter-spacing:0.05em; }}
[data-testid="stTabs"] button[role="tab"] {{ color:{MUTED}; font-size:0.79rem; font-weight:500;
    border-bottom:2px solid transparent; padding:6px 14px; }}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {{ color:{BLUE_P};
    border-bottom:2px solid {BLUE_P}; font-weight:700; }}
[data-testid="stExpander"] {{ border:1px solid {GRID} !important; border-radius:8px !important;
    background:#ffffff !important; }}
[data-testid="stMetric"] {{ background:#ffffff; border:1px solid {GRID}; border-radius:10px;
    padding:14px 16px; box-shadow:0 1px 6px rgba(16,57,94,.07); }}
[data-testid="stMetricLabel"] span {{ color:{MUTED} !important; font-size:0.7rem !important;
    text-transform:uppercase; letter-spacing:.07em; font-weight:600; }}
[data-testid="stMetricValue"] {{ color:{NAVY} !important; font-size:1.4rem !important; font-weight:700; }}
[data-testid="stDownloadButton"] button {{ background:{BLUE_P} !important; color:#fff !important;
    border:none !important; border-radius:6px; font-size:0.78rem; font-weight:600; padding:6px 14px; }}
hr {{ border-color:{GRID} !important; margin:12px 0; }}
[data-testid="stCaptionContainer"] p {{ color:{MUTED}; font-size:0.76rem; }}
.kpi-card {{ background:#fff; border:1px solid {GRID}; border-radius:10px;
    padding:14px 16px 10px; box-shadow:0 1px 8px rgba(16,57,94,.07);
    margin-bottom:8px; min-height:108px; display:flex; flex-direction:column; gap:3px; }}
.kpi-label {{ color:{MUTED}; font-size:0.67rem; text-transform:uppercase;
    letter-spacing:.08em; font-weight:700; }}
.kpi-value {{ color:{NAVY}; font-size:1.3rem; font-weight:800; line-height:1.1; margin-top:3px; }}
.kpi-delta-pos {{ color:{GREEN}; font-size:0.72rem; font-weight:700; }}
.kpi-delta-neg {{ color:{RED}; font-size:0.72rem; font-weight:700; }}
</style>
"""

# ── Helpers ───────────────────────────────────────────────────────────────────

def fmt_usd(v):
    if v is None or (isinstance(v, float) and np.isnan(v)): return "N/A"
    if abs(v) >= 1e9:  return f"${v/1e9:.2f}B"
    if abs(v) >= 1e6:  return f"${v/1e6:.1f}M"
    if abs(v) >= 1e3:  return f"${v/1e3:,.0f}K"
    return f"${v:,.0f}"


def fmt_count(v):
    return f"{int(v):,}" if not (isinstance(v, float) and np.isnan(v)) else "—"


def _show_error(e, label=""):
    st.error(f"[{label}] {type(e).__name__}: {e}" if label else f"{type(e).__name__}: {e}")
    _tb.print_exc()


def _spark_svg(values, color=BLUE_P, fill="rgba(10,102,194,0.13)", w=84, h=34):
    vals = [float(v) for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if len(vals) < 2: return ""
    mn, mx = min(vals), max(vals); rng = mx - mn if mx != mn else 1
    n = len(vals)
    pts = [(i * w / (n-1), h - 2 - (v-mn)/rng*(h-6)) for i, v in enumerate(vals)]
    path = " ".join(f"{'M' if i==0 else 'L'}{x:.1f},{y:.1f}" for i,(x,y) in enumerate(pts))
    fp = f"{path} L{pts[-1][0]:.1f},{h} L0,{h} Z"
    return (f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" style="display:block;margin-top:6px">'
            f'<path d="{fp}" fill="{fill}"/>'
            f'<path d="{path}" stroke="{color}" stroke-width="1.8" fill="none" '
            f'stroke-linecap="round" stroke-linejoin="round"/></svg>')


def _kpi_card(col, label, value, delta=None, spark_values=None):
    spark = _spark_svg(spark_values) if spark_values else ""
    d_html = ""
    if delta is not None:
        arrow = "▲" if delta >= 0 else "▼"
        cls   = "kpi-delta-pos" if delta >= 0 else "kpi-delta-neg"
        d_html = f'<div class="{cls}">{arrow} {abs(delta):.1f}% YoY</div>'
    with col:
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-label">{label}</div>'
            f'<div class="kpi-value">{value}</div>{d_html}{spark}</div>',
            unsafe_allow_html=True)


def _section_header(num, title, subtitle="", scope=""):
    scope_badge = ""
    if scope:
        bg = BLUE_P if "full" in scope.lower() else "#1a7a4a"
        scope_badge = (f'<div style="margin-left:auto;background:{bg};color:#fff;'
                       f'font-size:0.68rem;font-weight:700;padding:3px 10px;border-radius:12px;'
                       f'letter-spacing:0.04em;white-space:nowrap">{scope}</div>')
    sub = (f'<div style="color:{BLUE_L};font-size:0.77rem;margin-top:3px">{subtitle}</div>'
           if subtitle else "")
    st.markdown(
        f'<div style="background:linear-gradient(90deg,{NAVY} 0%,#1a527d 100%);'
        f'border-radius:8px;padding:14px 22px;margin:22px 0 10px;'
        f'display:flex;align-items:center;gap:14px">'
        f'<div style="color:{BLUE_L};font-size:1.6rem;font-weight:900;opacity:.5;line-height:1">§{num}</div>'
        f'<div style="flex:1"><div style="color:#fff;font-size:1.05rem;font-weight:700">'
        f'{title}</div>{sub}</div>{scope_badge}</div>',
        unsafe_allow_html=True)


def _explain(what, example=""):
    ex_html = (f'<br><br><strong>Worked example:</strong> {example}') if example else ""
    st.markdown(
        f'<div style="background:#f0f7ff;border-left:3px solid {BLUE_M};'
        f'border-radius:0 6px 6px 0;padding:10px 14px;margin:4px 0 10px;font-size:0.82rem;color:{NAVY}">'
        f'<strong>What this shows:</strong> {what}{ex_html}</div>',
        unsafe_allow_html=True)


def _missing_note():
    st.caption(_MISSING_NOTE)


def _theme(fig, title="", height=370, secondary_y=False):
    ax = dict(gridcolor=GRID, linecolor="#d0dae6", zerolinecolor=GRID,
              tickcolor=MUTED, tickfont=dict(color=MUTED, size=11),
              title_font=dict(color=MUTED, size=11), zeroline=False)
    upd = dict(
        plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
        font=dict(family="Inter,'Helvetica Neue',Arial,sans-serif", color="#1b2a3a", size=12),
        title=dict(text=f"<b>{title}</b>" if title else None,
                   font=dict(color=NAVY, size=13), x=0, xanchor="left", y=0.97),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=11), title_font=dict(size=11)),
        margin=dict(t=52 if title else 20, b=22, l=6, r=6),
        height=height, xaxis=ax, yaxis=ax,
        hoverlabel=dict(bgcolor=NAVY, font_color="#fff", font_size=12),
        colorway=[BLUE_P, BLUE_M, BLUE_L, GREEN, "#e8a020", RED, "#9b59b6"],
    )
    if secondary_y:
        upd["yaxis2"] = dict(**ax)
    fig.update_layout(**upd)
    return fig


def colour_map(labels):
    cmap, ci = {}, 0
    for lbl in labels:
        if lbl == "China":            cmap[lbl] = CHINA_COLOR
        elif lbl == "Rest":           cmap[lbl] = REST_COLOR
        elif lbl == "Hong Kong SAR":  cmap[lbl] = HK_COLOR
        else:
            cmap[lbl] = COMP_BLUES[ci % len(COMP_BLUES)]; ci += 1
    return cmap


def _html_table(df, title="", bold_last_row=False):
    hdr = "".join(
        f'<th style="background:{NAVY};color:#fff;padding:9px 14px;font-size:0.73rem;'
        f'font-weight:700;letter-spacing:.04em;text-align:left;white-space:nowrap">{c}</th>'
        for c in df.columns)
    rows_html = ""
    for i, (_, row) in enumerate(df.iterrows()):
        is_last = bold_last_row and (i == len(df) - 1)
        bg = NAVY if is_last else ("#f4f7fb" if i % 2 else "#ffffff")
        tc = "#fff" if is_last else NAVY
        fw = "700" if is_last else "400"
        cells = "".join(
            f'<td style="padding:7px 14px;font-size:0.82rem;color:{tc};font-weight:{fw};'
            f'border-bottom:1px solid {GRID};background:{bg}">{v}</td>'
            for v in row)
        rows_html += f"<tr>{cells}</tr>"
    ttl = (f'<div style="font-size:0.85rem;font-weight:700;color:{NAVY};margin-bottom:8px">'
           f'{title}</div>') if title else ""
    return (f'{ttl}<div style="border:1px solid {GRID};border-radius:8px;overflow:auto;'
            f'margin:6px 0;max-height:480px">'
            f'<table style="width:100%;border-collapse:collapse;font-family:Inter,Arial,sans-serif">'
            f'<thead><tr>{hdr}</tr></thead><tbody>{rows_html}</tbody></table></div>')


def _pivot_html(df_display, title="", idx_label="Country", total_label="Grand Total", note=""):
    """Render a pivot with navy header, zebra body, bold navy total row + highlighted total column."""
    cols = list(df_display.columns)
    hdr_cells = (f'<th style="background:{NAVY};color:#fff;padding:9px 12px;font-size:0.72rem;'
                 f'font-weight:700;text-align:left;white-space:nowrap;'
                 f'position:sticky;top:0;z-index:2">{idx_label}</th>')
    for c in cols:
        bg2 = "#0d2d4a" if str(c) == total_label else NAVY
        hdr_cells += (f'<th style="background:{bg2};color:#fff;padding:9px 12px;font-size:0.72rem;'
                      f'font-weight:700;text-align:right;white-space:nowrap;'
                      f'position:sticky;top:0;z-index:2">{c}</th>')

    body = ""
    for i, (idx, row) in enumerate(df_display.iterrows()):
        is_tot = str(idx) == total_label
        row_bg = NAVY if is_tot else ("#f4f7fb" if i % 2 else "#fff")
        row_tc = "#fff" if is_tot else NAVY
        row_fw = "700" if is_tot else "400"
        cells = (f'<td style="background:{row_bg};color:{row_tc};font-weight:{row_fw};'
                 f'padding:7px 12px;font-size:0.8rem;text-align:left;white-space:nowrap;'
                 f'border-bottom:1px solid {GRID}">{idx}</td>')
        for c in cols:
            is_tc = str(c) == total_label
            if is_tot and is_tc:
                cbg, ctc, cfw = "#0d2d4a", "#fff", "700"
            elif is_tot:
                cbg, ctc, cfw = NAVY, "#fff", "700"
            elif is_tc:
                cbg, ctc, cfw = "#e8f0fb", NAVY, "700"
            else:
                cbg, ctc, cfw = row_bg, row_tc, row_fw
            cells += (f'<td style="background:{cbg};color:{ctc};font-weight:{cfw};'
                      f'padding:7px 12px;font-size:0.8rem;text-align:right;white-space:nowrap;'
                      f'border-bottom:1px solid {GRID}">{row[c]}</td>')
        body += f"<tr>{cells}</tr>"

    ttl = (f'<div style="font-size:0.88rem;font-weight:700;color:{NAVY};margin-bottom:7px">'
           f'{title}</div>') if title else ""
    nt  = (f'<div style="font-size:0.72rem;color:{MUTED};margin-top:6px;font-style:italic">'
           f'{note}</div>') if note else ""
    return (f'{ttl}<div style="border:1px solid {GRID};border-radius:8px;overflow:auto;'
            f'max-height:520px;margin:6px 0">'
            f'<table style="width:100%;border-collapse:collapse;font-family:Inter,Arial,sans-serif">'
            f'<thead><tr>{hdr_cells}</tr></thead><tbody>{body}</tbody></table></div>{nt}')


def _hhi_box(h, lbl, unit_desc, worked_example):
    """Render one HHI result box: value + label + explanation + worked example."""
    st.markdown(
        f'<div style="background:#fff;border:1px solid {GRID};border-radius:8px;'
        f'padding:16px 20px;margin:8px 0">'
        f'<div style="font-size:1.6rem;font-weight:800;color:{BLUE_P}">{h:.4f}</div>'
        f'<div style="font-size:0.95rem;font-weight:700;color:{NAVY};margin:2px 0">{lbl}</div>'
        f'<div style="font-size:0.8rem;color:{MUTED};margin-bottom:10px">{unit_desc}</div>'
        f'<div style="font-size:0.8rem;color:{NAVY};border-top:1px solid {GRID};'
        f'padding-top:10px">{worked_example}</div></div>',
        unsafe_allow_html=True)


# ── Page config + CSS + data ──────────────────────────────────────────────────
st.set_page_config(page_title="Chinese Companies in LAC MDB Procurement",
                   layout="wide", initial_sidebar_state="expanded")
st.markdown(_CSS, unsafe_allow_html=True)
df_all = get_data()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Filters")
    st.caption("Applied to Sections 2 – 5. Section 1 has its own local filters.")
    yr_min = int(df_all["year_awarded"].min())
    yr_max = int(df_all["year_awarded"].max())
    year_range    = st.slider("Year range", yr_min, yr_max, (yr_min, yr_max))
    all_sources   = sorted(df_all["data_source"].dropna().unique())
    sel_sources   = st.multiselect("MDB source", all_sources, default=all_sources)
    all_sectors   = sorted(df_all["project_sector"].dropna().unique())
    sel_sectors   = st.multiselect("Sector", all_sectors, default=all_sectors)
    all_countries = sorted(df_all["borrower country"].dropna().unique())
    sel_countries = st.multiselect("Borrower country", all_countries, default=all_countries)
    st.divider()
    st.markdown("**Section 5 options**")
    top_n = st.slider("Top-N comparators", 3, 15, 8)

# ── Apply global filters ──────────────────────────────────────────────────────
fmask = (
    df_all["year_awarded"].between(year_range[0], year_range[1])
    & df_all["data_source"].isin(sel_sources)
    & df_all["project_sector"].isin(sel_sectors)
    & df_all["borrower country"].isin(sel_countries)
)
df    = df_all[fmask].copy()          # full dataset (all contractors)
df_cn = df[df["is_chinese"]].copy()  # Chinese only
df4   = df.copy()                     # Section 5: full dataset

# ── Dashboard header ──────────────────────────────────────────────────────────
st.markdown(f"""
<div style="background:linear-gradient(100deg,{NAVY} 0%,#1a527d 100%);
            padding:22px 28px 18px;margin:-1.5rem -1.5rem 0;border-bottom:3px solid {BLUE_P}">
  <div style="color:#fff;font-size:1.45rem;font-weight:800;line-height:1.2">
    Chinese Companies in MDB Public Procurement — Latin America
  </div>
  <div style="color:{BLUE_L};font-size:0.82rem;margin-top:5px">
    World Bank · IDB · CDB &nbsp;|&nbsp; 2000–2026 &nbsp;|&nbsp; Descriptive analysis
  </div>
</div>
<div style="height:18px"></div>
""", unsafe_allow_html=True)

# ── KPI row ───────────────────────────────────────────────────────────────────
n_cn  = len(df_cn)
val   = df_cn["contract_value_usd"].sum()
yr_df_kpi = cn_by_year(df_cn) if n_cn > 0 else pd.DataFrame()

def _delta(series):
    s = series.dropna()
    if len(s) < 2 or s.iloc[-2] == 0: return None
    return (s.iloc[-1] - s.iloc[-2]) / abs(s.iloc[-2]) * 100

if len(yr_df_kpi) > 0:
    spark_cnt  = yr_df_kpi["contracts"].tolist()
    spark_val  = yr_df_kpi["total_value"].tolist()
    spark_avg  = yr_df_kpi["avg_value"].tolist()
    ctry_yr    = df_cn.groupby("year_awarded")["borrower country"].nunique()
    spark_ctry = ctry_yr.reindex(yr_df_kpi["year_awarded"]).tolist()
    sec_yr     = df_cn.groupby("year_awarded")["project_sector"].nunique()
    spark_sec  = sec_yr.reindex(yr_df_kpi["year_awarded"]).tolist()
    da_yr      = df_cn.groupby("year_awarded")["is_direct_award"].mean() * 100
    spark_da   = da_yr.reindex(yr_df_kpi["year_awarded"]).tolist()
    avg_val    = df_cn["contract_value_usd"].mean()
    da_share   = df_cn["is_direct_award"].mean() * 100 if n_cn > 0 else float("nan")
else:
    spark_cnt = spark_val = spark_avg = spark_ctry = spark_sec = spark_da = []
    avg_val = float("nan"); da_share = float("nan")

k = st.columns(6)
_kpi_card(k[0], "Chinese contracts",  f"{n_cn:,}",        _delta(yr_df_kpi.get("contracts", pd.Series())), spark_cnt)
_kpi_card(k[1], "Total value (USD)",  fmt_usd(val),        _delta(yr_df_kpi.get("total_value", pd.Series())), spark_val)
_kpi_card(k[2], "Avg contract value", fmt_usd(avg_val),    _delta(yr_df_kpi.get("avg_value", pd.Series())), spark_avg)
_kpi_card(k[3], "Borrower countries", str(df_cn["borrower country"].nunique()), None, spark_ctry)
_kpi_card(k[4], "Sectors",            str(df_cn["project_sector"].nunique()),   None, spark_sec)
_kpi_card(k[5], "Direct-award share", f"{da_share:.1f}%" if not np.isnan(da_share) else "—", None, spark_da)

st.markdown('<div style="height:4px"></div>', unsafe_allow_html=True)

with st.expander("📋 Methodology & data notes", expanded=False):
    rpt = get_cleaning_report()
    st.markdown(f"""
**Data source:** `worldbank_idb_cdb_merged_0614V2.xlsx` — IDB (158,272 rows) · World Bank (78,950) · CDB (429) · **237,651 total** · 2000–2026.

**Chinese companies** = `contractor_country` contains "china" (case-insensitive), including joint-venture combos. **Hong Kong SAR excluded** (tracked separately).
Results: **{rpt['n_chinese']} Chinese contracts**, {rpt['n_hk']} Hong Kong SAR contracts.

**Cleaning:** (1) Typos fixed — BRICKES→BRICS, mixed-case labels standardised.
(2) {rpt['n_null_values']} contracts have no USD value (unavailable conversion rate) — included in counts, excluded from value totals.
(3) Procurement methods harmonised into 5 buckets: Open/Competitive, Limited/Shopping, Direct/Single-Source, Consultant Selection, Unknown. World Bank records mostly fall into Unknown.

**Dashboard structure:** Section 1 & 5 = full dataset (all contractors). Sections 2–4 = Chinese companies only.

**HHI** = Herfindahl-Hirschman Index (0–1). Formula: Σ(shareᵢ²). Thresholds: <0.01 highly competitive · 0.01–0.15 unconcentrated · 0.15–0.25 moderate · >0.25 highly concentrated.
    """)

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1 — PIVOT TABLES  (full dataset)
# ═════════════════════════════════════════════════════════════════════════════
_section_header(1, "Pivot Tables — Full Dataset Overview",
                "Fixed cross-tabulations of the full contract database (all contractor countries)",
                scope="Full dataset")

st.markdown(
    f'<div style="background:{BLUE_PALE};border-left:3px solid {BLUE_P};border-radius:0 6px 6px 0;'
    f'padding:10px 16px;font-size:0.82rem;color:{NAVY};margin-bottom:12px">'
    f'This section covers <strong>all contractors</strong> (full dataset), unlike Sections 2–4 which are China-only. '
    f'Use the filters below to narrow the data. Counts = number of contracts; '
    f'value = total USD. Percentages show each cell\'s share of its column total.</div>',
    unsafe_allow_html=True)

# Section 1 local filters
with st.container():
    f1a, f1b, f1c = st.columns([1, 1, 1])
    all_groups = sorted(df_all["contractor_country_group"].dropna().unique())
    s1_groups  = f1a.multiselect("Contractor group", all_groups, default=all_groups, key="s1_groups")
    s1_china   = f1c.checkbox("Chinese contractors only", value=False, key="s1_china")

df1_base = df_cn if s1_china else df
df1 = df1_base[df1_base["contractor_country_group"].isin(s1_groups)] if s1_groups != all_groups else df1_base

try:
    tab_p1, tab_p2, tab_p3 = st.tabs([
        "Pivot 1 — Count by MDB source",
        "Pivot 2 — Value (USD) by MDB source",
        "Pivot 3 — Count by JV status",
    ])

    # ── Pivot 1: Count by borrower country × MDB source ──────────────────────
    with tab_p1:
        try:
            ct1 = pd.crosstab(df1["borrower country"], df1["data_source"],
                              margins=True, margins_name="Grand Total")
            pct1 = (ct1.div(ct1.loc["Grand Total"]) * 100).round(2)

            # Format: "23,345\n(14.75%)" — use HTML <br>
            disp1 = pd.DataFrame(index=ct1.index, columns=ct1.columns)
            for r in ct1.index:
                for c in ct1.columns:
                    cnt = ct1.loc[r, c]
                    p   = pct1.loc[r, c]
                    if r == "Grand Total":
                        disp1.loc[r, c] = f"{cnt:,}"
                    else:
                        disp1.loc[r, c] = f"{cnt:,} ({p:.2f}%)"

            st.markdown(
                _pivot_html(disp1,
                    title="Contract count by borrower country × MDB source",
                    idx_label="Borrower country",
                    total_label="Grand Total",
                    note="Numbers show: count (% of column total). Column totals show grand count."),
                unsafe_allow_html=True)

            _explain(
                "Each cell = number of contracts between that borrower country (row) and MDB bank (column). "
                "The percentage in brackets shows each country's share of that bank's total contracts. "
                "Grand Total row shows each bank's total contract count.",
                example=(f"The Grand Total column shows the overall contract count per country. "
                         f"Grand Total row shows IDB total = {fmt_count(ct1.loc['Grand Total', 'IDB'] if 'IDB' in ct1.columns else 0)} contracts "
                         f"across all borrower countries.")
                if "IDB" in ct1.columns else "")

            csv1 = ct1.reset_index().to_csv(index=False).encode()
            st.download_button("⬇ Download Pivot 1 (CSV)", csv1, "pivot1_count_by_source.csv", "text/csv")
        except Exception as e:
            _show_error(e, "S1 Pivot 1")

    # ── Pivot 2: Value (USD) by borrower country × MDB source ────────────────
    with tab_p2:
        try:
            pv2 = (df1.groupby(["borrower country", "data_source"], observed=True)
                   ["contract_value_usd"].sum().unstack(fill_value=0))
            pv2["Grand Total"] = pv2.sum(axis=1)
            tot_row = pv2.sum().rename("Grand Total")
            pv2 = pd.concat([pv2, tot_row.to_frame().T])
            grand_total = pv2.loc["Grand Total", "Grand Total"]

            disp2 = pd.DataFrame(index=pv2.index, columns=pv2.columns)
            for r in pv2.index:
                for c in pv2.columns:
                    v = pv2.loc[r, c]
                    col_tot = pv2.loc["Grand Total", c]
                    col_pct = (v / col_tot * 100) if col_tot > 0 else 0
                    gt_pct  = (v / grand_total * 100) if grand_total > 0 else 0
                    if r == "Grand Total" and c == "Grand Total":
                        disp2.loc[r, c] = fmt_usd(v)
                    elif r == "Grand Total":
                        disp2.loc[r, c] = fmt_usd(v)
                    elif c == "Grand Total":
                        disp2.loc[r, c] = f"{fmt_usd(v)} ({gt_pct:.1f}% of total)"
                    else:
                        disp2.loc[r, c] = f"{fmt_usd(v)} ({col_pct:.1f}%)"

            st.markdown(
                _pivot_html(disp2,
                    title="Contract value (USD) by borrower country × MDB source",
                    idx_label="Borrower country",
                    total_label="Grand Total",
                    note="Non-Grand-Total cells: value (% of column total). Grand Total column: value (% of grand total). " + _MISSING_NOTE),
                unsafe_allow_html=True)

            _explain(
                "Each cell = total USD value of contracts between that country and that MDB. "
                "The Grand Total column shows each country's share of ALL contract value across all banks.",
                example=(f"Grand Total for all countries and all banks = {fmt_usd(grand_total)}."))

            csv2 = pv2.reset_index().to_csv(index=False).encode()
            st.download_button("⬇ Download Pivot 2 (CSV)", csv2, "pivot2_value_by_source.csv", "text/csv")
        except Exception as e:
            _show_error(e, "S1 Pivot 2")

    # ── Pivot 3: Count by borrower country × JV status ───────────────────────
    with tab_p3:
        try:
            ct3 = pd.crosstab(df1["borrower country"], df1["if_joint_venture"],
                              margins=True, margins_name="Grand Total")
            pct3 = (ct3.div(ct3.loc["Grand Total"]) * 100).round(2)

            disp3 = pd.DataFrame(index=ct3.index, columns=ct3.columns)
            for r in ct3.index:
                for c in ct3.columns:
                    cnt = ct3.loc[r, c]; p = pct3.loc[r, c]
                    if r == "Grand Total":
                        disp3.loc[r, c] = f"{cnt:,}"
                    else:
                        disp3.loc[r, c] = f"{cnt:,} ({p:.2f}%)"

            st.markdown(
                _pivot_html(disp3,
                    title="Contract count by borrower country × joint-venture status",
                    idx_label="Borrower country",
                    total_label="Grand Total",
                    note="Numbers show: count (% of column total)."),
                unsafe_allow_html=True)

            _explain(
                "Each cell = number of contracts in that country that were either a joint venture "
                "(two or more contractor companies bidding together) or a single contractor. "
                "JV contracts are much rarer — most contracts are awarded to a single firm.",
                example=("Grand Total row shows: across all countries, "
                         f"Joint Venture = {fmt_count(ct3.loc['Grand Total','Joint Venture'] if 'Joint Venture' in ct3.columns else 0)}, "
                         f"Non-Joint Venture = {fmt_count(ct3.loc['Grand Total','Non-Joint Venture'] if 'Non-Joint Venture' in ct3.columns else 0)}."))

            csv3 = ct3.reset_index().to_csv(index=False).encode()
            st.download_button("⬇ Download Pivot 3 (CSV)", csv3, "pivot3_count_by_jv.csv", "text/csv")
        except Exception as e:
            _show_error(e, "S1 Pivot 3")

except Exception as e:
    _show_error(e, "Section 1")

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2 — RAW PICTURE (Chinese only)
# ═════════════════════════════════════════════════════════════════════════════
_section_header(2, "Chinese Companies — The Raw Picture",
                "Contract counts and values by MDB source, sector, and procurement method",
                scope="Chinese only")

try:
    if n_cn == 0:
        st.warning("No Chinese contracts match the current filters.")
    else:
        tab_src, tab_sec, tab_meth, tab_tbl = st.tabs(
            ["By MDB Source", "By Sector", "By Procurement Method", "Contract Data"])

        with tab_src:
            try:
                src_df = cn_by_source(df_cn)
                c1, c2 = st.columns(2)
                fig_sc = go.Figure()
                fig_sc.add_trace(go.Bar(x=src_df["data_source"].tolist(),
                    y=src_df["contracts"].tolist(), marker_color=BLUE_P, marker_line_width=0))
                _theme(fig_sc, "Chinese contracts by MDB source — number of contracts")
                fig_sc.update_layout(xaxis_title="MDB source", yaxis_title="Number of contracts")
                c1.plotly_chart(fig_sc, use_container_width=True)

                fig_sv = go.Figure()
                fig_sv.add_trace(go.Bar(x=src_df["data_source"].tolist(),
                    y=src_df["value_usd"].tolist(), marker_color=BLUE_P, marker_line_width=0))
                _theme(fig_sv, "Chinese contracts by MDB source — total value (USD)")
                fig_sv.update_layout(xaxis_title="MDB source", yaxis_title="Contract value (USD)")
                c2.plotly_chart(fig_sv, use_container_width=True)
                _missing_note()
                _explain(
                    "The left bar shows how many contracts Chinese companies won through each MDB. "
                    "The right bar shows the combined dollar value. If a bank's value bar is much taller "
                    "than its count bar (relative to others), Chinese contracts through that bank "
                    "tend to be individually larger.",
                    example=(f"IDB: {fmt_count(src_df.loc[src_df.data_source=='IDB','contracts'].sum())} contracts "
                             f"= {fmt_usd(src_df.loc[src_df.data_source=='IDB','value_usd'].sum())} total.")
                    if "IDB" in src_df["data_source"].values else "")
            except Exception as e:
                _show_error(e, "S2 Source")

        with tab_sec:
            try:
                sec_df = cn_by_sector(df_cn)
                c1, c2 = st.columns(2)
                sd_c = sec_df.sort_values("contracts")
                fig_scc = go.Figure()
                fig_scc.add_trace(go.Bar(y=sd_c["project_sector"].tolist(),
                    x=sd_c["contracts"].tolist(), orientation="h",
                    marker_color=BLUE_P, marker_line_width=0))
                fig_scc.update_layout(yaxis={"categoryorder":"total ascending"})
                _theme(fig_scc, "Chinese contracts by sector — number of contracts")
                fig_scc.update_layout(xaxis_title="Number of contracts", yaxis_title="Project sector")
                c1.plotly_chart(fig_scc, use_container_width=True)

                sd_v = sec_df.sort_values("value_usd")
                fig_scv = go.Figure()
                fig_scv.add_trace(go.Bar(y=sd_v["project_sector"].tolist(),
                    x=sd_v["value_usd"].tolist(), orientation="h",
                    marker_color=BLUE_P, marker_line_width=0))
                fig_scv.update_layout(yaxis={"categoryorder":"total ascending"})
                _theme(fig_scv, "Chinese contracts by sector — total value (USD)")
                fig_scv.update_layout(xaxis_title="Contract value (USD)", yaxis_title="Project sector")
                c2.plotly_chart(fig_scv, use_container_width=True)
                _missing_note()
                top_v = sec_df.iloc[0]; top_c = sec_df.sort_values("contracts",ascending=False).iloc[0]
                _explain(
                    "Bars are sorted largest-to-smallest. If a sector ranks much higher by value than "
                    "by count, Chinese contracts in that sector are individually very large.",
                    example=(f"By total value, {top_v['project_sector']} leads "
                             f"({fmt_usd(top_v['value_usd'])}). "
                             f"By contract count, {top_c['project_sector']} leads "
                             f"({fmt_count(top_c['contracts'])} contracts)."))
            except Exception as e:
                _show_error(e, "S2 Sector")

        with tab_meth:
            try:
                meth_df = cn_by_method(df_cn)
                colors_m = [METHOD_COLORS.get(m,"#95a5a6") for m in meth_df["procurement_method"]]
                c1, c2 = st.columns(2)
                fig_mc = go.Figure()
                fig_mc.add_trace(go.Bar(x=meth_df["procurement_method"].tolist(),
                    y=meth_df["contracts"].tolist(), marker_color=colors_m, marker_line_width=0))
                _theme(fig_mc, "Chinese contracts by procurement method — number of contracts")
                fig_mc.update_layout(xaxis_title="Procurement method", yaxis_title="Number of contracts",
                    legend_title_text="Method")
                c1.plotly_chart(fig_mc, use_container_width=True)

                fig_mv = go.Figure()
                fig_mv.add_trace(go.Bar(x=meth_df["procurement_method"].tolist(),
                    y=meth_df["value_usd"].tolist(), marker_color=colors_m, marker_line_width=0))
                _theme(fig_mv, "Chinese contracts by procurement method — total value (USD)")
                fig_mv.update_layout(xaxis_title="Procurement method", yaxis_title="Contract value (USD)")
                c2.plotly_chart(fig_mv, use_container_width=True)
                _missing_note()
                _explain(
                    "The five colour-coded buckets show how Chinese companies were selected. "
                    "'Unknown' is large because World Bank doesn't log detailed methods. "
                    "Red = Direct/Single-Source = no competitive bidding. "
                    "Green = Open/Competitive = open bidding process.",
                    example=(f"Direct/Single-Source: "
                             f"{fmt_count(meth_df.loc[meth_df.procurement_method=='Direct/Single-Source','contracts'].sum() if 'Direct/Single-Source' in meth_df.procurement_method.values else 0)} "
                             f"Chinese contracts awarded without a competitive tender."))
                jv_n = (df_cn["if_joint_venture"] == "Joint Venture").sum()
                da_n = df_cn["is_direct_award"].sum()
                m1,m2,m3 = st.columns(3)
                m1.metric("Joint ventures",        f"{jv_n} ({100*jv_n/n_cn:.1f}%)")
                m2.metric("Direct / single-source",f"{da_n} ({100*da_n/n_cn:.1f}%)")
                m3.metric("Total Chinese contracts",f"{n_cn:,}")
            except Exception as e:
                _show_error(e, "S2 Method")

        with tab_tbl:
            try:
                cols_ = ["year_awarded","borrower country","data_source","project_sector",
                         "procurement_method","contract_value_usd","if_joint_venture",
                         "contractor_country","contract_name"]
                st.dataframe(df_cn[cols_].sort_values("year_awarded",ascending=False),
                             use_container_width=True, height=440)
                st.download_button("⬇ Download Chinese contracts CSV",
                    df_cn[cols_].to_csv(index=False).encode(),
                    "chinese_contracts.csv","text/csv")
            except Exception as e:
                _show_error(e, "S2 Table")

except Exception as e:
    _show_error(e, "Section 2")

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3 — BY BORROWER COUNTRY (Chinese only)
# ═════════════════════════════════════════════════════════════════════════════
_section_header(3, "Chinese Companies — By Borrower Country",
                "Which LAC countries receive Chinese-backed contracts, and how concentrated is that?",
                scope="Chinese only")

try:
    if n_cn == 0:
        st.warning("No Chinese contracts match the current filters.")
    else:
        ctry_df = cn_by_country(df_cn)

        tab_vc, tab_avg, tab_tbl3, tab_hhi3, tab_pen = st.tabs([
            "Value & Count", "Avg & Median", "Country Table",
            "Concentration (HHI)", "Market Penetration"])

        with tab_vc:
            try:
                c1, c2 = st.columns(2)
                sd_cnt = ctry_df.sort_values("contracts")
                fig_cc = go.Figure()
                fig_cc.add_trace(go.Bar(y=sd_cnt["borrower country"].tolist(),
                    x=sd_cnt["contracts"].tolist(), orientation="h",
                    marker_color=BLUE_P, marker_line_width=0))
                fig_cc.update_layout(yaxis={"categoryorder":"total ascending"})
                _theme(fig_cc, "Chinese contracts by borrower country — count", height=max(320, len(ctry_df)*24))
                fig_cc.update_layout(xaxis_title="Number of contracts", yaxis_title="Borrower country")
                c1.plotly_chart(fig_cc, use_container_width=True)

                sd_val = ctry_df.sort_values("total_value")
                fig_cv = go.Figure()
                fig_cv.add_trace(go.Bar(y=sd_val["borrower country"].tolist(),
                    x=sd_val["total_value"].tolist(), orientation="h",
                    marker_color=BLUE_M, marker_line_width=0))
                fig_cv.update_layout(yaxis={"categoryorder":"total ascending"})
                _theme(fig_cv, "Chinese contracts by borrower country — total value (USD)", height=max(320, len(ctry_df)*24))
                fig_cv.update_layout(xaxis_title="Contract value (USD)", yaxis_title="Borrower country")
                c2.plotly_chart(fig_cv, use_container_width=True)
                _missing_note()
                top_c = ctry_df.iloc[0]; top_by_cnt = ctry_df.sort_values("contracts",ascending=False).iloc[0]
                _explain(
                    "The left bar ranks countries by how many Chinese contracts they received. "
                    "The right bar ranks by total dollar value. A country ranking higher on value "
                    "than count received fewer but larger contracts.",
                    example=(f"{top_c['borrower country']} leads by total value "
                             f"({fmt_usd(top_c['total_value'])} across {fmt_count(top_c['contracts'])} contracts). "
                             f"{top_by_cnt['borrower country']} leads by contract count "
                             f"({fmt_count(top_by_cnt['contracts'])} contracts)."))
            except Exception as e:
                _show_error(e, "S3 Value & Count")

        with tab_avg:
            try:
                c1, c2 = st.columns(2)
                sa = ctry_df.sort_values("avg_value").dropna(subset=["avg_value"])
                fig_av = go.Figure()
                fig_av.add_trace(go.Bar(y=sa["borrower country"].tolist(),
                    x=sa["avg_value"].tolist(), orientation="h",
                    marker_color=BLUE_P, marker_line_width=0))
                fig_av.update_layout(yaxis={"categoryorder":"total ascending"})
                _theme(fig_av, "Average contract value for Chinese contracts by country (USD)", height=max(320,len(ctry_df)*24))
                fig_av.update_layout(xaxis_title="Average contract value (USD)", yaxis_title="Borrower country")
                c1.plotly_chart(fig_av, use_container_width=True)

                sm = ctry_df.sort_values("median_value").dropna(subset=["median_value"])
                fig_me = go.Figure()
                fig_me.add_trace(go.Bar(y=sm["borrower country"].tolist(),
                    x=sm["median_value"].tolist(), orientation="h",
                    marker_color=BLUE_M, marker_line_width=0))
                fig_me.update_layout(yaxis={"categoryorder":"total ascending"})
                _theme(fig_me, "Median contract value for Chinese contracts by country (USD)", height=max(320,len(ctry_df)*24))
                fig_me.update_layout(xaxis_title="Median contract value (USD)", yaxis_title="Borrower country")
                c2.plotly_chart(fig_me, use_container_width=True)
                _missing_note()
                _explain(
                    "Average (left) is pulled upward by a few very large contracts. "
                    "Median (right) is the middle contract — half of contracts are worth more, "
                    "half worth less. Where average >> median, one or two giant contracts inflate the average.")
            except Exception as e:
                _show_error(e, "S3 Avg/Median")

        with tab_tbl3:
            try:
                disp = ctry_df.copy()
                disp["total_value"]  = disp["total_value"].map(fmt_usd)
                disp["avg_value"]    = disp["avg_value"].map(fmt_usd)
                disp["median_value"] = disp["median_value"].map(fmt_usd)
                disp["contracts"]    = disp["contracts"].map(fmt_count)
                disp.columns = ["Borrower country","Contracts","Total value (USD)","Avg value (USD)","Median value (USD)"]
                st.markdown(_html_table(disp, "Chinese contracts by borrower country — summary"),
                            unsafe_allow_html=True)
                _missing_note()
                st.download_button("⬇ Download country table (CSV)",
                    ctry_df.to_csv(index=False).encode(),
                    "chinese_by_country.csv","text/csv")
            except Exception as e:
                _show_error(e, "S3 Table")

        with tab_hhi3:
            try:
                by_ctry_v = df_cn.groupby("borrower country")["contract_value_usd"].sum().dropna().sort_values(ascending=False)
                by_sec_v  = df_cn.groupby("project_sector")["contract_value_usd"].sum().dropna().sort_values(ascending=False)
                h_c = hhi(by_ctry_v); lbl_c = hhi_label(h_c)
                h_s = hhi(by_sec_v);  lbl_s = hhi_label(h_s)

                # Build worked examples from real data
                total_c = by_ctry_v.sum()
                top3c   = by_ctry_v.head(3)
                ex_c_parts = [f"{c}: {v/total_c*100:.1f}% → squared {(v/total_c)**2:.4f}"
                              for c, v in top3c.items()]
                rest_c_hhi = h_c - sum((v/total_c)**2 for v in top3c.values)
                ex_c = (f"Top 3 countries: {'; '.join(ex_c_parts)}. "
                        f"Plus remaining countries contribute {rest_c_hhi:.4f}. "
                        f"Sum = {h_c:.4f} → {lbl_c}. "
                        f"This means China's geographic spending is {lbl_c.lower()} — "
                        f"no single country completely dominates its portfolio.")

                total_s = by_sec_v.sum()
                top3s   = by_sec_v.head(3)
                ex_s_parts = [f"{c}: {v/total_s*100:.1f}% → squared {(v/total_s)**2:.4f}"
                              for c, v in top3s.items()]
                rest_s_hhi = h_s - sum((v/total_s)**2 for v in top3s.values)
                ex_s = (f"Top 3 sectors: {'; '.join(ex_s_parts)}. "
                        f"Plus remaining sectors {rest_s_hhi:.4f}. "
                        f"Sum = {h_s:.4f} → {lbl_s}. "
                        f"China's sector spending is {lbl_s.lower()} — a few sectors absorb most of the value.")

                st.markdown(
                    f'<div style="background:#f0f7ff;border-left:3px solid {BLUE_M};border-radius:0 6px 6px 0;'
                    f'padding:12px 16px;margin-bottom:12px;font-size:0.82rem;color:{NAVY}">'
                    f'<strong>What is the HHI?</strong> It measures how concentrated a distribution is. '
                    f'<strong>Formula:</strong> Take every group\'s share of the total, square it, and sum all squared shares. '
                    f'Result is between 0 (perfectly spread) and 1 (all in one group).<br>'
                    f'<strong>Thresholds:</strong> &lt;0.01 = highly fragmented · 0.01–0.15 = unconcentrated · '
                    f'0.15–0.25 = moderate · &gt;0.25 = highly concentrated.</div>',
                    unsafe_allow_html=True)

                hc1, hc2 = st.columns(2)
                with hc1:
                    _hhi_box(h_c, lbl_c,
                        "Unit = borrower country | Question: 'Is China spreading its contracts across many LAC countries, or concentrating in a few?'",
                        f"<strong>Worked example (geographic):</strong> {ex_c}")
                with hc2:
                    _hhi_box(h_s, lbl_s,
                        "Unit = project sector | Question: 'Is China winning contracts across many sectors, or focused on a few?'",
                        f"<strong>Worked example (sector):</strong> {ex_s}")

                st.markdown(f"""
<div style="background:#fff;border:1px solid {GRID};border-radius:8px;padding:12px 18px;
            font-size:0.8rem;color:{NAVY};margin:10px 0">
<strong>Why two different HHIs?</strong><br>
The <strong>geographic HHI ({h_c:.4f})</strong> answers: <em>Does China spread its MDB contracts evenly
across Latin America, or cluster in a handful of countries?</em> A lower number means broader geographic reach.<br><br>
The <strong>sector HHI ({h_s:.4f})</strong> answers: <em>Does China win contracts across all project types,
or specialise in certain sectors?</em> A higher number means China's value is funnelled into fewer sectors
(primarily infrastructure and energy in this dataset). These are independent questions with different answers.
</div>
""", unsafe_allow_html=True)

                if len(by_ctry_v) > 0 and len(by_sec_v) > 0:
                    c1, c2 = st.columns(2)
                    share_c = (by_ctry_v / total_c * 100).reset_index()
                    share_c.columns = ["borrower country","share_pct"]
                    fig_pc = go.Figure(go.Pie(
                        labels=share_c["borrower country"].tolist(),
                        values=share_c["share_pct"].tolist(), hole=0.42,
                        textposition="inside", textinfo="percent+label",
                        marker=dict(colors=px.colors.sequential.Blues_r[:len(share_c)],
                                    line=dict(color="#fff",width=1.5))))
                    _theme(fig_pc, "Composition of China's contract value by borrower country", height=400)
                    fig_pc.update_layout(showlegend=False)
                    c1.plotly_chart(fig_pc, use_container_width=True)
                    c1.caption("Composition: share of China's OWN total value. Denominator = all Chinese contracts.")

                    share_s = (by_sec_v / total_s * 100).reset_index()
                    share_s.columns = ["project_sector","share_pct"]
                    fig_ps = go.Figure(go.Pie(
                        labels=share_s["project_sector"].tolist(),
                        values=share_s["share_pct"].tolist(), hole=0.42,
                        textposition="inside", textinfo="percent+label",
                        marker=dict(colors=SECTOR_PAL[:len(share_s)],
                                    line=dict(color="#fff",width=1.5))))
                    _theme(fig_ps, "Composition of China's contract value by sector", height=400)
                    fig_ps.update_layout(showlegend=False)
                    c2.plotly_chart(fig_ps, use_container_width=True)
                    c2.caption("Composition: share of China's OWN total value. Denominator = all Chinese contracts.")

                    _explain("These donuts show how China's own portfolio is divided up — "
                             "they answer 'Where does China put its contracts?' not "
                             "'How important is China in each country?' "
                             "For the latter, see the Market Penetration tab.")
            except Exception as e:
                _show_error(e, "S3 HHI")

        with tab_pen:
            try:
                # Penetration = China's share of ALL contract value/count in each borrower country
                all_by_ctry = df.groupby("borrower country", observed=True).agg(
                    all_value=("contract_value_usd","sum"),
                    all_count=("notice_id","count")).reset_index()
                cn_by_ctry  = df_cn.groupby("borrower country", observed=True).agg(
                    cn_value=("contract_value_usd","sum"),
                    cn_count=("notice_id","count")).reset_index()
                pen = all_by_ctry.merge(cn_by_ctry, on="borrower country", how="inner")
                pen["value_pct"] = (pen["cn_value"] / pen["all_value"] * 100).round(2)
                pen["count_pct"] = (pen["cn_count"] / pen["all_count"] * 100).round(2)
                pen = pen.sort_values("value_pct", ascending=False)

                c1, c2 = st.columns(2)
                fig_pv = go.Figure()
                fig_pv.add_trace(go.Bar(y=pen["borrower country"].tolist(),
                    x=pen["value_pct"].tolist(), orientation="h",
                    marker_color=BLUE_P, marker_line_width=0))
                fig_pv.update_layout(yaxis={"categoryorder":"total ascending"})
                _theme(fig_pv, "China's share of ALL contractor value in each country (%)",
                       height=max(320,len(pen)*24))
                fig_pv.update_layout(xaxis_title="China's value penetration (%)",
                                      yaxis_title="Borrower country")
                c1.plotly_chart(fig_pv, use_container_width=True)
                c1.caption("Penetration: denominator = ALL contractors in that country (not just China).")

                fig_pc2 = go.Figure()
                pen_c = pen.sort_values("count_pct", ascending=False)
                fig_pc2.add_trace(go.Bar(y=pen_c["borrower country"].tolist(),
                    x=pen_c["count_pct"].tolist(), orientation="h",
                    marker_color=BLUE_M, marker_line_width=0))
                fig_pc2.update_layout(yaxis={"categoryorder":"total ascending"})
                _theme(fig_pc2, "China's share of ALL contractor count in each country (%)",
                       height=max(320,len(pen_c)*24))
                fig_pc2.update_layout(xaxis_title="China's count penetration (%)",
                                       yaxis_title="Borrower country")
                c2.plotly_chart(fig_pc2, use_container_width=True)
                c2.caption("Penetration: denominator = ALL contracts in that country (not just Chinese ones).")

                _missing_note()
                top_pen = pen.iloc[0]
                _explain(
                    "Penetration answers 'How important are Chinese firms in each country's MDB market?' "
                    "Unlike the composition donuts (which show where China spends), "
                    "penetration shows China's share of the total market in each country. "
                    "A 5% penetration means Chinese firms won 5 cents of every dollar "
                    "of MDB-financed contracts in that country.",
                    example=(f"In {top_pen['borrower country']}, Chinese firms won "
                             f"{top_pen['value_pct']:.1f}% of all MDB contract value "
                             f"({fmt_usd(top_pen['cn_value'])} out of "
                             f"{fmt_usd(top_pen['all_value'])} awarded to all contractors)."))

                # Firm-level analysis
                st.markdown(f"<hr><div style='font-size:0.85rem;font-weight:700;color:{NAVY};margin:8px 0'>Firm-level analysis</div>", unsafe_allow_html=True)
                firm_cols = [c for c in df_cn.columns if "supplier" in c.lower() or "firm" in c.lower()
                             or ("contractor" in c.lower() and "country" not in c.lower() and "label" not in c.lower() and "type" not in c.lower() and "group" not in c.lower())]
                firm_cols = [c for c in firm_cols if c not in ("contractor_country","number_of_contractor","number_of_contractor_country","contractor_country_type","contractor_country_group","contractor_label","is_direct_award","is_single_bidder")]
                if firm_cols:
                    pass  # would add firm-level analysis here
                else:
                    st.info(
                        "**No firm-name column found in the data.** "
                        "The dataset identifies contractors by *country* (`contractor_country`) "
                        "and the contract itself (`contract_name`), but does not include a "
                        "supplier or company name field. Firm-level analysis (which Chinese company "
                        "wins the most contracts, or the highest value) requires a "
                        "supplier/contractor name column not present in this dataset.")
            except Exception as e:
                _show_error(e, "S3 Penetration")

except Exception as e:
    _show_error(e, "Section 3")

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4 — BY YEAR AND BY COUNTRY (Chinese only)
# ═════════════════════════════════════════════════════════════════════════════
_section_header(4, "Chinese Companies — By Year and by Country",
                "How has Chinese participation changed over time? Is geographic reach growing?",
                scope="Chinese only")

try:
    if n_cn == 0:
        st.warning("No Chinese contracts match the current filters.")
    else:
        yr_df = cn_by_year(df_cn)
        if len(yr_df) == 0:
            st.warning("No year-level data for the current filters.")
        else:
            tab_trend, tab_heat, tab_conc = st.tabs(
                ["Annual Trends", "Year × Country Heatmap", "Concentration over Time"])

            with tab_trend:
                try:
                    all_yrs = yr_df["year_awarded"].tolist()
                    c1, c2 = st.columns(2)

                    fig_cnt = go.Figure()
                    fig_cnt.add_trace(go.Scatter(
                        x=all_yrs, y=yr_df["contracts"].tolist(),
                        mode="lines+markers", line=dict(color=BLUE_P, width=2.5),
                        fill="tozeroy", fillcolor=CHINA_FILL, marker=dict(size=6, color=BLUE_P)))
                    _theme(fig_cnt, "Chinese contracts per year — number of contracts")
                    fig_cnt.update_layout(
                        xaxis=dict(title="Year", tickmode="array", tickvals=all_yrs,
                                   tickangle=-45, tickfont=dict(size=10)),
                        yaxis_title="Number of contracts")
                    c1.plotly_chart(fig_cnt, use_container_width=True)

                    fig_val = go.Figure()
                    fig_val.add_trace(go.Scatter(
                        x=all_yrs, y=yr_df["total_value"].tolist(),
                        mode="lines+markers", line=dict(color=BLUE_P, width=2.5),
                        fill="tozeroy", fillcolor=CHINA_FILL, marker=dict(size=6, color=BLUE_P)))
                    _theme(fig_val, "Chinese contracts per year — total value (USD)")
                    fig_val.update_layout(
                        xaxis=dict(title="Year", tickmode="array", tickvals=all_yrs,
                                   tickangle=-45, tickfont=dict(size=10)),
                        yaxis_title="Contract value (USD)")
                    c2.plotly_chart(fig_val, use_container_width=True)
                    _missing_note()
                    _explain(
                        "Left: how many contracts Chinese companies won each year. "
                        "Right: the combined dollar value per year. "
                        "A year with few contracts but a tall value bar "
                        "means a small number of very large projects dominated that year.")

                    avg_df = yr_df.dropna(subset=["avg_value"])
                    if len(avg_df) > 0:
                        fig_avg = go.Figure()
                        fig_avg.add_trace(go.Bar(
                            x=avg_df["year_awarded"].tolist(), y=avg_df["avg_value"].tolist(),
                            marker_color=BLUE_M, marker_line_width=0))
                        _theme(fig_avg, "Average value per Chinese contract by year (USD)", height=310)
                        fig_avg.update_layout(
                            xaxis=dict(title="Year", tickmode="array",
                                       tickvals=avg_df["year_awarded"].tolist(),
                                       tickangle=-45, tickfont=dict(size=10)),
                            yaxis_title="Average contract value (USD)")
                        st.plotly_chart(fig_avg, use_container_width=True)
                        _missing_note()
                        _explain(
                            "This shows the typical size of a Chinese contract in each year. "
                            "Tall bars = a few very large contracts dominated. "
                            "Short bars = many smaller contracts. Cross-check with the count chart "
                            "to understand whether China's contracts are getting larger or smaller over time.")
                except Exception as e:
                    _show_error(e, "S4 Trends")

            with tab_heat:
                try:
                    metric_choice = st.radio("Metric:", ["Total value (USD)", "Contract count"], horizontal=True)
                    metric_key = "total_value" if "value" in metric_choice else "contracts"
                    piv = cn_country_year_pivot(df_cn, metric_key)
                    if piv.empty:
                        st.info("No data.")
                    else:
                        country_order = piv.sum(axis=0).sort_values(ascending=False).index.tolist()
                        piv_s = piv[country_order]
                        z_data = piv_s.values.T
                        hover  = z_data.copy()

                        if metric_key == "total_value":
                            z_plot = np.where(z_data > 0, np.log10(z_data + 1), 0)
                            cs = [[0,"#f4f7fb"],[0.2,BLUE_PALE],[0.5,BLUE_L],[0.75,BLUE_M],[1.0,NAVY]]
                            cb_t = "log₁₀(USD+1)"
                            htmpl = "Year: %{x}<br>Country: %{y}<br>Value (USD): $%{customdata:,.0f}<extra></extra>"
                        else:
                            z_plot = z_data.astype(float)
                            cs = [[0,"#f4f7fb"],[0.3,BLUE_PALE],[0.6,BLUE_L],[0.85,BLUE_M],[1.0,BLUE_P]]
                            cb_t = "Number of contracts"
                            htmpl = "Year: %{x}<br>Country: %{y}<br>Contracts: %{customdata}<extra></extra>"

                        fig_h = go.Figure(go.Heatmap(
                            z=z_plot, x=piv_s.index.tolist(), y=country_order,
                            colorscale=cs,
                            colorbar=dict(title=dict(text=cb_t, side="right"), thickness=14),
                            hovertemplate=htmpl, customdata=hover,
                            xgap=1, ygap=1))
                        _theme(fig_h, f"Chinese contracts — year × borrower country ({metric_choice})",
                               height=max(420, len(country_order)*22+120))
                        fig_h.update_layout(
                            xaxis=dict(title="Year", tickmode="array",
                                       tickvals=piv_s.index.tolist(),
                                       tickangle=-45, tickfont=dict(size=10)),
                            yaxis_title="Borrower country",
                            plot_bgcolor="#f4f7fb")
                        st.plotly_chart(fig_h, use_container_width=True)
                        _missing_note()
                        _explain(
                            "Each cell = one year (column) × one borrower country (row). "
                            "Darker blue = more activity. White = no Chinese contracts that year. "
                            "Countries are sorted top-to-bottom by total Chinese contract value. "
                            "Thin gridlines separate every cell. Value is on a log scale — hover for the exact figure.",
                            example="A sustained dark column across many rows = China was active in many countries that year. A single isolated dark cell = one large project in an otherwise inactive relationship.")
                except Exception as e:
                    _show_error(e, "S4 Heatmap")

            with tab_conc:
                try:
                    hhi_yr = cn_hhi_by_year(df_cn)
                    if hhi_yr.empty or hhi_yr["hhi"].isna().all():
                        st.info("Not enough data to compute HHI per year.")
                    else:
                        fig_hhi = go.Figure()
                        fig_hhi.add_trace(go.Scatter(
                            x=hhi_yr["year_awarded"].tolist(), y=hhi_yr["hhi"].tolist(),
                            mode="lines+markers",
                            line=dict(color=BLUE_P, width=2.5), marker=dict(size=7, color=BLUE_P),
                            name="HHI (by borrower country)"))
                        for thresh, ann, col in [
                            (0.01,"< 0.01 — highly fragmented",MUTED),
                            (0.15,"0.15 — moderate",          "#e8a020"),
                            (0.25,"0.25 — highly concentrated",RED)]:
                            fig_hhi.add_hline(y=thresh, line_dash="dot", line_color=col,
                                              annotation_text=ann,
                                              annotation_font=dict(color=col, size=10),
                                              annotation_position="right")
                        _theme(fig_hhi, "China's geographic concentration (HHI by borrower country) per year",
                               height=400)
                        fig_hhi.update_layout(
                            xaxis=dict(title="Year", tickmode="array",
                                       tickvals=hhi_yr["year_awarded"].tolist(),
                                       tickangle=-45, tickfont=dict(size=10)),
                            yaxis=dict(title="HHI (0 = spread across all countries, 1 = all in one country)",
                                       range=[0,1.05]),
                            legend=dict(title_text="Metric"))
                        st.plotly_chart(fig_hhi, use_container_width=True)

                        # Compute worked example for one year
                        yr_example = ""
                        good = hhi_yr[hhi_yr["hhi"].between(0.1, 0.9)].sort_values("year_awarded", ascending=False)
                        if len(good) > 0:
                            ex_row = good.iloc[0]
                            ex_yr  = int(ex_row["year_awarded"])
                            ex_h   = ex_row["hhi"]
                            yr_data = df_cn[df_cn["year_awarded"] == ex_yr]
                            yr_ctry = (yr_data.groupby("borrower country")["contract_value_usd"]
                                       .sum().dropna().sort_values(ascending=False))
                            yr_total = yr_ctry.sum()
                            n_c = len(yr_ctry)
                            if yr_total > 0 and n_c > 0:
                                parts = [f"{c}: {v/yr_total*100:.0f}% (sq={( v/yr_total)**2:.3f})"
                                         for c, v in yr_ctry.head(min(3,n_c)).items()]
                                rest = ex_h - sum((v/yr_total)**2 for v in yr_ctry.head(min(3,n_c)).values)
                                yr_example = (f"In {ex_yr}, China had contracts in {n_c} borrower "
                                              f"{'country' if n_c==1 else 'countries'}. "
                                              f"{'; '.join(parts)}"
                                              f"{f'; remaining {rest:.3f}' if n_c > 3 else ''}. "
                                              f"Sum = {ex_h:.3f} → {hhi_label(ex_h)}.")

                        _explain(
                            "This line measures, for each year, how concentrated China's contracts "
                            "were across borrower countries. "
                            "A high point means most of China's value in that year went to just one or "
                            "two countries. A low point means China had active contracts in many countries "
                            "that year. Early years often spike near 1.0 because China had only one "
                            "or two contracts total.",
                            example=yr_example)

                        st.markdown(
                            f'<div style="background:#fff;border:1px solid {GRID};border-radius:8px;'
                            f'padding:12px 18px;font-size:0.8rem;color:{NAVY};margin:10px 0">'
                            f'<strong>How the per-year HHI is calculated:</strong> '
                            f'For each year, look only at Chinese contracts in that year. '
                            f'For each borrower country, divide its value by the year\'s total Chinese value '
                            f'to get a share. Square each share. Sum all squared shares = that year\'s HHI. '
                            f'A year where one country gets 100% → HHI = 1.0²  = 1.0 (fully concentrated). '
                            f'A year where two countries split 50/50 → 0.5²+0.5² = 0.50 (moderately concentrated). '
                            f'A year where 10 countries split evenly → 10×(0.1)² = 0.10 (unconcentrated).</div>',
                            unsafe_allow_html=True)
                except Exception as e:
                    _show_error(e, "S4 Concentration")

except Exception as e:
    _show_error(e, "Section 4")

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5 — COMPARISON METRICS (full dataset, all contractors)
# ═════════════════════════════════════════════════════════════════════════════
_section_header(5, "Comparison Metrics — China vs All Other Contractor Countries",
                "China (dark blue, highlighted) compared to top-N comparators from the full dataset",
                scope="Full dataset")

try:
    if len(df4) == 0:
        st.warning("No data for comparison.")
    else:
        top_labels = top_n_labels(df4, top_n)
        all_labels = ["China"] + top_labels
        cmap = colour_map(all_labels + ["Rest"])

        (tab_sh, tab_st, tab_dist, tab_rank,
         tab_sec5, tab_gr, tab_proc, tab_mhhi) = st.tabs([
            "Market Share", "Value Stack", "Value Distribution",
            "Rank Trajectory", "Sector Mix",
            "Growth", "Procurement", "Market HHI"])

        with tab_sh:
            try:
                sh = annual_share(df4)
                fig_sh = go.Figure()
                fig_sh.add_trace(go.Scatter(
                    x=sh["year_awarded"].tolist(), y=sh["value_share_pct"].tolist(),
                    name="Value share (%)", line=dict(color=BLUE_P, width=3),
                    mode="lines+markers", marker=dict(size=6, color=BLUE_P)))
                fig_sh.add_trace(go.Scatter(
                    x=sh["year_awarded"].tolist(), y=sh["count_share_pct"].tolist(),
                    name="Count share (%)", line=dict(color=BLUE_L, width=2, dash="dash"),
                    mode="lines+markers", marker=dict(size=5, color=BLUE_L)))
                _theme(fig_sh, "China's share of total MDB contract value and count per year (%)", height=400)
                fig_sh.update_layout(
                    xaxis=dict(title="Year", tickmode="array",
                               tickvals=sh["year_awarded"].tolist(),
                               tickangle=-45, tickfont=dict(size=10)),
                    yaxis_title="Share of all contractor activity (%)",
                    legend=dict(title_text="Metric"))
                st.plotly_chart(fig_sh, use_container_width=True)
                _missing_note()
                _explain(
                    "The solid blue line shows China's share of ALL MDB contract value each year "
                    "(denominator = every contractor in the full dataset). "
                    "The dashed line shows China's share of contract count. "
                    "When value share > count share, Chinese contracts are larger than the market average.")

                prem = sh.dropna(subset=["premium_ratio"])
                if len(prem) > 0:
                    fig_pr = go.Figure()
                    fig_pr.add_trace(go.Bar(
                        x=prem["year_awarded"].tolist(), y=prem["premium_ratio"].tolist(),
                        marker_color=[GREEN if v>=1 else RED for v in prem["premium_ratio"]],
                        marker_line_width=0))
                    fig_pr.add_hline(y=1, line_dash="dot", line_color=MUTED,
                                     annotation_text="1.0 = market average contract size",
                                     annotation_font=dict(color=MUTED,size=11))
                    _theme(fig_pr, "China's contract-size premium ratio (value share ÷ count share)", height=310)
                    fig_pr.update_layout(
                        xaxis=dict(title="Year", tickmode="array",
                                   tickvals=prem["year_awarded"].tolist(),
                                   tickangle=-45, tickfont=dict(size=10)),
                        yaxis_title="Size premium ratio")
                    st.plotly_chart(fig_pr, use_container_width=True)
                    _explain(
                        "This ratio = China's value share ÷ China's count share each year. "
                        "A ratio of 2.0 means Chinese contracts were, on average, twice the market "
                        "average contract size that year. Bars above 1.0 (green) = large-than-average; "
                        "below 1.0 (red) = smaller-than-average.")
            except Exception as e:
                _show_error(e, "S5 Market Share")

        with tab_st:
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
                        line=dict(color=cmap.get(lbl,REST_COLOR), width=2 if lbl=="China" else 1),
                        fillcolor=cmap.get(lbl,REST_COLOR)))
                _theme(fig_st, f"Annual contract value: China / top-{top_n} / Rest (USD) — full dataset",
                       height=440)
                fig_st.update_layout(
                    xaxis=dict(title="Year", tickmode="array",
                               tickvals=sa["year_awarded"].tolist(),
                               tickangle=-45, tickfont=dict(size=10)),
                    yaxis_title="Total contract value (USD)",
                    legend=dict(title_text="Contractor country"))
                st.plotly_chart(fig_st, use_container_width=True)
                _missing_note()
                _explain(
                    "Total height of the stack = all MDB contract value that year (all contractors). "
                    "China (darkest blue, at the bottom) is easiest to read off the y-axis directly. "
                    "Watch whether China's band is growing, stable, or shrinking relative to "
                    "the total and to named comparators.")
            except Exception as e:
                _show_error(e, "S5 Stack")

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
                            boxpoints="outliers"))
                    _theme(fig_box, "Contract value distribution by contractor country — log scale", height=440)
                    fig_box.update_layout(
                        yaxis=dict(type="log", title="Contract value (USD, log scale)"),
                        xaxis_title="Contractor country",
                        legend=dict(title_text="Contractor"))
                    st.plotly_chart(fig_box, use_container_width=True)
                    _missing_note()
                    _explain(
                        "Each box = one contractor country. "
                        "Box spans the middle 50% of contract values; line inside = median. "
                        "Dots above whiskers = unusually large contracts (outliers). "
                        "Log scale: each gridline = 10× the one below.")
            except Exception as e:
                _show_error(e, "S5 Distribution")

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
                        name="China's rank (1 = highest value)"))
                    _theme(fig_rk, "China's rank among all contractor countries by annual contract value",
                           height=420)
                    fig_rk.update_layout(
                        xaxis=dict(title="Year", tickmode="array",
                                   tickvals=rnk["year_awarded"].tolist(),
                                   tickangle=-45, tickfont=dict(size=10)),
                        yaxis=dict(autorange="reversed",
                                   title="Rank (1 = largest total value that year)",
                                   gridcolor=GRID),
                        legend=dict(title_text="Series"))
                    st.plotly_chart(fig_rk, use_container_width=True)

                    # Worked example from real data
                    best_ex = rnk[rnk["rank"] > 1].sort_values("year_awarded", ascending=False)
                    ex_row = best_ex.iloc[0] if len(best_ex) > 0 else rnk.iloc[-1]
                    ex_yr   = int(ex_row["year_awarded"])
                    ex_rank = int(ex_row["rank"])
                    ex_val  = ex_row["china_value"]
                    n_ctrs_yr = df4[df4["year_awarded"] == ex_yr]["contractor_label"].nunique()
                    _explain(
                        "Each year, every contractor country is ranked by its total contract value "
                        "(rank 1 = most value). The y-axis is inverted: moving up the chart means "
                        "a better (lower) rank. Missing years = no Chinese contracts that year.",
                        example=(f"In {ex_yr}, China ranked #{ex_rank} of {n_ctrs_yr} contractor countries "
                                 f"by total value ({fmt_usd(ex_val)}), meaning {ex_rank-1} "
                                 f"{'country' if ex_rank-1==1 else 'countries'} won more total "
                                 f"contract value that year."))

                    rnk_d = rnk.copy()
                    rnk_d["china_value"] = rnk_d["china_value"].map(fmt_usd)
                    rnk_d["rank"] = rnk_d["rank"].astype(int)
                    rnk_d.columns = ["Year","China's rank","China total value (USD)"]
                    st.markdown(_html_table(rnk_d, "China's rank by year"), unsafe_allow_html=True)
            except Exception as e:
                _show_error(e, "S5 Rank")

        with tab_sec5:
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
                        labels={"share_pct":"Share of contracts (%)","contractor_label":"Contractor country",
                                "sector":"Project sector"},
                        color_discrete_sequence=SECTOR_PAL,
                        category_orders={"contractor_label":lo})
                    _theme(fig_sm, "Sector mix by contractor — each row = 100% of that country's contracts",
                           height=max(300,len(all_labels)*44))
                    fig_sm.update_layout(barmode="stack", xaxis_range=[0,100],
                        xaxis_title="Share of contracts (%)",
                        yaxis_title="Contractor country",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0,
                                    title_text="Sector"))
                    st.plotly_chart(fig_sm, use_container_width=True)
                    _explain(
                        "Each row = 100% of that contractor country's contracts. "
                        "The coloured segments show what fraction went to each project sector. "
                        "Compare China's row to others — a very different colour pattern suggests "
                        "China specialises in different project types than other contractor countries.")
            except Exception as e:
                _show_error(e, "S5 Sector Mix")

        with tab_gr:
            try:
                cagr_df = cagr(df4, all_labels)
                yoy_df  = yoy_growth(df4, all_labels)

                # CAGR worked example
                china_cagr_row = cagr_df[cagr_df["contractor_label"]=="China"]
                cagr_example = ""
                if len(china_cagr_row) > 0:
                    cr = china_cagr_row.iloc[0]
                    if not np.isnan(cr["cagr_pct"]) and cr["start_year"] and cr["end_year"]:
                        sy, ey = int(cr["start_year"]), int(cr["end_year"])
                        # look up actual values
                        yr_lookup = yr_df.set_index("year_awarded")["total_value"]
                        sv = yr_lookup.get(sy, np.nan); ev = yr_lookup.get(ey, np.nan)
                        cagr_example = (f"China's CAGR = {cr['cagr_pct']:.1f}%. "
                                        f"Chinese contract value went from {fmt_usd(sv)} in {sy} "
                                        f"to {fmt_usd(ev)} in {ey} ({ey-sy} years), "
                                        f"an average annual change of {cr['cagr_pct']:+.1f}% per year. "
                                        f"A positive CAGR means value grew on average; negative means it shrank.")

                cp = cagr_df.dropna(subset=["cagr_pct"])
                if len(cp) > 0:
                    cp_s = cp.sort_values("cagr_pct")
                    fig_cg = go.Figure()
                    fig_cg.add_trace(go.Bar(
                        y=cp_s["contractor_label"].tolist(), x=cp_s["cagr_pct"].tolist(),
                        orientation="h", marker_line_width=0,
                        marker_color=[BLUE_P if v>=0 else RED for v in cp_s["cagr_pct"]]))
                    fig_cg.add_vline(x=0, line_dash="dot", line_color=MUTED)
                    fig_cg.update_layout(yaxis={"categoryorder":"total ascending"})
                    _theme(fig_cg, "Compound Annual Growth Rate (CAGR) of contract value — full period (%)",
                           height=340)
                    fig_cg.update_layout(xaxis_title="CAGR (%)", yaxis_title="Contractor country")
                    st.plotly_chart(fig_cg, use_container_width=True)
                    _missing_note()
                    _explain(
                        "CAGR = average yearly growth rate over the full available period. "
                        "Blue bars (right of 0) = value grew on average each year. "
                        "Red bars (left of 0) = value shrank on average.",
                        example=cagr_example)

                fig_yoy = go.Figure()
                for lbl in all_labels:
                    if lbl not in yoy_df.columns: continue
                    fig_yoy.add_trace(go.Scatter(
                        x=yoy_df["year_awarded"].tolist(), y=yoy_df[lbl].tolist(),
                        name=lbl,
                        line=dict(color=cmap.get(lbl,REST_COLOR),
                                  width=3 if lbl=="China" else 1.5,
                                  dash="solid" if lbl=="China" else "dot"),
                        mode="lines", opacity=1.0 if lbl=="China" else 0.55))
                fig_yoy.add_hline(y=0, line_color=GRID, line_width=1)
                _theme(fig_yoy, "Year-over-year (YoY) growth of contract value (%) — China solid blue",
                       height=400)
                fig_yoy.update_layout(
                    xaxis=dict(title="Year", tickmode="array",
                               tickvals=yoy_df["year_awarded"].tolist(),
                               tickangle=-45, tickfont=dict(size=10)),
                    yaxis_title="Year-over-year change in contract value (%)",
                    legend=dict(title_text="Contractor country"))
                st.plotly_chart(fig_yoy, use_container_width=True)
                _explain(
                    "Each point = how much that contractor country's contract value changed "
                    "vs the prior year. China (solid, fully opaque) is the focal series. "
                    "Extreme swings for China often reflect one large project entering or leaving — "
                    "always cross-check with the count and value trend charts in Section 4.")
            except Exception as e:
                _show_error(e, "S5 Growth")

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
                        color="method", orientation="h", color_discrete_map=mc,
                        labels={"share_pct":"Share of contracts (%)","contractor_label":"Contractor country",
                                "method":"Procurement method"},
                        category_orders={"contractor_label":lo})
                    _theme(fig_pm,
                           "Procurement method mix per contractor — % of contracts (IDB/CDB meaningful; WB = Unknown)",
                           height=max(300,len(all_labels)*44))
                    fig_pm.update_layout(barmode="stack", xaxis_range=[0,100],
                        xaxis_title="Share of contracts (%)", yaxis_title="Contractor country",
                        legend=dict(title_text="Method"))
                    st.plotly_chart(fig_pm, use_container_width=True)
                    _explain(
                        "Each row = 100% of that contractor's contracts. "
                        "Grey ('Unknown') is large because World Bank doesn't record detailed methods. "
                        "Green = Open/Competitive (most transparent). Red = Direct/Single-Source (no competition). "
                        "Focus on IDB/CDB rows for meaningful comparisons.")

                bl = proc[["contractor_label","direct_award_pct","jv_pct"]].copy()
                bl_l = bl.melt(id_vars="contractor_label", var_name="indicator", value_name="pct")
                bl_l["indicator"] = bl_l["indicator"].map({
                    "direct_award_pct":"Direct/single-source (%)",
                    "jv_pct":          "Joint venture (%)"})
                fig_bl = px.bar(bl_l, x="contractor_label", y="pct", color="indicator",
                    barmode="group",
                    labels={"pct":"Share of contracts (%)","contractor_label":"Contractor country",
                            "indicator":"Indicator"},
                    category_orders={"contractor_label":["China"]+[l for l in all_labels if l!="China"]},
                    color_discrete_map={"Direct/single-source (%)":RED,"Joint venture (%)":BLUE_M})
                _theme(fig_bl, "Direct-award share and joint-venture share by contractor (%)", height=360)
                fig_bl.update_layout(
                    xaxis_title="Contractor country", yaxis_title="Share of contracts (%)",
                    legend=dict(title_text="Indicator"))
                st.plotly_chart(fig_bl, use_container_width=True)
                _explain(
                    "For each contractor country: red = share of contracts awarded directly without "
                    "competitive bidding; blue = share that are joint ventures.")
            except Exception as e:
                _show_error(e, "S5 Procurement")

        with tab_mhhi:
            try:
                hhi_sec, hhi_ctr = market_hhi(df4)
                c1, c2 = st.columns(2)

                fig_hs = go.Figure()
                fig_hs.add_trace(go.Bar(x=hhi_sec["project_sector"].tolist(),
                    y=hhi_sec["hhi"].tolist(), name="Market HHI",
                    marker_color=BLUE_M, marker_line_width=0))
                fig_hs.add_trace(go.Scatter(
                    x=hhi_sec["project_sector"].tolist(),
                    y=hhi_sec["china_share_pct"].tolist(),
                    name="China's share (%)", yaxis="y2", mode="markers+lines",
                    marker=dict(color=BLUE_P, size=10, symbol="diamond"),
                    line=dict(color=BLUE_P, dash="dot", width=2)))
                _theme(fig_hs, "Market HHI and China's share by project sector", height=390, secondary_y=True)
                fig_hs.update_layout(
                    xaxis=dict(title="Project sector", tickangle=-30),
                    yaxis_title="Market HHI (0–1)",
                    yaxis2=dict(title="China's share of sector value (%)",
                                overlaying="y", side="right", range=[0,100],
                                gridcolor=GRID, tickfont=dict(color=MUTED,size=11)),
                    legend=dict(title_text="Series"))
                c1.plotly_chart(fig_hs, use_container_width=True)

                fig_hc = go.Figure()
                fig_hc.add_trace(go.Bar(x=hhi_ctr["borrower country"].tolist(),
                    y=hhi_ctr["hhi"].tolist(), name="Market HHI",
                    marker_color=BLUE_M, marker_line_width=0))
                fig_hc.add_trace(go.Scatter(
                    x=hhi_ctr["borrower country"].tolist(),
                    y=hhi_ctr["china_share_pct"].tolist(),
                    name="China's share (%)", yaxis="y2", mode="markers+lines",
                    marker=dict(color=BLUE_P, size=8, symbol="diamond"),
                    line=dict(color=BLUE_P, dash="dot", width=2)))
                _theme(fig_hc, "Market HHI and China's share by borrower country", height=420, secondary_y=True)
                fig_hc.update_layout(
                    xaxis=dict(title="Borrower country", tickangle=-45),
                    yaxis_title="Market HHI (0–1)",
                    yaxis2=dict(title="China's share of country value (%)",
                                overlaying="y", side="right", range=[0,100],
                                gridcolor=GRID, tickfont=dict(color=MUTED,size=11)),
                    legend=dict(title_text="Series"))
                c2.plotly_chart(fig_hc, use_container_width=True)
                _missing_note()

                # Worked example from real data
                top_sec = hhi_sec.sort_values("total_value", ascending=False).iloc[0] if len(hhi_sec)>0 else None
                mhhi_example = ""
                if top_sec is not None:
                    mhhi_example = (f"In {top_sec['project_sector']}, the market HHI is "
                                    f"{top_sec['hhi']:.3f} ({top_sec['hhi_label']}), meaning the market "
                                    f"across all contractor countries is {top_sec['hhi_label'].lower()}. "
                                    f"Chinese contractors hold {top_sec['china_share_pct']:.1f}% of "
                                    f"that sector's total contract value ({fmt_usd(top_sec['total_value'])} "
                                    f"across all contractors).")

                _explain(
                    "Blue bars (left axis) = market HHI for each sector/country — "
                    "how concentrated the overall contractor market is "
                    "(unit = contractor country: are contracts dominated by one nationality?). "
                    "Blue diamonds (right axis) = China's specific share of value. "
                    "High HHI + High China share → China dominates. "
                    "High HHI + Low China share → Someone else dominates. "
                    "Low HHI + High China share → Competitive market where China is still significant.",
                    example=mhhi_example)

                # Detailed tables
                hd_sec = hhi_sec[["project_sector","hhi","hhi_label","china_share_pct",
                                   "n_contractors","total_value"]].copy()
                hd_sec["total_value"] = hd_sec["total_value"].map(fmt_usd)
                hd_sec["hhi"] = hd_sec["hhi"].round(4)
                hd_sec["china_share_pct"] = hd_sec["china_share_pct"].round(2)
                hd_sec.columns = ["Sector","HHI","Interpretation","China share (%)","# contractor countries","Total value (all contractors)"]
                st.markdown(_html_table(hd_sec,"Market concentration by sector"),unsafe_allow_html=True)

                hd_ctr = hhi_ctr[["borrower country","hhi","hhi_label","china_share_pct",
                                   "n_contractors","total_value"]].copy()
                hd_ctr["total_value"] = hd_ctr["total_value"].map(fmt_usd)
                hd_ctr["hhi"] = hd_ctr["hhi"].round(4)
                hd_ctr["china_share_pct"] = hd_ctr["china_share_pct"].round(2)
                hd_ctr.columns = ["Country","HHI","Interpretation","China share (%)","# contractor countries","Total value (all contractors)"]
                st.markdown(_html_table(hd_ctr,"Market concentration by borrower country"),unsafe_allow_html=True)
                st.caption("'# contractor countries' = the number of distinct contractor nationalities active in that sector/country — this is the unit over which the HHI is computed.")
            except Exception as e:
                _show_error(e, "S5 Market HHI")

except Exception as e:
    _show_error(e, "Section 5")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="background:{NAVY};border-radius:8px;padding:14px 22px;margin-top:28px;
            color:{BLUE_L};font-size:0.75rem;text-align:center">
  World Bank Capstone · <em>Chinese Companies' Participation in MDB Public Procurement
  in Latin America</em> · IDB · World Bank · CDB · 2000–2026 · Descriptive analysis
</div><div style="height:20px"></div>
""", unsafe_allow_html=True)
