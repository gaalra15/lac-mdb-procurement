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
ORANGE_LINE = "#e8702a"   # high-contrast overlay line — never blue-on-blue
# Colorblind-friendly categorical palette for non-China comparators
COMP_COLORS = ["#e67e22","#27ae60","#8e44ad","#1abc9c","#e74c3c","#f39c12","#d35400","#2c3e50",
               "#16a085","#c0392b","#7f8c8d","#2980b9"]
SECTOR_PAL  = [BLUE_P,"#1a9e5f","#e67e22","#e8a020","#9b59b6",RED,"#1abc9c","#20a0a0","#f39c12","#8e44ad"]
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
            cmap[lbl] = COMP_COLORS[ci % len(COMP_COLORS)]; ci += 1
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
**Data source:** `worldbank_idb_cdb_merged_0620.xlsx` — IDB · World Bank · CDB · **237,651 total** · 2000–2026.

**Chinese companies** = `contractor_country_unique == "China"` (exact match on the clean canonical field). **Hong Kong SAR** ("Hong Kong SAR, China") is tracked separately and never folded into China.
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
                "Each box in this grid = how many contracts that country (the row) got from that bank (the column). "
                "The % shows how big each country's slice is out of that bank's total contracts. "
                "The Grand Total row at the bottom adds up the full count for each bank.",
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
                "Same grid as Pivot 1, but in dollars instead of counts. "
                "The Grand Total column on the right shows each country's share of all the money across every bank combined.",
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
                "Each box shows how many contracts in that country went to a single company vs. a team bid "
                "(joint venture = two or more companies bidding together as partners). "
                "Joint ventures are rare — most contracts go to one company alone.",
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
                    "Two bars per bank: left = number of contracts Chinese firms won, right = total dollars. "
                    "If a bank's dollar bar is way taller than its count bar (compared to the other banks), "
                    "that bank tends to hand out bigger contracts to Chinese firms.",
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
                    "Left = how many contracts. Right = total dollars. "
                    "A sector ranked higher on the right chart than the left means each contract there is individually large — "
                    "think one big dam worth more than ten road repairs.",
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
                    "This shows HOW Chinese firms were selected. "
                    "Open/Competitive (green) = an open bidding race was held and China won. "
                    "Direct/Single-Source (red) = the contract went to China with no competition. "
                    "The big grey Unknown chunk is because the World Bank doesn't record which method was used for most contracts.",
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
                    "Left = how many Chinese contracts each country received. Right = total dollars. "
                    "A country sitting higher on the right chart than the left gets fewer Chinese contracts "
                    "but each one is worth a lot more money.",
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
                    "Average (left) = total dollars ÷ number of contracts. "
                    "One single giant contract can make this number look huge. "
                    "Median (right) = the middle contract — half are bigger, half are smaller, so outliers don't skew it. "
                    "Where average >> median, one or two mega-projects are pulling the number way up.")
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

                    _explain("These donuts answer 'Where does China put its own contracts?' "
                             "A big slice for one country means China sends a lot of its work there. "
                             "This is NOT the same as asking how dominant China is in each country — "
                             "for that, go to the Market Penetration tab.")
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
                    "Penetration asks: out of ALL contracts in this country (Chinese AND everyone else), "
                    "what fraction went to Chinese firms? "
                    "A 5% penetration = Chinese firms won 5 cents of every MDB dollar in that country. "
                    "The bigger the bar, the more dominant Chinese firms are in that country's market.",
                    example=(f"In {top_pen['borrower country']}, Chinese firms won "
                             f"{top_pen['value_pct']:.1f}% of all MDB contract value "
                             f"({fmt_usd(top_pen['cn_value'])} out of "
                             f"{fmt_usd(top_pen['all_value'])} awarded to all contractors)."))

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
                        "Right: total dollars those contracts were worth. "
                        "A year with a short count bar but a tall dollar bar = just a few very big projects came through that year.")

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
                            "This shows the typical size of one Chinese contract in each year. "
                            "Tall bar = the average project was huge that year. "
                            "Short bar = lots of smaller projects. "
                            "Cross-check with the count chart: if both are falling, China is winning fewer AND smaller contracts.")
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
                            "Rows = countries, columns = years. "
                            "Darker blue = more Chinese contract money that year. White = no Chinese contracts. "
                            "Countries are sorted top to bottom by total value over all years, so the busiest ones sit at the top. "
                            "Hover any cell to see the exact amount.",
                            example="A dark column across many rows = China was active in many countries that year. A single dark cell surrounded by white = one big project in an otherwise quiet relationship.")
                except Exception as e:
                    _show_error(e, "S4 Heatmap")

            with tab_conc:
                try:
                    hhi_yr = cn_hhi_by_year(df_cn)
                    if hhi_yr.empty or hhi_yr["hhi"].isna().all():
                        st.info("Not enough data to compute HHI per year.")
                    else:
                        all_yrs_hhi = hhi_yr["year_awarded"].tolist()

                        fig_hhi = go.Figure()
                        fig_hhi.add_trace(go.Scatter(
                            x=all_yrs_hhi, y=hhi_yr["hhi"].tolist(),
                            mode="lines+markers",
                            line=dict(color=ORANGE_LINE, width=3),
                            marker=dict(size=8, color=ORANGE_LINE),
                            name="HHI per year"))
                        _theme(fig_hhi, "China's geographic concentration per year — HHI score (0–1)",
                               height=400)
                        fig_hhi.update_layout(
                            xaxis=dict(title="Year", tickmode="array",
                                       tickvals=all_yrs_hhi,
                                       tickangle=-45, tickfont=dict(size=10)),
                            yaxis=dict(title="HHI score (0 = money spread across many countries, 1 = all in one country)",
                                       range=[0, 1.05]),
                            showlegend=False)
                        st.plotly_chart(fig_hhi, use_container_width=True)

                        # Dynamic worked example
                        yr_example = ""
                        good = hhi_yr[hhi_yr["hhi"].between(0.1, 0.9)].sort_values("year_awarded", ascending=False)
                        if len(good) > 0:
                            ex_row  = good.iloc[0]
                            ex_yr   = int(ex_row["year_awarded"])
                            ex_h    = ex_row["hhi"]
                            yr_data = df_cn[df_cn["year_awarded"] == ex_yr]
                            yr_ctry = (yr_data.groupby("borrower country")["contract_value_usd"]
                                       .sum().dropna().sort_values(ascending=False))
                            yr_total = yr_ctry.sum()
                            n_c = len(yr_ctry)
                            if yr_total > 0 and n_c > 0:
                                top3 = yr_ctry.head(min(3, n_c))
                                parts = [f"{c} got {v/yr_total*100:.0f}% of the money ({v/yr_total:.2f}² = {(v/yr_total)**2:.3f})"
                                         for c, v in top3.items()]
                                rest_hhi = ex_h - sum((v/yr_total)**2 for v in top3.values)
                                rest_str = f" The remaining countries add {rest_hhi:.3f}." if n_c > 3 else ""
                                conc_word = "high" if ex_h > 0.25 else "medium" if ex_h > 0.15 else "low"
                                conc_desc = "most money went to just a couple of countries" if ex_h > 0.25 else "the money was spread around reasonably well"
                                yr_example = (
                                    f"In {ex_yr}, China had contracts in {n_c} {'country' if n_c==1 else 'countries'}. "
                                    f"{'; '.join(parts)}.{rest_str} "
                                    f"Square each share and add them up → {ex_h:.2f}. "
                                    f"That's a {conc_word} score — {conc_desc}.")

                        _explain(
                            "Every year, China wins contracts in different countries. This line tells you "
                            "whether that year China's money was spread across many countries, or piled "
                            "into just one or two. Think of splitting a pizza: if China gave almost all "
                            "its money to ONE country, the line is HIGH (close to 1) — one giant slice. "
                            "If China spread money across MANY countries, the line is LOW — lots of small "
                            "slices. One warning: when the line is high in an early year, it's usually "
                            "just because China only had one or two contracts that whole year, so by default "
                            "'all of it' went to one place — that's not a strategy, it's just very little activity.",
                            example=yr_example)
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
         tab_sec5, tab_gr, tab_mhhi) = st.tabs([
            "Market Share", "Value Stack", "Value Distribution",
            "Rank Trajectory", "Sector Mix",
            "Growth", "Market HHI"])

        with tab_sh:
            try:
                sh = annual_share(df4)
                sh_yrs = sh["year_awarded"].tolist()
                c_sha, c_shb = st.columns(2)

                # (a) Value share
                fig_sha = go.Figure()
                fig_sha.add_trace(go.Scatter(
                    x=sh_yrs, y=sh["value_share_pct"].tolist(),
                    name="China's value share (%)", line=dict(color=BLUE_P, width=3),
                    mode="lines+markers", marker=dict(size=7, color=BLUE_P)))
                _theme(fig_sha, "China's share of total contract VALUE, by year (%)", height=380)
                fig_sha.update_layout(
                    xaxis=dict(title="Year", tickmode="array", tickvals=sh_yrs,
                               tickangle=-45, tickfont=dict(size=10)),
                    yaxis_title="China's share of all contractor value (%)")
                c_sha.plotly_chart(fig_sha, use_container_width=True)
                c_sha.caption("Denominator = total value awarded to ALL contractor nationalities that year.")

                # (b) Count share
                fig_shb = go.Figure()
                fig_shb.add_trace(go.Scatter(
                    x=sh_yrs, y=sh["count_share_pct"].tolist(),
                    name="China's count share (%)", line=dict(color=ORANGE_LINE, width=3),
                    mode="lines+markers", marker=dict(size=7, color=ORANGE_LINE)))
                _theme(fig_shb, "China's share of total contract COUNT, by year (%)", height=380)
                fig_shb.update_layout(
                    xaxis=dict(title="Year", tickmode="array", tickvals=sh_yrs,
                               tickangle=-45, tickfont=dict(size=10)),
                    yaxis_title="China's share of all contractor contracts (%)")
                c_shb.plotly_chart(fig_shb, use_container_width=True)
                c_shb.caption("Denominator = total contracts awarded to ALL contractor nationalities that year.")

                _missing_note()

                # Key insight with worked example from latest data year
                sh_latest = sh.dropna(subset=["value_share_pct", "count_share_pct"]).sort_values("year_awarded", ascending=False)
                insight_ex = ""
                if len(sh_latest) > 0:
                    row_l = sh_latest.iloc[0]
                    yr_l  = int(row_l["year_awarded"])
                    vs_l  = row_l["value_share_pct"]
                    cs_l  = row_l["count_share_pct"]
                    pr_l  = row_l.get("premium_ratio", float("nan"))
                    if not np.isnan(pr_l):
                        insight_ex = (f"In {yr_l}: value share = {vs_l:.1f}%, count share = {cs_l:.1f}% "
                                      f"→ premium ratio {pr_l:.2f}× — Chinese contracts were "
                                      f"{'larger' if pr_l > 1 else 'smaller'} than the market average that year.")
                    else:
                        insight_ex = f"In {yr_l}: value share = {vs_l:.1f}%, count share = {cs_l:.1f}%."

                _explain(
                    "Blue = what percentage of all MDB contract dollars went to Chinese firms that year. "
                    "Orange = what percentage of all contracts (by number) went to Chinese firms. "
                    "If the blue line sits above the orange, China wins fewer contracts but each one is bigger than average. "
                    "If the two lines are close together, China's contracts are roughly market size.",
                    example=insight_ex)

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
                        "This is simply the blue line ÷ the orange line from the chart above. "
                        "A ratio of 2.0 means Chinese contracts were, on average, twice the market size that year. "
                        "Green bars (above 1.0) = China's contracts are bigger than average. "
                        "Red bars (below 1.0) = smaller than average.")
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
                    "The full height of the pile = all MDB money spent that year by every contractor country combined. "
                    "China sits at the bottom in blue — you can read its dollar value straight off the y-axis. "
                    "Watch whether China's band is getting taller, staying flat, or shrinking compared to the rest.")
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
                        "The line in the middle = the median contract (half are bigger, half are smaller). "
                        "The box shows the middle 50% of contracts. "
                        "Dots above the whisker = unusually large individual contracts. "
                        "The y-axis is on a log scale — each gridline is 10× the one below.")
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
                        "Every year, all contractor countries are ranked by total dollars (Rank 1 = most money). "
                        "The y-axis is flipped — moving UP the chart means a better (lower number) rank. "
                        "A gap in a year means China had no contracts at all that year.",
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
                        "Each row = 100% of that country's contracts. "
                        "The colours show what fraction went to each type of project. "
                        "Compare China's row to others — a very different colour pattern means "
                        "China specialises in different sectors than its competitors.")
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
                        "CAGR = the average yearly growth rate from the first year to the last, "
                        "like a compound interest rate. "
                        "Blue bars (right of 0) = value grew on average each year. "
                        "Red bars (left of 0) = value shrank. "
                        "A CAGR of +15% means the value roughly doubled every 5 years.",
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
                    "Each dot = how much that country's contract value changed compared to the year before. "
                    "China's line is solid and fully opaque so it's easy to follow. "
                    "A huge spike or crash usually means one big project started or ended — "
                    "always check the count and value charts in Section 4 to understand why.")
            except Exception as e:
                _show_error(e, "S5 Growth")

        with tab_mhhi:
            try:
                hhi_sec, _hhi_ctr = market_hhi(df4)  # country chart removed per design

                fig_hs = go.Figure()
                fig_hs.add_trace(go.Bar(x=hhi_sec["project_sector"].tolist(),
                    y=hhi_sec["hhi"].tolist(), name="Market HHI",
                    marker_color=BLUE_M, marker_line_width=0))
                fig_hs.add_trace(go.Scatter(
                    x=hhi_sec["project_sector"].tolist(),
                    y=hhi_sec["china_share_pct"].tolist(),
                    name="China's share of sector value (%)", yaxis="y2",
                    mode="markers+lines",
                    marker=dict(color=ORANGE_LINE, size=10, symbol="diamond"),
                    line=dict(color=ORANGE_LINE, width=3)))
                _theme(fig_hs, "Market HHI and China's share by project sector", height=420, secondary_y=True)
                fig_hs.update_layout(
                    xaxis=dict(title="Project sector", tickangle=-30),
                    yaxis_title="Market HHI (0–1)",
                    yaxis2=dict(title="China's share of sector value (%)",
                                overlaying="y", side="right", range=[0, 100],
                                gridcolor=GRID, tickfont=dict(color=MUTED, size=11)),
                    legend=dict(title_text="Series"))
                st.plotly_chart(fig_hs, use_container_width=True)
                _missing_note()

                top_sec = hhi_sec.sort_values("total_value", ascending=False).iloc[0] if len(hhi_sec) > 0 else None
                mhhi_example = ""
                if top_sec is not None:
                    mhhi_example = (f"In {top_sec['project_sector']}, the market HHI is "
                                    f"{top_sec['hhi']:.3f} ({top_sec['hhi_label']}), meaning the overall "
                                    f"contractor market is {top_sec['hhi_label'].lower()}. "
                                    f"Chinese contractors hold {top_sec['china_share_pct']:.1f}% of "
                                    f"that sector's total contract value ({fmt_usd(top_sec['total_value'])} "
                                    f"across all contractors).")

                _explain(
                    "Each sector has two questions: (1) Is the market dominated by one or two countries, "
                    "or do many countries compete? The blue bars answer that — a tall bar means very few "
                    "countries dominate; a short bar means the pie is split many ways. "
                    "(2) How big is China's slice? The orange diamonds answer that. "
                    "Put them together: tall bar + high diamond = China (or someone) has most of the market. "
                    "Short bar + high diamond = China is strong but faces real competition.",
                    example=mhhi_example)

                hd_sec = hhi_sec[["project_sector", "hhi", "hhi_label", "china_share_pct",
                                   "n_contractors", "total_value"]].copy()
                hd_sec["total_value"]      = hd_sec["total_value"].map(fmt_usd)
                hd_sec["hhi"]              = hd_sec["hhi"].round(4)
                hd_sec["china_share_pct"]  = hd_sec["china_share_pct"].round(2)
                hd_sec.columns = ["Sector", "HHI", "Interpretation", "China share (%)",
                                   "# contractor countries", "Total value (all contractors)"]
                st.markdown(_html_table(hd_sec, "Market concentration by sector"), unsafe_allow_html=True)
                st.markdown(
                    f'<div style="background:#f8fafd;border:1px solid {GRID};border-radius:8px;'
                    f'padding:14px 18px;font-size:0.8rem;color:{NAVY};margin:10px 0;line-height:1.7">'
                    f'<strong>Column guide</strong><br>'
                    f'<b>Sector</b> — the area of work (Transport, Energy, Water, etc.).<br>'
                    f'<b>HHI</b> — a number from 0 to 1. Close to 0 = many countries share the work. '
                    f'Close to 1 = one country has almost everything.<br>'
                    f'<b>Interpretation</b> — plain-English label for the HHI score (Highly competitive → '
                    f'Unconcentrated → Moderately concentrated → Highly concentrated).<br>'
                    f'<b>China share (%)</b> — out of every dollar spent in this sector across all '
                    f'contractors, what share went to Chinese firms.<br>'
                    f'<b># contractor countries</b> — how many different nationalities won at least one '
                    f'contract in this sector. This is the number of "players" the HHI is based on.<br>'
                    f'<b>Total value (all contractors)</b> — the total USD value of all contracts in this '
                    f'sector, from every country combined.</div>',
                    unsafe_allow_html=True)
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
