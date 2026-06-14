"""
app.py — Streamlit dashboard: Chinese Companies in MDB Procurement, LAC
Run: streamlit run app.py
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

# ── Visual constants ──────────────────────────────────────────────────────────
CHINA_COLOR  = "#c0392b"
CHINA_FILL   = "rgba(192, 57, 43, 0.2)"
HK_COLOR     = "#e67e22"
REST_COLOR   = "#bdc3c7"
COMP_PALETTE = px.colors.qualitative.Set2
METHOD_COLORS = {
    "Open/Competitive":     "#2ecc71",
    "Limited/Shopping":     "#3498db",
    "Direct/Single-Source": "#e74c3c",
    "Consultant Selection": "#9b59b6",
    "Unknown":              "#95a5a6",
}
SECTOR_COLORS = px.colors.qualitative.Pastel

# ── Helpers ───────────────────────────────────────────────────────────────────

def fmt_usd(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    if abs(v) >= 1e9:  return f"${v/1e9:.2f} B"
    if abs(v) >= 1e6:  return f"${v/1e6:.1f} M"
    if abs(v) >= 1e3:  return f"${v/1e3:.0f} K"
    return f"${v:,.0f}"


def _show_error(e, label=""):
    prefix = f"[{label}] " if label else ""
    st.error(f"{prefix}{type(e).__name__}: {e}")
    _tb.print_exc()


def _hhi_primer():
    """Reusable plain-language HHI primer block."""
    st.markdown("""
> **What is the HHI?**
> The Herfindahl-Hirschman Index (HHI) is a number between **0 and 1** that measures
> how concentrated or spread out something is.
>
> **How it is calculated:** Take every group's share of the total, square each share,
> and add all those squared values together.
> Example: if Bolivia = 25 % of China's contracts → 0.25² = 0.0625.
> Do that for every country and sum them up.
>
> **How to read it:**
> | HHI | Meaning |
> |---|---|
> | < 0.01 | Highly spread out — many groups, none dominant |
> | 0.01 – 0.15 | Unconcentrated — reasonably balanced |
> | 0.15 – 0.25 | Moderately concentrated — a few groups stand out |
> | > 0.25 | Highly concentrated — one or two groups dominate |
>
> **Why it matters here:** An HHI close to 1 would mean China sends almost all its
> contracts to one country or one sector. An HHI close to 0 means the contracts are
> evenly distributed. Tracking it over time reveals whether China's engagement is
> deepening in specific places or broadening across the region.
>
> ⚠️ *The unit being measured changes by section — always check the label above each result.*
    """)


def bar_count_value(data, cat_col, title_count, title_value,
                    china_color=True, horizontal=False):
    color = CHINA_COLOR if china_color else None
    kw = dict(color_discrete_sequence=[color] if color else None)
    if horizontal:
        fc = px.bar(data, y=cat_col, x="contracts", orientation="h",
                    title=title_count, labels={"contracts": "Contracts", cat_col: ""}, **kw)
        fc.update_layout(yaxis={"categoryorder": "total ascending"})
        fv = px.bar(data, y=cat_col, x="value_usd", orientation="h",
                    title=title_value, labels={"value_usd": "Value (USD)", cat_col: ""}, **kw)
        fv.update_layout(yaxis={"categoryorder": "total ascending"})
    else:
        fc = px.bar(data, x=cat_col, y="contracts", title=title_count,
                    labels={"contracts": "Contracts", cat_col: ""}, **kw)
        fv = px.bar(data, x=cat_col, y="value_usd", title=title_value,
                    labels={"value_usd": "Value (USD)", cat_col: ""}, **kw)
    for fig in (fc, fv):
        fig.update_layout(margin=dict(t=40, b=10, l=10, r=10), height=340)
    return fc, fv


def colour_map(labels):
    cmap, ci = {}, 0
    for lbl in labels:
        if lbl == "China":          cmap[lbl] = CHINA_COLOR
        elif lbl == "Rest":         cmap[lbl] = REST_COLOR
        elif lbl == "Hong Kong SAR": cmap[lbl] = HK_COLOR
        else:
            cmap[lbl] = COMP_PALETTE[ci % len(COMP_PALETTE)]; ci += 1
    return cmap


# ── Page config + data load ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Chinese Companies in LAC MDB Procurement",
    layout="wide",
    initial_sidebar_state="expanded",
)
df_all = get_data()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Filters")
    st.caption("Applied to all sections. The Chinese segment is always shown regardless of selection.")

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
        help="Which non-China contractor groups appear in comparison charts.",
    )
    top_n = st.slider("Top-N comparator countries", 3, 15, 8)

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

# ── Title + KPI row ───────────────────────────────────────────────────────────
st.title("Chinese Companies in MDB Public Procurement — Latin America")
st.caption(
    "World Bank · IDB · CDB · 2000–2026 | Descriptive analysis | "
    "Research question: trends and characteristics of Chinese companies' participation"
)

n_cn  = len(df_cn)
val   = df_cn["contract_value_usd"].sum()
yr_sp = (f"{int(df_cn['year_awarded'].min())}–{int(df_cn['year_awarded'].max())}"
         if n_cn > 0 else "—")

k1,k2,k3,k4,k5,k6 = st.columns(6)
k1.metric("Chinese contracts",   f"{n_cn:,}")
k2.metric("Total value (USD)",   fmt_usd(val))
k3.metric("Borrower countries",  df_cn["borrower country"].nunique())
k4.metric("Sectors",             df_cn["project_sector"].nunique())
k5.metric("MDB sources",         df_cn["data_source"].nunique())
k6.metric("Year span",           yr_sp)

st.caption(
    "These headline numbers reflect **Chinese companies only** — firms whose "
    "`contractor_country` contains 'China', including joint-venture combinations, "
    "but excluding Hong Kong SAR (treated as a separate category). "
    "Adjust the sidebar filters to zoom into a specific time period, sector, or country."
)

# ── Methodology expander ──────────────────────────────────────────────────────
with st.expander("📋 How was this data prepared? (Methodology & data notes)", expanded=False):
    rpt = get_cleaning_report()
    st.markdown(f"""
### Where does the data come from?

This dashboard draws from a single combined dataset — **`worldbank_idb_cdb_merged_0614.xlsx`**,
sheet *"Data as of 15 June"* — which merges contract-award records from three multilateral
development banks (MDBs) that finance public projects across Latin America and the Caribbean:

| Bank | Abbreviation | Contracts in dataset |
|---|---|---|
| Inter-American Development Bank | IDB | 158,272 |
| World Bank | WB | 78,950 |
| Caribbean Development Bank | CDB | 429 |
| **Total** | | **237,651** |

The dataset covers contract awards from **2000 to 2026** (2026 is partial).

---

### How were Chinese companies identified?

A company is counted as **Chinese** if its `contractor_country` field contains the word
*"china"* (case-insensitive). This captures both pure Chinese firms and
joint ventures that include China (e.g. "China; Germany", "Peru; China").

**Hong Kong SAR, China is NOT counted as Chinese** — it is treated as its own separate
category, because it operates under a distinct legal and economic system.

Result: **{rpt['n_chinese']} Chinese contracts** identified, **{rpt['n_hk']} Hong Kong SAR
contracts** kept separate.

---

### What was cleaned or fixed?

**1. Typos and inconsistent labelling** were corrected:
- `BRICKES` (a typo) → `BRICS`
- `"the global south"` / `"the global north"` appearing in the wrong column → reclassified as `"Others"`
- Mixed capitalisation in joint-venture and contractor-type fields → standardised

**2. Contract values** — three contracts had *negative* dollar values (Bolivia –$3.4M,
Suriname –$132K, Costa Rica –$19K — all IDB records). These were excluded from any
money-based calculations but still counted in contract counts.
An additional **190 contracts had no dollar value recorded** at all — also excluded from
value sums but included in counts.

**3. Procurement method** — the three banks record procurement methods very differently.
The World Bank labels almost everything generically as *"Project procurement contracts"*,
which tells us nothing about competition. IDB and CDB use detailed categories (open bidding,
direct award, shopping, etc.). We harmonised everything into five plain-language buckets:
- **Open/Competitive** — any form of open competitive bidding
- **Limited/Shopping** — restricted bidding or price quotations
- **Direct/Single-Source** — awarded directly to one supplier without competition
- **Consultant Selection** — standard methods for selecting individual consultants or firms
- **Unknown** — World Bank records and anything else that couldn't be classified

This means **method charts are meaningful only for IDB and CDB contracts**.

**4. No contracts were removed by date.** All years 2000–2026 are included.
The year slider in the sidebar lets you focus on any sub-period.

---

### A note on the "single bidder" flag

The data includes a field called `number_of_contractor`. We flag contracts where this
equals 1 as "single bidder." However, in practice **99.8% of all contracts** have
`number_of_contractor = 1` — because this field counts the number of contractor
*entities* on the award (1 = single firm, 2+ = joint venture), not the number of
competing bids received. It is not a reliable measure of competition; use the
procurement method buckets instead.

---

### How does the Herfindahl-Hirschman Index (HHI) work?

See the **"What is the HHI?"** explainer box inside each HHI section — it is shown
in plain language every time the index appears.
    """)

st.divider()

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Chinese companies: raw picture
# ═════════════════════════════════════════════════════════════════════════════
st.header("Section 1 — Chinese Companies: The Raw Picture")
st.caption(
    "All charts in this section show **Chinese contracts only**. "
    "The goal is to understand the basic composition: how many contracts, "
    "how much money, through which banks, in which sectors, and via which procurement routes."
)

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
                fc, fv = bar_count_value(src_df, "data_source",
                    "Number of Chinese contracts by MDB source",
                    "Total value of Chinese contracts by MDB source (USD)")
                c1.plotly_chart(fc, use_container_width=True)
                c2.plotly_chart(fv, use_container_width=True)
                st.info(
                    "**How to read this:** Each bar is one of the three MDB sources. "
                    "The left chart counts how many contracts Chinese companies won; "
                    "the right chart shows the total dollar value. "
                    "If the value bar is much taller than the count bar (relative to other banks), "
                    "it means Chinese contracts through that bank tend to be larger on average. "
                    "\n\n"
                    "**Note:** The World Bank does not record detailed procurement methods "
                    "— all WB rows are labelled 'Unknown' in the Procurement Method tab."
                )
            except Exception as e:
                _show_error(e, "S1 By MDB Source")

        with tab_sec:
            try:
                sec_df = cn_by_sector(df_cn)
                c1, c2 = st.columns(2)
                fc, fv = bar_count_value(sec_df, "project_sector",
                    "Number of Chinese contracts by sector",
                    "Total value of Chinese contracts by sector (USD)",
                    horizontal=True)
                c1.plotly_chart(fc, use_container_width=True)
                c2.plotly_chart(fv, use_container_width=True)
                top_val_sec = sec_df.iloc[0]["project_sector"] if len(sec_df) > 0 else "—"
                top_cnt_sec = sec_df.sort_values("contracts", ascending=False).iloc[0]["project_sector"] if len(sec_df) > 0 else "—"
                st.info(
                    "**How to read this:** Bars are sorted from largest to smallest. "
                    "Compare the count chart (left) with the value chart (right) — "
                    "if a sector appears much higher on the right than on the left, "
                    "Chinese contracts in that sector tend to be very large individually. "
                    f"\n\n**Key finding:** By total value, **{top_val_sec}** is the dominant sector. "
                    f"By number of contracts, **{top_cnt_sec}** leads."
                )
            except Exception as e:
                _show_error(e, "S1 By Sector")

        with tab_meth:
            try:
                meth_df = cn_by_method(df_cn)
                colors_meth = [METHOD_COLORS.get(m, "#95a5a6") for m in meth_df["procurement_method"]]
                fig_mc = go.Figure(); fig_mc.add_trace(go.Bar(
                    x=meth_df["procurement_method"], y=meth_df["contracts"], marker_color=colors_meth))
                fig_mc.update_layout(title="Number of Chinese contracts by procurement method",
                                     height=340, margin=dict(t=40, b=10))
                fig_mv = go.Figure(); fig_mv.add_trace(go.Bar(
                    x=meth_df["procurement_method"], y=meth_df["value_usd"], marker_color=colors_meth))
                fig_mv.update_layout(title="Value of Chinese contracts by procurement method (USD)",
                                     height=340, margin=dict(t=40, b=10))
                c1, c2 = st.columns(2)
                c1.plotly_chart(fig_mc, use_container_width=True)
                c2.plotly_chart(fig_mv, use_container_width=True)

                st.info(
                    "**How to read this:** The five colour-coded buckets summarise *how* "
                    "Chinese companies were selected for these contracts. "
                    "'Unknown' is large because the World Bank does not report "
                    "detailed procurement methods. Among the contracts where we *do* know "
                    "the method (IDB/CDB), look at the share of "
                    "'Direct/Single-Source' (red) — this means no competitive bidding took place. "
                    "A high direct-award share could indicate preferred-supplier relationships "
                    "or specialised procurement, but the small total numbers for China "
                    "make this hard to generalise."
                )

                jv_n = (df_cn["if_joint_venture"] == "Joint Venture").sum()
                sb_n = df_cn["is_single_bidder"].sum()
                da_n = df_cn["is_direct_award"].sum()
                m1, m2, m3 = st.columns(3)
                m1.metric("Joint ventures", f"{jv_n} ({100*jv_n/n_cn:.1f}%)")
                m2.metric("Single-entity contractor", f"{sb_n} ({100*sb_n/n_cn:.1f}%)",
                          help="number_of_contractor == 1 per data dictionary — see data notes.")
                m3.metric("Direct / single-source awards", f"{da_n} ({100*da_n/n_cn:.1f}%)")
                st.caption(
                    "⚠️ 'Single-entity contractor' covers 99.8% of ALL contracts in the dataset "
                    "(not just Chinese ones), because this field counts firms on the award, "
                    "not competing bidders. It is not a useful competition indicator on its own."
                )
            except Exception as e:
                _show_error(e, "S1 Procurement Method")

        with tab_tbl:
            try:
                display_cols = ["year_awarded","borrower country","data_source","project_sector",
                                "procurement_method","contract_value_usd","if_joint_venture",
                                "contractor_country","contract_name","notice_id"]
                st.dataframe(df_cn[display_cols].sort_values("year_awarded", ascending=False),
                             use_container_width=True, height=480)
                st.download_button("⬇ Download Chinese contracts CSV",
                    df_cn[display_cols].to_csv(index=False).encode(),
                    "chinese_contracts.csv", "text/csv")
            except Exception as e:
                _show_error(e, "S1 Data Table")

except Exception as e:
    _show_error(e, "Section 1")

st.divider()

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Chinese companies: by borrower country
# ═════════════════════════════════════════════════════════════════════════════
st.header("Section 2 — Chinese Companies: By Borrower Country")
st.caption(
    "Which Latin American countries receive the most Chinese-backed contracts? "
    "And is China's engagement spread across the region or concentrated in a handful of countries?"
)

try:
    if n_cn == 0:
        st.warning("No Chinese contracts match the current filters.")
    else:
        ctry_df = cn_by_country(df_cn)

        tab_val, tab_avg, tab_tbl2, tab_hhi = st.tabs(
            ["Ranked by Total Value", "Ranked by Avg Value", "Country Table", "Concentration (HHI)"]
        )

        with tab_val:
            try:
                fig_tv = px.bar(ctry_df.sort_values("total_value"),
                    y="borrower country", x="total_value", orientation="h",
                    title="Total value of Chinese contracts by borrower country (USD)",
                    labels={"total_value": "Total Value (USD)", "borrower country": ""},
                    color_discrete_sequence=[CHINA_COLOR])
                fig_tv.update_layout(height=max(300, len(ctry_df)*22), margin=dict(t=40))
                st.plotly_chart(fig_tv, use_container_width=True)
                top_ctry = ctry_df.iloc[0]
                st.info(
                    f"**How to read this:** Each bar is a borrower country — the country that "
                    f"received the MDB loan and therefore the country where the project happened. "
                    f"A longer bar means Chinese contractors won more *value* (in USD) there. "
                    f"\n\n**Key finding:** **{top_ctry['borrower country']}** received the most "
                    f"Chinese contract value ({fmt_usd(top_ctry['total_value'])}) across "
                    f"{int(top_ctry['contracts'])} contracts."
                )
            except Exception as e:
                _show_error(e, "S2 Total Value")

        with tab_avg:
            try:
                c1, c2 = st.columns(2)
                fig_av = px.bar(ctry_df.sort_values("avg_value").dropna(subset=["avg_value"]),
                    y="borrower country", x="avg_value", orientation="h",
                    title="Average contract value per Chinese contract by country (USD)",
                    labels={"avg_value": "Avg Value (USD)", "borrower country": ""},
                    color_discrete_sequence=[CHINA_COLOR])
                fig_av.update_layout(height=max(300, len(ctry_df)*22), margin=dict(t=40))
                c1.plotly_chart(fig_av, use_container_width=True)

                fig_med = px.bar(ctry_df.sort_values("median_value").dropna(subset=["median_value"]),
                    y="borrower country", x="median_value", orientation="h",
                    title="Median contract value per Chinese contract by country (USD)",
                    labels={"median_value": "Median Value (USD)", "borrower country": ""},
                    color_discrete_sequence=["#e67e22"])
                fig_med.update_layout(height=max(300, len(ctry_df)*22), margin=dict(t=40))
                c2.plotly_chart(fig_med, use_container_width=True)
                st.info(
                    "**Average vs. Median — why show both?** "
                    "The average (mean) is pulled up by a few very large contracts. "
                    "The median is the 'middle' contract — half of contracts are above it, "
                    "half below. If average >> median in a country, it means there are "
                    "one or two giant contracts inflating the average. "
                    "Countries with a high median tend to receive large contracts *consistently*."
                )
            except Exception as e:
                _show_error(e, "S2 Avg/Median")

        with tab_tbl2:
            try:
                display = ctry_df.copy()
                display["total_value"]  = display["total_value"].map(fmt_usd)
                display["avg_value"]    = display["avg_value"].map(fmt_usd)
                display["median_value"] = display["median_value"].map(fmt_usd)
                display.columns = ["Country","Contracts","Total Value","Avg Value","Median Value"]
                st.dataframe(display, use_container_width=True, height=400)
            except Exception as e:
                _show_error(e, "S2 Table")

        with tab_hhi:
            try:
                _hhi_primer()
                st.markdown("---")
                st.markdown("#### Applying HHI to China's own contracts")
                st.caption(
                    "Here we ask: *within China's own portfolio of contracts, "
                    "how concentrated is the spending?* "
                    "The 'units' being measured are borrower countries and project sectors."
                )

                by_ctry_v = df_cn.groupby("borrower country")["contract_value_usd"].sum().dropna()
                h_c = hhi(by_ctry_v); lbl_c = hhi_label(h_c)
                by_sec_v  = df_cn.groupby("project_sector")["contract_value_usd"].sum().dropna()
                h_s = hhi(by_sec_v);  lbl_s = hhi_label(h_s)

                hc1, hc2 = st.columns(2)
                hc1.metric("HHI — across borrower countries", f"{h_c:.4f}",
                           delta=lbl_c, delta_color="off")
                hc2.metric("HHI — across project sectors",    f"{h_s:.4f}",
                           delta=lbl_s, delta_color="off")

                st.markdown(f"""
**What do these numbers mean?**

- **HHI by country = {h_c:.4f} → {lbl_c}.**
  China's contract value is reasonably spread across borrower countries.
  No single country absorbs so much of China's total that it dominates the portfolio.
  This suggests China participates across the region, though some countries (like Bolivia)
  receive disproportionately more than others.

- **HHI by sector = {h_s:.4f} → {lbl_s}.**
  China's contracts are **heavily skewed toward a small number of sectors**.
  Infrastructure & Transport and Energy, Climate & Environment together account for
  the large majority of Chinese contract *value*, while sectors like Education or
  Public Administration receive very little. This is consistent with China's global
  reputation for infrastructure-focused development financing.
                """)

                if len(by_ctry_v) > 0:
                    share_df = (by_ctry_v / by_ctry_v.sum() * 100).reset_index()
                    share_df.columns = ["borrower country","share_pct"]
                    fig_pie = px.pie(share_df, names="borrower country", values="share_pct",
                        title="China's contract value — share by borrower country",
                        color_discrete_sequence=px.colors.qualitative.Set3, hole=0.35)
                    fig_pie.update_traces(textposition="inside", textinfo="percent+label")
                    fig_pie.update_layout(showlegend=False, height=420)
                    st.plotly_chart(fig_pie, use_container_width=True)
                    st.caption(
                        "Each slice shows what percentage of China's *total contract value* "
                        "went to that country. Larger slices = more concentrated there."
                    )

                if len(by_sec_v) > 0:
                    share_s = (by_sec_v / by_sec_v.sum() * 100).reset_index()
                    share_s.columns = ["project_sector","share_pct"]
                    fig_ss = px.pie(share_s, names="project_sector", values="share_pct",
                        title="China's contract value — share by sector",
                        color_discrete_sequence=SECTOR_COLORS, hole=0.35)
                    fig_ss.update_traces(textposition="inside", textinfo="percent+label")
                    fig_ss.update_layout(showlegend=False, height=420)
                    st.plotly_chart(fig_ss, use_container_width=True)
                    st.caption(
                        "This pie shows which sectors absorb most of China's contract value. "
                        "A large sector slice combined with a high HHI score confirms that "
                        "Chinese MDB procurement is sector-specific, not broadly distributed."
                    )
            except Exception as e:
                _show_error(e, "S2 HHI")

except Exception as e:
    _show_error(e, "Section 2")

st.divider()

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Chinese companies: by year and by country
# ═════════════════════════════════════════════════════════════════════════════
st.header("Section 3 — Chinese Companies: By Year and by Country")
st.caption(
    "How has Chinese participation changed over time? "
    "Which countries absorbed Chinese contracts in which years? "
    "And has the geographic concentration increased or decreased?"
)

try:
    if n_cn == 0:
        st.warning("No Chinese contracts match the current filters.")
    else:
        with st.expander("Section 3 debug info", expanded=False):
            yr_df_dbg = cn_by_year(df_cn)
            st.write(f"df_cn shape: {df_cn.shape}")
            st.write(f"Year range: {int(df_cn['year_awarded'].min())} – {int(df_cn['year_awarded'].max())}")
            st.write(f"yr_df shape: {yr_df_dbg.shape}")
            st.dataframe(yr_df_dbg, use_container_width=True)

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
                        line=dict(color=CHINA_COLOR, width=2),
                        fill="tozeroy", fillcolor=CHINA_FILL,
                    ))
                    fig_cnt.update_layout(title="Chinese contract count per year",
                        xaxis_title="Year", yaxis_title="Number of contracts",
                        height=340, margin=dict(t=40,b=10))
                    c1.plotly_chart(fig_cnt, use_container_width=True)

                    fig_val = go.Figure()
                    fig_val.add_trace(go.Scatter(
                        x=yr_df["year_awarded"].tolist(), y=yr_df["total_value"].tolist(),
                        mode="lines+markers", name="Value (USD)",
                        line=dict(color=CHINA_COLOR, width=2),
                        fill="tozeroy", fillcolor=CHINA_FILL,
                    ))
                    fig_val.update_layout(title="Chinese contract value per year (USD)",
                        xaxis_title="Year", yaxis_title="Total Value (USD)",
                        height=340, margin=dict(t=40,b=10))
                    c2.plotly_chart(fig_val, use_container_width=True)

                    st.info(
                        "**How to read these:** Each point is one calendar year. "
                        "The left chart counts how many contracts Chinese companies won that year; "
                        "the right chart shows the combined dollar value. "
                        "A year with few contracts but high value means a small number of very large "
                        "individual contracts dominated. A year with many contracts but low value "
                        "means many smaller contracts. "
                        "Look for peaks and troughs — they may correspond to specific projects "
                        "or changes in MDB lending priorities."
                    )

                    avg_df = yr_df.dropna(subset=["avg_value"])
                    if len(avg_df) > 0:
                        fig_avg = go.Figure()
                        fig_avg.add_trace(go.Bar(
                            x=avg_df["year_awarded"].tolist(),
                            y=avg_df["avg_value"].tolist(),
                            marker_color=CHINA_COLOR,
                        ))
                        fig_avg.update_layout(
                            title="Average value per Chinese contract by year (USD)",
                            xaxis_title="Year", yaxis_title="Avg Contract Value (USD)",
                            height=320, margin=dict(t=40,b=10))
                        st.plotly_chart(fig_avg, use_container_width=True)
                        st.info(
                            "**How to read this:** This bar shows the *average size* of a Chinese "
                            "contract in each year. High bars mean China won a few very large "
                            "contracts that year. Low bars mean many smaller contracts. "
                            "Comparing this to the count and value charts above helps you understand "
                            "whether China's contracts are getting larger or smaller over time."
                        )
                except Exception as e:
                    _show_error(e, "S3 Trends")

            with tab_heatmap:
                try:
                    metric_choice = st.radio(
                        "Show in heatmap:", ["Total value (USD)", "Contract count"], horizontal=True)
                    metric_key = "total_value" if "value" in metric_choice else "contracts"

                    piv = cn_country_year_pivot(df_cn, metric_key)
                    if piv.empty:
                        st.info("No data to display.")
                    else:
                        country_order = piv.sum(axis=0).sort_values(ascending=False).index.tolist()
                        piv_s  = piv[country_order]
                        z_data = piv_s.values.T
                        hover  = z_data.copy()

                        if metric_key == "total_value":
                            z_plot     = np.where(z_data > 0, np.log10(z_data + 1), 0)
                            colorscale = "Reds"
                            cb_title   = "log₁₀(USD+1)"
                            hover_tmpl = "Year: %{x}<br>Country: %{y}<br>Value: $%{customdata:,.0f}<extra></extra>"
                        else:
                            z_plot     = z_data.astype(float)
                            colorscale = "Blues"
                            cb_title   = "Contracts"
                            hover_tmpl = "Year: %{x}<br>Country: %{y}<br>Contracts: %{customdata}<extra></extra>"

                        fig_heat = go.Figure(go.Heatmap(
                            z=z_plot, x=piv_s.index.tolist(), y=country_order,
                            colorscale=colorscale, colorbar=dict(title=cb_title),
                            hovertemplate=hover_tmpl, customdata=hover,
                        ))
                        fig_heat.update_layout(
                            title=f"Chinese contracts: year × borrower country ({metric_choice})",
                            xaxis_title="Year", yaxis_title="",
                            height=max(400, len(country_order)*22+120),
                            margin=dict(t=50,b=10))
                        st.plotly_chart(fig_heat, use_container_width=True)
                        st.info(
                            "**How to read this heatmap:** Each cell is a year (columns) × country (rows) "
                            "combination. Darker red (or blue) = more activity. White or very light = "
                            "no Chinese contracts that year in that country. "
                            "Countries are sorted top-to-bottom by total Chinese contract value across all years. "
                            "\n\nLook for: (1) Which countries have sustained dark cells across many years "
                            "(consistent recipients); (2) Which have only isolated dark cells (one-off large projects); "
                            "(3) Whether activity is spreading to new countries over time or remaining fixed. "
                            "\n\nNote: value is shown on a **log scale** to prevent one very large contract "
                            "from washing out all smaller ones — hover over any cell to see the exact dollar figure."
                        )
                except Exception as e:
                    _show_error(e, "S3 Heatmap")

            with tab_conc:
                try:
                    _hhi_primer()
                    st.markdown("---")
                    st.markdown(
                        "#### HHI of China's contract value across borrower countries — per year\n"
                        "*(Unit = borrower country; question = 'In this year, how concentrated was "
                        "China's spending across countries?')*"
                    )

                    hhi_yr = cn_hhi_by_year(df_cn)
                    if hhi_yr.empty or hhi_yr["hhi"].isna().all():
                        st.info("Not enough data to compute HHI per year.")
                    else:
                        fig_hhi = go.Figure()
                        fig_hhi.add_trace(go.Scatter(
                            x=hhi_yr["year_awarded"].tolist(), y=hhi_yr["hhi"].tolist(),
                            mode="lines+markers", line=dict(color=CHINA_COLOR, width=2), name="HHI",
                        ))
                        for thresh, ann in [
                            (0.01, "0.01 — competitive"),
                            (0.15, "0.15 — moderate"),
                            (0.25, "0.25 — concentrated"),
                        ]:
                            fig_hhi.add_hline(y=thresh, line_dash="dot", line_color="gray",
                                              annotation_text=ann, annotation_position="right")
                        fig_hhi.update_layout(
                            title="HHI of China's contract value by borrower country per year",
                            xaxis_title="Year", yaxis_title="HHI (0 = spread out, 1 = all in one country)",
                            height=380, margin=dict(t=50,b=10), yaxis_range=[0,1.05])
                        st.plotly_chart(fig_hhi, use_container_width=True)
                        st.info(
                            "**How to read this:** Each point is one year. "
                            "The HHI answers: *'In this year, how spread out were China's contracts "
                            "across borrower countries?'* "
                            "\n\n"
                            "**Going up** = that year's Chinese contracts were concentrated in fewer countries. "
                            "**Going down** = contracts were spread across more countries. "
                            "**Years with HHI near 1.0** often had only one or two countries receiving "
                            "any Chinese contracts at all (common in early years when China's "
                            "participation was just beginning). "
                            "**Years with lower HHI** reflect broader regional engagement. "
                            "\n\n"
                            "The dotted horizontal lines mark the standard interpretation thresholds "
                            "(below 0.15 = unconcentrated; above 0.25 = highly concentrated). "
                            "Hover over any point to see the exact year and HHI value."
                        )
                except Exception as e:
                    _show_error(e, "S3 Concentration")

except Exception as e:
    _show_error(e, "Section 3")

st.divider()

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Comparison: China vs other countries
# ═════════════════════════════════════════════════════════════════════════════
st.header("Section 4 — Comparison: China vs Other Countries")
st.caption(
    "China is always the **focal, highlighted series** (red). "
    "Comparators are the top-N other contractor countries by total value, "
    "drawn from the groups selected in the sidebar. "
    "All dollar values in USD."
)

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
                    line=dict(color=CHINA_COLOR, width=3), mode="lines+markers"))
                fig_sh.add_trace(go.Scatter(
                    x=sh["year_awarded"].tolist(), y=sh["count_share_pct"].tolist(),
                    name="China — count share %",
                    line=dict(color=CHINA_COLOR, width=2, dash="dash"), mode="lines+markers"))
                fig_sh.update_layout(
                    title="China's share of total MDB contract value and count per year (%)",
                    xaxis_title="Year", yaxis_title="%", height=400,
                    margin=dict(t=50,b=10),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02))
                st.plotly_chart(fig_sh, use_container_width=True)
                st.info(
                    "**How to read this:** The solid red line is China's share of the *total dollar value* "
                    "of all MDB contracts in each year. The dashed red line is China's share of the "
                    "*total number* of contracts. "
                    "\n\n**When value share > count share** (solid line above dashed): "
                    "Chinese contracts are *larger than average* — China wins fewer contracts "
                    "but they tend to be bigger. This 'value premium' is explored in the chart below. "
                    "\n\n**When they track together**: China's contracts are close to the market average size."
                )

                prem = sh.dropna(subset=["premium_ratio"])
                if len(prem) > 0:
                    fig_pr = go.Figure()
                    fig_pr.add_trace(go.Bar(
                        x=prem["year_awarded"].tolist(), y=prem["premium_ratio"].tolist(),
                        marker_color=CHINA_COLOR))
                    fig_pr.add_hline(y=1, line_dash="dot", line_color="gray",
                                     annotation_text="1.0 = same size as market average")
                    fig_pr.update_layout(
                        title="China's value-share ÷ count-share (the 'size premium' ratio)",
                        xaxis_title="Year", yaxis_title="Ratio", height=320,
                        margin=dict(t=50,b=10))
                    st.plotly_chart(fig_pr, use_container_width=True)
                    st.info(
                        "**How to read the size premium:** This bar divides China's value share "
                        "by its count share each year. "
                        "\n\n"
                        "- **Bar above 1.0**: Chinese contracts were *larger* than the overall market "
                        "average that year (e.g. a ratio of 2.0 means China's average contract "
                        "was twice the market average). "
                        "- **Bar below 1.0**: Chinese contracts were *smaller* than average. "
                        "- **Bar at 1.0**: Chinese contracts matched the market average exactly. "
                        "\n\n"
                        "Very large ratios in early years often reflect a single massive contract "
                        "in a year when China had very few contracts overall — use alongside "
                        "the count and value trend charts for context."
                    )
            except Exception as e:
                _show_error(e, "S4 Market Share")

        with tab_stack:
            try:
                sa = stacked_area_data(df4, top_labels)
                cols_order = (["China"] + [l for l in top_labels if l in sa.columns]
                              + (["Rest"] if "Rest" in sa.columns else []))
                fig_st = go.Figure()
                for lbl in cols_order:
                    if lbl not in sa.columns: continue
                    fig_st.add_trace(go.Scatter(
                        x=sa["year_awarded"].tolist(), y=sa[lbl].tolist(),
                        name=lbl, stackgroup="one", mode="lines",
                        line=dict(color=cmap.get(lbl,REST_COLOR),
                                  width=3 if lbl=="China" else 1),
                        fillcolor=cmap.get(lbl,REST_COLOR)))
                fig_st.update_layout(
                    title=f"Annual contract value by contractor group: China / top-{top_n} / Rest (USD)",
                    xaxis_title="Year", yaxis_title="Value (USD)", height=440,
                    margin=dict(t=50,b=10),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
                st.plotly_chart(fig_st, use_container_width=True)
                st.info(
                    "**How to read this stacked area chart:** The total height of the stack "
                    "each year = total MDB contract value across all contractor countries. "
                    "China (red, at the bottom) starts from zero, making it easiest to read "
                    "its absolute value directly from the y-axis. The areas above it are "
                    "the top comparator countries and the 'Rest' bucket (all other countries combined). "
                    "\n\n"
                    "Look for: (1) Whether China's red area is growing, shrinking, or stable; "
                    "(2) Whether the total stack is growing (more MDB lending overall); "
                    "(3) China's size relative to other named comparators."
                )
            except Exception as e:
                _show_error(e, "S4 Value Stack")

        with tab_dist:
            try:
                dist_df = df4[df4["contractor_label"].isin(all_labels)].dropna(subset=["contract_value_usd"])
                if len(dist_df) == 0:
                    st.info("No value data for selected contractors.")
                else:
                    fig_box = go.Figure()
                    for lbl in all_labels:
                        vals = dist_df.loc[dist_df["contractor_label"]==lbl, "contract_value_usd"]
                        if len(vals)==0: continue
                        fig_box.add_trace(go.Box(
                            y=vals.tolist(), name=lbl,
                            marker_color=cmap.get(lbl,"#7f7f7f"),
                            line_width=2 if lbl=="China" else 1,
                            boxpoints="outliers"))
                    fig_box.update_layout(
                        title="Contract value distribution by contractor country (log scale)",
                        yaxis_title="Contract Value (USD)", yaxis_type="log",
                        height=440, margin=dict(t=50,b=10))
                    st.plotly_chart(fig_box, use_container_width=True)
                    st.info(
                        "**How to read a box plot:** The box spans the middle 50% of contract values "
                        "(from the 25th to 75th percentile). The line inside the box is the median. "
                        "The whiskers extend to the smallest/largest values within 1.5× the box range. "
                        "Dots outside the whiskers are outliers — unusually large or small contracts. "
                        "The y-axis is on a **log scale** (each gridline = 10× the one below) "
                        "because contract values span several orders of magnitude. "
                        "\n\n"
                        "Compare China's box to others: is its median higher or lower? "
                        "Does it have more outlier dots (unusually sized contracts)? "
                        "A higher box overall means that country tends to win larger contracts."
                    )

                    am = (dist_df[dist_df["contractor_label"].isin(all_labels)]
                          .groupby("contractor_label", observed=True)["contract_value_usd"]
                          .agg(avg="mean", median="median")
                          .reindex(all_labels).reset_index())
                    c1,c2 = st.columns(2)
                    fig_avg = px.bar(am.sort_values("avg",ascending=True),
                        y="contractor_label", x="avg", orientation="h",
                        title="Average contract value by contractor (USD)",
                        labels={"avg":"Avg (USD)","contractor_label":""},
                        color="contractor_label", color_discrete_map=cmap)
                    fig_avg.update_layout(showlegend=False, height=340, margin=dict(t=40))
                    c1.plotly_chart(fig_avg, use_container_width=True)
                    fig_med = px.bar(am.sort_values("median",ascending=True),
                        y="contractor_label", x="median", orientation="h",
                        title="Median contract value by contractor (USD)",
                        labels={"median":"Median (USD)","contractor_label":""},
                        color="contractor_label", color_discrete_map=cmap)
                    fig_med.update_layout(showlegend=False, height=340, margin=dict(t=40))
                    c2.plotly_chart(fig_med, use_container_width=True)
                    st.caption(
                        "China's bar (red) compared to others reveals whether Chinese contracts "
                        "tend to be larger or smaller than those won by other contractor countries. "
                        "Average can be skewed by outliers; median gives a more typical picture."
                    )
            except Exception as e:
                _show_error(e, "S4 Distribution")

        with tab_rank:
            try:
                rnk = rank_trajectory(df4)
                if len(rnk)==0:
                    st.info("No rank data available.")
                else:
                    fig_rk = go.Figure()
                    fig_rk.add_trace(go.Scatter(
                        x=rnk["year_awarded"].tolist(), y=rnk["rank"].tolist(),
                        mode="lines+markers+text",
                        line=dict(color=CHINA_COLOR, width=3),
                        marker=dict(size=8, color=CHINA_COLOR),
                        text=rnk["rank"].astype(int).astype(str).tolist(),
                        textposition="top center", name="China rank"))
                    fig_rk.update_yaxes(autorange="reversed",
                                        title="Rank (1 = highest contract value that year)")
                    fig_rk.update_layout(
                        title="China's rank among all contractor countries by annual contract value",
                        xaxis_title="Year", height=400, margin=dict(t=50,b=10))
                    st.plotly_chart(fig_rk, use_container_width=True)
                    st.info(
                        "**How to read this:** Rank 1 means China had the *highest total contract value* "
                        "among all contractor countries in that year. The y-axis is *inverted* — "
                        "moving up the chart = better rank. "
                        "\n\n"
                        "A downward trend (rank rising, line falling on the inverted axis) means "
                        "more contractor countries are winning more value than China over time. "
                        "An upward trend means China is rising in relative importance. "
                        "Missing years are years where China had zero contracts."
                    )
                    rnk_d = rnk.copy()
                    rnk_d["china_value"] = rnk_d["china_value"].map(fmt_usd)
                    rnk_d["rank"] = rnk_d["rank"].astype(int)
                    rnk_d.columns = ["Year","Rank","China Value (USD)"]
                    st.dataframe(rnk_d, use_container_width=True, height=280)
            except Exception as e:
                _show_error(e, "S4 Rank")

        with tab_spread:
            try:
                sp = spread_by_label(df4, all_labels)
                if len(sp)==0:
                    st.info("No spread data available.")
                else:
                    fig_sc = px.scatter(sp, x="n_countries", y="n_sectors",
                        text="contractor_label", size="total_value",
                        color="contractor_label", color_discrete_map=cmap,
                        title="Geographic vs. sectoral spread — bubble size = total value (USD)",
                        labels={"n_countries":"# Borrower countries","n_sectors":"# Sectors"})
                    fig_sc.update_traces(textposition="top center")
                    fig_sc.update_layout(showlegend=False, height=440, margin=dict(t=50,b=10))
                    st.plotly_chart(fig_sc, use_container_width=True)
                    st.info(
                        "**How to read this scatter plot:** Each bubble is one contractor country. "
                        "Position to the **right** = active in more borrower countries. "
                        "Position **higher** = active in more sectors. "
                        "**Bubble size** = total contract value (larger = more money). "
                        "China (red) is highlighted. "
                        "\n\n"
                        "A contractor in the top-right corner has broad reach — many countries AND sectors. "
                        "A contractor in the bottom-left is narrow — concentrated in few places and sectors. "
                        "This tells you whether China's MDB participation is broad-based or niche."
                    )
                    c1,c2 = st.columns(2)
                    fig_c = px.bar(sp.sort_values("n_countries"),
                        y="contractor_label", x="n_countries", orientation="h",
                        title="# Borrower countries per contractor",
                        labels={"n_countries":"Countries","contractor_label":""},
                        color="contractor_label", color_discrete_map=cmap)
                    fig_c.update_layout(showlegend=False, height=340, margin=dict(t=40))
                    c1.plotly_chart(fig_c, use_container_width=True)
                    fig_s = px.bar(sp.sort_values("n_sectors"),
                        y="contractor_label", x="n_sectors", orientation="h",
                        title="# Sectors per contractor",
                        labels={"n_sectors":"Sectors","contractor_label":""},
                        color="contractor_label", color_discrete_map=cmap)
                    fig_s.update_layout(showlegend=False, height=340, margin=dict(t=40))
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
                    label_order = ["China"]+[l for l in all_labels if l!="China"]
                    fig_sm = px.bar(smix_l, x="share_pct", y="contractor_label",
                        color="sector", orientation="h",
                        title="Sector mix per contractor (each bar = 100% of that country's contracts)",
                        labels={"share_pct":"Share (%)","contractor_label":"","sector":"Sector"},
                        color_discrete_sequence=SECTOR_COLORS,
                        category_orders={"contractor_label":label_order})
                    fig_sm.update_layout(barmode="stack", xaxis_range=[0,100],
                        height=max(300,len(all_labels)*44), margin=dict(t=50,b=10),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
                    st.plotly_chart(fig_sm, use_container_width=True)
                    st.info(
                        "**How to read this 100% stacked bar:** Each row is one contractor country. "
                        "The full bar always = 100% of that country's contracts (by count). "
                        "The coloured segments show what share went to each sector. "
                        "\n\n"
                        "Compare China's row to others: does China have a different sector mix? "
                        "A country heavily tilted toward Infrastructure will have a large segment "
                        "of that colour. If China's bar looks very different from LATAM domestic "
                        "contractors, it suggests China specialises in specific project types."
                    )
            except Exception as e:
                _show_error(e, "S4 Sector Mix")

        with tab_growth:
            try:
                cagr_df = cagr(df4, all_labels)
                yoy_df  = yoy_growth(df4, all_labels)

                cp = cagr_df.dropna(subset=["cagr_pct"])
                if len(cp)>0:
                    fig_cg = px.bar(cp.sort_values("cagr_pct"),
                        y="contractor_label", x="cagr_pct", orientation="h",
                        title="Compound Annual Growth Rate (CAGR) of contract value over the full period (%)",
                        labels={"cagr_pct":"CAGR (%)","contractor_label":""},
                        color="contractor_label", color_discrete_map=cmap)
                    fig_cg.add_vline(x=0, line_dash="dot", line_color="gray")
                    fig_cg.update_layout(showlegend=False, height=340, margin=dict(t=40))
                    st.plotly_chart(fig_cg, use_container_width=True)
                    st.info(
                        "**What is CAGR?** The Compound Annual Growth Rate is the average yearly "
                        "growth rate of contract value over the entire period shown. "
                        "Think of it like an interest rate: a CAGR of 10% means, on average, "
                        "the total value grew by 10% each year from start to end. "
                        "A negative CAGR means the value shrank on average over time. "
                        "The dotted line at 0% separates growing (right) from shrinking (left). "
                        "China (red) is compared to each of the top comparator countries."
                    )

                fig_yoy = go.Figure()
                for lbl in all_labels:
                    if lbl not in yoy_df.columns: continue
                    fig_yoy.add_trace(go.Scatter(
                        x=yoy_df["year_awarded"].tolist(), y=yoy_df[lbl].tolist(),
                        name=lbl,
                        line=dict(color=cmap.get(lbl,REST_COLOR),
                                  width=3 if lbl=="China" else 1.5,
                                  dash="solid" if lbl=="China" else "dot"),
                        mode="lines", opacity=1.0 if lbl=="China" else 0.6))
                fig_yoy.add_hline(y=0, line_color="gray", line_width=1)
                fig_yoy.update_layout(
                    title="Year-over-year (YoY) growth of contract value (%) — China in solid red",
                    xaxis_title="Year", yaxis_title="YoY growth (%)", height=400,
                    margin=dict(t=50,b=10),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
                st.plotly_chart(fig_yoy, use_container_width=True)
                st.info(
                    "**How to read YoY growth:** Each point shows how much the contract value for "
                    "that contractor country changed *compared to the previous year*. "
                    "Above 0% = grew; below 0% = shrank. China (solid red, fully opaque) "
                    "is highlighted; comparators are dashed and semi-transparent. "
                    "\n\n"
                    "Because China has few contracts (147 total), any single large project "
                    "can cause extreme swings. A +500% jump may just mean one giant contract "
                    "that wasn't there the year before. Treat extreme values with caution "
                    "and cross-check with the count and value trend charts in Section 3."
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
                    fig_pm = px.bar(pl, x="share_pct", y="contractor_label",
                        color="method", orientation="h",
                        title="Procurement method mix per contractor (% of contracts, IDB/CDB only)",
                        labels={"share_pct":"Share (%)","contractor_label":""},
                        category_orders={"contractor_label":lo})
                    fig_pm.update_layout(barmode="stack", xaxis_range=[0,100],
                        height=max(300,len(all_labels)*44), margin=dict(t=50,b=10),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
                    st.plotly_chart(fig_pm, use_container_width=True)
                    st.info(
                        "**How to read this:** Each row = 100% of that contractor country's contracts. "
                        "The colour segments show how those contracts were awarded. "
                        "'Unknown' (grey) is large because World Bank contracts have no detailed method recorded. "
                        "\n\n"
                        "Among the known methods: **Open/Competitive** (green) = competitive bidding — "
                        "the most transparent. **Direct/Single-Source** (red) = no competition, "
                        "awarded directly. If China's row has a larger red segment than others, "
                        "it could indicate preferential treatment — but the small sample size "
                        "means this must be interpreted cautiously."
                    )

                bl = proc[["contractor_label","direct_award_pct","single_bidder_pct","jv_pct"]].copy()
                bl_l = bl.melt(id_vars="contractor_label", var_name="indicator", value_name="pct")
                bl_l["indicator"] = bl_l["indicator"].map({
                    "direct_award_pct":  "Direct / single-source (%)",
                    "single_bidder_pct": "Single bidder (number_of_contractor == 1) (%)",
                    "jv_pct":            "Joint venture (%)"})
                lo = ["China"]+[l for l in all_labels if l!="China"]
                fig_bl = px.bar(bl_l, x="contractor_label", y="pct", color="indicator",
                    barmode="group",
                    title="Direct awards / single bidder / joint-venture share by contractor (%)",
                    labels={"pct":"%","contractor_label":""},
                    category_orders={"contractor_label":lo},
                    color_discrete_sequence=["#e74c3c","#3498db","#9b59b6"])
                fig_bl.update_layout(height=380, margin=dict(t=50,b=10),
                                     legend=dict(orientation="h", yanchor="bottom", y=1.02))
                st.plotly_chart(fig_bl, use_container_width=True)
                st.info(
                    "**How to read this grouped bar:** For each contractor country, three bars "
                    "show: (red) share of contracts awarded directly without competition; "
                    "(blue) share with a single contractor entity; "
                    "(purple) share that are joint ventures. "
                    "\n\n"
                    "⚠️ The blue 'single bidder' bar covers nearly 100% of contracts for ALL countries — "
                    "this is because the underlying field counts contractor firms on the award, "
                    "not competing bidders, making it uninformative as a competition indicator. "
                    "Focus on the red (direct award) and purple (JV) bars instead."
                )

                st.markdown("#### By MDB source (IDB/CDB have meaningful procurement detail)")
                for src in sorted(df4["data_source"].unique()):
                    sub = df4[(df4["data_source"]==src)&(df4["contractor_label"].isin(all_labels))]
                    if len(sub)==0: continue
                    with st.expander(f"{src}"):
                        pp = procurement_profile(sub, all_labels)
                        st.dataframe(
                            pp[["contractor_label","contracts","direct_award_pct",
                                "single_bidder_pct","jv_pct"]].rename(columns={
                                "contractor_label":"Contractor","contracts":"Contracts",
                                "direct_award_pct":"Direct/SSS (%)","single_bidder_pct":"Single bidder (%)","jv_pct":"JV (%)"}),
                            use_container_width=True)
            except Exception as e:
                _show_error(e, "S4 Procurement")

        with tab_mhhi:
            try:
                _hhi_primer()
                st.markdown("---")
                st.markdown(
                    "#### HHI of contract value **across contractor countries** — by sector and by borrower country\n"
                    "*(Unit = contractor country; question = 'Within this sector/country, "
                    "is one nationality winning most contracts?')*"
                )
                st.caption(
                    "This is a different use of HHI from Sections 2–3. Here we are measuring "
                    "**market concentration** — how dominant any single contractor nationality is "
                    "within each sector or recipient country. China's own share is shown alongside "
                    "so you can see whether China specifically is the dominant player."
                )

                hhi_sec, hhi_ctr = market_hhi(df4)

                c1,c2 = st.columns(2)
                fig_hs = go.Figure()
                fig_hs.add_trace(go.Bar(x=hhi_sec["project_sector"].tolist(),
                    y=hhi_sec["hhi"].tolist(), name="HHI", marker_color="#3498db"))
                fig_hs.add_trace(go.Scatter(x=hhi_sec["project_sector"].tolist(),
                    y=hhi_sec["china_share_pct"].tolist(),
                    name="China share (%)", yaxis="y2", mode="markers+lines",
                    marker=dict(color=CHINA_COLOR, size=10, symbol="diamond"),
                    line=dict(color=CHINA_COLOR, dash="dot")))
                fig_hs.update_layout(title="Market HHI and China's share by sector",
                    yaxis_title="HHI (0–1)", yaxis2=dict(title="China share (%)",
                    overlaying="y", side="right", range=[0,100]),
                    height=380, margin=dict(t=50,b=80),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                    xaxis_tickangle=-30)
                c1.plotly_chart(fig_hs, use_container_width=True)

                fig_hc = go.Figure()
                fig_hc.add_trace(go.Bar(x=hhi_ctr["borrower country"].tolist(),
                    y=hhi_ctr["hhi"].tolist(), name="HHI", marker_color="#3498db"))
                fig_hc.add_trace(go.Scatter(x=hhi_ctr["borrower country"].tolist(),
                    y=hhi_ctr["china_share_pct"].tolist(),
                    name="China share (%)", yaxis="y2", mode="markers+lines",
                    marker=dict(color=CHINA_COLOR, size=8, symbol="diamond"),
                    line=dict(color=CHINA_COLOR, dash="dot")))
                fig_hc.update_layout(title="Market HHI and China's share by borrower country",
                    yaxis_title="HHI (0–1)", yaxis2=dict(title="China share (%)",
                    overlaying="y", side="right", range=[0,100]),
                    height=420, margin=dict(t=50,b=100),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                    xaxis_tickangle=-45)
                c2.plotly_chart(fig_hc, use_container_width=True)

                st.info(
                    "**How to read these dual-axis charts:** "
                    "The **blue bars** (left y-axis) show the HHI for that sector or country — "
                    "how concentrated the overall contractor market is. "
                    "The **red diamond line** (right y-axis, 0–100%) shows China's specific share. "
                    "\n\n"
                    "Four combinations to watch for: "
                    "\n- **High HHI + High China share**: China dominates this sector/country's market. "
                    "\n- **High HHI + Low China share**: One country dominates, but it's not China. "
                    "\n- **Low HHI + High China share**: China has a significant share but the market "
                    "is generally competitive with many players. "
                    "\n- **Low HHI + Low China share**: A competitive, fragmented market where China "
                    "plays a minor role."
                )

                st.markdown("**Detailed table — by sector**")
                st.dataframe(hhi_sec[["project_sector","hhi","hhi_label","china_share_pct",
                                       "n_contractors","total_value"]].rename(columns={
                    "project_sector":"Sector","hhi":"HHI","hhi_label":"What this means",
                    "china_share_pct":"China's share (%)","n_contractors":"# Contractor countries",
                    "total_value":"Total Value (USD)"}),
                    use_container_width=True, height=280)

                st.markdown("**Detailed table — by borrower country**")
                st.dataframe(hhi_ctr[["borrower country","hhi","hhi_label","china_share_pct",
                                       "n_contractors","total_value"]].rename(columns={
                    "borrower country":"Country","hhi":"HHI","hhi_label":"What this means",
                    "china_share_pct":"China's share (%)","n_contractors":"# Contractor countries",
                    "total_value":"Total Value (USD)"}),
                    use_container_width=True, height=380)
                st.caption(
                    "The 'What this means' column applies the standard HHI thresholds: "
                    "Unconcentrated (< 0.15), Moderately concentrated (0.15–0.25), "
                    "Highly concentrated (> 0.25). "
                    "These thresholds are widely used in competition economics "
                    "(e.g. by the US Department of Justice for antitrust analysis)."
                )
            except Exception as e:
                _show_error(e, "S4 Market HHI")

except Exception as e:
    _show_error(e, "Section 4")

st.divider()
st.caption(
    "World Bank Capstone · 'Chinese Companies' Participation in MDB Public Procurement "
    "in Latin America' · Descriptive analysis · IDB · World Bank · CDB · 2000–2026"
)
