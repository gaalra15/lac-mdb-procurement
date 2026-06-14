"""
metrics.py — reusable aggregation and concentration functions.

HHI is computed on a 0-1 scale (sum of squared market shares).
The *unit* of concentration differs by context — always labelled in the UI:
  • Sections 1-3: unit = borrower country or sector  (how concentrated China's own spending is)
  • Section 4:    unit = contractor country           (how concentrated the overall market is)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ── Core HHI ─────────────────────────────────────────────────────────────────

def hhi(values_by_unit: pd.Series) -> float:
    """Herfindahl-Hirschman Index on [0, 1] scale.

    Parameters
    ----------
    values_by_unit : Series of non-negative numbers indexed by unit (e.g. country name).
                     Nulls and non-positive entries are excluded.

    Returns
    -------
    float in [0, 1], or NaN if total <= 0.
    """
    v = values_by_unit.dropna()
    v = v[v > 0]
    total = v.sum()
    if total <= 0:
        return float("nan")
    shares = v / total
    return float((shares ** 2).sum())


def hhi_label(h: float) -> str:
    """Plain-language interpretation of an HHI value."""
    if np.isnan(h):
        return "N/A"
    if h < 0.01:
        return "Highly competitive (fragmented)"
    if h < 0.15:
        return "Unconcentrated"
    if h < 0.25:
        return "Moderately concentrated"
    return "Highly concentrated"


# ── Chinese-only aggregations (Sections 1-3) ─────────────────────────────────

def cn_by_source(df_cn: pd.DataFrame) -> pd.DataFrame:
    """Count and total value by MDB data source."""
    return (
        df_cn.groupby("data_source", observed=True)
        .agg(contracts=("notice_id", "count"),
             value_usd=("contract_value_usd", "sum"))
        .reset_index()
        .sort_values("contracts", ascending=False)
    )


def cn_by_sector(df_cn: pd.DataFrame) -> pd.DataFrame:
    """Count and total value by project sector."""
    return (
        df_cn.groupby("project_sector", observed=True)
        .agg(contracts=("notice_id", "count"),
             value_usd=("contract_value_usd", "sum"))
        .reset_index()
        .sort_values("value_usd", ascending=False)
    )


def cn_by_method(df_cn: pd.DataFrame) -> pd.DataFrame:
    """Count and total value by harmonised procurement method."""
    return (
        df_cn.groupby("procurement_method", observed=True)
        .agg(contracts=("notice_id", "count"),
             value_usd=("contract_value_usd", "sum"))
        .reset_index()
        .sort_values("contracts", ascending=False)
    )


def cn_by_country(df_cn: pd.DataFrame) -> pd.DataFrame:
    """Per borrower-country: count, total value, average value, median value."""
    g = df_cn.groupby("borrower country", observed=True)
    agg = g.agg(
        contracts=("notice_id", "count"),
        total_value=("contract_value_usd", "sum"),
        avg_value=("contract_value_usd", "mean"),
        median_value=("contract_value_usd", "median"),
    ).reset_index()
    agg = agg.sort_values("total_value", ascending=False)
    return agg


def cn_by_year(df_cn: pd.DataFrame) -> pd.DataFrame:
    """Per year: count, total value, average value."""
    g = df_cn.groupby("year_awarded", observed=True)
    agg = g.agg(
        contracts=("notice_id", "count"),
        total_value=("contract_value_usd", "sum"),
        avg_value=("contract_value_usd", "mean"),
    ).reset_index().sort_values("year_awarded")
    return agg


def cn_hhi_by_year(df_cn: pd.DataFrame) -> pd.DataFrame:
    """HHI of China's value across borrower countries, per year."""
    rows = []
    for yr, grp in df_cn.groupby("year_awarded", observed=True):
        by_country = grp.groupby("borrower country")["contract_value_usd"].sum()
        rows.append({"year_awarded": yr, "hhi": hhi(by_country)})
    return pd.DataFrame(rows).sort_values("year_awarded")


def cn_country_year_pivot(
    df_cn: pd.DataFrame, metric: str = "total_value"
) -> pd.DataFrame:
    """Pivot table: rows = year, columns = borrower country, values = contracts or value."""
    if metric == "contracts":
        piv = (
            df_cn.groupby(["year_awarded", "borrower country"], observed=True)
            ["notice_id"].count()
            .unstack(fill_value=0)
        )
    else:
        piv = (
            df_cn.groupby(["year_awarded", "borrower country"], observed=True)
            ["contract_value_usd"].sum()
            .unstack(fill_value=0)
        )
    return piv


# ── Comparison aggregations (Section 4) ──────────────────────────────────────

def top_n_labels(df: pd.DataFrame, n: int = 8) -> list[str]:
    """Top-N non-China contractor labels by total contract value."""
    totals = (
        df[~df["is_chinese"]]
        .groupby("contractor_label", observed=True)["contract_value_usd"]
        .sum()
        .nlargest(n)
    )
    return totals.index.tolist()


def annual_share(df: pd.DataFrame) -> pd.DataFrame:
    """Per year: China's share of total value and total count, plus premium ratio."""
    g = df.groupby("year_awarded", observed=True)
    total_val = g["contract_value_usd"].sum()
    total_cnt = g["notice_id"].count()

    cn = df[df["is_chinese"]].groupby("year_awarded", observed=True)
    cn_val = cn["contract_value_usd"].sum()
    cn_cnt = cn["notice_id"].count()

    out = pd.DataFrame({
        "total_value": total_val,
        "total_count": total_cnt,
        "china_value": cn_val,
        "china_count": cn_cnt,
    }).fillna(0)
    out["value_share_pct"] = 100 * out["china_value"] / out["total_value"].replace(0, np.nan)
    out["count_share_pct"] = 100 * out["china_count"] / out["total_count"].replace(0, np.nan)
    out["premium_ratio"] = out["value_share_pct"] / out["count_share_pct"].replace(0, np.nan)
    return out.reset_index()


def stacked_area_data(df: pd.DataFrame, top_labels: list[str]) -> pd.DataFrame:
    """Annual value for China, each top label, and Rest — ready for area chart."""
    annual = (
        df.groupby(["year_awarded", "contractor_label"], observed=True)
        ["contract_value_usd"].sum().reset_index()
    )

    def _group(lbl: str) -> str:
        if lbl == "China":
            return "China"
        return lbl if lbl in top_labels else "Rest"

    annual["label_group"] = annual["contractor_label"].map(_group)
    pivot = (
        annual.groupby(["year_awarded", "label_group"], observed=True)
        ["contract_value_usd"].sum()
        .unstack(fill_value=0)
    )
    # Ensure China column exists
    for col in ["China", "Rest"]:
        if col not in pivot.columns:
            pivot[col] = 0.0
    return pivot.reset_index()


def rank_trajectory(df: pd.DataFrame) -> pd.DataFrame:
    """China's rank by total contract value per year (1 = highest)."""
    annual = (
        df.groupby(["year_awarded", "contractor_label"], observed=True)
        ["contract_value_usd"].sum().reset_index()
    )
    annual["rank"] = annual.groupby("year_awarded", observed=True)["contract_value_usd"].rank(
        ascending=False, method="min"
    )
    china = annual[annual["contractor_label"] == "China"][["year_awarded", "rank", "contract_value_usd"]]
    return china.sort_values("year_awarded").rename(columns={"contract_value_usd": "china_value"})


def spread_by_label(df: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    """# distinct borrower countries and sectors per contractor label."""
    subset = df[df["contractor_label"].isin(labels)]
    rows = []
    for lbl, grp in subset.groupby("contractor_label", observed=True):
        rows.append({
            "contractor_label": lbl,
            "n_countries": grp["borrower country"].nunique(),
            "n_sectors": grp["project_sector"].nunique(),
            "total_value": grp["contract_value_usd"].sum(),
            "contracts": len(grp),
        })
    return pd.DataFrame(rows).sort_values("total_value", ascending=False)


def sector_mix(df: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    """100%-stacked sector mix: rows = contractor_label, cols = sector, values = share."""
    subset = df[df["contractor_label"].isin(labels)]
    counts = (
        subset.groupby(["contractor_label", "project_sector"], observed=True)
        ["notice_id"].count()
        .unstack(fill_value=0)
    )
    totals = counts.sum(axis=1)
    shares = counts.div(totals, axis=0) * 100
    return shares.reset_index()


def yoy_growth(df: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    """Year-over-year growth of contract value for each label."""
    annual = (
        df[df["contractor_label"].isin(labels)]
        .groupby(["year_awarded", "contractor_label"], observed=True)
        ["contract_value_usd"].sum()
        .unstack(fill_value=np.nan)
    )
    growth = annual.pct_change() * 100
    return growth.reset_index()


def cagr(df: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    """CAGR of contract value for each label over the available year range."""
    annual = (
        df[df["contractor_label"].isin(labels)]
        .groupby(["year_awarded", "contractor_label"], observed=True)
        ["contract_value_usd"].sum()
        .unstack(fill_value=np.nan)
    )
    rows = []
    for lbl in annual.columns:
        series = annual[lbl].dropna()
        series = series[series > 0]
        if len(series) < 2:
            rows.append({"contractor_label": lbl, "cagr_pct": np.nan,
                         "start_year": None, "end_year": None})
            continue
        y0, y1 = series.index[0], series.index[-1]
        n = y1 - y0
        v0, v1 = series.iloc[0], series.iloc[-1]
        rows.append({
            "contractor_label": lbl,
            "cagr_pct": ((v1 / v0) ** (1 / n) - 1) * 100,
            "start_year": y0,
            "end_year": y1,
        })
    return pd.DataFrame(rows).sort_values("cagr_pct", ascending=False)


def procurement_profile(df: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    """Procurement characteristics per contractor label (shares, %)."""
    subset = df[df["contractor_label"].isin(labels)]
    rows = []
    for lbl, grp in subset.groupby("contractor_label", observed=True):
        n = len(grp)
        rows.append({
            "contractor_label": lbl,
            "contracts": n,
            "direct_award_pct": 100 * grp["is_direct_award"].sum() / n if n else np.nan,
            "single_bidder_pct": 100 * grp["is_single_bidder"].sum() / n if n else np.nan,
            "jv_pct": 100 * (grp["if_joint_venture"] == "Joint Venture").sum() / n if n else np.nan,
        })
    out = pd.DataFrame(rows)
    # Method mix columns
    method_mix = (
        subset.groupby(["contractor_label", "procurement_method"], observed=True)
        ["notice_id"].count()
        .unstack(fill_value=0)
    )
    method_pct = method_mix.div(method_mix.sum(axis=1), axis=0) * 100
    method_pct.columns = [f"method_{c.lower().replace('/', '_')}" for c in method_pct.columns]
    out = out.merge(method_pct.reset_index(), on="contractor_label", how="left")
    return out.sort_values("contracts", ascending=False)


def market_hhi(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """HHI of contract value across contractor labels, within each sector and borrower country.

    Returns (hhi_by_sector, hhi_by_country) DataFrames with china_share column.
    """
    def _compute(group_col: str) -> pd.DataFrame:
        rows = []
        for grp_val, grp in df.groupby(group_col, observed=True):
            by_label = grp.groupby("contractor_label", observed=True)["contract_value_usd"].sum().dropna()
            total = by_label.sum()
            h = hhi(by_label)
            china_share = (by_label.get("China", 0.0) / total * 100) if total > 0 else np.nan
            rows.append({
                group_col: grp_val,
                "hhi": h,
                "hhi_label": hhi_label(h),
                "china_share_pct": china_share,
                "total_value": total,
                "n_contractors": len(by_label),
            })
        return pd.DataFrame(rows).sort_values("hhi", ascending=False)

    return _compute("project_sector"), _compute("borrower country")
