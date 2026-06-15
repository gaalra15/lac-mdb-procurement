"""
data_loader.py — single load-and-clean step with parquet cache.

All cleaning decisions are documented in METHODOLOGY.md.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

BASE = Path(__file__).parent
RAW_XLSX = BASE / "worldbank_idb_cdb_merged_0614V2.xlsx"
PARQUET = BASE / "data" / "clean.parquet"
SHEET = "Sheet1"

# ── Casing / spelling normalisation maps ─────────────────────────────────────

_GROUP_MAP: dict[str, str] = {
    "BRICS": "BRICS",
    "BRICKES": "BRICS",          # typo in source
    "G7": "G7",
    "Others": "Others",
    "the global south": "Others",  # misclassified rows (see METHODOLOGY.md)
    "the global north": "Others",
}

_TYPE_MAP: dict[str, str] = {
    "The Global South": "Global South",
    "The Global North": "Global North",
    "others": "Other",
}

_JV_MAP: dict[str, str] = {
    "Joint Venture": "Joint Venture",
    "Non-Joint Venture": "Non-Joint Venture",
    "non-joint venture": "Non-Joint Venture",
}

# ── Procurement harmonisation ─────────────────────────────────────────────────

# Values that are structurally non-informative about competition method
_EXPLICIT_UNKNOWN: set[str] = {
    ".",
    "project procurement contracts",  # WB catch-all, non-granular
    "others",
    "n/a (sistema nacional)",
    "consultancy",          # project types, not procurement methods
    "works",
    "goods",
    "personal services hiring",
    "procurement 100% funded by agency",
    "national system - advance procurement",
}


def _harmonize_procurement(series: pd.Series) -> pd.Series:
    """Map raw procurement_channel to a 5-bucket procurement_method column."""
    s = series.fillna(".").astype(str).str.strip()
    lo = s.str.lower()
    out = pd.Series("Unknown", index=series.index, dtype=str)

    # Apply in ascending priority (later rules overwrite earlier ones)

    # Consultant Selection
    out[lo.str.contains(
        r"individual consultant|quality.and.cost|qcbs|qbs\b|cqs\b|lcs\b|fbs\b"
        r"|fixed budget|least.cost|consultant.*qualif|consultores individ",
        regex=True, na=False,
    )] = "Consultant Selection"

    # Limited / Shopping
    out[lo.str.contains(
        r"shopping|quotation|framework agree|convenio|comparaci|subasta"
        r"|limited tender|limited bidding|tomada de pre",
        regex=True, na=False,
    )] = "Limited/Shopping"

    # Open / Competitive
    out[lo.str.contains(
        r"international competitive|national comp.*bidding|competitive bidding"
        r"|pregao|licitaci|licitacion|concorr",
        regex=True, na=False,
    )] = "Open/Competitive"

    # Direct / Single-Source  (highest priority among competitive buckets)
    out[lo.str.contains(
        r"direct contract|single.source|force account",
        regex=True, na=False,
    )] = "Direct/Single-Source"

    # Explicit unknowns override all pattern matches
    out[lo.isin(_EXPLICIT_UNKNOWN)] = "Unknown"

    return out


# ── Core cleaning ─────────────────────────────────────────────────────────────

def _clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Apply all normalisation and derivation steps.

    Returns (cleaned_df, data_quality_report_dict).
    """
    report: dict = {}

    # 1. contractor_country_group
    df["contractor_country_group"] = (
        df["contractor_country_group"].map(_GROUP_MAP).fillna("Others")
    )

    # 2. contractor_country_type
    df["contractor_country_type"] = (
        df["contractor_country_type"].map(_TYPE_MAP).fillna("Other")
    )

    # 3. if_joint_venture
    df["if_joint_venture"] = (
        df["if_joint_venture"].map(_JV_MAP).fillna("Non-Joint Venture")
    )

    # 4. project_type — drop junk "." sentinel
    df["project_type"] = df["project_type"].replace({"." : "Unknown"})

    # 5. contract_value_usd — exclude negatives from value aggregations
    neg_mask = df["contract_value_usd"] < 0
    report["n_negative"] = int(neg_mask.sum())
    report["negative_rows"] = df.loc[neg_mask, [
        "notice_id", "borrower country", "contractor_country",
        "contract_value_usd", "data_source",
    ]].to_dict("records")
    df.loc[neg_mask, "contract_value_usd"] = np.nan
    report["n_null_values"] = int(df["contract_value_usd"].isna().sum())

    # 6. China flag
    #    Chinese = contractor_country contains "china" (case-insensitive)
    #              AND does NOT contain "Hong Kong"
    cc = df["contractor_country"].fillna("")
    is_hk = cc.str.contains("Hong Kong", case=False)
    df["is_chinese"] = cc.str.contains("china", case=False) & ~is_hk
    df["is_hk"] = is_hk
    report["n_chinese"] = int(df["is_chinese"].sum())
    report["n_hk"] = int(is_hk.sum())

    # Unified contractor label for comparison charts
    df["contractor_label"] = df["contractor_country"].copy().astype(str)
    df.loc[df["is_chinese"], "contractor_label"] = "China"
    df.loc[is_hk, "contractor_label"] = "Hong Kong SAR"

    # 7. Procurement harmonisation + boolean flags
    df["procurement_method"] = _harmonize_procurement(df["procurement_channel"])
    df["is_direct_award"] = df["procurement_method"] == "Direct/Single-Source"
    # number_of_contractor == 1 per data dictionary (see METHODOLOGY.md §5)
    df["is_single_bidder"] = df["number_of_contractor"] == 1

    return df, report


# ── Public API ────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading and cleaning data…")
def get_data() -> pd.DataFrame:
    """Return cleaned DataFrame, loading from parquet cache when available."""
    if PARQUET.exists():
        return pd.read_parquet(PARQUET)

    raw = pd.read_excel(RAW_XLSX, sheet_name=SHEET, engine="openpyxl")
    df, _report = _clean(raw)
    try:
        PARQUET.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(PARQUET, index=False)
    except Exception:
        pass  # read-only filesystem (e.g. Streamlit Cloud) — skip cache write
    return df


def get_cleaning_report() -> dict:
    """Return the data-quality report dict (runs clean on first call)."""
    if PARQUET.exists():
        # Report is static — return the known values from the DQ analysis
        return {
            "n_negative": 3,
            "negative_rows": [
                {"notice_id": "BO-L1209-P00011-C01", "borrower country": "Bolivia",
                 "contractor_country": "Bolivia", "contract_value_usd": -3422967.3,
                 "data_source": "IDB"},
                {"notice_id": "SU-L1052-P00064-C01", "borrower country": "Suriname",
                 "contractor_country": "Suriname", "contract_value_usd": -131976.0,
                 "data_source": "IDB"},
                {"notice_id": "CR-L1137-P00354-C01", "borrower country": "Costa Rica",
                 "contractor_country": "Costa Rica", "contract_value_usd": -19300.0,
                 "data_source": "IDB"},
            ],
            "n_null_values": 193,   # 190 original + 3 set to NaN
            "n_chinese": 147,
            "n_hk": 5,
        }
    raw = pd.read_excel(RAW_XLSX, sheet_name=SHEET, engine="openpyxl")
    _df, report = _clean(raw)
    return report
