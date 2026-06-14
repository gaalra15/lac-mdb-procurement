# Methodology — Chinese Companies in MDB Procurement, LAC

World Bank Capstone · Descriptive Analysis · Pipeline version: June 2026

---

## 1. Data source

| Item | Detail |
|---|---|
| File | `worldbank_idb_cdb_merged_0614.xlsx` |
| Sheet used | `Data as of 15 June` |
| Rows | 237,651 |
| Columns | 19 |
| Year range | 2000–2026 |
| **Sheets ignored** | Count by Source, Values by Source, Count by JV (pre-built pivots) |

All figures in the dashboard are recomputed from the raw sheet. Pre-built pivot sheets are not used.

---

## 2. China identification rule

```python
cc = df["contractor_country"].fillna("")
is_hk = cc.str.contains("Hong Kong", case=False)
df["is_chinese"] = cc.str.contains("china", case=False) & ~is_hk
```

- **Included**: any `contractor_country` value containing the string "china" (case-insensitive), including joint-venture combos such as "China; Germany", "Peru; China", "Netherlands; China", "China; St. Lucia", "Paraguay; China".
- **Excluded**: "Hong Kong SAR, China" — Hong Kong is treated as a separate contractor category and is never folded into the Chinese total.
- Result (full dataset): **147 Chinese contracts**, **5 Hong Kong SAR contracts**.

A unified `contractor_label` column is derived:
- `is_chinese == True` → `"China"`
- `is_hk == True` → `"Hong Kong SAR"`
- Otherwise → original `contractor_country` value

---

## 3. Negative and null contract values

| Issue | Count | Action |
|---|---|---|
| Null `contract_value_usd` | 190 | Kept as NaN. Excluded from sums and averages. |
| Negative `contract_value_usd` | **3** | Set to NaN. Excluded from value aggregations. |

Negative rows (all IDB source):

| notice_id | Country | Value (USD) |
|---|---|---|
| BO-L1209-P00011-C01 | Bolivia | −3,422,967.30 |
| SU-L1052-P00064-C01 | Suriname | −131,976.00 |
| CR-L1137-P00354-C01 | Costa Rica | −19,300.00 |

**Contract counts always use `notice_id`**, not value, so these 3 rows are still counted.

After setting negatives to NaN: **193 total null values** in `contract_value_usd`.

---

## 4. Casing / spelling normalisation

### 4.1 `contractor_country_group` → `{BRICS, G7, Others}`

| Raw value | Count | Cleaned → |
|---|---|---|
| `Others` | 212,001 | `Others` |
| `BRICKES` | 16,055 | **`BRICS`** (typo in source data) |
| `G7` | 4,634 | `G7` |
| `the global south` | 4,503 | **`Others`** |
| `the global north` | 458 | **`Others`** |

The 4,961 rows with `the global south` / `the global north` in the group column are the identical 4,961 rows that have `others` in `contractor_country_type`. These are misclassified rows (group-column values were filled with type-column labels). Since the correct group cannot be reliably derived, they are mapped to `Others` — the most appropriate catch-all group — and treated as subcategories of "Others" in Section 4 analysis.

### 4.2 `contractor_country_type` → `{Global South, Global North, Other}`

| Raw | Cleaned |
|---|---|
| `The Global South` | `Global South` |
| `The Global North` | `Global North` |
| `others` | `Other` |

### 4.3 `if_joint_venture` → `{Joint Venture, Non-Joint Venture}`

| Raw | Cleaned |
|---|---|
| `Joint Venture` | `Joint Venture` |
| `Non-Joint Venture` | `Non-Joint Venture` |
| `non-joint venture` | `Non-Joint Venture` |

### 4.4 `project_type`

Two rows contain the junk sentinel `"."`. These are relabelled `"Unknown"`.

---

## 5. Procurement method harmonisation

Raw `procurement_channel` contains 50+ variants across IDB, World Bank, and CDB. A harmonised `procurement_method` column is derived with **5 buckets**:

| Bucket | Key source values |
|---|---|
| `Open/Competitive` | International Competitive Bidding, National Competitive Bidding (all spellings), Competitive Bidding variants, SN-Pregao, SN-Licitacion (various) |
| `Limited/Shopping` | Shopping/Quotations (all spellings), Limited Bidding, Framework Agreement, SN-Convenios Marco, SN-Comparación de Precios, SN-Subasta |
| `Direct/Single-Source` | Direct Contracting (all variants), Single-Source Selection of Individuals/Firms/Audit (SSS), Force Account |
| `Consultant Selection` | Individual Consultant Selection (3CV, Open Invitation), QCBS, QBS, CQS, LCS, FBS, Least-Cost Selection, Consultores Individuales |
| `Unknown` | `"."` (119,677 IDB/CDB rows), `"Project procurement contracts"` (78,950 WB rows — non-granular catch-all), Others, N/A, Consultancy/Works/Goods (misclassified project types), Personal Services Hiring, Procurement 100% funded by Agency, National system entries |

**World Bank limitation**: All 78,950 WB rows carry `"Project procurement contracts"` → mapped to `Unknown`. Procurement method breakdowns are therefore meaningful only for IDB (158,272 rows) and CDB (429 rows). The dashboard labels this explicitly.

**Boolean flags derived**:
- `is_direct_award = (procurement_method == "Direct/Single-Source")`
- `is_single_bidder = (number_of_contractor == 1)` — per data dictionary

### ⚠ Note on `is_single_bidder`

`number_of_contractor == 1` is true for **237,174 out of 237,651 rows (99.8%)**. The column tracks the number of contractor *entities* on the award (1 = single firm; 2+ = joint venture), not the number of bids received in a competitive process. Interpreting this as a proxy for competitive intensity requires caution: it will not distinguish between a competitively awarded single-firm contract and a sole-source award. The flag is retained per the data dictionary but is labelled prominently in the dashboard.

---

## 6. No date filtering

All years 2000–2026 are loaded. The year slider in the sidebar narrows charts interactively without dropping any rows from the underlying dataset. Default view shows the full 2000–2026 range.

---

## 7. HHI — Herfindahl-Hirschman Index

```python
def hhi(values_by_unit):
    total = values_by_unit.sum()
    if total <= 0: return float("nan")
    shares = values_by_unit / total
    return float((shares ** 2).sum())
```

Scale: **0–1** (sum of squared shares). Thresholds:
- < 0.01 — Highly competitive (fragmented)
- 0.01–0.15 — Unconcentrated
- 0.15–0.25 — Moderately concentrated
- > 0.25 — Highly concentrated

**The unit of concentration differs by use-case — always labelled in the dashboard:**

| Use | Unit | Interpretation |
|---|---|---|
| Sections 2–3: China's HHI by country | Borrower country | How concentrated China's own procurement is across recipient countries |
| Section 3: HHI per year | Borrower country (within year) | Whether China is concentrating on fewer countries over time |
| Section 4: Market HHI by sector | Contractor country | How monopolised each sector is by any one contractor country |
| Section 4: Market HHI by borrower country | Contractor country | How monopolised each national market is by any one contractor country |

---

## 8. Parquet cache

On first run, the app reads `worldbank_idb_cdb_merged_0614.xlsx`, applies all cleaning steps, and writes `data/clean.parquet`. Subsequent runs load from parquet (faster). Delete `data/clean.parquet` to force a re-clean.

---

## 9. Replication

```
pip install -r requirements.txt
streamlit run app.py
```

The pipeline is fully deterministic: same input Excel → same output parquet → same dashboard figures.
