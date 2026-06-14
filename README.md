# Chinese Companies in MDB Public Procurement — Latin America

Interactive Streamlit dashboard for the World Bank capstone:
*"Chinese Companies' Participation in MDB Public Procurement in Latin America"*

## Run

```
streamlit run app.py
```

First run loads `worldbank_idb_cdb_merged_0614.xlsx`, cleans it, and caches `data/clean.parquet`.
Subsequent runs load from the parquet file (much faster).

## Requirements

```
pip install -r requirements.txt
```

Python 3.10+ · Anaconda environment recommended.

## Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit dashboard (one page, four sections) |
| `data_loader.py` | Data loading, cleaning, parquet cache, China flag |
| `metrics.py` | Aggregations and HHI functions |
| `METHODOLOGY.md` | All cleaning decisions documented |
| `requirements.txt` | Python dependencies |
| `data/clean.parquet` | Generated on first run — delete to re-clean |

## Dashboard sections

| Section | Scope | Key charts |
|---|---|---|
| 1 — Raw picture | Chinese only | KPIs, by MDB source, by sector, by procurement method, data table |
| 2 — By country | Chinese only | Ranked bars, avg/median by country, HHI by country and sector |
| 3 — By year | Chinese only | Count/value trends, year×country heatmap, HHI over time |
| 4 — Comparison | China vs comparators | Market share, stacked area, value distribution, rank trajectory, sector mix, growth, procurement profile, market HHI |

## Data

Source: `worldbank_idb_cdb_merged_0614.xlsx` sheet "Data as of 15 June"
237,651 rows · IDB (158,272) · World Bank (78,950) · CDB (429) · 2000–2026

See `METHODOLOGY.md` for all cleaning decisions, China identification rule, and HHI definitions.
