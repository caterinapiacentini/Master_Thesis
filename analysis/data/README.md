# Analysis Data

Place input files here before running the analysis scripts.

## data/gep/
GEP index outputs from the cluster pipeline (copy from `~/Final_Thesis_Clean/output/`):
- `GEP_Daily_Robust_min2.csv`
- `GEP_Monthly_Robust_min2.csv`

## data/robustness/
Robustness variant indices:
- `GEP_Daily_Updated.csv`       — min-1 threshold
- `GEP_Monthly_Updated.csv`     — min-1 threshold
- `GEP_Daily_Robust_min3.csv`   — min-3 threshold
- `GEP_Monthly_Robust_min3.csv` — min-3 threshold
- `GEP_Daily_Robust_min4.csv`   — min-4 threshold
- `GEP_Monthly_Robust_min4.csv` — min-4 threshold
- `GEP_Daily_Robust_min2_v2.csv`   — GTM v2 (alternative seeds) [optional]
- `GEP_Monthly_Robust_min2_v2.csv` — GTM v2 (alternative seeds) [optional]

## data/countries/
Country-level GEP indices (copy from cluster output):
- `GEP_Monthly_JAPAN_min2.csv`
- `GEP_Monthly_UK_min2.csv`
- `GEP_Monthly_GERMANY_min2.csv`
- `GEP_Monthly_RUSSIA_min2.csv`
- `GEP_Monthly_IRAN_min2.csv`
- `GEP_Monthly_CHINA_min2.csv`

## data/external/
Literature data files (download manually):
- `data_gpr_daily_recent.xls`         — Caldara & Iacoviello (2022) GPR daily
- `data_gpr_export.xls`               — Caldara & Iacoviello (2022) GPR monthly/export
- `All_Daily_Policy_Data.csv`          — Baker, Bloom & Davis EPU daily
- `US_Policy_Uncertainty_Data.xlsx`    — Baker, Bloom & Davis EPU monthly
- `[usa]_[all_factors]_[daily]_[vw_cap].csv`   — JKP factors daily (optional)
- `[usa]_[all_factors]_[monthly]_[vw_cap].csv` — JKP factors monthly (optional)

Fama-French data (FF3, FF49) is downloaded automatically via `pandas_datareader`.
VIX and S&P 500 data is downloaded automatically via `yfinance`.
