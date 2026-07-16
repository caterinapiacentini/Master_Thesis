#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Downloads and caches VIX, S&P 500, Fama-French, and JKP data to
data/external/cached/ so the analysis scripts don't hit the network.
Run once; use --force to re-download.
"""

import argparse
import numpy as np
import pandas as pd
import yfinance as yf
import pandas_datareader.data as web
from pathlib import Path

parser = argparse.ArgumentParser(description="Download and cache external market data.")
parser.add_argument("--force", action="store_true",
                    help="Re-download even if cached files already exist")
args, _ = parser.parse_known_args()

try:
    HERE = Path(__file__).parent
except NameError:
    HERE = Path.cwd()
REPO  = next((p for p in [HERE, *HERE.parents] if (p / "data" / "gep_us").exists()), HERE.parent)
EXT   = REPO / "data" / "external"
CACHE = EXT / "cached"
CACHE.mkdir(parents=True, exist_ok=True)

START = "1990-01-01"

FACTOR_RENAME = {
    "market_equity": "Size (SMB)",              "be_me": "Book-to-Market (HML)",
    "ope_be": "Operating Profitability (RMW)",   "at_gr1": "Asset Growth (CMA)",
    "ret_60_12": "Long-Term Reversals",          "ivol_ff3_21d": "Residual Variance (RVAR)",
    "qmj": "Quality Minus Junk (QMJ)",           "betabab_1260d": "Low Beta (BAB)",
    "ami_126d": "Amihud Illiquidity",            "age": "Firm Age",
    "prc": "Nominal Price",                      "dolvol_126d": "High Volume Premium",
    "gp_at": "Gross Profitability",              "ni_be": "Return on Equity",
    "niq_at": "Return on Assets",               "ebit_sale": "Profit Margin",
    "at_turnover": "Change in Asset Turnover",   "oaccruals_at": "Accruals Factor",
    "noa_at": "Net Operating Assets",            "cowc_gr1a": "Net Working Capital Changes",
    "ocf_me": "Cash Flow to Price",              "ni_me": "Earnings to Price",
    "ebitda_mev": "Enterprise Multiple",         "sale_me": "Sales to Price",
    "inv_gr1": "Growth in Inventory",            "sale_gr1": "Sales Growth",
    "dsale_dinv": "Growth in Sales/Inventory",   "capex_abn": "Abnormal Investment",
    "capx_gr1": "CAPX Growth Rate",              "dbnetis_at": "Debt Issuance Factor",
    "at_be": "Leverage Factor",                  "chcsho_12m": "1-Year Share Issuance",
    "netis_at": "Total External Financing",      "o_score": "Ohlson O-Score",
    "z_score": "Altman Z-Score",                 "f_score": "Piotroski F-Score",
}


def save(path, build_fn):
    if path.exists() and not args.force:
        print(f"  [SKIP]  {path.name}")
        return
    print(f"  [GET]   {path.name} ...", end=" ", flush=True)
    df = build_fn()
    df.to_csv(path)
    print(f"{len(df):,} rows saved")


# ── VIX ──────────────────────────────────────────────────────────────────────
def _vix_daily():
    df = yf.download("^VIX", start=START, end="2025-12-31",
                     interval="1d", auto_adjust=False, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Close"]].rename(columns={"Close": "VIX"}).dropna()
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    return df

def _vix_monthly():
    df = yf.download("^VIX", start=START, end="2025-12-31",
                     interval="1mo", auto_adjust=False, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Close"]].rename(columns={"Close": "VIX"}).dropna()
    df.index = pd.to_datetime(df.index).to_period("M").to_timestamp()
    df.index.name = "Date"
    return df

print("\nVIX")
save(CACHE / "vix_daily.csv",   _vix_daily)
save(CACHE / "vix_monthly.csv", _vix_monthly)

# ── S&P 500 ──────────────────────────────────────────────────────────────────
def _sp500_daily():
    df = yf.download("^GSPC", start="1995-12-01", end="2025-12-31",
                     interval="1d", auto_adjust=True, progress=False)[["Close"]].copy()
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    df.columns = ["sp500"]
    df["log_ret"] = np.log(df["sp500"] / df["sp500"].shift(1))
    return df.dropna()

def _sp500_monthly():
    df = yf.download("^GSPC", start="1995-12-01", end="2025-12-31",
                     interval="1mo", auto_adjust=True, progress=False)[["Close"]].copy()
    df.index = df.index.to_period("M").to_timestamp()
    df.index.name = "Date"
    df.columns = ["sp500"]
    df["log_ret"] = np.log(df["sp500"] / df["sp500"].shift(1))
    return df.dropna()

print("\nS&P 500")
save(CACHE / "sp500_daily.csv",   _sp500_daily)
save(CACHE / "sp500_monthly.csv", _sp500_monthly)

# ── Fama-French 3 factors ─────────────────────────────────────────────────────
def _ff3_daily():
    df = web.DataReader("F-F_Research_Data_Factors_daily", "famafrench", start=START)[0]
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    return df

def _ff3_monthly():
    df = web.DataReader("F-F_Research_Data_Factors", "famafrench", start=START)[0]
    df.index = df.index.to_timestamp()
    df.index.name = "Date"
    return df

print("\nFama-French 3 factors")
save(CACHE / "ff3_daily.csv",   _ff3_daily)
save(CACHE / "ff3_monthly.csv", _ff3_monthly)

# ── FF 49 industry portfolios ─────────────────────────────────────────────────
def _ff49_daily():
    df = web.DataReader("49_Industry_Portfolios_Daily", "famafrench", start=START)[0]
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    return df

def _ff49_monthly():
    df = web.DataReader("49_Industry_Portfolios", "famafrench", start=START)[0]
    df.index = df.index.to_timestamp()
    df.index.name = "Date"
    return df

print("\nFF 49 industry portfolios")
save(CACHE / "ff49_daily.csv",   _ff49_daily)
save(CACHE / "ff49_monthly.csv", _ff49_monthly)

# ── JKP factors (selected, pivoted) ──────────────────────────────────────────
JKP_DAILY   = EXT / "[usa]_[all_factors]_[daily]_[vw_cap].csv"
JKP_MONTHLY = EXT / "[usa]_[all_factors]_[monthly]_[vw_cap].csv"

def _jkp(path):
    raw  = pd.read_csv(path, usecols=["date", "name", "ret"])
    wide = raw.pivot_table(index="date", columns="name", values="ret", aggfunc="first")
    wide.columns.name = None
    wide.index = pd.to_datetime(wide.index)
    wide.index.name = "Date"
    keep = [c for c in FACTOR_RENAME if c in wide.columns]
    return wide[keep].rename(columns=FACTOR_RENAME)

print("\nJKP factors")
if JKP_DAILY.exists():
    save(CACHE / "jkp_daily_factors.csv",   lambda: _jkp(JKP_DAILY))
else:
    print(f"  [SKIP]  jkp_daily_factors.csv — source file not found: {JKP_DAILY.name}")

if JKP_MONTHLY.exists():
    save(CACHE / "jkp_monthly_factors.csv", lambda: _jkp(JKP_MONTHLY))
else:
    print(f"  [SKIP]  jkp_monthly_factors.csv — source file not found: {JKP_MONTHLY.name}")

print(f"\nDone — files saved to {CACHE.relative_to(REPO)}")
