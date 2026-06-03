#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_industries.py

GEP regressions on FF49 industry portfolios and JKP factors.

DATA layout (relative to this script):
  data/gep/GEP_Monthly_Robust_min2.csv
  data/gep/GEP_Daily_Robust_min2.csv
  data/external/data_gpr_daily_recent.xls
  data/external/data_gpr_export.xls
  data/external/[usa]_[all_factors]_[daily]_[vw_cap].csv   (JKP daily factors)
  data/external/[usa]_[all_factors]_[monthly]_[vw_cap].csv (JKP monthly factors)

External downloads via pandas_datareader: FF49 industry portfolios, FF3 factors.

Outputs saved to output/industries/
  GEP_Industry_Contemp_Daily.png           (FF49 + GPR, levels, contemporaneous)
  GEP_Industry_Predic_Daily.png            (FF49 + GPR, levels, predictive)
  GEP_Industry_Contemp_Daily_DELTA.png     (FF49 + ΔGPR, first-diff, contemporaneous)
  GEP_Industry_Predic_Daily_DELTA.png      (FF49 + ΔGPR, first-diff, predictive)
  GEP_Industry_Contemp_Daily_no_GPR.png    (FF49 only, no GPR, contemporaneous)
  GEP_Industry_Predic_Daily_no_GPR.png     (FF49 only, no GPR, predictive)
  GEP_Factor_Contemp_Daily.png             (JKP, levels, contemporaneous)
  GEP_Factor_Predic_Daily.png              (JKP, levels, predictive)
  GEP_Factor_Contemp_Daily_DELTA.png       (JKP, first-diff, contemporaneous)
  GEP_Factor_Predic_Daily_DELTA.png        (JKP, first-diff, predictive)
"""

import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

warnings.filterwarnings("ignore")

HERE = Path(__file__).parent
DATA = HERE.parent / "data"
GEP  = DATA / "gep_us"
EXT  = DATA / "external"
OUT  = HERE / "output" / "industries"
OUT.mkdir(parents=True, exist_ok=True)

GPR_RECENT = EXT / "data_gpr_daily_recent.xls"
GPR_EXPORT = EXT / "data_gpr_export.xls"
GPR_DAILY_PATH   = GPR_RECENT if GPR_RECENT.exists() else GPR_EXPORT
GPR_MONTHLY_PATH = GPR_EXPORT

JKP_DAILY   = EXT / "[usa]_[all_factors]_[daily]_[vw_cap].csv"
JKP_MONTHLY = EXT / "[usa]_[all_factors]_[monthly]_[vw_cap].csv"

DARK_BLUE  = "#1a3a5c"
LIGHT_BLUE = "#a8c4e0"
SIG_LEVEL  = 0.10

# ─────────────────────────────────────────────────────────────────────────────
# Load GEP
# ─────────────────────────────────────────────────────────────────────────────
daily_gep = pd.read_csv(GEP / "GEP_Daily_Robust_min2.csv")
daily_gep["date"] = pd.to_datetime(daily_gep["date"])
daily_gep = daily_gep.set_index("date").sort_index()

monthly_gep = pd.read_csv(GEP / "GEP_Monthly_Robust_min2.csv")
monthly_gep["month"] = pd.to_datetime(monthly_gep["month"])
monthly_gep = monthly_gep.set_index("month").sort_index()

# ─────────────────────────────────────────────────────────────────────────────
# Load GPR
# ─────────────────────────────────────────────────────────────────────────────
gpr_d_raw = pd.read_excel(GPR_DAILY_PATH)
gpr_d_raw["date"] = pd.to_datetime(gpr_d_raw["date"], dayfirst=True)
gpr_daily = (gpr_d_raw[["date", "GPRD"]].rename(columns={"GPRD": "GPR_daily"})
             .set_index("date").sort_index())

gpr_m_raw = pd.read_excel(GPR_MONTHLY_PATH)
gpr_m_raw["month"] = pd.to_datetime(gpr_m_raw["month"], dayfirst=True)
gpr_monthly = (gpr_m_raw[["month", "GPR"]].rename(columns={"GPR": "GPR_monthly"})
               .set_index("month").sort_index())

# ─────────────────────────────────────────────────────────────────────────────
# First-differenced series
# ─────────────────────────────────────────────────────────────────────────────
def diff_series(df, col):
    d = df[[col]].copy(); d[col] = d[col].diff(); return d.dropna()

dgep_daily   = diff_series(daily_gep,   "GEP_daily")
dgpr_daily   = diff_series(gpr_daily,   "GPR_daily")
dgep_monthly = diff_series(monthly_gep, "GEP_monthly")
dgpr_monthly = diff_series(gpr_monthly, "GPR_monthly")

# ─────────────────────────────────────────────────────────────────────────────
# Download FF data
# ─────────────────────────────────────────────────────────────────────────────
print("Downloading Fama-French data...")
ind_d   = web.DataReader("49_Industry_Portfolios_Daily",    "famafrench", start="1990-01-01")[0] / 100.0
ff3_d   = web.DataReader("F-F_Research_Data_Factors_daily", "famafrench", start="1990-01-01")[0] / 100.0
ind_m_r = web.DataReader("49_Industry_Portfolios",   "famafrench", start="1990-01-01")[0] / 100.0
ff3_m_r = web.DataReader("F-F_Research_Data_Factors","famafrench", start="1990-01-01")[0] / 100.0
ind_m = ind_m_r.copy(); ind_m.index = ind_m.index.to_timestamp()
ff3_m = ff3_m_r.copy(); ff3_m.index = ff3_m.index.to_timestamp()

FF49_NAMES = {
    "Agric": "Agriculture",           "Food":  "Food Products",
    "Soda":  "Candy & Soda",          "Beer":  "Beer & Liquor",
    "Smoke": "Tobacco Products",      "Toys":  "Recreation",
    "Fun":   "Entertainment",         "Books": "Printing & Publishing",
    "Hshld": "Consumer Goods",        "Clths": "Apparel",
    "Hlth":  "Healthcare",            "MedEq": "Medical Equipment",
    "Drugs": "Pharmaceutical Products","Chems": "Chemicals",
    "Rubbr": "Rubber & Plastic",      "Txtls": "Textiles",
    "BldMt": "Construction Materials","Cnstr": "Construction",
    "Steel": "Steel Works",           "FabPr": "Fabricated Products",
    "Mach":  "Machinery",             "ElcEq": "Electrical Equipment",
    "Autos": "Automobiles & Trucks",  "Aero":  "Aircraft",
    "Ships": "Shipbuilding & Railroad Equip.","Guns": "Defense",
    "Gold":  "Precious Metals & Mining","Mines":"Industrial Metal Mining",
    "Coal":  "Coal",                  "Oil":   "Petroleum & Natural Gas",
    "Util":  "Utilities",             "Telcm": "Telecommunications",
    "PerSv": "Personal Services",     "BusSv": "Business Services",
    "Hardw": "Computers & Hardware",  "Softw": "Computer Software",
    "Chips": "Electronic Equipment",  "LabEq": "Lab Equipment",
    "Paper": "Paper & Paper Products","Boxes": "Shipping Containers",
    "Trans": "Transportation",        "Whlsl": "Wholesale",
    "Rtail": "Retail",                "Meals": "Restaurants, Hotels & Motels",
    "Banks": "Banking",               "Insur": "Insurance",
    "RlEst": "Real Estate",           "Fin":   "Finance",
    "Other": "Other",
}

def full_name(short): return FF49_NAMES.get(short.strip(), short.strip())
def stars(p): return "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.10 else ""))

# ─────────────────────────────────────────────────────────────────────────────
# Regression engines
# ─────────────────────────────────────────────────────────────────────────────
def run_ind_regressions(ind_df, ff_df, gep_series, gpr_series, with_gpr=True):
    """Industry-level regressions: contemp + predictive."""
    results = []
    gep_col = gep_series.columns[0]
    for industry in ind_df.columns:
        pieces = [ind_df[[industry]], gep_series, ff_df]
        if with_gpr: pieces.append(gpr_series)
        df = pd.concat(pieces, axis=1).dropna()
        cols = ["RET", "GEP", "MktRF", "SMB", "HML", "RF"]
        if with_gpr: cols.append("GPR")
        df.columns = cols
        df["y"]       = df["RET"] - df["RF"]
        df["gep_var"] = df["GEP"] * 100
        ctrl = ["MktRF", "SMB", "HML"] + (["GPR"] if with_gpr else [])

        X1 = sm.add_constant(df[["gep_var"] + ctrl])
        m1 = sm.OLS(df["y"], X1).fit(cov_type="HC3")

        df["gep_lag"] = df["gep_var"].shift(1); df_lag = df.dropna()
        X2 = sm.add_constant(df_lag[["gep_lag"] + ctrl])
        m2 = sm.OLS(df_lag["y"], X2).fit(cov_type="HC3")

        results.append({"Industry":     industry.strip(),
                         "Contemp_Beta": m1.params["gep_var"],
                         "Contemp_Pval": m1.pvalues["gep_var"],
                         "Predic_Beta":  m2.params["gep_lag"],
                         "Predic_Pval":  m2.pvalues["gep_lag"],
                         "R2":           m1.rsquared})
    return pd.DataFrame(results)


def plot_exposure(df, beta_col, pval_col, title, filename, subtitle_note=""):
    plot_df = df[["Industry", beta_col, pval_col]].copy()
    plot_df["label"] = plot_df["Industry"].apply(full_name)
    raw = plot_df[beta_col] * 10_000
    plot_df["exposure"] = (raw - raw.mean()) / raw.std()
    plot_df["sig"]      = plot_df[pval_col] < SIG_LEVEL
    plot_df = plot_df.sort_values("exposure", ascending=False).reset_index(drop=True)
    colors  = [DARK_BLUE if s else LIGHT_BLUE for s in plot_df["sig"]]

    fig, ax = plt.subplots(figsize=(16, 10))
    ax.barh(range(len(plot_df)), plot_df["exposure"],
            color=colors, edgecolor="white", linewidth=0.3, height=0.7)
    ax.set_yticks([])
    ax.invert_yaxis()
    x_min, x_max = plot_df["exposure"].min(), plot_df["exposure"].max()
    pad = (x_max - x_min) * 0.012
    for i, (val, name) in enumerate(zip(plot_df["exposure"], plot_df["label"])):
        if val >= 0: ax.text(val + pad, i, name, va="center", ha="left",  fontsize=7.2, color="black")
        else:        ax.text(val - pad, i, name, va="center", ha="right", fontsize=7.2, color="black")
    ax.set_xlim(x_min - (x_max - x_min) * 0.28, x_max + (x_max - x_min) * 0.28)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Average Exposure (standardised, ×10 000 bps)", fontsize=10)
    ax.set_title(title, fontsize=12, pad=12)
    ax.legend(handles=[
        mpatches.Patch(color=DARK_BLUE,  label=f"Significant (p < {SIG_LEVEL})"),
        mpatches.Patch(color=LIGHT_BLUE, label=f"Not significant (p ≥ {SIG_LEVEL})"),
    ], loc="lower right", fontsize=8)
    if subtitle_note:
        fig.text(0.5, 0.01, subtitle_note, ha="center", fontsize=7.5, style="italic")
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(filename, dpi=200, bbox_inches="tight")
    print(f"Saved: {filename.name}")
    plt.close()


# ═════════════════════════════════════════════════════════════════════════════
# FF49 with GPR (levels + delta)
# ═════════════════════════════════════════════════════════════════════════════
print("\nRunning FF49 regressions — levels (with GPR)...")
res_d_lev = run_ind_regressions(ind_d, ff3_d,
                                daily_gep[["GEP_daily"]],   gpr_daily[["GPR_daily"]],  with_gpr=True)
res_d_dlt = run_ind_regressions(ind_d, ff3_d,
                                dgep_daily[["GEP_daily"]],  dgpr_daily[["GPR_daily"]], with_gpr=True)

for res, beta_c, pval_c, title, fname, note in [
    (res_d_lev, "Contemp_Beta", "Contemp_Pval",
     "GEP Industry Exposure — Daily Contemporaneous [LEVELS]",
     OUT / "GEP_Industry_Contemp_Daily.png",
     "Beta on GEP (×10 000, std). Controls: FF3 + GPR daily (levels). SE: HC3."),
    (res_d_lev, "Predic_Beta", "Predic_Pval",
     "GEP Industry Exposure — Daily Predictive [LEVELS]",
     OUT / "GEP_Industry_Predic_Daily.png",
     "Beta on lagged GEP (×10 000, std). Controls: FF3 + GPR daily (levels). SE: HC3."),
    (res_d_dlt, "Contemp_Beta", "Contemp_Pval",
     "ΔGEP Industry Exposure — Daily Contemporaneous [FIRST DIFFERENCES]",
     OUT / "GEP_Industry_Contemp_Daily_DELTA.png",
     "Beta on ΔGEP (×10 000, std). Controls: FF3 + ΔGPR daily. SE: HC3."),
    (res_d_dlt, "Predic_Beta", "Predic_Pval",
     "ΔGEP Industry Exposure — Daily Predictive [FIRST DIFFERENCES]",
     OUT / "GEP_Industry_Predic_Daily_DELTA.png",
     "Beta on lagged ΔGEP (×10 000, std). Controls: FF3 + ΔGPR daily. SE: HC3."),
]:
    plot_exposure(res, beta_c, pval_c, title, fname, note)

# ═════════════════════════════════════════════════════════════════════════════
# FF49 without GPR
# ═════════════════════════════════════════════════════════════════════════════
print("\nRunning FF49 regressions — levels (no GPR)...")
res_d_no_gpr = run_ind_regressions(ind_d, ff3_d,
                                   daily_gep[["GEP_daily"]], None, with_gpr=False)

for res, beta_c, pval_c, title, fname, note in [
    (res_d_no_gpr, "Contemp_Beta", "Contemp_Pval",
     "GEP Industry Exposure — Daily Contemporaneous [NO GPR]",
     OUT / "GEP_Industry_Contemp_Daily_no_GPR.png",
     "Beta on GEP (×10 000, std). Controls: FF3 only. SE: HC3."),
    (res_d_no_gpr, "Predic_Beta", "Predic_Pval",
     "GEP Industry Exposure — Daily Predictive [NO GPR]",
     OUT / "GEP_Industry_Predic_Daily_no_GPR.png",
     "Beta on lagged GEP (×10 000, std). Controls: FF3 only. SE: HC3."),
]:
    plot_exposure(res, beta_c, pval_c, title, fname, note)


# ═════════════════════════════════════════════════════════════════════════════
# JKP Factors
# ═════════════════════════════════════════════════════════════════════════════
FACTOR_RENAME = {
    "market_equity": "Size (SMB)",     "be_me": "Book-to-Market (HML)",
    "ope_be": "Operating Profitability (RMW)", "at_gr1": "Asset Growth (CMA)",
    "ret_60_12": "Long-Term Reversals", "ivol_ff3_21d": "Residual Variance (RVAR)",
    "qmj": "Quality Minus Junk (QMJ)", "betabab_1260d": "Low Beta (BAB)",
    "ami_126d": "Amihud Illiquidity",  "age": "Firm Age",
    "prc": "Nominal Price",            "dolvol_126d": "High Volume Premium",
    "gp_at": "Gross Profitability",    "ni_be": "Return on Equity",
    "niq_at": "Return on Assets",      "ebit_sale": "Profit Margin",
    "at_turnover": "Change in Asset Turnover", "oaccruals_at": "Accruals Factor",
    "noa_at": "Net Operating Assets",  "cowc_gr1a": "Net Working Capital Changes",
    "ocf_me": "Cash Flow to Price",    "ni_me": "Earnings to Price",
    "ebitda_mev": "Enterprise Multiple","sale_me": "Sales to Price",
    "inv_gr1": "Growth in Inventory",  "sale_gr1": "Sales Growth",
    "dsale_dinv": "Growth in Sales/Inventory", "capex_abn": "Abnormal Investment",
    "capx_gr1": "CAPX Growth Rate",    "dbnetis_at": "Debt Issuance Factor",
    "at_be": "Leverage Factor",        "chcsho_12m": "1-Year Share Issuance",
    "netis_at": "Total External Financing", "o_score": "Ohlson O-Score",
    "z_score": "Altman Z-Score",       "f_score": "Piotroski F-Score",
}

def load_jkp(path, freq="monthly"):
    print(f"Loading JKP {freq} factors...")
    raw  = pd.read_csv(path, usecols=["date", "name", "ret"])
    wide = raw.pivot_table(index="date", columns="name", values="ret", aggfunc="first")
    wide.columns.name = None
    wide.index = pd.to_datetime(wide.index)
    wide = wide.sort_index()
    keep = [c for c in FACTOR_RENAME if c in wide.columns]
    return wide[keep].rename(columns=FACTOR_RENAME)

def run_factor_regressions(gep_series, factor_df, gpr_ctrl, monthly=False):
    results = []
    if monthly:
        gep_series = gep_series.copy()
        gep_series.index = pd.to_datetime(gep_series.index).to_period("M").to_timestamp()
        factor_df  = factor_df.copy()
        factor_df.index  = pd.to_datetime(factor_df.index).to_period("M").to_timestamp()
        gpr_ctrl   = gpr_ctrl.copy()
        gpr_ctrl.index = pd.to_datetime(gpr_ctrl.index).to_period("M").to_timestamp()

    for factor in factor_df.columns:
        df = pd.concat([factor_df[factor], gep_series, gpr_ctrl], axis=1).dropna()
        if len(df) < 10: continue
        df.columns = ["FAC", "GEP", "GPR"]
        df["gep_pct"] = df["GEP"] * 100

        X1 = sm.add_constant(df[["gep_pct", "GPR"]])
        m1 = sm.OLS(df["FAC"], X1).fit(cov_type="HC3")

        df["gep_lag"] = df["gep_pct"].shift(1); df_lag = df.dropna()
        X2 = sm.add_constant(df_lag[["gep_lag", "GPR"]])
        m2 = sm.OLS(df_lag["FAC"], X2).fit(cov_type="HC3")

        results.append({"Factor": factor, "N": int(m1.nobs),
                         "Contemp_Beta": m1.params["gep_pct"],
                         "Contemp_Pval": m1.pvalues["gep_pct"],
                         "Predic_Beta":  m2.params["gep_lag"],
                         "Predic_Pval":  m2.pvalues["gep_lag"],
                         "R2_Contemp":   m1.rsquared})
    return pd.DataFrame(results)

def plot_factor_exposure(df, beta_col, pval_col, title, filename, note=""):
    plot_df = df[["Factor", beta_col, pval_col]].copy()
    raw = plot_df[beta_col] * 10_000
    plot_df["exposure"] = (raw - raw.mean()) / raw.std()
    plot_df["sig"]      = plot_df[pval_col] < SIG_LEVEL
    plot_df = plot_df.sort_values("exposure", ascending=False).reset_index(drop=True)
    colors  = [DARK_BLUE if s else LIGHT_BLUE for s in plot_df["sig"]]
    fig, ax = plt.subplots(figsize=(15, max(8, len(plot_df) * 0.28)))
    ax.barh(range(len(plot_df)), plot_df["exposure"],
            color=colors, edgecolor="white", linewidth=0.3, height=0.72)
    ax.set_yticks([])
    ax.invert_yaxis()
    x_min, x_max = plot_df["exposure"].min(), plot_df["exposure"].max()
    pad = (x_max - x_min) * 0.015
    for i, (val, name) in enumerate(zip(plot_df["exposure"], plot_df["Factor"])):
        if val >= 0: ax.text(val + pad, i, name, va="center", ha="left",  fontsize=7.5, color="black")
        else:        ax.text(val - pad, i, name, va="center", ha="right", fontsize=7.5, color="black")
    ax.set_xlim(x_min - (x_max - x_min) * 0.30, x_max + (x_max - x_min) * 0.30)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("GEP Exposure (standardised β ×10 000 bps)", fontsize=10)
    ax.set_title(title, fontsize=12, pad=12)
    ax.legend(handles=[
        mpatches.Patch(color=DARK_BLUE,  label=f"Significant (p < {SIG_LEVEL})"),
        mpatches.Patch(color=LIGHT_BLUE, label=f"Not significant (p ≥ {SIG_LEVEL})"),
    ], loc="lower right", fontsize=8)
    if note: fig.text(0.5, 0.01, note, ha="center", fontsize=7.5, style="italic")
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(filename, dpi=200, bbox_inches="tight")
    print(f"Saved: {filename.name}")
    plt.close()


if JKP_DAILY.exists() and JKP_MONTHLY.exists():
    factors_d = load_jkp(JKP_DAILY,   freq="daily")
    factors_m = load_jkp(JKP_MONTHLY, freq="monthly")

    dgpr_d_jkp = diff_series(gpr_daily,   "GPR_daily")
    dgep_d_jkp = diff_series(daily_gep,   "GEP_daily")

    print("\nRunning JKP factor regressions...")
    res_jkp_d     = run_factor_regressions(daily_gep["GEP_daily"],   factors_d, gpr_daily,   monthly=False)
    res_jkp_d_dlt = run_factor_regressions(dgep_d_jkp["GEP_daily"],  factors_d, dgpr_d_jkp,  monthly=False)

    note_lev = "Controls: GPR only (levels). Beta on GEP (×10 000, std). SE: HC3."
    note_dlt = "Controls: ΔGPR only. Beta on ΔGEP (×10 000, std). SE: HC3."

    for res, beta_c, pval_c, title, fname, note in [
        (res_jkp_d,     "Contemp_Beta", "Contemp_Pval",
         "GEP Factor Exposure — Daily Contemporaneous",
         OUT / "GEP_Factor_Contemp_Daily.png", f"Daily — {note_lev}"),
        (res_jkp_d,     "Predic_Beta",  "Predic_Pval",
         "GEP Factor Exposure — Daily Predictive (Lagged GEP)",
         OUT / "GEP_Factor_Predic_Daily.png", f"Daily — {note_lev}"),
        (res_jkp_d_dlt, "Contemp_Beta", "Contemp_Pval",
         "ΔGEP Factor Exposure — Daily Contemporaneous [FIRST DIFFERENCES]",
         OUT / "GEP_Factor_Contemp_Daily_DELTA.png", f"Daily — {note_dlt}"),
        (res_jkp_d_dlt, "Predic_Beta",  "Predic_Pval",
         "ΔGEP Factor Exposure — Daily Predictive (Lagged ΔGEP) [FIRST DIFFERENCES]",
         OUT / "GEP_Factor_Predic_Daily_DELTA.png", f"Daily — {note_dlt}"),
    ]:
        plot_factor_exposure(res, beta_c, pval_c, title, fname, note)
else:
    print(f"\n[WARNING] JKP factor files not found in {EXT}.")
    print("  Expected: [usa]_[all_factors]_[daily]_[vw_cap].csv")
    print("  Expected: [usa]_[all_factors]_[monthly]_[vw_cap].csv")
    print("  Skipping JKP factor regressions.")

print("\n═══ All industry/factor plots saved to output/industries/ ═══")
