#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GEP exposure regressions on FF49 industry portfolios and JKP factors,
levels and first differences, contemporaneous and predictive.
Writes bar-chart plots to output/industries/.
"""

import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

warnings.filterwarnings("ignore")

try:
    HERE = Path(__file__).parent
except NameError:
    HERE = Path.cwd()
REPO = next((p for p in [HERE, *HERE.parents] if (p / "data" / "gep_us").exists()), HERE.parent)
DATA = REPO / "data"
GEP  = DATA / "gep_us"
EXT  = DATA / "external"
OUT  = REPO / "analysis" / "output" / "industries"
OUT.mkdir(parents=True, exist_ok=True)

GPR_RECENT = EXT / "data_gpr_daily_recent.xls"
GPR_EXPORT = EXT / "data_gpr_export.xls"
GPR_DAILY_PATH   = GPR_RECENT if GPR_RECENT.exists() else GPR_EXPORT
GPR_MONTHLY_PATH = GPR_EXPORT

CACHE = EXT / "cached"

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
print("Loading Fama-French data from cache...")
ind_d = pd.read_csv(CACHE / "ff49_daily.csv",   index_col="Date", parse_dates=True) / 100.0
ff3_d = pd.read_csv(CACHE / "ff3_daily.csv",    index_col="Date", parse_dates=True) / 100.0
ind_m = pd.read_csv(CACHE / "ff49_monthly.csv", index_col="Date", parse_dates=True) / 100.0
ff3_m = pd.read_csv(CACHE / "ff3_monthly.csv",  index_col="Date", parse_dates=True) / 100.0

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
def run_ind_regressions(ind_df, ff_df, gep_series, gpr_series):
    """Industry-level regressions: contemp + predictive, with FF3 + GPR controls."""
    results = []
    for industry in ind_df.columns:
        df = pd.concat([ind_df[[industry]], gep_series, ff_df, gpr_series], axis=1).dropna()
        df.columns = ["RET", "GEP", "MktRF", "SMB", "HML", "RF", "GPR"]
        df["y"]       = df["RET"] - df["RF"]
        df["gep_var"] = df["GEP"] * 100
        ctrl = ["MktRF", "SMB", "HML", "GPR"]

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

    # Square canvas (49 industries need ~10in of vertical room at a legible
    # font, so the square is sized to that rather than to \textwidth; it
    # will be scaled down a bit more on insertion than a width-matched
    # figure, but stays square and keeps the per-row spacing that avoids
    # label collisions).
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.barh(range(len(plot_df)), plot_df["exposure"],
            color=colors, edgecolor="white", linewidth=0.3, height=0.72)
    ax.set_yticks([])
    ax.invert_yaxis()
    x_min, x_max = plot_df["exposure"].min(), plot_df["exposure"].max()
    pad = (x_max - x_min) * 0.012
    for i, (val, name) in enumerate(zip(plot_df["exposure"], plot_df["label"])):
        if val >= 0: ax.text(val + pad, i, name, va="center", ha="left",  fontsize=8, color="black")
        else:        ax.text(val - pad, i, name, va="center", ha="right", fontsize=8, color="black")
    ax.set_xlim(x_min - (x_max - x_min) * 0.34, x_max + (x_max - x_min) * 0.34)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Average Exposure (standardised, ×10 000 bps)", fontsize=10.5)
    ax.set_title(title, fontsize=13, pad=14)
    ax.legend(handles=[
        mpatches.Patch(color=DARK_BLUE,  label=f"Significant (p < {SIG_LEVEL})"),
        mpatches.Patch(color=LIGHT_BLUE, label=f"Not significant (p ≥ {SIG_LEVEL})"),
    ], loc="lower right", fontsize=8.5)
    if subtitle_note:
        fig.text(0.5, 0.005, subtitle_note, ha="center", fontsize=8, style="italic")
    plt.tight_layout(rect=[0, 0.025, 1, 1])
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"Saved: {filename.name}")
    plt.close()


# FF49 with GPR (levels + delta)
print("\nRunning FF49 regressions — levels (with GPR)...")
res_d_lev = run_ind_regressions(ind_d, ff3_d, daily_gep[["GEP_daily"]],  gpr_daily[["GPR_daily"]])
res_d_dlt = run_ind_regressions(ind_d, ff3_d, dgep_daily[["GEP_daily"]], dgpr_daily[["GPR_daily"]])

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

# FF49 MONTHLY — console output only (ALL results)
print("\nRunning FF49 regressions — monthly levels (with GPR)...")
res_m_lev = run_ind_regressions(ind_m, ff3_m, monthly_gep[["GEP_monthly"]], gpr_monthly[["GPR_monthly"]])

print("\nRunning FF49 regressions — monthly first differences (with ΔGPR)...")
res_m_dlt = run_ind_regressions(ind_m, ff3_m, dgep_monthly[["GEP_monthly"]], dgpr_monthly[["GPR_monthly"]])

for res, label_contemp, label_predic in [
    (res_m_lev,
     "FF49 MONTHLY — LEVELS — CONTEMPORANEOUS",
     "FF49 MONTHLY — LEVELS — PREDICTIVE"),
    (res_m_dlt,
     "FF49 MONTHLY — FIRST DIFFERENCES — CONTEMPORANEOUS",
     "FF49 MONTHLY — FIRST DIFFERENCES — PREDICTIVE"),
]:
    print("\n" + "="*70)
    print(label_contemp)
    print("="*70)
    print(f"  {'Industry':<40}  {'Beta':>10}  {'SE':>10}  {'p':>8}  {'Sig'}")
    print("  " + "-"*70)
    for _, row in res.sort_values("Contemp_Pval").iterrows():
        s = stars(row["Contemp_Pval"])
        print(f"  {full_name(row['Industry']):<40}  "
              f"{row['Contemp_Beta']:>+10.6f}  "
              f"p={row['Contemp_Pval']:>6.3f}  {s}")

    print("\n" + "="*70)
    print(label_predic)
    print("="*70)
    print(f"  {'Industry':<40}  {'Beta':>10}  {'p':>8}  {'Sig'}")
    print("  " + "-"*70)
    for _, row in res.sort_values("Predic_Pval").iterrows():
        s = stars(row["Predic_Pval"])
        print(f"  {full_name(row['Industry']):<40}  "
              f"{row['Predic_Beta']:>+10.6f}  "
              f"p={row['Predic_Pval']:>6.3f}  {s}")

# JKP Factors
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
    # Square canvas, side scaled to the factor count (see plot_exposure for
    # the sizing rationale)
    side = max(9, len(plot_df) * 0.26)
    fig, ax = plt.subplots(figsize=(side, side))
    ax.barh(range(len(plot_df)), plot_df["exposure"],
            color=colors, edgecolor="white", linewidth=0.3, height=0.74)
    ax.set_yticks([])
    ax.invert_yaxis()
    x_min, x_max = plot_df["exposure"].min(), plot_df["exposure"].max()
    pad = (x_max - x_min) * 0.015
    for i, (val, name) in enumerate(zip(plot_df["exposure"], plot_df["Factor"])):
        if val >= 0: ax.text(val + pad, i, name, va="center", ha="left",  fontsize=8, color="black")
        else:        ax.text(val - pad, i, name, va="center", ha="right", fontsize=8, color="black")
    ax.set_xlim(x_min - (x_max - x_min) * 0.32, x_max + (x_max - x_min) * 0.32)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("GEP Exposure (standardised β ×10 000 bps)", fontsize=10.5)
    ax.set_title(title, fontsize=13, pad=14)
    ax.legend(handles=[
        mpatches.Patch(color=DARK_BLUE,  label=f"Significant (p < {SIG_LEVEL})"),
        mpatches.Patch(color=LIGHT_BLUE, label=f"Not significant (p ≥ {SIG_LEVEL})"),
    ], loc="lower right", fontsize=8.5)
    if note: fig.text(0.5, 0.005, note, ha="center", fontsize=8, style="italic")
    plt.tight_layout(rect=[0, 0.02, 1, 1])
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"Saved: {filename.name}")
    plt.close()


if (CACHE / "jkp_daily_factors.csv").exists() and (CACHE / "jkp_monthly_factors.csv").exists():
    factors_d = pd.read_csv(CACHE / "jkp_daily_factors.csv",   index_col="Date", parse_dates=True)
    factors_m = pd.read_csv(CACHE / "jkp_monthly_factors.csv", index_col="Date", parse_dates=True)

    dgpr_d_jkp = diff_series(gpr_daily,   "GPR_daily")
    dgep_d_jkp = diff_series(daily_gep,   "GEP_daily")

    print("\nRunning JKP factor regressions — daily...")
    res_jkp_d     = run_factor_regressions(daily_gep["GEP_daily"],   factors_d, gpr_daily,   monthly=False)
    res_jkp_d_dlt = run_factor_regressions(dgep_d_jkp["GEP_daily"],  factors_d, dgpr_d_jkp,  monthly=False)

    print("\nRunning JKP factor regressions — monthly...")
    res_jkp_m     = run_factor_regressions(monthly_gep["GEP_monthly"], factors_m, gpr_monthly, monthly=True)
    res_jkp_m_dlt = run_factor_regressions(dgep_monthly["GEP_monthly"], factors_m, dgpr_monthly, monthly=True)

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

    for res, label_contemp, label_predic in [
        (res_jkp_m,
         "JKP MONTHLY — LEVELS — CONTEMPORANEOUS",
         "JKP MONTHLY — LEVELS — PREDICTIVE"),
        (res_jkp_m_dlt,
         "JKP MONTHLY — FIRST DIFFERENCES — CONTEMPORANEOUS",
         "JKP MONTHLY — FIRST DIFFERENCES — PREDICTIVE"),
    ]:
        print("\n" + "="*70)
        print(label_contemp)
        print("="*70)
        print(f"  {'Factor':<45}  {'Beta':>10}  {'p':>8}  {'Sig'}")
        print("  " + "-"*70)
        for _, row in res.sort_values("Contemp_Pval").iterrows():
            s = stars(row["Contemp_Pval"])
            print(f"  {row['Factor']:<45}  "
                  f"{row['Contemp_Beta']:>+10.6f}  "
                  f"p={row['Contemp_Pval']:>6.3f}  {s}")

        print("\n" + "="*70)
        print(label_predic)
        print("="*70)
        print(f"  {'Factor':<45}  {'Beta':>10}  {'p':>8}  {'Sig'}")
        print("  " + "-"*70)
        for _, row in res.sort_values("Predic_Pval").iterrows():
            s = stars(row["Predic_Pval"])
            print(f"  {row['Factor']:<45}  "
                  f"{row['Predic_Beta']:>+10.6f}  "
                  f"p={row['Predic_Pval']:>6.3f}  {s}")

else:
    print(f"\n[WARNING] JKP cached files not found. Run fetch_data.py first.")
    print("  Skipping JKP factor regressions.")