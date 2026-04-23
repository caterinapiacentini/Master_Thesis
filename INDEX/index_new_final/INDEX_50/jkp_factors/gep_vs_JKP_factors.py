#!/usr/bin/env python3
"""
GEP Factor Regressions
----------------------
Runs contemporaneous & predictive OLS regressions of factor returns on the
GEP index, controlling for GPR only (MktRF excluded because market_equity
is itself one of the factors under study).

Two model families:
  LEVELS : GEP_t  (or GEP_{t-1})  + GPR_t
  DELTA  : ΔGEP_t (or ΔGEP_{t-1}) + ΔGPR_t   ← first differences

Plots:  daily only (contemp + predictive), both families.
Prints: regression tables for both daily and monthly, both families.
"""

import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
GEP_BASE             = "/Users/catepiacentini/Desktop/tesi/Master_Thesis/INDEX/index_new_final/INDEX_50"
GPR_DAILY_PATH       = "/Users/catepiacentini/Desktop/tesi/literature/data_gpr_daily_recent.xls"
GPR_MONTHLY_PATH     = "/Users/catepiacentini/Desktop/tesi/literature/data_gpr_export.xls"
FACTORS_DAILY_PATH   = "/Users/catepiacentini/Desktop/tesi/literature/[usa]_[all_factors]_[daily]_[vw_cap].csv"
FACTORS_MONTHLY_PATH = "/Users/catepiacentini/Desktop/tesi/literature/[usa]_[all_factors]_[monthly]_[vw_cap].csv"
OUTPUT_DIR = GEP_BASE

# ── Factor rename map ──────────────────────────────────────────────────────────
FACTOR_RENAME = {
    "market_equity":   "Size (SMB)",
    "be_me":           "Book-to-Market (HML)",
    "ope_be":          "Operating Profitability (RMW)",
    "at_gr1":          "Asset Growth (CMA)",
    "ret_60_12":       "Long-Term Reversals",
    "ivol_ff3_21d":    "Residual Variance (RVAR)",
    "qmj":             "Quality Minus Junk (QMJ)",
    "betabab_1260d":   "Low Beta (BAB)",
    "ami_126d":        "Amihud Illiquidity",
    "age":             "Firm Age",
    "prc":             "Nominal Price",
    "dolvol_126d":     "High Volume Premium",
    "gp_at":           "Gross Profitability",
    "ni_be":           "Return on Equity",
    "niq_at":          "Return on Assets",
    "ebit_sale":       "Profit Margin",
    "at_turnover":     "Change in Asset Turnover",
    "oaccruals_at":    "Accruals Factor",
    "noa_at":          "Net Operating Assets",
    "cowc_gr1a":       "Net Working Capital Changes",
    "ocf_me":          "Cash Flow to Price",
    "ni_me":           "Earnings to Price",
    "ebitda_mev":      "Enterprise Multiple",
    "sale_me":         "Sales to Price",
    "inv_gr1":         "Growth in Inventory",
    "sale_gr1":        "Sales Growth",
    "dsale_dinv":      "Growth in Sales/Inventory",
    "capex_abn":       "Abnormal Investment",
    "capx_gr1":        "CAPX Growth Rate",
    "dbnetis_at":      "Debt Issuance Factor",
    "at_be":           "Leverage Factor",
    "chcsho_12m":      "1-Year Share Issuance",
    "netis_at":        "Total External Financing",
    "o_score":         "Ohlson O-Score",
    "z_score":         "Altman Z-Score",
    "f_score":         "Piotroski F-Score",
}

# ── Significance stars ─────────────────────────────────────────────────────────
def stars(pval):
    if pval < 0.01:   return "***"
    elif pval < 0.05: return "**"
    elif pval < 0.10: return "*"
    return ""

# ── 1. Load factor files ───────────────────────────────────────────────────────
def load_factors(path, freq="monthly"):
    print(f"Loading {freq} factors...")
    raw  = pd.read_csv(path, usecols=["date", "name", "ret"])
    wide = raw.pivot_table(index="date", columns="name", values="ret", aggfunc="first")
    wide.columns.name = None
    wide.index = pd.to_datetime(wide.index)
    wide = wide.sort_index()
    wide = wide[wide.index >= "1963-01-01"]
    keep_cols = [c for c in FACTOR_RENAME if c in wide.columns]
    wide = wide[keep_cols].rename(columns=FACTOR_RENAME)
    print(f"  → {wide.shape[1]} factors | {wide.shape[0]} obs "
          f"({wide.index[0].date()} – {wide.index[-1].date()})")
    return wide

factors_d = load_factors(FACTORS_DAILY_PATH,   freq="daily")
factors_m = load_factors(FACTORS_MONTHLY_PATH, freq="monthly")

# ── 2. Load GEP ───────────────────────────────────────────────────────────────
daily_gep = pd.read_csv(os.path.join(GEP_BASE, "data", "GEP_Daily_Robust_min2.csv"))
daily_gep["date"] = pd.to_datetime(daily_gep["date"])
daily_gep = daily_gep.set_index("date").sort_index()

monthly_gep = pd.read_csv(os.path.join(GEP_BASE, "data", "GEP_Monthly_Robust_min2.csv"))
monthly_gep["month"] = pd.to_datetime(monthly_gep["month"])
monthly_gep = monthly_gep.set_index("month").sort_index()

# ── 3. Load GPR ───────────────────────────────────────────────────────────────
gpr_d_raw = pd.read_excel(GPR_DAILY_PATH)
gpr_d_raw["date"] = pd.to_datetime(gpr_d_raw["date"], dayfirst=True)
gpr_daily = (gpr_d_raw[["date", "GPRD"]]
             .rename(columns={"GPRD": "GPR"})
             .set_index("date").sort_index())

gpr_m_raw = pd.read_excel(GPR_MONTHLY_PATH)
gpr_m_raw["month"] = pd.to_datetime(gpr_m_raw["month"], dayfirst=True)
gpr_monthly = (gpr_m_raw[["month", "GPR"]]
               .set_index("month").sort_index())

# ── 4. First-differenced (delta) series ───────────────────────────────────────
dgep_daily             = daily_gep[["GEP_daily"]].copy()
dgep_daily["GEP_daily"] = dgep_daily["GEP_daily"].diff()
dgep_daily             = dgep_daily.dropna()

dgpr_daily             = gpr_daily.copy()
dgpr_daily["GPR"]      = dgpr_daily["GPR"].diff()
dgpr_daily             = dgpr_daily.dropna()

dgep_monthly              = monthly_gep[["GEP_monthly"]].copy()
dgep_monthly["GEP_monthly"] = dgep_monthly["GEP_monthly"].diff()
dgep_monthly              = dgep_monthly.dropna()

dgpr_monthly           = gpr_monthly.copy()
dgpr_monthly["GPR"]    = dgpr_monthly["GPR"].diff()
dgpr_monthly           = dgpr_monthly.dropna()

# ── Helper ────────────────────────────────────────────────────────────────────
def to_month_start(s):
    s = s.copy()
    s.index = pd.to_datetime(s.index).to_period("M").to_timestamp()
    return s

# ── 5. Regression engine ──────────────────────────────────────────────────────
def run_factor_regressions(gep_series, factor_df, gpr_ctrl, monthly=False):
    """
    y = factor return
    X = const + GEP_pct + GPR
    Controls: GPR only (MktRF excluded — market_equity is itself a factor).
    Works for both levels and delta series; caller passes the right inputs.
    """
    results = []
    if monthly:
        gep_series = to_month_start(gep_series)
        factor_df  = factor_df.copy()
        factor_df.index = pd.to_datetime(factor_df.index).to_period("M").to_timestamp()
        gpr_ctrl   = to_month_start(gpr_ctrl)

    for factor in factor_df.columns:
        df = pd.concat([factor_df[factor], gep_series, gpr_ctrl], axis=1).dropna()
        if len(df) < 10:
            print(f"  [SKIP] {factor}: only {len(df)} overlapping obs")
            continue
        df.columns = ["FAC", "GEP", "GPR"]
        df["gep_pct"] = df["GEP"] * 100

        # Contemporaneous
        X1 = sm.add_constant(df[["gep_pct", "GPR"]])
        m1 = sm.OLS(df["FAC"], X1).fit(cov_type="HC3")

        # Predictive
        df["gep_lag"] = df["gep_pct"].shift(1)
        df_lag = df.dropna()
        X2 = sm.add_constant(df_lag[["gep_lag", "GPR"]])
        m2 = sm.OLS(df_lag["FAC"], X2).fit(cov_type="HC3")

        results.append({
            "Factor":       factor,
            "N":            int(m1.nobs),
            "Contemp_Beta": m1.params["gep_pct"],
            "Contemp_SE":   m1.bse["gep_pct"],
            "Contemp_Pval": m1.pvalues["gep_pct"],
            "Predic_Beta":  m2.params["gep_lag"],
            "Predic_SE":    m2.bse["gep_lag"],
            "Predic_Pval":  m2.pvalues["gep_lag"],
            "R2_Contemp":   m1.rsquared,
            "R2_Predic":    m2.rsquared,
        })
    return pd.DataFrame(results)

# ── 6. Run all regressions ────────────────────────────────────────────────────
print("\nRunning LEVELS regressions — Daily...")
res_d = run_factor_regressions(
    daily_gep["GEP_daily"], factors_d, gpr_daily, monthly=False)

print("Running LEVELS regressions — Monthly...")
res_m = run_factor_regressions(
    monthly_gep["GEP_monthly"], factors_m, gpr_monthly, monthly=True)

print("Running DELTA regressions — Daily...")
res_d_dlt = run_factor_regressions(
    dgep_daily["GEP_daily"], factors_d, dgpr_daily, monthly=False)

print("Running DELTA regressions — Monthly...")
res_m_dlt = run_factor_regressions(
    dgep_monthly["GEP_monthly"], factors_m, dgpr_monthly, monthly=True)

# ── 7. Print tables ───────────────────────────────────────────────────────────
def print_results(res, label, controls_note):
    print(f"\n{'='*110}")
    print(f" REGRESSION RESULTS: {label}")
    print(f" Controls: {controls_note}")
    print(f" Significance: *** p<0.01  ** p<0.05  * p<0.10   |   SE: HC3")
    print(f"{'='*110}")
    df = res.copy()
    df["Contemp"]    = df.apply(
        lambda r: f"{r['Contemp_Beta']:+.6f}{stars(r['Contemp_Pval'])}  (p={r['Contemp_Pval']:.4f})", axis=1)
    df["Predictive"] = df.apply(
        lambda r: f"{r['Predic_Beta']:+.6f}{stars(r['Predic_Pval'])}  (p={r['Predic_Pval']:.4f})", axis=1)
    df["R²_Contemp"] = df["R2_Contemp"].map("{:.2%}".format)
    df["N"]          = df["N"].astype(str)
    print(df[["Factor", "N", "Contemp", "Predictive", "R²_Contemp"]].to_string(index=False))

print_results(res_d,     "DAILY   — LEVELS  (GEP, GPR)",   "GPR_daily (levels)")
print_results(res_m,     "MONTHLY — LEVELS  (GEP, GPR)",   "GPR_monthly (levels)")
print_results(res_d_dlt, "DAILY   — DELTA   (ΔGEP, ΔGPR)", "ΔGPR_daily")
print_results(res_m_dlt, "MONTHLY — DELTA   (ΔGEP, ΔGPR)", "ΔGPR_monthly")

# ── 8. Plot function ──────────────────────────────────────────────────────────
DARK_BLUE  = "#1a3a5c"
LIGHT_BLUE = "#a8c4e0"
SIG_LEVEL  = 0.10

def plot_factor_exposure(df, beta_col, pval_col, title, filename, note=""):
    plot_df = df[["Factor", beta_col, pval_col]].copy()
    raw = plot_df[beta_col] * 10_000
    plot_df["exposure"] = (raw - raw.mean()) / raw.std()
    plot_df["sig"]      = plot_df[pval_col] < SIG_LEVEL
    plot_df = plot_df.sort_values("exposure", ascending=False).reset_index(drop=True)

    colors = [DARK_BLUE if s else LIGHT_BLUE for s in plot_df["sig"]]
    n      = len(plot_df)
    fig_h  = max(8, n * 0.28)

    fig, ax = plt.subplots(figsize=(15, fig_h))
    ax.barh(range(n), plot_df["exposure"],
            color=colors, edgecolor="white", linewidth=0.3, height=0.72)
    ax.set_yticks([])
    ax.invert_yaxis()

    x_min, x_max = plot_df["exposure"].min(), plot_df["exposure"].max()
    pad = (x_max - x_min) * 0.015
    for i, (val, name) in enumerate(zip(plot_df["exposure"], plot_df["Factor"])):
        if val >= 0:
            ax.text(val + pad, i, name, va="center", ha="left",  fontsize=7.5, color="black")
        else:
            ax.text(val - pad, i, name, va="center", ha="right", fontsize=7.5, color="black")

    ax.set_xlim(x_min - (x_max - x_min) * 0.30,
                x_max + (x_max - x_min) * 0.30)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("GEP Exposure (standardised β ×10 000 bps)", fontsize=10)
    ax.set_title(title, fontsize=12, pad=12)
    ax.legend(handles=[
        mpatches.Patch(color=DARK_BLUE,  label=f"Significant (p < {SIG_LEVEL})"),
        mpatches.Patch(color=LIGHT_BLUE, label=f"Not significant (p ≥ {SIG_LEVEL})"),
    ], loc="lower right", fontsize=8)
    if note:
        fig.text(0.5, 0.01, note, ha="center", fontsize=7.5, style="italic")
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(filename, dpi=200, bbox_inches="tight")
    print(f"Saved: {filename}")
    plt.close()

# ── 9. Generate plots ─────────────────────────────────────────────────────────
base_note_lev = "Controls: GPR only (levels). Beta of factor return on GEP index (×10 000, standardised). SE: HC3."
base_note_dlt = "Controls: ΔGPR only. Beta of factor return on ΔGEP (day-over-day change, ×10 000, standardised). SE: HC3."

# — LEVELS (original names unchanged) —
plot_factor_exposure(
    res_d, "Contemp_Beta", "Contemp_Pval",
    title    = "GEP Exposure by Factor — Daily Contemporaneous",
    filename = os.path.join(OUTPUT_DIR, "GEP_Factor_Contemp_Daily.png"),
    note     = f"Daily — {base_note_lev}",
)
plot_factor_exposure(
    res_d, "Predic_Beta", "Predic_Pval",
    title    = "GEP Exposure by Factor — Daily Predictive (Lagged GEP)",
    filename = os.path.join(OUTPUT_DIR, "GEP_Factor_Predic_Daily.png"),
    note     = f"Daily — {base_note_lev}",
)

# — DELTA (new plots) —
plot_factor_exposure(
    res_d_dlt, "Contemp_Beta", "Contemp_Pval",
    title    = "ΔGEP Exposure by Factor — Daily Contemporaneous  [FIRST DIFFERENCES]",
    filename = os.path.join(OUTPUT_DIR, "GEP_Factor_Contemp_Daily_DELTA.png"),
    note     = f"Daily — {base_note_dlt}",
)
plot_factor_exposure(
    res_d_dlt, "Predic_Beta", "Predic_Pval",
    title    = "ΔGEP Exposure by Factor — Daily Predictive (Lagged ΔGEP)  [FIRST DIFFERENCES]",
    filename = os.path.join(OUTPUT_DIR, "GEP_Factor_Predic_Daily_DELTA.png"),
    note     = f"Daily — {base_note_dlt}",
)

print("\nAll done.")