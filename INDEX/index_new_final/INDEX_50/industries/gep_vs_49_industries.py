#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gep_industry_regressions.py

Regresses FF49 daily/monthly industry excess returns on GEP (Robust min-2),
controlling for FF3 factors and GPR. Runs TWO model families:

  LEVELS  : GEP_t (or GEP_{t-1}) + GPR_t
  DELTA   : ΔGEP_t (or ΔGEP_{t-1}) + ΔGPR_t   ← first differences

Within each family: Contemporaneous AND Predictive (lagged GEP / ΔGEP).

Produces 4 plots (daily levels + daily delta, contemp + predictive):
  GEP_Industry_Contemp_Daily_LEVELS.png
  GEP_Industry_Predic_Daily_LEVELS.png
  GEP_Industry_Contemp_Daily_DELTA.png
  GEP_Industry_Predic_Daily_DELTA.png
"""

import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Paths ──────────────────────────────────────────────────────────────────────
GEP_BASE         = "/Users/catepiacentini/Desktop/tesi/Master_Thesis/INDEX/index_new_final/INDEX_50"
GPR_DAILY_PATH   = "/Users/catepiacentini/Desktop/tesi/literature/data_gpr_daily_recent.xls"
GPR_MONTHLY_PATH = "/Users/catepiacentini/Desktop/tesi/literature/data_gpr_export.xls"

# ── 1. Load GEP Index ──────────────────────────────────────────────────────────
daily_gep = pd.read_csv(os.path.join(GEP_BASE, "data", "GEP_Daily_Robust_min2.csv"))
daily_gep["date"] = pd.to_datetime(daily_gep["date"])
daily_gep = daily_gep.set_index("date").sort_index()

monthly_gep = pd.read_csv(os.path.join(GEP_BASE, "data", "GEP_Monthly_Robust_min2.csv"))
monthly_gep["month"] = pd.to_datetime(monthly_gep["month"])
monthly_gep = monthly_gep.set_index("month").sort_index()

# ── 2. Load GPR Controls ───────────────────────────────────────────────────────
gpr_d_raw = pd.read_excel(GPR_DAILY_PATH)
gpr_d_raw['date'] = pd.to_datetime(gpr_d_raw['date'], dayfirst=True)
gpr_daily = (gpr_d_raw[['date', 'GPRD']]
             .rename(columns={'GPRD': 'GPR_daily'})
             .set_index('date').sort_index())

gpr_m_raw = pd.read_excel(GPR_MONTHLY_PATH)
gpr_m_raw['month'] = pd.to_datetime(gpr_m_raw['month'], dayfirst=True)
gpr_monthly = (gpr_m_raw[['month', 'GPR']]
               .rename(columns={'GPR': 'GPR_monthly'})
               .set_index('month').sort_index())

# ── 3. Compute first-differenced (delta) series ────────────────────────────────
# ΔGEP: daily
dgep_daily = daily_gep[["GEP_daily"]].copy()
dgep_daily["DGEP_daily"] = dgep_daily["GEP_daily"].diff()
dgep_daily = dgep_daily.dropna()

# ΔGPR: daily
dgpr_daily = gpr_daily.copy()
dgpr_daily["DGPR_daily"] = dgpr_daily["GPR_daily"].diff()
dgpr_daily = dgpr_daily.dropna()

# ΔGEP: monthly
dgep_monthly = monthly_gep[["GEP_monthly"]].copy()
dgep_monthly["DGEP_monthly"] = dgep_monthly["GEP_monthly"].diff()
dgep_monthly = dgep_monthly.dropna()

# ΔGPR: monthly
dgpr_monthly = gpr_monthly.copy()
dgpr_monthly["DGPR_monthly"] = dgpr_monthly["GPR_monthly"].diff()
dgpr_monthly = dgpr_monthly.dropna()

# ── 4. Download Fama-French Data ───────────────────────────────────────────────
print("Downloading Fama-French Industry Portfolios and Factors...")

ind_d   = web.DataReader('49_Industry_Portfolios_Daily',    'famafrench', start='1990-01-01')[0] / 100.0
ff3_d   = web.DataReader('F-F_Research_Data_Factors_daily', 'famafrench', start='1990-01-01')[0] / 100.0

ind_m_raw = web.DataReader('49_Industry_Portfolios',   'famafrench', start='1990-01-01')[0] / 100.0
ff3_m_raw = web.DataReader('F-F_Research_Data_Factors','famafrench', start='1990-01-01')[0] / 100.0

ind_m = ind_m_raw.copy(); ind_m.index = ind_m.index.to_timestamp()
ff3_m = ff3_m_raw.copy(); ff3_m.index = ff3_m.index.to_timestamp()

# ── 5. FF49 full name mapping ──────────────────────────────────────────────────
FF49_NAMES = {
    'Agric': 'Agriculture',           'Food':  'Food Products',
    'Soda':  'Candy & Soda',          'Beer':  'Beer & Liquor',
    'Smoke': 'Tobacco Products',      'Toys':  'Recreation',
    'Fun':   'Entertainment',         'Books': 'Printing & Publishing',
    'Hshld': 'Consumer Goods',        'Clths': 'Apparel',
    'Hlth':  'Healthcare',            'MedEq': 'Medical Equipment',
    'Drugs': 'Pharmaceutical Products','Chems': 'Chemicals',
    'Rubbr': 'Rubber & Plastic',      'Txtls': 'Textiles',
    'BldMt': 'Construction Materials','Cnstr': 'Construction',
    'Steel': 'Steel Works',           'FabPr': 'Fabricated Products',
    'Mach':  'Machinery',             'ElcEq': 'Electrical Equipment',
    'Autos': 'Automobiles & Trucks',  'Aero':  'Aircraft',
    'Ships': 'Shipbuilding & Railroad Equip.', 'Guns': 'Defense',
    'Gold':  'Precious Metals & Mining','Mines':'Industrial Metal Mining',
    'Coal':  'Coal',                  'Oil':   'Petroleum & Natural Gas',
    'Util':  'Utilities',             'Telcm': 'Telecommunications',
    'PerSv': 'Personal Services',     'BusSv': 'Business Services',
    'Hardw': 'Computers & Hardware',  'Softw': 'Computer Software',
    'Chips': 'Electronic Equipment',  'LabEq': 'Lab Equipment',
    'Paper': 'Paper & Paper Products','Boxes': 'Shipping Containers',
    'Trans': 'Transportation',        'Whlsl': 'Wholesale',
    'Rtail': 'Retail',                'Meals': 'Restaurants, Hotels & Motels',
    'Banks': 'Banking',               'Insur': 'Insurance',
    'RlEst': 'Real Estate',           'Fin':   'Finance',
    'Other': 'Other',
}

def full_name(short):
    return FF49_NAMES.get(short.strip(), short.strip())

# ── 6. Helpers ─────────────────────────────────────────────────────────────────
def stars(pval):
    if pval < 0.01:   return '***'
    elif pval < 0.05: return '**'
    elif pval < 0.10: return '*'
    return ''

# ── 7. Generic regression engine ──────────────────────────────────────────────
def run_regressions(ind_df, ff_df,
                    gep_col,  gep_series,
                    gpr_col,  gpr_series,
                    freq_label="daily"):
    """
    Runs contemp + predictive OLS for every industry.

    Parameters
    ----------
    gep_col    : name of the GEP variable in the merged df (e.g. 'gep_pct' or 'dgep_pct')
    gep_series : pd.DataFrame with that column
    gpr_col    : name of the GPR control column
    gpr_series : pd.DataFrame with that column
    """
    results = []
    controls = ['MktRF', 'SMB', 'HML', gpr_col]

    for industry in ind_df.columns:
        pieces = [ind_df[[industry]], gep_series, ff_df, gpr_series]
        df = pd.concat(pieces, axis=1).dropna()

        # Rename columns predictably
        df.columns = ['RET'] + list(gep_series.columns) + ['MktRF', 'SMB', 'HML', 'RF'] + list(gpr_series.columns)
        df['y']       = df['RET'] - df['RF']
        df['gep_var'] = df[gep_series.columns[0]] * 100   # scale to pct
        df['gpr_var'] = df[gpr_series.columns[0]]

        # Contemporaneous
        X1 = sm.add_constant(df[['gep_var', 'MktRF', 'SMB', 'HML', 'gpr_var']])
        m1 = sm.OLS(df['y'], X1).fit(cov_type='HC3')

        # Predictive: lagged GEP variable
        df['gep_lag'] = df['gep_var'].shift(1)
        df_lag = df.dropna()
        X2 = sm.add_constant(df_lag[['gep_lag', 'MktRF', 'SMB', 'HML', 'gpr_var']])
        m2 = sm.OLS(df_lag['y'], X2).fit(cov_type='HC3')

        results.append({
            'Industry':     industry.strip(),
            'Contemp_Beta': m1.params['gep_var'],
            'Contemp_Pval': m1.pvalues['gep_var'],
            'Predic_Beta':  m2.params['gep_lag'],
            'Predic_Pval':  m2.pvalues['gep_lag'],
            'R2_contemp':   m1.rsquared,
            'R2_predic':    m2.rsquared,
        })

    return pd.DataFrame(results)

# ── 8. Run all four model families ────────────────────────────────────────────
print("\nRunning LEVELS regressions — Daily...")
res_d_lev = run_regressions(
    ind_d, ff3_d,
    gep_col='GEP_daily',  gep_series=daily_gep[["GEP_daily"]],
    gpr_col='GPR_daily',  gpr_series=gpr_daily[["GPR_daily"]],
    freq_label="daily"
)

print("Running DELTA regressions — Daily...")
res_d_dlt = run_regressions(
    ind_d, ff3_d,
    gep_col='DGEP_daily', gep_series=dgep_daily[["DGEP_daily"]],
    gpr_col='DGPR_daily', gpr_series=dgpr_daily[["DGPR_daily"]],
    freq_label="daily"
)

print("Running LEVELS regressions — Monthly...")
res_m_lev = run_regressions(
    ind_m, ff3_m,
    gep_col='GEP_monthly',  gep_series=monthly_gep[["GEP_monthly"]],
    gpr_col='GPR_monthly',  gpr_series=gpr_monthly[["GPR_monthly"]],
    freq_label="monthly"
)

print("Running DELTA regressions — Monthly...")
res_m_dlt = run_regressions(
    ind_m, ff3_m,
    gep_col='DGEP_monthly', gep_series=dgep_monthly[["DGEP_monthly"]],
    gpr_col='DGPR_monthly', gpr_series=dgpr_monthly[["DGPR_monthly"]],
    freq_label="monthly"
)

# ── 9. Print results ───────────────────────────────────────────────────────────
def print_results(res, label, controls_note):
    print(f"\n{'='*110}")
    print(f" REGRESSION RESULTS: {label}")
    print(f" Controls: {controls_note}")
    print(f" Significance: *** p<0.01  ** p<0.05  * p<0.10   |   SE: HC3")
    print(f"{'='*110}")
    df = res.copy()
    df['Contemp_Beta'] = df.apply(lambda r: f"{r['Contemp_Beta']:+.6f}{stars(r['Contemp_Pval'])}", axis=1)
    df['Contemp_Pval'] = df['Contemp_Pval'].map('{:.4f}'.format)
    df['Predic_Beta']  = df.apply(lambda r: f"{r['Predic_Beta']:+.6f}{stars(r['Predic_Pval'])}", axis=1)
    df['Predic_Pval']  = df['Predic_Pval'].map('{:.4f}'.format)
    df['R2_contemp']   = df['R2_contemp'].map('{:.2%}'.format)
    df['R2_predic']    = df['R2_predic'].map('{:.2%}'.format)
    print(df[['Industry', 'Contemp_Beta', 'Contemp_Pval',
              'Predic_Beta', 'Predic_Pval', 'R2_contemp', 'R2_predic']].to_string(index=False))

print_results(res_d_lev, "DAILY — LEVELS  (GEP, GPR)",  "FF3 + GPR_daily (levels)")
print_results(res_d_dlt, "DAILY — DELTA   (ΔGEP, ΔGPR)", "FF3 + ΔGPR_daily")
print_results(res_m_lev, "MONTHLY — LEVELS  (GEP, GPR)",  "FF3 + GPR_monthly (levels)")
print_results(res_m_dlt, "MONTHLY — DELTA   (ΔGEP, ΔGPR)", "FF3 + ΔGPR_monthly")

# ── 10. Plot function ─────────────────────────────────────────────────────────
DARK_BLUE  = '#1a3a5c'
LIGHT_BLUE = '#a8c4e0'
SIG_LEVEL  = 0.10

def plot_exposure(df, beta_col, pval_col, title, subtitle_note, filename):
    plot_df = df[['Industry', beta_col, pval_col]].copy()
    plot_df['label'] = plot_df['Industry'].apply(full_name)

    # Standardise betas (×10 000 to convert to bps)
    raw = plot_df[beta_col] * 10_000
    plot_df['exposure'] = (raw - raw.mean()) / raw.std()
    plot_df['sig']      = plot_df[pval_col] < SIG_LEVEL
    plot_df = plot_df.sort_values('exposure', ascending=False).reset_index(drop=True)

    colors = [DARK_BLUE if s else LIGHT_BLUE for s in plot_df['sig']]

    fig, ax = plt.subplots(figsize=(16, 10))
    ax.barh(range(len(plot_df)), plot_df['exposure'],
            color=colors, edgecolor='white', linewidth=0.3, height=0.7)
    ax.set_yticks([])
    ax.invert_yaxis()

    x_min, x_max = plot_df['exposure'].min(), plot_df['exposure'].max()
    padding = (x_max - x_min) * 0.012

    for i, (val, name) in enumerate(zip(plot_df['exposure'], plot_df['label'])):
        if val >= 0:
            ax.text(val + padding, i, name, va='center', ha='left',  fontsize=7.2, color='black')
        else:
            ax.text(val - padding, i, name, va='center', ha='right', fontsize=7.2, color='black')

    ax.set_xlim(x_min - (x_max - x_min) * 0.28,
                x_max + (x_max - x_min) * 0.28)
    ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_xlabel('Average Exposure (standardised, ×10 000 bps)', fontsize=10)
    ax.set_title(title, fontsize=12, pad=12)
    ax.legend(handles=[
        mpatches.Patch(color=DARK_BLUE,  label=f'Significant (p < {SIG_LEVEL})'),
        mpatches.Patch(color=LIGHT_BLUE, label=f'Not significant (p ≥ {SIG_LEVEL})'),
    ], loc='lower right', fontsize=8)

    fig.text(0.5, 0.01, subtitle_note,
             ha='center', fontsize=7.5, style='italic')
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()

# ── 11. Generate all 4 plots ──────────────────────────────────────────────────

# — LEVELS, Contemporaneous —
plot_exposure(
    res_d_lev, 'Contemp_Beta', 'Contemp_Pval',
    title    = 'GEP Exposure by Industry — Daily Contemporaneous  [LEVELS]',
    subtitle_note = (
        "Beta of daily excess industry return on GEP index (×10 000, standardised). "
        "Controls: FF3 (MktRF, SMB, HML) + GPR daily (levels). SE: HC3."
    ),
    filename = os.path.join(GEP_BASE, 'GEP_Industry_Contemp_Daily.png'),
)

# — LEVELS, Predictive —
plot_exposure(
    res_d_lev, 'Predic_Beta', 'Predic_Pval',
    title    = 'GEP Exposure by Industry — Daily Predictive (Lagged GEP)  [LEVELS]',
    subtitle_note = (
        "Beta of daily excess industry return on lagged GEP index (×10 000, standardised). "
        "Controls: FF3 (MktRF, SMB, HML) + GPR daily (levels). SE: HC3."
    ),
    filename = os.path.join(GEP_BASE, 'GEP_Industry_Predic_Daily.png'),
)

# — DELTA, Contemporaneous —
plot_exposure(
    res_d_dlt, 'Contemp_Beta', 'Contemp_Pval',
    title    = 'ΔGEP Exposure by Industry — Daily Contemporaneous  [FIRST DIFFERENCES]',
    subtitle_note = (
        "Beta of daily excess industry return on ΔGEP (day-over-day change, ×10 000, standardised). "
        "Controls: FF3 (MktRF, SMB, HML) + ΔGPR daily. SE: HC3."
    ),
    filename = os.path.join(GEP_BASE, 'GEP_Industry_Contemp_Daily_DELTA.png'),
)

# — DELTA, Predictive —
plot_exposure(
    res_d_dlt, 'Predic_Beta', 'Predic_Pval',
    title    = 'ΔGEP Exposure by Industry — Daily Predictive (Lagged ΔGEP)  [FIRST DIFFERENCES]',
    subtitle_note = (
        "Beta of daily excess industry return on lagged ΔGEP (day-over-day change, ×10 000, standardised). "
        "Controls: FF3 (MktRF, SMB, HML) + ΔGPR daily. SE: HC3."
    ),
    filename = os.path.join(GEP_BASE, 'GEP_Industry_Predic_Daily_DELTA.png'),
)

print("\nDone. All results printed and 4 plots saved to:", GEP_BASE)