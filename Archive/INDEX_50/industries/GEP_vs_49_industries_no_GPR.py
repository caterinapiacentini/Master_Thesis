#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Paths ──────────────────────────────────────────────────────────────────────
GEP_BASE = "/Users/catepiacentini/Desktop/tesi/Master_Thesis/INDEX/index_new_final/INDEX_50"

# ── 1. Load GEP Index ──────────────────────────────────────────────────────────
daily_gep = pd.read_csv(os.path.join(GEP_BASE, "data", "GEP_Daily_Robust_min2.csv"))
daily_gep["date"] = pd.to_datetime(daily_gep["date"])
daily_gep = daily_gep.set_index("date").sort_index()

monthly_gep = pd.read_csv(os.path.join(GEP_BASE, "data", "GEP_Monthly_Robust_min2.csv"))
monthly_gep["month"] = pd.to_datetime(monthly_gep["month"])
monthly_gep = monthly_gep.set_index("month").sort_index()

# ── 2. Download Fama-French Data ───────────────────────────────────────────────
print("Downloading Fama-French Industry Portfolios and Factors...")

# Daily
ind_d = web.DataReader('49_Industry_Portfolios_Daily', 'famafrench', start='1990-01-01')[0] / 100.0
ff3_d = web.DataReader('F-F_Research_Data_Factors_daily', 'famafrench', start='1990-01-01')[0] / 100.0

# Monthly
ind_m = web.DataReader('49_Industry_Portfolios', 'famafrench', start='1990-01-01')[0] / 100.0
ind_m.index = ind_m.index.to_timestamp()

ff3_m = web.DataReader('F-F_Research_Data_Factors', 'famafrench', start='1990-01-01')[0] / 100.0
ff3_m.index = ff3_m.index.to_timestamp()

# ── 3. FF49 full name mapping ──────────────────────────────────────────────────
FF49_NAMES = {
    'Agric': 'Agriculture',            'Food':  'Food Products',
    'Soda':  'Candy & Soda',           'Beer':  'Beer & Liquor',
    'Smoke': 'Tobacco Products',       'Toys':  'Recreation',
    'Fun':   'Entertainment',          'Books': 'Printing & Publishing',
    'Hshld': 'Consumer Goods',         'Clths': 'Apparel',
    'Hlth':  'Healthcare',             'MedEq': 'Medical Equipment',
    'Drugs': 'Pharmaceutical Products','Chems': 'Chemicals',
    'Rubbr': 'Rubber & Plastic',       'Txtls': 'Textiles',
    'BldMt': 'Construction Materials', 'Cnstr': 'Construction',
    'Steel': 'Steel Works',            'FabPr': 'Fabricated Products',
    'Mach':  'Machinery',              'ElcEq': 'Electrical Equipment',
    'Autos': 'Automobiles & Trucks',   'Aero':  'Aircraft',
    'Ships': 'Shipbuilding & Railroad Equip.', 'Guns': 'Defense',
    'Gold':  'Precious Metals & Mining','Mines': 'Industrial Metal Mining',
    'Coal':  'Coal',                   'Oil':   'Petroleum & Natural Gas',
    'Util':  'Utilities',              'Telcm': 'Telecommunications',
    'PerSv': 'Personal Services',      'BusSv': 'Business Services',
    'Hardw': 'Computers & Hardware',   'Softw': 'Computer Software',
    'Chips': 'Electronic Equipment',   'LabEq': 'Lab Equipment',
    'Paper': 'Paper & Paper Products', 'Boxes': 'Shipping Containers',
    'Trans': 'Transportation',         'Whlsl': 'Wholesale',
    'Rtail': 'Retail',                 'Meals': 'Restaurants, Hotels & Motels',
    'Banks': 'Banking',                'Insur': 'Insurance',
    'RlEst': 'Real Estate',            'Fin':   'Finance',
    'Other': 'Other',
}

def full_name(short):
    return FF49_NAMES.get(short.strip(), short.strip())

# ── 4. Significance stars helper ───────────────────────────────────────────────
def stars(pval):
    if pval < 0.01:   return '***'
    elif pval < 0.05: return '**'
    elif pval < 0.10: return '*'
    return ''

# ── 5. Regression engines ──────────────────────────────────────────────────────
def run_regressions_daily(gep_series, ind_df, ff_df):
    """Daily regression: controls = FF3 only"""
    results = []
    for industry in ind_df.columns:
        df = pd.concat([ind_df[industry], gep_series, ff_df], axis=1).dropna()
        df.columns = ['RET', 'GEP', 'MktRF', 'SMB', 'HML', 'RF']

        df['y']       = df['RET'] - df['RF']
        df['gep_pct'] = df['GEP'] * 100

        controls = ['MktRF', 'SMB', 'HML']

        # Contemporaneous
        X1 = sm.add_constant(df[['gep_pct'] + controls])
        m1 = sm.OLS(df['y'], X1).fit(cov_type='HC3')

        # Predictive (lagged GEP)
        df['gep_lag'] = df['gep_pct'].shift(1)
        df_lag = df.dropna()
        X2 = sm.add_constant(df_lag[['gep_lag'] + controls])
        m2 = sm.OLS(df_lag['y'], X2).fit(cov_type='HC3')

        results.append({
            'Industry':     industry.strip(),
            'Contemp_Beta': m1.params['gep_pct'],
            'Contemp_Pval': m1.pvalues['gep_pct'],
            'Predic_Beta':  m2.params['gep_lag'],
            'Predic_Pval':  m2.pvalues['gep_lag'],
            'R2':           m1.rsquared,
        })
    return pd.DataFrame(results)


def run_regressions_monthly(gep_series, ind_df, ff_df):
    """Monthly regression: controls = FF3 only"""
    results = []
    for industry in ind_df.columns:
        df = pd.concat([ind_df[industry], gep_series, ff_df], axis=1).dropna()
        df.columns = ['RET', 'GEP', 'MktRF', 'SMB', 'HML', 'RF']

        df['y']       = df['RET'] - df['RF']
        df['gep_pct'] = df['GEP'] * 100

        controls = ['MktRF', 'SMB', 'HML']

        # Contemporaneous
        X1 = sm.add_constant(df[['gep_pct'] + controls])
        m1 = sm.OLS(df['y'], X1).fit(cov_type='HC3')

        # Predictive (lagged GEP)
        df['gep_lag'] = df['gep_pct'].shift(1)
        df_lag = df.dropna()
        X2 = sm.add_constant(df_lag[['gep_lag'] + controls])
        m2 = sm.OLS(df_lag['y'], X2).fit(cov_type='HC3')

        results.append({
            'Industry':     industry.strip(),
            'Contemp_Beta': m1.params['gep_pct'],
            'Contemp_Pval': m1.pvalues['gep_pct'],
            'Predic_Beta':  m2.params['gep_lag'],
            'Predic_Pval':  m2.pvalues['gep_lag'],
            'R2':           m1.rsquared,
        })
    return pd.DataFrame(results)

# ── 6. Run regressions ─────────────────────────────────────────────────────────
res_d = run_regressions_daily(daily_gep["GEP_daily"], ind_d, ff3_d)
res_m = run_regressions_monthly(monthly_gep["GEP_monthly"], ind_m, ff3_m)

# ── 7. Print summary results ───────────────────────────────────────────────────
def print_results(res, label, controls_note):
    print(f"\n{'='*100}")
    print(f" REGRESSION RESULTS: {label}")
    print(f" Controls: {controls_note}")
    print(f" Significance: *** p<0.01  ** p<0.05  * p<0.10   |   SE: HC3")
    print(f"{'='*100}")
    df = res.copy()
    df['Contemp_Beta'] = df.apply(lambda r: f"{r['Contemp_Beta']:+.6f}{stars(r['Contemp_Pval'])}", axis=1)
    df['Contemp_Pval'] = df['Contemp_Pval'].map('{:.4f}'.format)
    df['Predic_Beta']  = df.apply(lambda r: f"{r['Predic_Beta']:+.6f}{stars(r['Predic_Pval'])}", axis=1)
    df['Predic_Pval']  = df['Predic_Pval'].map('{:.4f}'.format)
    df['R2']           = df['R2'].map('{:.2%}'.format)
    print(df[['Industry', 'Contemp_Beta', 'Contemp_Pval',
              'Predic_Beta', 'Predic_Pval', 'R2']].to_string(index=False))

print_results(res_d, "DAILY INDUSTRY PORTFOLIOS",   "FF3 (MktRF, SMB, HML)")
print_results(res_m, "MONTHLY INDUSTRY PORTFOLIOS", "FF3 (MktRF, SMB, HML)")

# ── 8. Full coefficient tables: Daily (Contemp + Predictive) ──────────────────
print(f"\n{'='*100}")
print(" FULL COEFFICIENTS: DAILY CONTEMPORANEOUS")
print(f"{'='*100}")

for industry in ind_d.columns:
    df = pd.concat([ind_d[industry], daily_gep["GEP_daily"], ff3_d], axis=1).dropna()
    df.columns = ['RET', 'GEP', 'MktRF', 'SMB', 'HML', 'RF']
    df['y']       = df['RET'] - df['RF']
    df['gep_pct'] = df['GEP'] * 100
    X = sm.add_constant(df[['gep_pct', 'MktRF', 'SMB', 'HML']])
    m = sm.OLS(df['y'], X).fit(cov_type='HC3')
    print(f"\n--- {industry.strip()} ---")
    summary = pd.DataFrame({'Beta': m.params, 'Pval': m.pvalues})
    summary['Beta_str'] = summary.apply(lambda r: f"{r['Beta']:+.6f}{stars(r['Pval'])}", axis=1)
    summary['Pval_str'] = summary['Pval'].map('{:.4f}'.format)
    print(summary[['Beta_str', 'Pval_str']].to_string())

print(f"\n{'='*100}")
print(" FULL COEFFICIENTS: DAILY PREDICTIVE")
print(f"{'='*100}")

for industry in ind_d.columns:
    df = pd.concat([ind_d[industry], daily_gep["GEP_daily"], ff3_d], axis=1).dropna()
    df.columns = ['RET', 'GEP', 'MktRF', 'SMB', 'HML', 'RF']
    df['y']       = df['RET'] - df['RF']
    df['gep_pct'] = df['GEP'] * 100
    df['gep_lag'] = df['gep_pct'].shift(1)
    df = df.dropna()
    X = sm.add_constant(df[['gep_lag', 'MktRF', 'SMB', 'HML']])
    m = sm.OLS(df['y'], X).fit(cov_type='HC3')
    print(f"\n--- {industry.strip()} ---")
    summary = pd.DataFrame({'Beta': m.params, 'Pval': m.pvalues})
    summary['Beta_str'] = summary.apply(lambda r: f"{r['Beta']:+.6f}{stars(r['Pval'])}", axis=1)
    summary['Pval_str'] = summary['Pval'].map('{:.4f}'.format)
    print(summary[['Beta_str', 'Pval_str']].to_string())

# ── 9. Plot function (daily results only) ─────────────────────────────────────
DARK_BLUE  = '#1a3a5c'
LIGHT_BLUE = '#a8c4e0'
SIG_LEVEL  = 0.10

def plot_exposure(df, beta_col, pval_col, title, filename):
    plot_df = df[['Industry', beta_col, pval_col]].copy()
    plot_df['label'] = plot_df['Industry'].apply(full_name)

    raw = plot_df[beta_col] * 10_000
    plot_df['exposure'] = (raw - raw.mean()) / raw.std()

    plot_df['sig'] = plot_df[pval_col] < SIG_LEVEL
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
            ax.text(val + padding, i, name, va='center', ha='left', fontsize=7.2, color='black')
        else:
            ax.text(val - padding, i, name, va='center', ha='right', fontsize=7.2, color='black')

    ax.set_xlim(x_min - (x_max - x_min) * 0.28, x_max + (x_max - x_min) * 0.28)
    ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_xlabel('Average Exposure (standardised, ×10 000 bps)', fontsize=10)
    ax.set_title(title, fontsize=12, pad=12)

    ax.legend(handles=[
        mpatches.Patch(color=DARK_BLUE,  label=f'Significant (p < {SIG_LEVEL})'),
        mpatches.Patch(color=LIGHT_BLUE, label=f'Not significant (p ≥ {SIG_LEVEL})'),
    ], loc='lower right', fontsize=8)

    fig.text(0.5, 0.01,
             "Note: Beta of daily excess industry return on GEP index (×10 000, standardised). "
             "Controls: FF3 (MktRF, SMB, HML). SE: HC3.",
             ha='center', fontsize=7.5, style='italic')

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.show()

# ── 10. Generate plots (daily only) ───────────────────────────────────────────
plot_exposure(res_d, 'Contemp_Beta', 'Contemp_Pval',
              'Figure: Exposure to GEP by Industry — Daily Contemporaneous',
              os.path.join(GEP_BASE, 'GEP_Industry_Contemp_Daily_no_GPR.png'))

plot_exposure(res_d, 'Predic_Beta', 'Predic_Pval',
              'Figure: Exposure to GEP by Industry — Daily Predictive (Lagged GEP)',
              os.path.join(GEP_BASE, 'GEP_Industry_Predic_Daily_no_GPR.png'))