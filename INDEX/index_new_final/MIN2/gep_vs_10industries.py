#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
import pandas_datareader.data as web

# ── Paths ──────────────────────────────────────────────────────────────────────
GEP_BASE = "/Users/catepiacentini/Desktop/tesi/Master_Thesis/INDEX/index_new_final/MIN2"

# ── 1. Load your local GEP Index ───────────────────────────────────────────────
monthly_gep = pd.read_csv(os.path.join(GEP_BASE, "GEP_Monthly_Robust_min2.csv"))
monthly_gep["month"] = pd.to_datetime(monthly_gep["month"])
monthly_gep = monthly_gep.set_index("month").sort_index()

daily_gep = pd.read_csv(os.path.join(GEP_BASE, "GEP_Daily_Robust_min2.csv"))
daily_gep["date"] = pd.to_datetime(daily_gep["date"])
daily_gep = daily_gep.set_index("date").sort_index()

# ── 2. Download Fama-French Data (Live) ────────────────────────────────────────
print("Downloading Fama-French Industry Portfolios and Factors...")

# 10 Industry Portfolios (Monthly and Daily)
# [0] usually fetches the Value-Weighted returns table
ind_mo_dict = web.DataReader('10_Industry_Portfolios', 'famafrench', start='1990-01-01')
ind_mo = ind_mo_dict[0] / 100.0
ind_mo.index = ind_mo.index.to_timestamp()

ind_d_dict = web.DataReader('10_Industry_Portfolios_Daily', 'famafrench', start='1990-01-01')
ind_d = ind_d_dict[0] / 100.0

# FF 3 Factors (Monthly and Daily)
ff3_mo = web.DataReader('F-F_Research_Data_Factors', 'famafrench', start='1990-01-01')[0] / 100.0
ff3_mo.index = ff3_mo.index.to_timestamp()

ff3_d = web.DataReader('F-F_Research_Data_Factors_daily', 'famafrench', start='1990-01-01')[0] / 100.0

# ── 3. Regression Engine ───────────────────────────────────────────────────────
def run_industry_regressions(gep_series, ind_df, ff_df, freq_label):
    print(f"\n{'='*85}")
    print(f" REGRESSION ANALYSIS: {freq_label} INDUSTRY PORTFOLIOS (Live Data)")
    print(f"{'='*85}")
    
    results = []
    
    for industry in ind_df.columns:
        # Merge GEP, Industry, and FF3 Factors
        df = pd.concat([ind_df[industry], gep_series, ff_df], axis=1).dropna()
        df.columns = ['RET', 'GEP', 'MktRF', 'SMB', 'HML', 'RF']
        
        # Calculate Excess Returns (y) and Scale GEP for readability
        df['y'] = df['RET'] - df['RF']
        df['gep_pct'] = df['GEP'] * 100
        
        # Model: Contemporaneous
        X1 = sm.add_constant(df[['gep_pct', 'MktRF', 'SMB', 'HML']])
        m1 = sm.OLS(df['y'], X1).fit()
        
        # Model: Predictive (Lagged GEP)
        df['gep_lag'] = df['gep_pct'].shift(1)
        df_lag = df.dropna()
        X2 = sm.add_constant(df_lag[['gep_lag', 'MktRF', 'SMB', 'HML']])
        m2 = sm.OLS(df_lag['y'], X2).fit()
        
        results.append({
            'Industry': industry.strip(),
            'Contemp_Beta': m1.params['gep_pct'],
            'Contemp_Pval': m1.pvalues['gep_pct'],
            'Predic_Beta': m2.params['gep_lag'],
            'Predic_Pval': m2.pvalues['gep_lag'],
            'R2': m1.rsquared
        })

    # Summary Table
    res_df = pd.DataFrame(results)
    print(res_df.to_string(index=False, formatters={
        'Contemp_Beta': '{:,.4f}'.format, 'Contemp_Pval': '{:,.4f}'.format,
        'Predic_Beta': '{:,.4f}'.format, 'Predic_Pval': '{:,.4f}'.format,
        'R2': '{:,.2%}'.format
    }))
    return res_df

# ── 4. Run Analysis ────────────────────────────────────────────────────────────
res_mo = run_industry_regressions(monthly_gep["GEP_monthly"], ind_mo, ff3_mo, "MONTHLY")
res_d  = run_industry_regressions(daily_gep["GEP_daily"], ind_d, ff3_d, "DAILY")