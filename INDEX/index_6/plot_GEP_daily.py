import pandas as pd
import matplotlib.pyplot as plt

# 1. Caricamento e preparazione dati GEP
df = pd.read_csv('INDEX/GEP_Daily_Index.csv')
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# --- NORMALIZATION LOGIC ---
# We calculate the global mean of the original score to use as the base for 100
global_mean = df['score'].mean()

# Normalizing the entire dataframe column: (Value / Mean) * 100
df['normalized_score'] = (df['score'] / global_mean) * 100

# Resampling monthly for the long-term plot
monthly_gep = df['normalized_score'].resample('ME').mean()

# ---------------------------------------------------------
# PLOT 1: LONG-TERM MONTHLY INDEX (1996 - 2025)
# ---------------------------------------------------------
plt.figure(figsize=(18, 8))

plt.plot(monthly_gep.index, monthly_gep, color='#1f77b4', linewidth=2, label='Normalized Monthly GEP')

long_term_anns = [
    ('1998-05-01', 'India-Pak Sanctions', 40),
    ('2003-03-01', 'Iraq War / UN Embargo', 50),
    ('2012-01-01', 'Iran Oil Embargo', 45),
    ('2018-03-01', 'Trump Section 232 Tariffs', 60),
    ('2022-02-01', 'Russia-Ukraine Sanctions', 50),
    ('2025-04-02', '"Liberation Day" Tariffs', 70)
]

for date, label, offset in long_term_anns:
    target_date = pd.to_datetime(date)
    if target_date in monthly_gep.index or target_date < monthly_gep.index.max():
        y_val = monthly_gep.asof(target_date)
        plt.annotate(label, xy=(target_date, y_val), xytext=(0, offset), 
                     textcoords='offset points', ha='center', fontsize=9, fontweight='bold',
                     arrowprops=dict(arrowstyle='->', color='black', alpha=0.6),
                     bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.8))

plt.title('Monthly Geoeconomic Pressure (GEP) Index: 1996 - 2025 (Normalized: Mean = 100)', fontsize=16, fontweight='bold')
plt.ylabel('GEP Index (Base 100)', fontsize=12)
plt.xlabel('Year', fontsize=12)
plt.grid(True, axis='y', linestyle='--', alpha=0.5)

# The mean is now 100 by definition
plt.axhline(100, color='red', linestyle=':', alpha=0.7, label='Historical Average (100)')
plt.legend(loc='upper left')

plt.tight_layout()
plt.savefig('GEP_Monthly_LongTerm_Normalized.png', dpi=300)
plt.show()

# ---------------------------------------------------------
# PLOT 2: HIGH-RESOLUTION 2025 VALIDATION (Daily)
# ---------------------------------------------------------
gep_2025 = df.loc['2025']['normalized_score']

plt.figure(figsize=(18, 10))
plt.plot(gep_2025.index, gep_2025, color='#d62728', linewidth=1.5, label='Daily Normalized GEP 2025')

annotations = [
    ('2025-01-23', 'Shipbuilding Findings', 40),
    ('2025-04-02', '"Liberation Day" Universal Tariffs', 70),
    ('2025-05-12', 'Geneva Tariff Truce (Pause)', -50),
    ('2025-06-23', 'Appliance Tariff Expansion', 30),
    ('2025-07-31', 'Truce Extension (Stockholm)', -40),
    ('2025-11-01', 'Trump-Xi Trade Deal', -60)
]

for date, label, offset in annotations:
    target_date = pd.to_datetime(date)
    y_val = gep_2025.asof(target_date)
    plt.annotate(label, xy=(target_date, y_val), xytext=(0, offset), 
                 textcoords='offset points', ha='center', fontsize=10, fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color='black', alpha=0.7),
                 bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='red' if offset < 0 else 'navy', alpha=0.1))

plt.title('2025 Case Study: Geoeconomic Pressure Response (Normalized: Mean = 100)', fontsize=18, fontweight='bold')
plt.ylabel('GEP Index (Base 100)', fontsize=14)
plt.xlabel('Date (2025)', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.4)

plt.axhline(100, color='black', linestyle='--', alpha=0.3, label='Historical Mean')
plt.xlim(pd.to_datetime('2025-01-01'), pd.to_datetime('2025-12-31'))
plt.ylim(0, gep_2025.max() * 1.3)
plt.legend()

plt.tight_layout()
plt.savefig('GEP_2025_Daily_Normalized.png', dpi=300)
plt.show()