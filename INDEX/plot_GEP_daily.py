import pandas as pd
import matplotlib.pyplot as plt

# 1. Load and prepare data (Universal for both plots)
df = pd.read_csv('GEP_Daily_Index.csv')
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# ---------------------------------------------------------
# PLOT 1: LONG-TERM MONTHLY INDEX (1996 - 2025)
# ---------------------------------------------------------
# Resample to Monthly and Scale
monthly_gep = df['score'].resample('ME').mean() * 10000

plt.figure(figsize=(18, 8))
plt.plot(monthly_gep.index, monthly_gep, color='#1f77b4', linewidth=2, label='Monthly GEP Index')

# Long-term Annotations
long_term_anns = [
    ('1998-05-01', 'India-Pakistan Sanctions', 40),
    ('2003-03-01', 'Iraq War / UN Embargo', 50),
    ('2012-01-01', 'Iran Oil Embargo', 45),
    ('2018-03-01', 'Trump Section 232 Tariffs', 60),
    ('2022-02-01', 'Russia-Ukraine Sanctions', 50),
    ('2025-04-02', '"Liberation Day" Tariffs', 70)
]

for date, label, offset in long_term_anns:
    target_date = pd.to_datetime(date)
    y_val = monthly_gep.asof(target_date)
    plt.annotate(label, xy=(target_date, y_val), xytext=(0, offset), 
                 textcoords='offset points', ha='center', fontsize=9, fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color='black', alpha=0.6),
                 bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.8))

plt.title('Monthly Geoeconomic Pressure (GEP) Index: 1996 - 2025', fontsize=16, fontweight='bold')
plt.ylabel('GEP Exposure Score (Scaled x10,000)', fontsize=12)
plt.grid(True, axis='y', linestyle='--', alpha=0.5)
plt.axhline(monthly_gep.mean(), color='red', linestyle=':', alpha=0.5, label='Historical Mean')
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig('GEP_Monthly_LongTerm.png', dpi=300)
plt.show()

# ---------------------------------------------------------
# PLOT 2: HIGH-RESOLUTION 2025 VALIDATION
# ---------------------------------------------------------
# Filter for 2025 only and maintain Daily resolution
gep_2025 = df.loc['2025']['score'] * 10000

plt.figure(figsize=(18, 10))
plt.plot(gep_2025.index, gep_2025, color='#d62728', linewidth=1.5, label='Daily GEP 2025')

# 2025 Specific Annotations (Positive and Negative Frictions)
# Sources: Proclamation 10895, Executive Order 14257, Geneva Truce
# 4. 2025-Specific Annotations (Positive and Negative Frictions)
annotations = [
    ('2025-01-23', 'Shipbuilding Findings', 40),
    ('2025-04-02', '"Liberation Day" Universal Tariffs', 70),
    ('2025-05-12', 'Geneva Tariff Truce (Pause)', -50), # Negative offset for dip
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

plt.title('2025 Case Study: Geoeconomic Pressure Response to Policy Shocks', fontsize=18, fontweight='bold')
plt.ylabel('Daily GEP Exposure Score (Scaled x10,000)', fontsize=14)
plt.xlabel('Date (2025)', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.4)
plt.xlim(pd.to_datetime('2025-01-01'), pd.to_datetime('2025-12-31'))
plt.ylim(0, gep_2025.max() * 1.5)
plt.tight_layout()
plt.savefig('GEP_2025_Daily_DeepDive.png', dpi=300)
plt.show()
