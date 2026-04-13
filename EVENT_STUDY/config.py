"""
Event Study Configuration
=========================
Edit this file to add/remove events and tune window parameters.
Each event needs a date (YYYY-MM-DD) and a short description.
"""

# =============================================================================
# PATHS
# =============================================================================
META_DIR  = "/home/h12429576/master_thesis/clean_txt/INFO_DATA1"
TEXT_DIR  = "/home/h12429576/master_thesis/clean_txt/DATA1"
OUT_DIR   = "/home/h12429576/master_thesis/output/EVENT_STUDY"

# =============================================================================
# EVENT WINDOW PARAMETERS  (in trading days, relative to event date = 0)
# =============================================================================
ESTIMATION_START = -252   # start of estimation window (≈ 1 year before event)
ESTIMATION_END   = -11    # end of estimation window   (leave gap before event)
EVENT_START      = -10    # start of event window
EVENT_END        = +10    # end of event window
MIN_EST_OBS      = 100    # skip event if fewer obs in estimation window

# CAR windows to report in summary table
CAR_WINDOWS = [(-1, 1), (-3, 3), (-5, 5), (-10, 10)]

# Number of days around event date to pull articles for validation
ARTICLE_WINDOW_DAYS = 3

## =============================================================================
# GEOECONOMIC PRESSURE EVENTS (Curated for Event Study)
# =============================================================================
EVENTS = {

    # ── Rare Earth Embargo (Early Geoeconomics) ──────────────────────────────
    "China_RareEarth_Embargo": {
        "date": "2010-09-22",
        "description": "China unofficially halts rare earth mineral exports to Japan amid diplomatic dispute",
        "category": "Export_Controls",
    },

    # ── Dawn of the US-China Trade War ───────────────────────────────────────
    "Sec301_China_Tariffs": {
        "date": "2018-03-22",
        "description": "Trump signs memorandum under Section 301 for tariffs on $50B of Chinese goods",
        "category": "Trade_Coercion",
    },

    # ── Re-weaponization of Secondary Sanctions ──────────────────────────────
    "Iran_JCPOA_Withdrawal": {
        "date": "2018-05-08",
        "description": "US withdraws from JCPOA, threatening secondary sanctions on global allies buying Iran oil",
        "category": "Sanctions",
    },

    # ── The Tech War Begins (Huawei) ─────────────────────────────────────────
    "Huawei_EntityList": {
        "date": "2019-05-16",
        "description": "US adds Huawei to Entity List, banning US technology exports and severing supply chains",
        "category": "Export_Controls",
    },

    # ── The Ultimate Financial Weapon (Russia SWIFT/CBR) ─────────────────────
    "Russia_SWIFT_Sanctions": {
        "date": "2022-02-28", # The Monday market reaction to weekend announcements
        "description": "US/EU freeze Russian Central Bank assets and cut major Russian banks from SWIFT",
        "category": "Sanctions",
    },

    # ── The Semiconductor Chokehold ──────────────────────────────────────────
    "BIS_Chip_Controls": {
        "date": "2022-10-07",
        "description": "Biden administration announces sweeping unilateral semiconductor export controls on China",
        "category": "Export_Controls",
    },

    # ── The Global Tariff Era ────────────────────────────────────────────────
    "Liberation_Day": {
        "date": "2025-04-02",
        "description": "Trump announces 'Liberation Day' sweeping reciprocal tariffs on all global trading partners",
        "category": "Trade_Coercion",
    },
}