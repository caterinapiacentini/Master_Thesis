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

# =============================================================================
# GEOECONOMIC PRESSURE EVENTS
# =============================================================================
EVENTS = {

    # ── US–China Trade War ──────────────────────────────────────────────────
    "USC_EO_Mar2018": {
        "date": "2018-03-22",
        "description": "Trump signs EO imposing tariffs on ~$60B Chinese goods (Section 301)",
        "category": "Trade_Coercion",
    },
    "USC_Tariffs_Wave1": {
        "date": "2018-07-06",
        "description": "US imposes 25% tariffs on $34B Chinese goods — first wave",
        "category": "Trade_Coercion",
    },
    "USC_Tariffs_Wave2": {
        "date": "2018-08-23",
        "description": "US imposes 25% tariffs on additional $16B Chinese goods",
        "category": "Trade_Coercion",
    },
    "USC_Tariffs_Wave3": {
        "date": "2018-09-24",
        "description": "US imposes 10% tariffs on $200B Chinese goods",
        "category": "Trade_Coercion",
    },
    "USC_Tariffs_Escalation_May2019": {
        "date": "2019-05-10",
        "description": "US raises tariffs from 10% to 25% on $200B Chinese goods",
        "category": "Trade_Coercion",
    },
    "USC_Tariffs_Wave4_Aug2019": {
        "date": "2019-08-01",
        "description": "Trump announces 10% tariffs on remaining $300B Chinese goods",
        "category": "Trade_Coercion",
    },
    "USC_Escalation_Aug2019": {
        "date": "2019-08-23",
        "description": "China retaliatory tariffs + Trump raises tariffs further",
        "category": "Retaliation",
    },

    # ── Huawei / Export Controls ─────────────────────────────────────────────
    "Huawei_EntityList": {
        "date": "2019-05-16",
        "description": "US adds Huawei to Entity List, banning US technology exports",
        "category": "Export_Controls",
    },
    "CHIPS_Act": {
        "date": "2022-08-09",
        "description": "Biden signs CHIPS Act, restricts semiconductor exports to China",
        "category": "Export_Controls",
    },
    "BIS_Oct2022_Chip_Rules": {
        "date": "2022-10-07",
        "description": "BIS sweeping new chip export controls targeting China",
        "category": "Export_Controls",
    },

    # ── Russia–Ukraine & Related Sanctions ──────────────────────────────────
    "Russia_Crimea_Sanctions": {
        "date": "2014-03-17",
        "description": "EU/US first sanctions package on Russia after Crimea annexation",
        "category": "Sanctions",
    },
    "Russia_Invasion_Sanctions": {
        "date": "2022-02-24",
        "description": "Russia invades Ukraine — US/EU/UK announce sweeping sanctions",
        "category": "Sanctions",
    },
    "Russia_SWIFT_Exclusion": {
        "date": "2022-02-26",
        "description": "Western allies exclude Russia from SWIFT messaging system",
        "category": "Financial_Coercion",
    },
    "Russia_Asset_Freeze": {
        "date": "2022-02-28",
        "description": "G7 freeze Russian central bank reserves (~$300B)",
        "category": "Financial_Coercion",
    },
    "Russia_Oil_Ban": {
        "date": "2022-03-11",
        "description": "US bans Russian oil imports; EU begins phased oil embargo",
        "category": "Embargo",
    },
    "Russia_Oil_Price_Cap": {
        "date": "2022-12-05",
        "description": "G7/EU implement $60/barrel price cap on Russian oil",
        "category": "Embargo",
    },

    # ── Iran ─────────────────────────────────────────────────────────────────
    "Iran_JCPOA_Withdrawal": {
        "date": "2018-05-08",
        "description": "US withdraws from JCPOA, reimpose full Iran nuclear sanctions",
        "category": "Sanctions",
    },
    "Iran_Oil_Waivers_End": {
        "date": "2019-04-22",
        "description": "US ends Iran oil sanction waivers, targets zero Iranian exports",
        "category": "Embargo",
    },

    # ── North Korea ──────────────────────────────────────────────────────────
    "DPRK_UNSC_Sanctions": {
        "date": "2017-08-05",
        "description": "UN Security Council toughest-ever sanctions on North Korea",
        "category": "Sanctions",
    },

    # ── Steel / Aluminium Tariffs ────────────────────────────────────────────
    "Steel_Aluminium_Tariffs": {
        "date": "2018-03-08",
        "description": "Trump announces 25% steel / 10% aluminium tariffs globally",
        "category": "Trade_Coercion",
    },

    # ── Liberation Day 2025 ──────────────────────────────────────────────────
    "Liberation_Day": {
        "date": "2025-04-02",
        "description": "Trump Liberation Day — sweeping reciprocal tariffs on all trading partners",
        "category": "Trade_Coercion",
    },
    "Liberation_Day_90day_Pause": {
        "date": "2025-04-09",
        "description": "Trump announces 90-day pause on reciprocal tariffs (except China)",
        "category": "Retaliation",
    },
}
