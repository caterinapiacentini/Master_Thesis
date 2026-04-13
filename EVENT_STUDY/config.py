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

    # ── Kosovo / Balkans ─────────────────────────────────────────────────────
    "Kosovo_NATO_Bombing": {
        "date": "1999-03-24",
        "description": "NATO begins bombing Yugoslavia — US/EU sanctions on Serbia",
        "category": "Sanctions",
    },

    # ── 9/11 & aftermath ─────────────────────────────────────────────────────
    "Sep11_Attack": {
        "date": "2001-09-11",
        # NYSE closed 9/11–9/14; code maps this to first trading day: 2001-09-17
        "description": "9/11 attacks — markets reopen 2001-09-17 amid financial sanctions on Al-Qaeda networks",
        "category": "Sanctions",
    },

    # ── Iraq War ─────────────────────────────────────────────────────────────
    "Iraq_War_Start": {
        "date": "2003-03-20",
        "description": "US-led invasion of Iraq begins — full trade/financial embargo on Iraq",
        "category": "Embargo",
    },

    # ── North Korea ──────────────────────────────────────────────────────────
    "DPRK_Nuclear_Test1": {
        "date": "2006-10-09",
        "description": "North Korea first nuclear test — UN/US sweeping sanctions response",
        "category": "Sanctions",
    },

    # ── Russia–Georgia War ───────────────────────────────────────────────────
    "Russia_Georgia_War": {
        "date": "2008-08-08",
        "description": "Russia invades Georgia — US/EU pressure campaign, export restrictions",
        "category": "Sanctions",
    },

    # ── Iran oil embargo ─────────────────────────────────────────────────────
    "EU_Iran_Oil_Embargo": {
        "date": "2012-01-23",
        "description": "EU announces full oil embargo on Iran, asset freeze on central bank",
        "category": "Embargo",
    },

    # ── Russia–Ukraine I (Crimea) ────────────────────────────────────────────
    "Russia_Crimea_Sanctions": {
        "date": "2014-03-17",
        "description": "EU/US first sanctions package on Russia after Crimea annexation",
        "category": "Sanctions",
    },

    # ── North Korea 2017 ─────────────────────────────────────────────────────
    "DPRK_UNSC_Sanctions_2017": {
        "date": "2017-08-05",
        "description": "UN Security Council toughest-ever sanctions on North Korea (coal/iron ban)",
        "category": "Sanctions",
    },

    # ── Steel / Aluminium Tariffs ────────────────────────────────────────────
    "Steel_Aluminium_Tariffs": {
        "date": "2018-03-08",
        "description": "Trump imposes 25% steel / 10% aluminium tariffs on all trading partners",
        "category": "Trade_Coercion",
    },

    # ── Iran JCPOA withdrawal ────────────────────────────────────────────────
    "Iran_JCPOA_Withdrawal": {
        "date": "2018-05-08",
        "description": "US withdraws from JCPOA, reimpose full Iran nuclear sanctions",
        "category": "Sanctions",
    },

    # ── US–China Trade War (single key date) ─────────────────────────────────
    "USC_Tariffs_Wave1": {
        "date": "2018-07-06",
        "description": "US imposes 25% tariffs on $34B Chinese goods — first wave of trade war",
        "category": "Trade_Coercion",
    },

    # ── Huawei / Export Controls ─────────────────────────────────────────────
    "Huawei_EntityList": {
        "date": "2019-05-16",
        "description": "US adds Huawei to Entity List, banning US technology exports",
        "category": "Export_Controls",
    },

    # ── Russia–Ukraine II (full invasion) ────────────────────────────────────
    "Russia_Invasion": {
        "date": "2022-02-24",
        "description": "Russia full-scale invasion of Ukraine — sweeping US/EU/UK sanctions, asset freeze, SWIFT exclusion",
        "category": "Sanctions",
    },

    # ── Semiconductor export controls ────────────────────────────────────────
    "BIS_Chip_Controls": {
        "date": "2022-10-07",
        "description": "BIS sweeping new chip and semiconductor export controls targeting China",
        "category": "Export_Controls",
    },

    # ── Liberation Day 2025 ──────────────────────────────────────────────────
    "Liberation_Day": {
        "date": "2025-04-02",
        "description": "Trump Liberation Day — sweeping reciprocal tariffs on all trading partners",
        "category": "Trade_Coercion",
    },
}
