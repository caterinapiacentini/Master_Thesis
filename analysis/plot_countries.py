#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Country-level GEP index plots: a 3x2 panel (Japan, UK, Germany, Russia,
Iran, China) plus one enlarged plot per country.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from pathlib import Path

# Configurazione font globale per uno stile accademico uniforme
plt.rcParams["font.family"] = "serif"

# Blu scuro istituzionale (Stile C&I)
COL_GEP = "#2b4c8c"

try:
    HERE = Path(__file__).parent
except NameError:
    HERE = Path.cwd()
REPO = next((p for p in [HERE, *HERE.parents] if (p / "data" / "gep_us").exists()), HERE.parent)
CTRY = REPO / "data" / "countries"
OUT  = REPO / "analysis" / "output" / "countries"
OUT.mkdir(parents=True, exist_ok=True)

START = "1996-01-01"
END   = "2025-12-31"

# Liste eventi ottimizzate e sfoltite (Focus su shock geopolitici idiosincratici)
# Liste eventi ottimizzate con offset (x, y) ricalibrati per evitare sovrapposizioni
COUNTRIES = [
    {
        "title":  "GEP Japan",
        "file":   "GEP_Monthly_JAPAN_min2.csv",
        "events": [
            ("9/11",              "2001-09-01", (0, 35)),
            ("Senkaku Tension",   "2010-09-01", (0, 45)),
            ("N.Korea Missiles",     "2017-08-01", (-55, 45)),  # up-left diagonal
            ("US-China Trade War",   "2018-07-01", (-100, 70)), # parallel diagonal to N.Korea Missiles, further out to the left
            ("Ukraine Invasion",  "2022-02-01", (0, 45)),
        ],
    },
    {
        "title":  "GEP United Kingdom",
        "file":   "GEP_Monthly_UK_min2.csv",
        "events": [
            ("9/11",              "2001-09-01", (0, 35)),
            ("Iraq War",          "2003-03-01", (45, 20)),
            ("Brexit Vote",       "2016-06-01", (-15, 55)),   # above the peak instead of below
            ("Ukraine Invasion",  "2022-02-01", (0, 45)),
        ],
    },
    {
        "title":  "GEP Germany",
        "file":   "GEP_Monthly_GERMANY_min2.csv",
        "events": [
            ("Iraq War",          "2003-03-01", (0, 35)),
            ("Eurozone Crisis",   "2010-05-01", (0, 45)),
            ("COVID-19",          "2020-03-01", (0, -50)),
            ("Ukraine Invasion",  "2022-02-01", (0, 45)),
        ],
    },
    {
        "title":  "GEP Russia",
        "file":   "GEP_Monthly_RUSSIA_min2.csv",
        "events": [
            ("Georgia War",        "2008-08-01", (0, 35)),
            ("Crimea Annexation",  "2014-03-01", (0, 40)),
            ("Syria Intervention", "2015-09-01", (55, -35)),  # to the right side
            ("Ukraine Invasion",   "2022-02-01", (-30, 45)),
        ],
    },
    {
        "title":  "GEP Iran",
        "file":   "GEP_Monthly_IRAN_min2.csv",
        "events": [
            ("Nuclear Crisis",    "2006-01-01", (0, 35)),
            ("JCPOA Withdrawal",  "2018-05-01", (-50, 40)),
            ("Soleimani",         "2020-01-01", (50, 40)),
            ("Ukraine Invasion",  "2022-02-01", (0, -50)),
        ],
    },
    {
        "title":  "GEP China",
        "file":   "GEP_Monthly_CHINA_min2.csv",
        "events": [
            ("Asian Crisis",              "1997-07-01", (35, 45)),
            ("South China Sea Dispute",   "2012-07-01", (-50, 45)),
            ("US-China Trade",            "2018-07-01", (0, 45)),
            ("COVID-19",                  "2020-03-01", (-45, -50)),
            ("Export Controls",           "2023-07-01", (45, -50)),
        ],
    },
]


def load_and_normalize(file_path):
    df = pd.read_csv(file_path)
    df["date"] = pd.to_datetime(df["month"], format="%Y-%m")
    df = df.sort_values("date").reset_index(drop=True)
    df = df[(df["date"] >= START) & (df["date"] <= END)].copy()
    df["GEP_norm"] = df["GEP_monthly"] / df["GEP_monthly"].mean() * 100
    return df


def annotate_events_academic(ax, df, events, font_size=6):
    """Disegna un punto solido sul picco (cercato in una finestra di +/-45 giorni
    attorno alla data dell'evento, cosi' l'etichetta si aggancia al vero massimo
    locale invece che al mese nominale) e posiziona l'etichetta con offset in punti."""
    for label, date_str, offset in events:
        event_date = pd.Timestamp(date_str)
        if event_date < df["date"].min() or event_date > df["date"].max():
            continue

        window = df[(df["date"] >= event_date - pd.Timedelta(days=45)) &
                    (df["date"] <= event_date + pd.Timedelta(days=45))]
        idx = window["GEP_norm"].idxmax() if not window.empty else (df["date"] - event_date).abs().idxmin()
        y_val = df.loc[idx, "GEP_norm"]
        x_val = df.loc[idx, "date"]

        # Punto solido sul picco della serie indici
        ax.scatter(x_val, y_val, s=20, color=COL_GEP, zorder=5, linewidths=0)
        
        # Attiva la freccia solo se l'offset sposta il testo in basso o lateralmente
        use_arrow = True if (abs(offset[0]) > 10 or abs(offset[1]) > 20 or offset[1] < 0) else False
        
        text_kwargs = dict(
            text=label.replace("\n", " "),
            xy=(x_val, y_val),
            xytext=offset,
            textcoords="offset points",
            fontsize=font_size,
            ha="center",
            va="center" if use_arrow else "bottom",
            color="black",
            alpha=0.9
        )
        
        if use_arrow:
            ax.annotate(
                **text_kwargs,
                arrowprops=dict(arrowstyle="-|>", color="black", lw=0.6, mutation_scale=7)
            )
        else:
            ax.annotate(**text_kwargs)


# PLOT 1 — Pannello 3 righe × 2 colonne (Layout ottimizzato per tesi/paper)
fig, axes = plt.subplots(3, 2, figsize=(20, 16), dpi=300)

for ax, country in zip(axes.flatten(), COUNTRIES):
    path = CTRY / country["file"]
    if not path.exists():
        print(f"[WARNING] Missing: {path.name}"); ax.set_visible(False); continue

    df = load_and_normalize(path)

    # Grafico principale della serie storica
    ax.plot(df["date"], df["GEP_norm"], color=COL_GEP, linewidth=1.3, alpha=0.95)
    annotate_events_academic(ax, df, country["events"], font_size=13)

    # Configurazione della scala logaritmica sull'asse Y
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x)}"))
    ax.set_yticks([50, 100, 200, 400, 800])
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())

    # Limiti e dettagli estetici minimali (headroom extra per le etichette piu' grandi)
    ax.set_xlim(pd.Timestamp(START), pd.Timestamp(END))
    ax.set_ylim(18, 1800)
    ax.set_title(country["title"], fontsize=17, pad=14)

    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_color("black")

    ax.tick_params(axis="y", colors=COL_GEP, labelsize=12, direction="out")
    ax.tick_params(axis="x", labelsize=12, direction="out", colors="black")

    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

plt.tight_layout()
plt.savefig(OUT / "GEP_6Countries_1996_2025.png", dpi=300, bbox_inches="tight")
print("Saved: GEP_6Countries_1996_2025.png")
plt.close()


# PLOT 2 — Grafici individuali ingranditi (Layout orizzontale)
for country in COUNTRIES:
    path = CTRY / country["file"]
    if not path.exists():
        print(f"[WARNING] Missing: {path.name}"); continue

    df = load_and_normalize(path)

    fig, ax = plt.subplots(figsize=(12, 5.5), dpi=300)
    ax.plot(df["date"], df["GEP_norm"], color=COL_GEP, linewidth=1.6, alpha=0.95)

    # Font leggermente più grandi per la versione standalone a pagina singola
    annotate_events_academic(ax, df, country["events"], font_size=11)

    # Configurazione scala logaritmica Y
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x)}"))
    ax.set_yticks([50, 100, 200, 400, 800])
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())

    # Dettagli estetici assi e spines
    ax.set_xlim(pd.Timestamp(START), pd.Timestamp(END))
    ax.set_ylim(25, 1600)
    ax.set_title(f"{country['title']} — Monthly GEP Index (1996–2025)", fontsize=13, pad=12)
    
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_color("black")
    
    ax.tick_params(axis="y", colors=COL_GEP, labelsize=10, direction="out")
    ax.tick_params(axis="x", labelsize=10, direction="out", colors="black")
    
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    short_name = country["file"].split("_")[2]
    plt.tight_layout()
    plt.savefig(OUT / f"GEP_{short_name}_1996_2025.png", dpi=300, bbox_inches="tight")
    print(f"Saved: GEP_{short_name}_1996_2025.png")
    plt.close()

print("\nAll country plots saved to output/countries/")