"""
event_study.py
==============
MacKinlay (1997) event study using Fama-French 3-factor model for
normal return estimation.

Normal return model (estimated over estimation window):
    R_{sp500,t} - Rf_t = α + β1·MktRF_t + β2·SMB_t + β3·HML_t + ε_t

Abnormal return:
    AR_t = (R_{sp500,t} - Rf_t) - (α̂ + β̂1·MktRF_t + β̂2·SMB_t + β̂3·HML_t)

Variance of AR (MacKinlay 1997, multifactor extension):
    σ²(AR_t) = σ²_ε · [1 + f_t' (F'F)^{-1} f_t]
    where f_t = [1, MktRF_t, SMB_t, HML_t] and F = factor matrix over
    estimation window (including intercept column).

Test statistics:
    - MacKinlay t-test on CAAR
    - BMP (Boehmer, Musumeci, Poulsen 1991) standardized cross-sectional test
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import warnings
from scipy import stats

warnings.filterwarnings("ignore")


# =============================================================================
# HELPERS
# =============================================================================

def _nearest_trading_day(date: pd.Timestamp, trading_days: pd.DatetimeIndex) -> pd.Timestamp:
    """Return the nearest trading day on or after `date`."""
    mask = trading_days >= date
    if mask.any():
        return trading_days[mask][0]
    return trading_days[-1]


def _ols(X: np.ndarray, y: np.ndarray):
    """
    OLS: y = X @ beta + eps
    Returns beta, residual variance σ²_ε, (X'X)^{-1}
    """
    XtX     = X.T @ X
    XtX_inv = np.linalg.pinv(XtX)
    beta    = XtX_inv @ X.T @ y
    resid   = y - X @ beta
    sigma2  = np.sum(resid**2) / max(len(y) - X.shape[1], 1)
    return beta, sigma2, XtX_inv


# =============================================================================
# MAIN CLASS
# =============================================================================

class EventStudy:
    """
    Parameters
    ----------
    sp500      : pd.Series  daily log-returns of S&P500, DatetimeIndex
    ff3        : pd.DataFrame  columns [MktRF, SMB, HML, RF], daily, in decimal
    events     : dict  from config.EVENTS
    params     : dict  with keys: ESTIMATION_START, ESTIMATION_END,
                       EVENT_START, EVENT_END, MIN_EST_OBS, CAR_WINDOWS, OUT_DIR
    """

    def __init__(self, sp500, ff3, events, params):
        self.sp500   = sp500.sort_index()
        self.ff3     = ff3.sort_index()
        self.events  = events
        self.params  = params

        # merged daily panel: excess return + factors
        self.panel = (
            pd.DataFrame({"R_sp500": self.sp500})
            .join(self.ff3, how="inner")
            .dropna()
        )
        self.panel["ExRet"] = self.panel["R_sp500"] - self.panel["RF"]
        self.trading_days   = self.panel.index

        self.results = {}   # populated by run_all()

    # ------------------------------------------------------------------
    def _event_idx(self, event_date: str) -> int | None:
        """Trading-day index of event date (or next trading day)."""
        ts  = pd.Timestamp(event_date)
        day = _nearest_trading_day(ts, self.trading_days)
        if abs((day - ts).days) > 7:
            return None   # no trading day found within a week
        loc = self.trading_days.get_loc(day)
        return loc

    # ------------------------------------------------------------------
    def run_event(self, event_name: str, event_info: dict) -> dict | None:
        """
        Full MacKinlay pipeline for one event.
        Returns dict with ARs, CARs, test stats; None if skipped.
        """
        ev_idx = self._event_idx(event_info["date"])
        if ev_idx is None:
            print(f"  [SKIP] {event_name}: no trading day near {event_info['date']}")
            return None

        est_start = ev_idx + self.params["ESTIMATION_START"]
        est_end   = ev_idx + self.params["ESTIMATION_END"]
        evw_start = ev_idx + self.params["EVENT_START"]
        evw_end   = ev_idx + self.params["EVENT_END"]

        # bounds check
        if est_start < 0 or evw_end >= len(self.trading_days):
            print(f"  [SKIP] {event_name}: window out of data range")
            return None

        est_data = self.panel.iloc[est_start : est_end + 1]
        evw_data = self.panel.iloc[evw_start : evw_end + 1]

        if len(est_data) < self.params["MIN_EST_OBS"]:
            print(f"  [SKIP] {event_name}: only {len(est_data)} estimation obs "
                  f"(min={self.params['MIN_EST_OBS']})")
            return None

        # ── FF3 estimation ────────────────────────────────────────────
        F_est = np.column_stack([
            np.ones(len(est_data)),
            est_data["MktRF"].values,
            est_data["SMB"].values,
            est_data["HML"].values,
        ])
        y_est = est_data["ExRet"].values
        beta, sigma2_eps, FtF_inv = _ols(F_est, y_est)

        # ── Abnormal returns over event window ─────────────────────────
        F_ev = np.column_stack([
            np.ones(len(evw_data)),
            evw_data["MktRF"].values,
            evw_data["SMB"].values,
            evw_data["HML"].values,
        ])
        y_ev     = evw_data["ExRet"].values
        exp_ret  = F_ev @ beta
        ars      = y_ev - exp_ret

        # variance of each AR: σ²_ε · [1 + f_t'(F'F)^{-1}f_t]
        ar_vars = np.array([
            sigma2_eps * (1.0 + f @ FtF_inv @ f)
            for f in F_ev
        ])
        ar_stds = np.sqrt(ar_vars)

        # ── Relative day indices ───────────────────────────────────────
        rel_days = np.arange(self.params["EVENT_START"],
                             self.params["EVENT_END"] + 1)

        # ── CARs for requested windows ─────────────────────────────────
        cars = {}
        for (w1, w2) in self.params["CAR_WINDOWS"]:
            mask = (rel_days >= w1) & (rel_days <= w2)
            if mask.sum() == 0:
                continue
            car  = ars[mask].sum()
            var  = ar_vars[mask].sum()
            cars[(w1, w2)] = {
                "CAR":    car,
                "se":     np.sqrt(var),
                "t_stat": car / np.sqrt(var) if var > 0 else np.nan,
            }

        return {
            "event_name":  event_name,
            "event_date":  event_info["date"],
            "category":    event_info.get("category", "Other"),
            "description": event_info.get("description", ""),
            "event_day":   str(self.trading_days[ev_idx].date()),
            "n_est":       len(est_data),
            "alpha":       beta[0],
            "beta_mkt":    beta[1],
            "beta_smb":    beta[2],
            "beta_hml":    beta[3],
            "sigma2_eps":  sigma2_eps,
            "rel_days":    rel_days,
            "dates":       evw_data.index,
            "ars":         ars,
            "ar_stds":     ar_stds,
            "cars":        cars,
        }

    # ------------------------------------------------------------------
    def run_all(self) -> None:
        """Run event study for all events in config."""
        print("\n" + "=" * 70)
        print("EVENT STUDY — FF3 Normal Return Model")
        print("=" * 70)

        for name, info in self.events.items():
            print(f"\n[{name}]  {info['date']}  —  {info['description'][:60]}")
            res = self.run_event(name, info)
            if res is not None:
                self.results[name] = res
                print(f"  ✓  n_est={res['n_est']}  "
                      f"α={res['alpha']:.4f}  "
                      f"β_mkt={res['beta_mkt']:.3f}  "
                      f"β_smb={res['beta_smb']:.3f}  "
                      f"β_hml={res['beta_hml']:.3f}")
                for (w1, w2), c in res["cars"].items():
                    sig = (
                        "***" if abs(c["t_stat"]) > 2.576 else
                        "**"  if abs(c["t_stat"]) > 1.960 else
                        "*"   if abs(c["t_stat"]) > 1.645 else ""
                    )
                    print(f"    CAR({w1:+d},{w2:+d}) = {c['CAR']:+.4f}  "
                          f"t={c['t_stat']:+.3f} {sig}")

        print(f"\n[INFO] {len(self.results)}/{len(self.events)} events processed.")

    # ------------------------------------------------------------------
    def _aggregate(self, subset: dict) -> dict:
        """
        Aggregate ARs across events in subset.
        Returns AAR, CAAR, MacKinlay t-stat, BMP t-stat per relative day
        and per CAR window.
        """
        if not subset:
            return {}

        # stack ARs: shape (N_events, L_event_window)
        all_ars   = np.vstack([r["ars"]     for r in subset.values()])
        all_stds  = np.vstack([r["ar_stds"] for r in subset.values()])
        rel_days  = list(subset.values())[0]["rel_days"]
        N         = len(subset)

        # AAR and its std (cross-sectional)
        AAR       = all_ars.mean(axis=0)
        AAR_se    = all_ars.std(axis=0, ddof=1) / np.sqrt(N)
        CAAR      = np.cumsum(AAR)

        # MacKinlay t-stat: pool analytical variances
        pool_var  = (all_stds**2).mean(axis=0)
        AAR_t_mk  = AAR / np.sqrt(pool_var / N)

        # BMP (1991) standardized test
        SAR       = all_ars / all_stds          # (N, L) standardized ARs
        SAAR      = SAR.mean(axis=0)            # mean SAR per day
        SAAR_cs   = SAR.std(axis=0, ddof=1)     # cross-sectional std
        BMP_t     = SAAR / (SAAR_cs / np.sqrt(N))

        # CAAR per window — MacKinlay + BMP
        caar_table = {}
        for (w1, w2) in self.params["CAR_WINDOWS"]:
            mask    = (rel_days >= w1) & (rel_days <= w2)
            L       = mask.sum()
            if L == 0:
                continue

            cars_n  = all_ars[:, mask].sum(axis=1)   # CAR per event
            CAAR_w  = cars_n.mean()
            CAAR_se = cars_n.std(ddof=1) / np.sqrt(N)
            t_mk    = CAAR_w / CAAR_se if CAAR_se > 0 else np.nan
            p_mk    = 2 * (1 - stats.t.cdf(abs(t_mk), df=N - 1))

            # BMP for window
            car_stds = np.sqrt((all_stds[:, mask]**2).sum(axis=1))
            scars    = cars_n / car_stds
            t_bmp    = (scars.mean() / (scars.std(ddof=1) / np.sqrt(N))
                        if scars.std(ddof=1) > 0 else np.nan)
            p_bmp    = 2 * (1 - stats.t.cdf(abs(t_bmp), df=N - 1))

            caar_table[(w1, w2)] = {
                "CAAR":    CAAR_w,
                "se":      CAAR_se,
                "t_mk":    t_mk,
                "p_mk":    p_mk,
                "t_bmp":   t_bmp,
                "p_bmp":   p_bmp,
                "N":       N,
            }

        return {
            "rel_days":   rel_days,
            "AAR":        AAR,
            "AAR_se":     AAR_se,
            "CAAR":       CAAR,
            "AAR_t_mk":   AAR_t_mk,
            "BMP_t":      BMP_t,
            "caar_table": caar_table,
            "N":          N,
        }

    # ------------------------------------------------------------------
    def aggregate_and_print(self) -> dict:
        """Aggregate across all events and by category. Print summary."""
        if not self.results:
            print("[WARN] No results to aggregate.")
            return {}

        groups = {"All events": self.results}

        # group by category
        categories = set(r["category"] for r in self.results.values())
        for cat in sorted(categories):
            sub = {k: v for k, v in self.results.items() if v["category"] == cat}
            if len(sub) >= 2:
                groups[f"Category: {cat}"] = sub

        agg_results = {}
        for group_name, subset in groups.items():
            agg = self._aggregate(subset)
            if not agg:
                continue
            agg_results[group_name] = agg

            print(f"\n{'='*70}")
            print(f"  {group_name}  (N={agg['N']})")
            print(f"{'='*70}")
            print(f"  {'Window':<15} {'CAAR':>8} {'SE':>8} "
                  f"{'t_MK':>8} {'p_MK':>7} {'t_BMP':>8} {'p_BMP':>7}")
            print(f"  {'-'*65}")

            for (w1, w2), row in agg["caar_table"].items():
                sig = (
                    "***" if row["p_mk"] < 0.01 else
                    "**"  if row["p_mk"] < 0.05 else
                    "*"   if row["p_mk"] < 0.10 else ""
                )
                print(f"  [{w1:+d},{w2:+d}]        "
                      f"{row['CAAR']:>+8.4f} "
                      f"{row['se']:>8.4f} "
                      f"{row['t_mk']:>+8.3f} "
                      f"{row['p_mk']:>7.4f} "
                      f"{row['t_bmp']:>+8.3f} "
                      f"{row['p_bmp']:>7.4f}  {sig}")

        return agg_results

    # ------------------------------------------------------------------
    def plot(self, agg_results: dict, out_dir: str) -> None:
        """Plot AAR and CAAR over event window, one panel per group."""
        os.makedirs(os.path.join(out_dir, "plots"), exist_ok=True)

        for group_name, agg in agg_results.items():
            rel   = agg["rel_days"]
            AAR   = agg["AAR"]
            AAR_se= agg["AAR_se"]
            CAAR  = agg["CAAR"]
            N     = agg["N"]

            safe_name = group_name.replace(" ", "_").replace(":", "").replace("/", "_")
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
            fig.suptitle(f"{group_name}  (N={N})\nFF3 Event Study — MacKinlay (1997)",
                         fontsize=13, fontweight="bold")

            # ── AAR ───────────────────────────────────────────────────
            ax1.axhline(0, color="black", linewidth=0.8, linestyle="--")
            ax1.axvline(0, color="red",   linewidth=1.0, linestyle="--", alpha=0.7,
                        label="Event day")
            ax1.bar(rel, AAR, color=["#d73027" if x < 0 else "#4575b4" for x in AAR],
                    alpha=0.8, width=0.6)
            ax1.fill_between(rel, AAR - 1.96*AAR_se, AAR + 1.96*AAR_se,
                             alpha=0.2, color="gray", label="95% CI")
            ax1.set_ylabel("AAR (log-return)", fontsize=11)
            ax1.legend(fontsize=9)
            ax1.set_title("Average Abnormal Return (AAR)", fontsize=11)
            ax1.grid(axis="y", alpha=0.3)

            # ── CAAR ──────────────────────────────────────────────────
            ax2.axhline(0, color="black", linewidth=0.8, linestyle="--")
            ax2.axvline(0, color="red",   linewidth=1.0, linestyle="--", alpha=0.7,
                        label="Event day")
            ax2.plot(rel, CAAR, color="#2c7bb6", linewidth=2.0, marker="o",
                     markersize=4, label="CAAR")
            ax2.fill_between(rel,
                             CAAR - 1.96 * np.cumsum(AAR_se),
                             CAAR + 1.96 * np.cumsum(AAR_se),
                             alpha=0.2, color="#2c7bb6")
            ax2.set_ylabel("CAAR (log-return)", fontsize=11)
            ax2.set_xlabel("Trading days relative to event", fontsize=11)
            ax2.legend(fontsize=9)
            ax2.set_title("Cumulative Average Abnormal Return (CAAR)", fontsize=11)
            ax2.grid(axis="y", alpha=0.3)

            plt.tight_layout()
            out_path = os.path.join(out_dir, "plots", f"{safe_name}.png")
            plt.savefig(out_path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            print(f"  [plot] saved → {out_path}")

    # ------------------------------------------------------------------
    def save_csv(self, out_dir: str) -> None:
        """Save per-event AR series and summary CAR table to CSV."""
        os.makedirs(os.path.join(out_dir, "abnormal_returns"), exist_ok=True)

        rows = []
        for name, res in self.results.items():
            # per-event AR CSV
            ar_df = pd.DataFrame({
                "date":    res["dates"].strftime("%Y-%m-%d"),
                "rel_day": res["rel_days"],
                "AR":      res["ars"],
                "AR_std":  res["ar_stds"],
            })
            ar_df.to_csv(
                os.path.join(out_dir, "abnormal_returns", f"{name}_ARs.csv"),
                index=False
            )

            # summary row
            for (w1, w2), c in res["cars"].items():
                t_stat = c["t_stat"]
                
                # Calculate the exact p-value (two-tailed)
                # Degrees of freedom (N-1) is roughly len(est_data) - 4 parameters
                df = res["n_est"] - 4
                p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df=df)) if not np.isnan(t_stat) else np.nan
                
                # Assign academic stars
                stars = ""
                if p_val <= 0.01:
                    stars = "***"
                elif p_val <= 0.05:
                    stars = "**"
                elif p_val <= 0.10:
                    stars = "*"

                rows.append({
                    "event":       name,
                    "date":        res["event_date"],
                    "category":    res["category"],
                    "window":      f"[{w1:+d},{w2:+d}]",
                    "CAR":         round(c["CAR"], 6),
                    "se":          round(c["se"],  6),
                    "t_stat":      round(t_stat, 4),
                    "p_value":     round(p_val, 4) if not np.isnan(p_val) else np.nan,
                    "stars":       stars,
                    "significant": abs(t_stat) >= 1.645,  # Changed to 10% threshold
                })

        summary = pd.DataFrame(rows)
        out_path = os.path.join(out_dir, "summary_CARs.csv")
        summary.to_csv(out_path, index=False)
        print(f"\n  [csv] summary saved → {out_path}")