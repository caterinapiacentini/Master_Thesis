#!/usr/bin/env Rscript
# =============================================================================
# Realized Volatility Regression — MIN2 Index
# Following: Zhang et al. "Geopolitical risk and stock market volatility:
#            A global perspective"
#
# RV_t   = sum_{j=1}^{M_t} r^2_{t,j}   (sum of squared daily returns in month t)
# LV_t   = log(RV_t)
#
# Main specification (Zhang et al., eq. with GEP replacing GPR):
#   LV_t = β0 + β1*LV_{t-1} + β2*dGEP_z_t + ε_t
#
# Also run baseline without AR term for comparison:
#   LV_t = β0 + β1*dGEP_z_t + ε_t
#
# h = 0 | Monthly | Newey-West HAC SEs
# Subperiods: 1996-2025 | 2005-2015 | 2015-2025
# =============================================================================

suppressPackageStartupMessages({
  library(dplyr)
  library(lubridate)
  library(readxl)
  library(quantmod)
  library(sandwich)
  library(lmtest)
  library(tseries)
})

# =============================================================================
# PATHS — edit only here
# =============================================================================
BASE_MIN2 <- "/Users/catepiacentini/Desktop/tesi/Master_Thesis/INDEX/index_new_final/MIN2"
GPR_PATH  <- "/Users/catepiacentini/Desktop/tesi/literature/data_gpr_export.xls"

# =============================================================================
# 1. S&P 500 — daily log returns → monthly RV (Zhang et al. eq. 1)
#    RV_t = sum_{j=1}^{M_t} r^2_{t,j}
# =============================================================================
getSymbols("^GSPC", from = "1995-12-01", to = "2025-12-31",
           auto.assign = TRUE, warnings = FALSE)

sp500_daily <- data.frame(
  date  = as.Date(index(GSPC)),
  close = as.numeric(Ad(GSPC))
) %>%
  arrange(date) %>%
  mutate(
    log_ret = c(NA, diff(log(close))),
    month   = floor_date(date, "month")
  ) %>%
  filter(!is.na(log_ret))

rv_monthly <- sp500_daily %>%
  group_by(month) %>%
  summarise(
    RV     = sum(log_ret^2, na.rm = TRUE),
    n_days = n(),
    .groups = "drop"
  ) %>%
  rename(date = month) %>%
  mutate(LV = log(RV))

# =============================================================================
# 2. GEP — MIN2 monthly
# =============================================================================
gep_raw <- read.csv(file.path(BASE_MIN2, "GEP_Monthly_Robust_min2.csv"),
                    stringsAsFactors = FALSE)
gep_raw$date <- as.Date(paste0(gep_raw$month, "-01"))

gep <- gep_raw %>%
  filter(!is.na(GEP_monthly)) %>%
  select(date, GEP = GEP_monthly) %>%
  mutate(date = floor_date(date, "month")) %>%
  arrange(date)

# =============================================================================
# 3. MONTHLY PANEL — merge, first-difference GEP, z-score, AR lag of LV
# =============================================================================
df <- rv_monthly %>%
  inner_join(gep, by = "date") %>%
  arrange(date) %>%
  filter(date >= as.Date("1996-01-01")) %>%
  mutate(
    LV_lag1 = lag(LV, 1),                                          # AR(1) term
    dGEP    = GEP - lag(GEP),
    dGEP_z  = (dGEP - mean(dGEP, na.rm = TRUE)) / sd(dGEP, na.rm = TRUE)
  ) %>%
  filter(!is.na(dGEP_z) & !is.na(LV_lag1))

cat(sprintf("Full panel: %d obs  (%s -> %s)\n",
            nrow(df), min(df$date), max(df$date)))

# =============================================================================
# 4. STATIONARITY CHECK
# =============================================================================
cat("\n", strrep("=", 70), "\n")
cat("STATIONARITY CHECK\n")
cat(strrep("=", 70), "\n")

adf_vars <- list(
  "LV      (levels)"      = df$LV,
  "dGEP_z  (differences)" = df$dGEP_z
)

for (vname in names(adf_vars)) {
  x   <- na.omit(adf_vars[[vname]])
  adf <- adf.test(x)
  cat(sprintf("%-32s  ADF=%.3f  p=%.4f  %s\n",
              vname, adf$statistic, adf$p.value,
              ifelse(adf$p.value < 0.05, "stationary [OK]", "NON-stationary")))
}

# =============================================================================
# 5. SUBPERIODS
# =============================================================================
subperiods <- list(
  "Full sample  (1996-2025)" = df,
  "2005-2015"                = df %>% filter(date >= as.Date("2005-01-01") &
                                               date <= as.Date("2015-12-31")),
  "2015-2025"                = df %>% filter(date >= as.Date("2015-01-01"))
)

# =============================================================================
# 6. REGRESSION RUNNER — Newey-West HAC SEs
#
#   m_base : LV_t ~ dGEP_z_t                          (no AR term, baseline)
#   m_ar   : LV_t ~ LV_{t-1} + dGEP_z_t              (Zhang et al. spec)
# =============================================================================
run_models <- function(d) {
  d <- d %>% filter(!is.na(LV) & !is.na(LV_lag1) & !is.na(dGEP_z))

  if (nrow(d) < 16) {
    cat(sprintf("  Skipping: only %d complete obs\n", nrow(d)))
    return(NULL)
  }

  nw_lag <- max(1, floor(4 * (nrow(d) / 100)^(2/9)))

  safe_nw <- function(formula, data) {
    tryCatch({
      fit <- lm(formula, data = data)
      ct  <- coeftest(fit, vcov = NeweyWest(fit, lag = nw_lag, prewhite = FALSE))
      list(fit = fit, ct = ct,
           r2     = summary(fit)$r.squared,
           adj_r2 = summary(fit)$adj.r.squared)
    }, error = function(e) NULL)
  }

  list(
    m_base = safe_nw(LV ~ dGEP_z,          d),
    m_ar   = safe_nw(LV ~ LV_lag1 + dGEP_z, d),
    n      = nrow(d),
    nw_lag = nw_lag
  )
}

# =============================================================================
# 7. RUN ALL MODELS
# =============================================================================
results <- lapply(subperiods, run_models)

# =============================================================================
# 8. PRINT RESULTS
# =============================================================================
cat("\n", strrep("=", 70), "\n")
cat("REALIZED VOLATILITY REGRESSIONS — MONTHLY (h = 0)\n")
cat("  LV_t   = log(RV_t) = log(sum of squared daily log-returns in month t)\n")
cat("  dGEP_z = standardised first difference of GEP (MIN2)\n")
cat("\n")
cat("  m_base : LV_t ~ dGEP_z_t                       [no AR, baseline]\n")
cat("  m_ar   : LV_t ~ LV_{t-1} + dGEP_z_t            [Zhang et al. spec]\n")
cat("  SEs    : Newey-West HAC\n")
cat(strrep("=", 70), "\n")

for (period_name in names(subperiods)) {
  models <- results[[period_name]]
  if (is.null(models)) next

  cat(sprintf("\n>> %s  |  N=%d  |  NW lag=%d\n",
              period_name, models$n, models$nw_lag))

  for (mn in c("m_base", "m_ar")) {
    m <- models[[mn]]
    if (is.null(m)) next
    cat(sprintf("  [%s]  R2=%.4f  Adj.R2=%.4f\n", mn, m$r2, m$adj_r2))
    print(m$ct)
  }
}

cat("\n", strrep("=", 70), "\n")
cat("DONE\n")
cat(strrep("=", 70), "\n")
