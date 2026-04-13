#!/usr/bin/env Rscript
# =============================================================================
# Realized Volatility Regression — MIN2 Index
# Following: Zhang et al. "Geopolitical risk and stock market volatility:
#            A global perspective"
#
# RV_t = sum_{j=1}^{M_t} r^2_{t,j}   (sum of squared daily returns in month t)
#
# Specifications (h = 1, monthly, Newey-West HAC SEs):
#   m1     : RV_{t+1} ~ dGEP_z_t
#   m2     : RV_{t+1} ~ dGEP_z_t + dGPR_z_t     [main spec]
#   m1_log : log(RV_{t+1}) ~ dGEP_z_t
#   m2_log : log(RV_{t+1}) ~ dGEP_z_t + dGPR_z_t
#
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
  mutate(log_RV = log(RV))

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
# 3. GPR — monthly
# =============================================================================
gpr_raw  <- read_excel(GPR_PATH)

date_col <- names(gpr_raw)[sapply(gpr_raw, function(x)
  inherits(x, "Date") || inherits(x, "POSIXct"))][1]
if (is.na(date_col)) date_col <- names(gpr_raw)[1]

gpr <- gpr_raw %>%
  rename(date = all_of(date_col)) %>%
  mutate(date = floor_date(as.Date(date), "month"),
         GPR  = as.numeric(GPR)) %>%
  filter(!is.na(GPR)) %>%
  select(date, GPR) %>%
  arrange(date)

# =============================================================================
# 4. MONTHLY PANEL — merge, first-difference, z-score, lead RV for h=1
# =============================================================================
df <- rv_monthly %>%
  inner_join(gep, by = "date") %>%
  inner_join(gpr, by = "date") %>%
  arrange(date) %>%
  filter(date >= as.Date("1996-01-01")) %>%
  mutate(
    dGEP   = GEP - lag(GEP),
    dGPR   = GPR - lag(GPR),
    dGEP_z = (dGEP - mean(dGEP, na.rm = TRUE)) / sd(dGEP, na.rm = TRUE),
    dGPR_z = (dGPR - mean(dGPR, na.rm = TRUE)) / sd(dGPR, na.rm = TRUE),
    # h = 1: next month's RV is the dependent variable
    RV_lead1     = lead(RV, 1),
    log_RV_lead1 = lead(log_RV, 1)
  ) %>%
  filter(!is.na(dGEP_z) & !is.na(dGPR_z))

cat(sprintf("Full panel: %d obs  (%s -> %s)\n",
            nrow(df), min(df$date), max(df$date)))

# =============================================================================
# 5. STATIONARITY CHECK
# =============================================================================
cat("\n", strrep("=", 70), "\n")
cat("STATIONARITY CHECK\n")
cat(strrep("=", 70), "\n")

adf_vars <- list(
  "RV          (levels)"  = df$RV,
  "log_RV      (levels)"  = df$log_RV,
  "dGEP_z  (differences)" = df$dGEP_z,
  "dGPR_z  (differences)" = df$dGPR_z
)

for (vname in names(adf_vars)) {
  x   <- na.omit(adf_vars[[vname]])
  adf <- adf.test(x)
  cat(sprintf("%-32s  ADF=%.3f  p=%.4f  %s\n",
              vname, adf$statistic, adf$p.value,
              ifelse(adf$p.value < 0.05, "stationary [OK]", "NON-stationary")))
}

# =============================================================================
# 6. SUBPERIODS
# =============================================================================
subperiods <- list(
  "Full sample  (1996-2025)" = df,
  "2005-2015"                = df %>% filter(date >= as.Date("2005-01-01") &
                                               date <= as.Date("2015-12-31")),
  "2015-2025"                = df %>% filter(date >= as.Date("2015-01-01"))
)

# =============================================================================
# 7. REGRESSION RUNNER — h = 1, Newey-West HAC SEs
#
#   m1     : RV_{t+1}     ~ dGEP_z_t
#   m2     : RV_{t+1}     ~ dGEP_z_t + dGPR_z_t
#   m1_log : log(RV_{t+1}) ~ dGEP_z_t
#   m2_log : log(RV_{t+1}) ~ dGEP_z_t + dGPR_z_t
# =============================================================================
run_models <- function(d) {
  d <- d %>%
    filter(!is.na(RV_lead1) & !is.na(log_RV_lead1) &
           !is.na(dGEP_z)   & !is.na(dGPR_z))

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
    m1     = safe_nw(RV_lead1     ~ dGEP_z,          d),
    m2     = safe_nw(RV_lead1     ~ dGEP_z + dGPR_z, d),
    m1_log = safe_nw(log_RV_lead1 ~ dGEP_z,          d),
    m2_log = safe_nw(log_RV_lead1 ~ dGEP_z + dGPR_z, d),
    n      = nrow(d),
    nw_lag = nw_lag
  )
}

# =============================================================================
# 8. RUN ALL MODELS
# =============================================================================
results <- lapply(subperiods, run_models)

# =============================================================================
# 9. PRINT RESULTS
# =============================================================================
cat("\n", strrep("=", 70), "\n")
cat("REALIZED VOLATILITY REGRESSIONS — MONTHLY (h = 1)\n")
cat("  RV_t   = sum of squared daily log-returns in month t (Zhang et al.)\n")
cat("  dGEP_z = standardised first difference of GEP (MIN2)\n")
cat("  dGPR_z = standardised first difference of GPR\n")
cat("\n")
cat("  m1     : RV_{t+1}     ~ dGEP_z_t\n")
cat("  m2     : RV_{t+1}     ~ dGEP_z_t + dGPR_z_t   [main spec]\n")
cat("  m1_log : log(RV_{t+1}) ~ dGEP_z_t\n")
cat("  m2_log : log(RV_{t+1}) ~ dGEP_z_t + dGPR_z_t\n")
cat("  SEs    : Newey-West HAC\n")
cat(strrep("=", 70), "\n")

for (period_name in names(subperiods)) {
  models <- results[[period_name]]
  if (is.null(models)) next

  cat(sprintf("\n>> %s  |  N=%d  |  NW lag=%d\n",
              period_name, models$n, models$nw_lag))

  for (mn in c("m1", "m2", "m1_log", "m2_log")) {
    m <- models[[mn]]
    if (is.null(m)) next
    cat(sprintf("  [%s]  R2=%.4f  Adj.R2=%.4f\n", mn, m$r2, m$adj_r2))
    print(m$ct)
  }
}

cat("\n", strrep("=", 70), "\n")
cat("DONE\n")
cat(strrep("=", 70), "\n")
