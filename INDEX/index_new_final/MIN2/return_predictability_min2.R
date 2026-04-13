#!/usr/bin/env Rscript
# =============================================================================
# Return Predictability Regression — MIN2 Index
# Monthly + Quarterly | Simple OLS → +GPR → +FF3 → +Lags
# Newey-West HAC SEs | Subperiods | Structural break
#
# NOTE ON LEVELS vs DIFFERENCES:
#   The old GEP index (raw score) was non-stationary → required first-differencing.
#   MIN2 is n_gep_articles / n_articles — a bounded ratio (0–1) that controls
#   for total article volume by construction. Bounded series cannot have a unit
#   root in the standard sense and visually mean-reverts after each geopolitical
#   spike. We therefore use GEP_z (levels) as the primary specification.
#   The ADF test below will confirm; dGEP_z is computed and included as a
#   robustness column (m_rob) for comparison.
# =============================================================================

suppressPackageStartupMessages({
  library(dplyr)
  library(lubridate)
  library(readxl)
  library(quantmod)
  library(frenchdata)
  library(sandwich)
  library(lmtest)
  library(tseries)
  library(strucchange)
})

# =============================================================================
# PATHS  — edit only here
# =============================================================================
BASE_MIN2 <- "/Users/catepiacentini/Desktop/tesi/Master_Thesis/INDEX/index_new_final/MIN2"
GPR_PATH  <- "/Users/catepiacentini/Desktop/tesi/literature/data_gpr_export.xls"

# =============================================================================
# 1. GEP — MIN2 monthly file (already monthly, no aggregation needed)
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
# 2. GPR — monthly file
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
# 3. S&P 500 — monthly log returns
# =============================================================================
getSymbols("^GSPC", from = "1995-12-01", to = "2025-12-31",
           auto.assign = TRUE, warnings = FALSE)

sp500 <- data.frame(
  date  = as.Date(index(GSPC)),
  close = as.numeric(Ad(GSPC))
) %>%
  arrange(date) %>%
  mutate(month = floor_date(date, "month")) %>%
  group_by(month) %>%
  slice_tail(n = 1) %>%
  ungroup() %>%
  arrange(month) %>%
  mutate(log_ret = c(NA, diff(log(close)))) %>%
  filter(!is.na(log_ret)) %>%
  select(date = month, log_ret)

# =============================================================================
# 4. Fama-French 3 Factors — monthly
# =============================================================================
ff3_raw <- download_french_data("Fama/French 3 Factors")
ff3 <- ff3_raw$subsets$data[[1]] %>%
  mutate(date = as.Date(paste0(as.character(date), "01"), format = "%Y%m%d"),
         date = floor_date(date, "month")) %>%
  rename(MktRF = `Mkt-RF`) %>%
  mutate(across(c(MktRF, SMB, HML, RF), ~ as.numeric(.) / 100)) %>%
  select(date, MktRF, SMB, HML, RF) %>%
  filter(!is.na(MktRF))

# =============================================================================
# 5. MONTHLY PANEL — merge, z-score, levels + differences, lags
# =============================================================================
df_m <- sp500 %>%
  inner_join(gep, by = "date") %>%
  inner_join(gpr, by = "date") %>%
  inner_join(ff3, by = "date") %>%
  arrange(date) %>%
  filter(date >= as.Date("1996-01-01")) %>%
  mutate(
    # PRIMARY: GEP in levels (stationary bounded ratio)
    GEP_z         = (GEP - mean(GEP, na.rm = TRUE)) / sd(GEP, na.rm = TRUE),
    GPR_z         = (GPR - mean(GPR, na.rm = TRUE)) / sd(GPR, na.rm = TRUE),
    GEP_z_lag1    = lag(GEP_z, 1),
    GPR_z_lag1    = lag(GPR_z, 1),
    # ROBUSTNESS: first difference (kept for comparison)
    dGEP_z        = GEP_z - lag(GEP_z, 1),
    dGEP_z_lag1   = lag(dGEP_z, 1),
    # leads
    log_ret_lead1 = lead(log_ret, 1)
  )

cat(sprintf("Monthly panel: %d obs  (%s -> %s)\n",
            nrow(df_m), min(df_m$date), max(df_m$date)))

# =============================================================================
# 6. QUARTERLY PANEL — aggregate monthly → quarterly
# =============================================================================
df_q <- df_m %>%
  mutate(quarter = floor_date(date, "quarter")) %>%
  group_by(quarter) %>%
  summarise(
    log_ret = sum(log_ret, na.rm = TRUE),
    GEP     = mean(GEP,    na.rm = TRUE),
    GPR     = mean(GPR,    na.rm = TRUE),
    MktRF   = sum(MktRF,   na.rm = TRUE),
    SMB     = sum(SMB,     na.rm = TRUE),
    HML     = sum(HML,     na.rm = TRUE),
    RF      = sum(RF,      na.rm = TRUE),
    .groups = "drop"
  ) %>%
  rename(date = quarter) %>%
  arrange(date) %>%
  mutate(
    GEP_z         = (GEP - mean(GEP, na.rm = TRUE)) / sd(GEP, na.rm = TRUE),
    GPR_z         = (GPR - mean(GPR, na.rm = TRUE)) / sd(GPR, na.rm = TRUE),
    GEP_z_lag1    = lag(GEP_z, 1),
    GPR_z_lag1    = lag(GPR_z, 1),
    dGEP_z        = GEP_z - lag(GEP_z, 1),
    dGEP_z_lag1   = lag(dGEP_z, 1),
    log_ret_lead1 = lead(log_ret, 1)
  )

cat(sprintf("Quarterly panel: %d obs  (%s -> %s)\n",
            nrow(df_q), min(df_q$date), max(df_q$date)))

# =============================================================================
# 7. STATIONARITY CHECK — this determines whether GEP levels are appropriate
# =============================================================================
cat("\n", strrep("=", 70), "\n")
cat("STATIONARITY CHECK\n")
cat("  If GEP_z (levels) is stationary -> use levels as primary spec.\n")
cat("  If non-stationary -> switch to dGEP_z.\n")
cat(strrep("=", 70), "\n")

adf_vars <- list(
  "GEP_z  (levels)"      = df_m$GEP_z,
  "dGEP_z (differences)" = df_m$dGEP_z,
  "GPR_z  (levels)"      = df_m$GPR_z,
  "log_ret (monthly)"    = df_m$log_ret
)

for (vname in names(adf_vars)) {
  x   <- na.omit(adf_vars[[vname]])
  adf <- adf.test(x)
  cat(sprintf("%-32s  ADF=%.3f  p=%.4f  %s\n",
              vname, adf$statistic, adf$p.value,
              ifelse(adf$p.value < 0.05, "stationary [OK: use levels]",
                     "NON-stationary [use differences]")))
}

# =============================================================================
# 8. DIAGNOSTICS on primary spec GEP_z (levels)
# =============================================================================
cat("\n", strrep("=", 70), "\n")
cat("DIAGNOSTICS — log_ret ~ GEP_z + GPR_z  (monthly, levels)\n")
cat(strrep("=", 70), "\n")

fit_diag <- lm(log_ret ~ GEP_z + GPR_z,
               data = df_m %>% filter(!is.na(GEP_z) & !is.na(GPR_z)))

bg     <- bgtest(fit_diag, order = 4)
bp_tst <- bptest(fit_diag)
jb     <- jarque.bera.test(residuals(fit_diag))

cat(sprintf("Breusch-Godfrey (autocorr., lag=4):  stat=%.3f  p=%.4f  %s\n",
            bg$statistic, bg$p.value,
            ifelse(bg$p.value < 0.05, "-> use NW SEs", "-> no autocorrelation")))
cat(sprintf("Breusch-Pagan   (heteroscedast.):    stat=%.3f  p=%.4f  %s\n",
            bp_tst$statistic, bp_tst$p.value,
            ifelse(bp_tst$p.value < 0.05, "-> use robust SEs", "-> homoscedastic")))
cat(sprintf("Jarque-Bera     (normality resid.):  stat=%.3f  p=%.4f  %s\n",
            jb$statistic, jb$p.value,
            ifelse(jb$p.value < 0.05, "-> fat tails (expected)", "-> approx. normal")))
cat("\nNote: NW SEs applied regardless as best practice for financial time series.\n")

# =============================================================================
# 9. STRUCTURAL BREAK — monthly, levels spec
# =============================================================================
cat("\n", strrep("=", 70), "\n")
cat("STRUCTURAL BREAK — Bai-Perron on GEP_z coefficient (monthly)\n")
cat(strrep("=", 70), "\n")

df_clean_m <- df_m %>% filter(!is.na(log_ret) & !is.na(GEP_z) & !is.na(GPR_z))
bp_struct  <- breakpoints(log_ret ~ GEP_z + GPR_z, data = df_clean_m, h = 0.15)
print(summary(bp_struct))

opt_m <- which.min(BIC(bp_struct)) - 1
cat(sprintf("\nOptimal breaks by BIC: m=%d\n", opt_m))
if (opt_m > 0 && !all(is.na(bp_struct$breakpoints))) {
  bp_dates <- df_clean_m$date[bp_struct$breakpoints]
  cat(sprintf("Breakpoint date(s): %s\n", paste(bp_dates, collapse = ", ")))
}

# =============================================================================
# 10. SUBPERIODS
# =============================================================================
subperiods_m <- list(
  "Full sample (1996-2025)"        = df_m,
  "Pre-GFC (1996-2007)"            = df_m %>% filter(date < as.Date("2008-01-01")),
  "GFC & aftermath (2008-2011)"    = df_m %>% filter(date >= as.Date("2008-01-01") &
                                                       date <= as.Date("2011-12-31")),
  "Post-GFC (2012-2021)"           = df_m %>% filter(date >= as.Date("2012-01-01") &
                                                       date <= as.Date("2021-12-31")),
  "Russia-Ukraine war (2022-2023)" = df_m %>% filter(date >= as.Date("2022-02-01") &
                                                       date <= as.Date("2023-12-01")),
  "2025 (Liberation Day shock)"    = df_m %>% filter(date >= as.Date("2025-01-01"))
)

subperiods_q <- list(
  "Full sample (1996-2025)"        = df_q,
  "Pre-GFC (1996-2007)"            = df_q %>% filter(date < as.Date("2008-01-01")),
  "GFC & aftermath (2008-2011)"    = df_q %>% filter(date >= as.Date("2008-01-01") &
                                                       date <= as.Date("2011-12-31")),
  "Post-GFC (2012-2021)"           = df_q %>% filter(date >= as.Date("2012-01-01") &
                                                       date <= as.Date("2021-12-31")),
  "Russia-Ukraine war (2022-2023)" = df_q %>% filter(date >= as.Date("2022-02-01") &
                                                       date <= as.Date("2023-12-01")),
  "2025 (Liberation Day shock)"    = df_q %>% filter(date >= as.Date("2025-01-01"))
)

# =============================================================================
# 11. REGRESSION RUNNER — 5 specifications, Newey-West SEs
#
#  PRIMARY (levels — appropriate for bounded MIN2 ratio):
#  m1: log_ret ~ GEP_z
#  m2: log_ret ~ GEP_z + GPR_z
#  m3: log_ret ~ GEP_z + GPR_z + MktRF + SMB + HML
#  m4: log_ret ~ GEP_z + GEP_z_lag1 + GPR_z + GPR_z_lag1 + MktRF + SMB + HML
#
#  ROBUSTNESS (first differences — for comparison with old index approach):
#  m_rob: log_ret ~ dGEP_z + GPR_z
# =============================================================================
run_models <- function(d, h) {
  y_col <- if (h == 0) "log_ret" else "log_ret_lead1"

  d <- d %>%
    filter(!is.na(.data[[y_col]]) &
           !is.na(GEP_z)    & !is.na(GEP_z_lag1)  &
           !is.na(dGEP_z)   & !is.na(dGEP_z_lag1) &
           !is.na(GPR_z)    & !is.na(GPR_z_lag1)  &
           !is.na(MktRF)    & !is.na(SMB) & !is.na(HML))

  min_obs <- 16
  if (nrow(d) < min_obs) {
    cat(sprintf("  Skipping h=%d: only %d complete obs\n", h, nrow(d)))
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
    m1    = safe_nw(as.formula(paste(y_col, "~ GEP_z")), d),
    m2    = safe_nw(as.formula(paste(y_col, "~ GEP_z + GPR_z")), d),
    m3    = safe_nw(as.formula(paste(y_col, "~ GEP_z + GPR_z + MktRF + SMB + HML")), d),
    m4    = safe_nw(as.formula(paste(y_col,
              "~ GEP_z + GEP_z_lag1 + GPR_z + GPR_z_lag1 + MktRF + SMB + HML")), d),
    m_rob = safe_nw(as.formula(paste(y_col, "~ dGEP_z + GPR_z")), d),
    n      = nrow(d),
    nw_lag = nw_lag,
    h      = h
  )
}

# =============================================================================
# 12. RUN ALL MODELS
# =============================================================================
results_m <- list()
results_q <- list()

for (period_name in names(subperiods_m)) {
  for (h in c(0, 1)) {
    key <- paste0(period_name, " | h=", h)
    results_m[[key]] <- run_models(subperiods_m[[period_name]], h)
    results_q[[key]] <- run_models(subperiods_q[[period_name]], h)
  }
}

# =============================================================================
# 13. PRINT RESULTS — plain text, all specs
# =============================================================================
print_results <- function(all_results, subperiods, freq_label) {
  cat("\n", strrep("=", 70), "\n")
  cat(sprintf("REGRESSION RESULTS — %s\n", freq_label))
  cat(sprintf("  m1: Y ~ GEP_z\n"))
  cat(sprintf("  m2: Y ~ GEP_z + GPR_z\n"))
  cat(sprintf("  m3: Y ~ GEP_z + GPR_z + MktRF + SMB + HML\n"))
  cat(sprintf("  m4: Y ~ GEP_z + GEP_z_lag1 + GPR_z + GPR_z_lag1 + MktRF + SMB + HML\n"))
  cat(sprintf("  m_rob: Y ~ dGEP_z + GPR_z  [robustness: differences]\n"))
  cat(strrep("=", 70), "\n")

  for (period_name in names(subperiods)) {
    for (h in c(0, 1)) {
      key    <- paste0(period_name, " | h=", h)
      models <- all_results[[key]]
      if (is.null(models)) next

      cat(sprintf("\n>> %s  |  h=%d  |  N=%d  |  NW lag=%d\n",
                  period_name, h, models$n, models$nw_lag))

      for (mn in c("m1", "m2", "m3", "m4", "m_rob")) {
        m <- models[[mn]]
        if (is.null(m)) next
        cat(sprintf("  [%s]  R2=%.4f  Adj.R2=%.4f\n", mn, m$r2, m$adj_r2))
        print(m$ct)
      }
    }
  }
}

print_results(results_m, subperiods_m, "MONTHLY")
print_results(results_q, subperiods_q, "QUARTERLY")

cat("\n", strrep("=", 70), "\n")
cat("DONE\n")
cat(strrep("=", 70), "\n")
