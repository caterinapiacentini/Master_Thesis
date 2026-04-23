#!/usr/bin/env Rscript
# =============================================================================
# Return Predictability Regression — MONTHLY
# ΔGEP_z (first difference) + GPR_z (levels) + Newey-West SEs
# =============================================================================

suppressPackageStartupMessages({
  library(dplyr)
  library(lubridate)
  library(readxl)
  library(quantmod)
  library(frenchdata)
  library(gt)
  library(sandwich)
  library(lmtest)
  library(tseries)
  library(strucchange)
})

BASE_GEP <- "/Users/catepiacentini/Desktop/tesi/Master_Thesis/INDEX/index_8_revised"
BASE_LIT <- "/Users/catepiacentini/Desktop/tesi/literature"

# =============================================================================
# 1. GEP — aggregate daily → monthly
# =============================================================================
gep_daily <- read.csv(file.path(BASE_GEP, "GEP_Daily_Index.csv"),
                      stringsAsFactors = FALSE)
gep_daily$date <- as.Date(gep_daily$date)

gep <- gep_daily %>%
  filter(n_articles > 0) %>%
  mutate(month = floor_date(date, "month")) %>%
  group_by(month) %>%
  summarise(GEP = mean(score, na.rm = TRUE), .groups = "drop") %>%
  rename(date = month) %>%
  arrange(date)

# =============================================================================
# 2. GPR — monthly file
# =============================================================================
gpr_path <- file.path(BASE_LIT, "data_gpr_export.xls")
gpr_raw  <- read_excel(gpr_path)

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
# 5. Merge + z-score + ΔGEP + lags
# =============================================================================
df <- sp500 %>%
  inner_join(gep, by = "date") %>%
  inner_join(gpr, by = "date") %>%
  inner_join(ff3, by = "date") %>%
  arrange(date) %>%
  filter(date >= as.Date("1996-01-01")) %>%
  mutate(
    # z-score on full sample
    GEP_z          = (GEP - mean(GEP, na.rm = TRUE)) / sd(GEP, na.rm = TRUE),
    GPR_z          = (GPR - mean(GPR, na.rm = TRUE)) / sd(GPR, na.rm = TRUE),
    # ΔGEP: first difference of z-scored GEP
    dGEP_z         = GEP_z - lag(GEP_z, 1),
    dGEP_z_lag1    = lag(dGEP_z, 1),
    dGEP_z_lag2    = lag(dGEP_z, 2),
    # GPR stays in levels (stationary), but add lag for spec 4
    GPR_z_lag1     = lag(GPR_z, 1),
    # lead return for h=1
    log_ret_lead1  = lead(log_ret, 1)
  )

cat(sprintf("Merged monthly: %d obs  (%s → %s)\n",
            nrow(df), min(df$date), max(df$date)))

# =============================================================================
# 6. CONFIRM STATIONARITY of dGEP_z
# =============================================================================
cat("\n", strrep("=", 70), "\n")
cat("STATIONARITY CHECK\n")
cat(strrep("=", 70), "\n")

adf_vars <- list(
  "GEP_z  (levels)"      = df$GEP_z,
  "dGEP_z (differences)" = df$dGEP_z,
  "GPR_z  (levels)"      = df$GPR_z,
  "log_ret"              = df$log_ret
)

for (vname in names(adf_vars)) {
  x   <- na.omit(adf_vars[[vname]])
  adf <- adf.test(x)
  cat(sprintf("%-30s  ADF=%.3f  p=%.4f  %s\n",
              vname, adf$statistic, adf$p.value,
              ifelse(adf$p.value < 0.05, "stationary ✓",
                     "NON-stationary ✗")))
}

# =============================================================================
# 7. DIAGNOSTICS on baseline model with correct specification
# =============================================================================
cat("\n", strrep("=", 70), "\n")
cat("DIAGNOSTICS — log_ret ~ dGEP_z + GPR_z\n")
cat(strrep("=", 70), "\n")

fit_diag <- lm(log_ret ~ dGEP_z + GPR_z,
               data = df %>% filter(!is.na(dGEP_z) & !is.na(GPR_z)))

bg <- bgtest(fit_diag, order = 4)
bp <- bptest(fit_diag)
jb <- jarque.bera.test(residuals(fit_diag))

cat(sprintf("Breusch-Godfrey (autocorr., lag=4):  stat=%.3f  p=%.4f  %s\n",
            bg$statistic, bg$p.value,
            ifelse(bg$p.value < 0.05, "→ use NW SEs", "→ no autocorrelation")))
cat(sprintf("Breusch-Pagan   (heteroscedast.):    stat=%.3f  p=%.4f  %s\n",
            bp$statistic, bp$p.value,
            ifelse(bp$p.value < 0.05, "→ use robust SEs", "→ homoscedastic")))
cat(sprintf("Jarque-Bera     (normality resid.):  stat=%.3f  p=%.4f  %s\n",
            jb$statistic, jb$p.value,
            ifelse(jb$p.value < 0.05, "→ fat tails (expected)", "→ approx. normal")))

cat("\nNote: NW SEs applied regardless as best practice for financial time series.\n")

# =============================================================================
# 8. STRUCTURAL BREAK
# =============================================================================
cat("\n", strrep("=", 70), "\n")
cat("STRUCTURAL BREAK — Bai-Perron on dGEP_z coefficient\n")
cat(strrep("=", 70), "\n")

df_clean <- df %>% filter(!is.na(log_ret) & !is.na(dGEP_z) & !is.na(GPR_z))
bp_struct <- breakpoints(log_ret ~ dGEP_z + GPR_z, data = df_clean, h = 0.15)
print(summary(bp_struct))

# optimal number of breaks by BIC
opt_m <- which.min(BIC(bp_struct)) - 1   # -1 because index 1 = m=0
cat(sprintf("\nOptimal breaks by BIC: m=%d\n", opt_m))
if (opt_m > 0 && !all(is.na(bp_struct$breakpoints))) {
  bp_dates <- df_clean$date[bp_struct$breakpoints]
  cat(sprintf("Breakpoint date(s): %s\n", paste(bp_dates, collapse = ", ")))
}

# =============================================================================
# 9. Subperiods
# =============================================================================
subperiods <- list(
  "Full sample (1996–2025)"        = df,
  "Pre-GFC (1996–2007)"            = df %>% filter(date < as.Date("2008-01-01")),
  "GFC & aftermath (2008–2011)"    = df %>% filter(date >= as.Date("2008-01-01") &
                                                    date <= as.Date("2011-12-31")),
  "Post-GFC (2012–2021)"           = df %>% filter(date >= as.Date("2012-01-01") &
                                                    date <= as.Date("2021-12-31")),
  "Russia–Ukraine war (2022–2023)" = df %>% filter(date >= as.Date("2022-02-01") &
                                                    date <= as.Date("2023-12-01")),
  "2025 (Liberation Day shock)"    = df %>% filter(date >= as.Date("2025-01-01"))
)

# =============================================================================
# 10. Regressions — 4 specifications, Newey-West SEs
#
#  m1: log_ret ~ dGEP_z
#  m2: log_ret ~ dGEP_z + GPR_z
#  m3: log_ret ~ dGEP_z + GPR_z + MktRF + SMB + HML
#  m4: log_ret ~ dGEP_z + dGEP_z_lag1 + GPR_z + GPR_z_lag1 + MktRF + SMB + HML
# =============================================================================
run_models <- function(d, h) {
  y_col <- if (h == 0) "log_ret" else "log_ret_lead1"

  d <- d %>%
    filter(!is.na(.data[[y_col]]) &
           !is.na(dGEP_z) & !is.na(dGEP_z_lag1) &
           !is.na(GPR_z)  & !is.na(GPR_z_lag1)  &
           !is.na(MktRF)  & !is.na(SMB) & !is.na(HML))

  if (nrow(d) < 24) {
    cat(sprintf("  Skipping h=%d: only %d complete obs\n", h, nrow(d)))
    return(NULL)
  }

  nw_lag <- max(1, floor(4 * (nrow(d) / 100)^(2/9)))

  safe_nw <- function(formula, data) {
    tryCatch({
      fit <- lm(formula, data = data)
      ct  <- coeftest(fit, vcov = NeweyWest(fit, lag = nw_lag, prewhite = FALSE))
      list(fit = fit, ct = ct, r2 = summary(fit)$r.squared,
           adj_r2 = summary(fit)$adj.r.squared)
    }, error = function(e) NULL)
  }

  list(
    m1 = safe_nw(as.formula(paste(y_col, "~ dGEP_z")), d),
    m2 = safe_nw(as.formula(paste(y_col, "~ dGEP_z + GPR_z")), d),
    m3 = safe_nw(as.formula(paste(y_col, "~ dGEP_z + GPR_z + MktRF + SMB + HML")), d),
    m4 = safe_nw(as.formula(paste(y_col,
         "~ dGEP_z + dGEP_z_lag1 + GPR_z + GPR_z_lag1 + MktRF + SMB + HML")), d),
    n      = nrow(d),
    nw_lag = nw_lag,
    h      = h
  )
}

all_results <- list()
for (period_name in names(subperiods)) {
  for (h in c(0, 1)) {
    key <- paste0(period_name, " | h=", h)
    all_results[[key]] <- run_models(subperiods[[period_name]], h)
  }
}

# =============================================================================
# 11. Print plain-text summaries
# =============================================================================
cat("\n", strrep("=", 70), "\n")
cat("REGRESSION RESULTS — plain text\n")
cat(strrep("=", 70), "\n")

for (period_name in names(subperiods)) {
  for (h in c(0, 1)) {
    key    <- paste0(period_name, " | h=", h)
    models <- all_results[[key]]
    if (is.null(models)) next

    cat(sprintf("\n▶ %s  |  h=%d  |  N=%d  |  NW lag=%d\n",
                period_name, h, models$n, models$nw_lag))

    for (mn in c("m1","m2","m3","m4")) {
      m <- models[[mn]]
      if (is.null(m)) next
      cat(sprintf("  %s  R²=%.4f  Adj.R²=%.4f\n", mn, m$r2, m$adj_r2))
      print(m$ct)
    }
  }
}

# =============================================================================
# 12. GT tables
# =============================================================================
extract_coef <- function(models, var) {
  sapply(1:4, function(i) {
    mn <- paste0("m", i)
    m  <- models[[mn]]
    if (is.null(m)) return("—")
    ct <- m$ct
    if (!var %in% rownames(ct)) return("—")
    est   <- ct[var, "Estimate"]
    pv    <- ct[var, "Pr(>|t|)"]
    stars <- ifelse(pv < 0.001, "***", ifelse(pv < 0.01, "**",
              ifelse(pv < 0.05, "*",   ifelse(pv < 0.10, ".", ""))))
    sprintf("%.4f%s", est, stars)
  })
}

extract_r2 <- function(models) {
  sapply(1:4, function(i) {
    m <- models[[paste0("m", i)]]
    if (is.null(m)) "—" else sprintf("%.4f", m$r2)
  })
}

vars_display <- c("dGEP_z", "dGEP_z_lag1", "GPR_z", "GPR_z_lag1",
                  "MktRF", "SMB", "HML", "(Intercept)")
var_labels   <- c("ΔGEP (z)", "ΔGEP lag1 (z)", "GPR (z)", "GPR lag1 (z)",
                  "Mkt-RF", "SMB", "HML", "Intercept")

for (h in c(0, 1)) {
  h_label <- if (h == 0) "h = 0  (contemporaneous)" else "h = 1  (next-month)"

  tbl_rows <- list()

  for (period_name in names(subperiods)) {
    key    <- paste0(period_name, " | h=", h)
    models <- all_results[[key]]
    n_obs  <- if (is.null(models)) 0 else models$n
    nw     <- if (is.null(models)) "—" else models$nw_lag

    # section header
    tbl_rows[[paste0(period_name, "_hdr")]] <- data.frame(
      Variable   = paste0("── ", period_name, "  (N=", n_obs, ", NW lag=", nw, ")"),
      `(1)`= "", `(2)`= "", `(3)`= "", `(4)`= "",
      check.names = FALSE
    )

    for (vi in seq_along(vars_display)) {
      row <- extract_coef(models, vars_display[vi])
      tbl_rows[[paste0(period_name, "_", vars_display[vi])]] <- data.frame(
        Variable = paste0("   ", var_labels[vi]),
        `(1)` = row[1], `(2)` = row[2], `(3)` = row[3], `(4)` = row[4],
        check.names = FALSE
      )
    }

    r2 <- extract_r2(models)
    tbl_rows[[paste0(period_name, "_R2")]] <- data.frame(
      Variable = "   R²",
      `(1)` = r2[1], `(2)` = r2[2], `(3)` = r2[3], `(4)` = r2[4],
      check.names = FALSE
    )
  }

  tbl <- do.call(rbind, tbl_rows)
  rownames(tbl) <- NULL

  gt_tbl <- tbl %>%
    gt() %>%
    tab_header(
      title    = md(paste0("**Return Predictability — Monthly — ", h_label, "**")),
      subtitle = md(paste0(
        "*Dep. var.: S&P 500 monthly log return. ",
        "ΔGEP = first difference of z-scored GEP (non-stationary in levels). ",
        "GPR in levels (stationary). Newey-West HAC SEs.*"
      ))
    ) %>%
    cols_label(
      Variable = "Variable",
      `(1)`    = "(1) ΔGEP",
      `(2)`    = "(2) +GPR",
      `(3)`    = "(3) +FF3",
      `(4)`    = "(4) +Lags"
    ) %>%
    tab_style(
      style     = cell_text(weight = "bold", color = "#1A3A5C"),
      locations = cells_body(rows = startsWith(Variable, "──"))
    ) %>%
    tab_style(
      style     = cell_fill(color = "#EAF2FB"),
      locations = cells_body(rows = startsWith(Variable, "──"))
    ) %>%
    tab_style(
      style     = cell_fill(color = "#F0F7FF"),
      locations = cells_body(rows = trimws(Variable) %in%
                               c("ΔGEP (z)", "ΔGEP lag1 (z)"))
    ) %>%
    tab_style(
      style     = cell_fill(color = "#F5F5F5"),
      locations = cells_body(rows = trimws(Variable) == "R²")
    ) %>%
    tab_source_note(
      "Significance: *** p<0.001  ** p<0.01  * p<0.05  . p<0.10  |  Newey-West HAC SEs"
    ) %>%
    tab_options(
      table.font.size           = px(12),
      column_labels.font.weight = "bold",
      data_row.padding          = px(3)
    )

  print(gt_tbl)

  out <- file.path(BASE_GEP, sprintf("regression_monthly_dGEP_h%d.html", h))
  gtsave(gt_tbl, out)
  cat(sprintf("Saved: %s\n", out))
}