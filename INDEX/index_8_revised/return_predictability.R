#!/usr/bin/env Rscript
# =============================================================================
# Return Predictability Regression
# R_{x,t+h} = α + β_GEP * GEP_t + β_GPR * GPR_t + β'_ctrl * X_t + u_{t+h}
#
# h = 0 : contemporaneous  (same-day return)
# h = 1 : predictive       (next-day return)
#
# Y  : S&P 500 daily log returns  (stationary, ~Normal — correct for OLS)
# X_t: Fama-French 3 Factors (Mkt-RF, SMB, HML) + risk-free rate
# Standard errors: Newey-West HAC (robust to heteroscedasticity & autocorr.)
# =============================================================================

.libPaths(c("/home/h12429576/R_libs", .libPaths()))

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(lubridate)
  library(readxl)
  library(quantmod)
  library(frenchdata)
  library(sandwich)
  library(lmtest)
  library(broom)
  library(gt)
})

BASE <- dirname(normalizePath(sys.frame(1)$ofile, mustWork = FALSE))
if (!nzchar(BASE) || BASE == ".") BASE <- getwd()

cat("Working directory:", BASE, "\n")

# =============================================================================
# 1. Load GEP daily index
# =============================================================================
gep <- read.csv(file.path(BASE, "GEP_Daily_Index.csv"), stringsAsFactors = FALSE)
gep$date <- as.Date(gep$date)
# keep trading days with articles; use 'score' (GEP_t^D in the thesis)
gep <- gep %>%
  filter(n_articles > 0) %>%
  select(date, GEP = score) %>%
  arrange(date)

cat(sprintf("GEP: %d trading days  (%s → %s)\n",
            nrow(gep), min(gep$date), max(gep$date)))

# =============================================================================
# 2. Download / load GPR daily index (Caldara & Iacoviello 2022)
# =============================================================================
gpr_path <- file.path(BASE, "data_gpr_daily_recent.xls")

if (!file.exists(gpr_path)) {
  cat("Downloading GPR daily index...\n")
  url <- "https://www.matteoiacoviello.com/gpr_files/data_gpr_daily_recent.xls"
  tryCatch(
    download.file(url, gpr_path, mode = "wb", quiet = TRUE),
    error = function(e) stop("Could not download GPR file. Download manually from:\n  ", url)
  )
}

gpr_raw <- read_excel(gpr_path)
# column names vary; find the date column and GPRD
date_col <- names(gpr_raw)[sapply(gpr_raw, function(x) inherits(x, "Date") || inherits(x, "POSIXct"))][1]
gpr_raw[[date_col]] <- as.Date(gpr_raw[[date_col]])
gpr <- gpr_raw %>%
  rename(date = all_of(date_col)) %>%
  filter(!is.na(GPRD)) %>%
  select(date, GPR = GPRD) %>%
  arrange(date)

cat(sprintf("GPR: %d days  (%s → %s)\n",
            nrow(gpr), min(gpr$date), max(gpr$date)))

# =============================================================================
# 3. Download S&P 500 daily prices → log returns
# =============================================================================
cat("Downloading S&P 500...\n")
getSymbols("^GSPC", from = "1995-12-29", to = "2025-12-31",
           auto.assign = TRUE, warnings = FALSE)
sp500 <- data.frame(
  date    = as.Date(index(GSPC)),
  close   = as.numeric(Ad(GSPC))   # adjusted close
)
sp500 <- sp500 %>%
  arrange(date) %>%
  mutate(log_ret = c(NA, diff(log(close)))) %>%
  filter(!is.na(log_ret)) %>%
  select(date, log_ret)

cat(sprintf("S&P 500: %d trading days  (%s → %s)\n",
            nrow(sp500), min(sp500$date), max(sp500$date)))

# =============================================================================
# 4. Download Fama-French 3 Factors (daily)
# =============================================================================
cat("Downloading Fama-French 3 factors...\n")
ff3_raw  <- download_french_data("Fama/French 3 Factors [Daily]")
ff3_data <- ff3_raw$subsets$data[[1]]

ff3 <- ff3_data %>%
  mutate(date = as.Date(as.character(date), format = "%Y%m%d")) %>%
  rename(MktRF = `Mkt-RF`) %>%
  mutate(across(c(MktRF, SMB, HML, RF), ~ as.numeric(.) / 100)) %>%  # % → decimal
  select(date, MktRF, SMB, HML, RF) %>%
  filter(!is.na(MktRF))

cat(sprintf("FF3: %d days  (%s → %s)\n",
            nrow(ff3), min(ff3$date), max(ff3$date)))

# =============================================================================
# 5. Merge all series on common trading days
# =============================================================================
df <- sp500 %>%
  inner_join(gep,  by = "date") %>%
  inner_join(gpr,  by = "date") %>%
  inner_join(ff3,  by = "date") %>%
  arrange(date) %>%
  filter(date >= as.Date("1996-01-01"))

cat(sprintf("Merged dataset: %d observations  (%s → %s)\n",
            nrow(df), min(df$date), max(df$date)))

# =============================================================================
# 6. Construct lead return for h = 1
# =============================================================================
df <- df %>%
  mutate(log_ret_lead1 = lead(log_ret, 1))

# =============================================================================
# 7. Regressions with Newey-West standard errors
#    h = 0: R_{t}   ~ GEP_t + GPR_t + MktRF_t + SMB_t + HML_t + RF_t
#    h = 1: R_{t+1} ~ GEP_t + GPR_t + MktRF_t + SMB_t + HML_t + RF_t
# =============================================================================
run_reg <- function(data, h) {
  if (h == 0) {
    y_col  <- "log_ret"
    y_label <- "R[t] (h=0)"
  } else {
    y_col  <- "log_ret_lead1"
    y_label <- "R[t+1] (h=1)"
  }

  d <- data %>% filter(!is.na(.data[[y_col]]))
  formula_str <- paste(y_col, "~ GEP + GPR + MktRF + SMB + HML + RF")
  fit <- lm(as.formula(formula_str), data = d)

  # Newey-West HAC: lag = floor(4*(T/100)^(2/9)) — standard rule of thumb
  T_obs <- nrow(d)
  nw_lag <- floor(4 * (T_obs / 100)^(2/9))
  cat(sprintf("\n[h=%d] N=%d  Newey-West lag=%d\n", h, T_obs, nw_lag))

  nw_se  <- NeweyWest(fit, lag = nw_lag, prewhite = FALSE)
  ct     <- coeftest(fit, vcov = nw_se)

  # Tidy into a data frame
  res <- as.data.frame(ct)
  res$term <- rownames(res)
  rownames(res) <- NULL
  names(res) <- c("estimate", "std_error", "t_stat", "p_value", "term")
  res <- res[, c("term", "estimate", "std_error", "t_stat", "p_value")]

  # Pretty term labels
  res$term <- recode(res$term,
    "(Intercept)" = "Intercept",
    "GEP"         = "GEP (β_GEP)",
    "GPR"         = "GPR (β_GPR)",
    "MktRF"       = "Mkt-RF",
    "SMB"         = "SMB",
    "HML"         = "HML",
    "RF"          = "Risk-Free Rate"
  )

  # Significance stars
  res$stars <- case_when(
    res$p_value < 0.001 ~ "***",
    res$p_value < 0.01  ~ "**",
    res$p_value < 0.05  ~ "*",
    res$p_value < 0.10  ~ ".",
    TRUE                ~ ""
  )

  list(fit = fit, coeftest = ct, tidy = res,
       r2 = summary(fit)$r.squared,
       adj_r2 = summary(fit)$adj.r.squared,
       n = T_obs, h = h)
}

reg_h0 <- run_reg(df, h = 0)
reg_h1 <- run_reg(df, h = 1)

# =============================================================================
# 8. Print plain-text summaries
# =============================================================================
cat("\n", strrep("=", 70), "\n")
cat("REGRESSION h=0 (contemporaneous)  — Newey-West HAC SEs\n")
cat(strrep("=", 70), "\n")
print(reg_h0$coeftest)
cat(sprintf("R²=%.4f   Adj. R²=%.4f   N=%d\n",
            reg_h0$r2, reg_h0$adj_r2, reg_h0$n))

cat("\n", strrep("=", 70), "\n")
cat("REGRESSION h=1 (next-day prediction) — Newey-West HAC SEs\n")
cat(strrep("=", 70), "\n")
print(reg_h1$coeftest)
cat(sprintf("R²=%.4f   Adj. R²=%.4f   N=%d\n",
            reg_h1$r2, reg_h1$adj_r2, reg_h1$n))

# =============================================================================
# 9. GT tables
# =============================================================================
make_gt <- function(reg, h_label) {
  footer <- sprintf(
    "N = %d  |  R² = %.4f  |  Adj. R² = %.4f  |  Newey-West HAC standard errors",
    reg$n, reg$r2, reg$adj_r2
  )

  reg$tidy %>%
    gt() %>%
    tab_header(
      title    = md(paste0("**Return Predictability Regression (", h_label, ")**")),
      subtitle = md("*R\\_{x,t+h} = α + β\\_{GEP} GEP\\_t + β\\_{GPR} GPR\\_t + β'X\\_t + u\\_{t+h}*")
    ) %>%
    cols_label(
      term      = "Variable",
      estimate  = "Coef.",
      std_error = "Std. Error",
      t_stat    = "t-stat",
      p_value   = "p-value",
      stars     = ""
    ) %>%
    fmt_number(columns = c(estimate, std_error, t_stat), decimals = 6) %>%
    fmt_number(columns = p_value, decimals = 4) %>%
    tab_style(
      style = cell_text(weight = "bold"),
      locations = cells_body(
        rows = term %in% c("GEP (β_GEP)", "GPR (β_GPR)")
      )
    ) %>%
    tab_style(
      style = cell_fill(color = "#F0F7FF"),
      locations = cells_body(
        rows = term %in% c("GEP (β_GEP)", "GPR (β_GPR)")
      )
    ) %>%
    tab_source_note(source_note = footer) %>%
    tab_source_note(
      source_note = "Significance: *** p<0.001  ** p<0.01  * p<0.05  . p<0.10"
    ) %>%
    tab_options(
      table.font.size       = px(13),
      heading.title.font.size = px(15),
      column_labels.font.weight = "bold"
    )
}

gt_h0 <- make_gt(reg_h0, "h = 0, contemporaneous")
gt_h1 <- make_gt(reg_h1, "h = 1, next-day prediction")

# Save as HTML
out_h0 <- file.path(BASE, "regression_h0.html")
out_h1 <- file.path(BASE, "regression_h1.html")
gtsave(gt_h0, out_h0)
gtsave(gt_h1, out_h1)
cat(sprintf("\nSaved: %s\n", out_h0))
cat(sprintf("Saved: %s\n", out_h1))
