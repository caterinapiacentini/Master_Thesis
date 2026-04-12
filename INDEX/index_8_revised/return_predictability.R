#!/usr/bin/env Rscript
# =============================================================================
# Return Predictability Regression — MONTHLY
# z-scored GEP & GPR, multiple subperiods
# =============================================================================

suppressPackageStartupMessages({
  library(dplyr)
  library(lubridate)
  library(readxl)
  library(quantmod)
  library(frenchdata)
  library(gt)
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

cat(sprintf("GEP monthly: %d months  (%s → %s)\n",
            nrow(gep), min(gep$date), max(gep$date)))

# =============================================================================
# 2. GPR — monthly file
# =============================================================================
gpr_path <- file.path(BASE_LIT, "data_gpr_export.xls")
gpr_raw  <- read_excel(gpr_path)

# inspect columns to find date and GPR value
cat("GPR columns:", paste(names(gpr_raw), collapse = ", "), "\n")

# find date column
date_col <- names(gpr_raw)[sapply(gpr_raw, function(x)
  inherits(x, "Date") || inherits(x, "POSIXct"))][1]

# if date col not auto-detected (sometimes stored as numeric/char), try first col
if (is.na(date_col)) date_col <- names(gpr_raw)[1]

gpr_raw[[date_col]] <- as.Date(gpr_raw[[date_col]])

# find GPR column — try common names
gpr <- gpr_raw %>%
  rename(date = all_of(date_col)) %>%
  mutate(
    date = floor_date(as.Date(date), "month"),
    GPR  = as.numeric(GPR)          # ← forza logical → numeric
  ) %>%
  filter(!is.na(GPR)) %>%
  select(date, GPR) %>%
  arrange(date)

cat(sprintf("GPR monthly: %d months  (%s → %s)\n",
            nrow(gpr), min(gpr$date), max(gpr$date)))



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
  slice_tail(n = 1) %>%          # last trading day of each month
  ungroup() %>%
  arrange(month) %>%
  mutate(log_ret = c(NA, diff(log(close)))) %>%
  filter(!is.na(log_ret)) %>%
  select(date = month, log_ret)

cat(sprintf("S&P 500 monthly: %d months  (%s → %s)\n",
            nrow(sp500), min(sp500$date), max(sp500$date)))

# =============================================================================
# 4. Fama-French 3 Factors — monthly
# =============================================================================
ff3_raw <- download_french_data("Fama/French 3 Factors")
ff3 <- ff3_raw$subsets$data[[1]] %>%
  mutate(date = as.Date(paste0(as.character(date), "01"), format = "%Y%m%d")) %>%
  mutate(date = floor_date(date, "month")) %>%
  rename(MktRF = `Mkt-RF`) %>%
  mutate(across(c(MktRF, SMB, HML, RF), ~ as.numeric(.) / 100)) %>%
  select(date, MktRF, SMB, HML, RF) %>%
  filter(!is.na(MktRF))

cat(sprintf("FF3 monthly: %d months  (%s → %s)\n",
            nrow(ff3), min(ff3$date), max(ff3$date)))

# =============================================================================
# 5. Merge + Z-score on full sample
# =============================================================================
df <- sp500 %>%
  inner_join(gep,  by = "date") %>%
  inner_join(gpr,  by = "date") %>%
  inner_join(ff3,  by = "date") %>%
  arrange(date) %>%
  filter(date >= as.Date("1996-01-01")) %>%
  mutate(
    GEP_z         = (GEP - mean(GEP, na.rm = TRUE)) / sd(GEP, na.rm = TRUE),
    GPR_z         = (GPR - mean(GPR, na.rm = TRUE)) / sd(GPR, na.rm = TRUE),
    log_ret_lead1 = lead(log_ret, 1)
  )

cat(sprintf("Merged monthly: %d obs  (%s → %s)\n",
            nrow(df), min(df$date), max(df$date)))

# =============================================================================
# 6. Subperiods
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
# 7. Regressions
# =============================================================================
run_models <- function(d, h) {
  y_col <- if (h == 0) "log_ret" else "log_ret_lead1"
  d     <- d %>% filter(!is.na(.data[[y_col]]) &
                        !is.na(GEP_z) & !is.na(GPR_z) &
                        !is.na(MktRF) & !is.na(SMB) & !is.na(HML))
  
  if (nrow(d) < 24) {
    cat(sprintf("  Skipping '%s' h=%d: only %d complete obs\n", y_col, h, nrow(d)))
    return(NULL)
  }

  safe_lm <- function(formula, data) {
    tryCatch(lm(formula, data = data), error = function(e) NULL)
  }

  list(
    m1 = safe_lm(as.formula(paste(y_col, "~ GEP_z")),                              d),
    m2 = safe_lm(as.formula(paste(y_col, "~ GEP_z + GPR_z")),                      d),
    m3 = safe_lm(as.formula(paste(y_col, "~ GEP_z + GPR_z + MktRF + SMB + HML")), d),
    n  = nrow(d),
    h  = h
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
# 8. Helpers
# =============================================================================
extract_coef <- function(models, var) {
  out <- character(3)
  for (i in 1:3) {
    mn <- paste0("m", i)
    if (is.null(models[[mn]])) { out[i] <- "—"; next }
    cf <- summary(models[[mn]])$coefficients
    if (!var %in% rownames(cf)) { out[i] <- "—"; next }
    est   <- cf[var, "Estimate"]
    pv    <- cf[var, "Pr(>|t|)"]
    stars <- ifelse(pv < 0.001, "***", ifelse(pv < 0.01, "**",
              ifelse(pv < 0.05, "*",   ifelse(pv < 0.10, ".", ""))))
    out[i] <- sprintf("%.4f%s", est, stars)
  }
  out
}

extract_r2 <- function(models) {
  sapply(1:3, function(i) {
    mn <- paste0("m", i)
    if (is.null(models[[mn]])) return("—")
    sprintf("%.4f", summary(models[[mn]])$r.squared)
  })
}

# =============================================================================
# 9. GT tables
# =============================================================================
vars_display <- c("GEP_z", "GPR_z", "MktRF", "SMB", "HML", "(Intercept)")
var_labels   <- c("GEP (z)", "GPR (z)", "Mkt-RF", "SMB", "HML", "Intercept")

for (h in c(0, 1)) {
  h_label <- if (h == 0) "h = 0  (contemporaneous)" else "h = 1  (next-month prediction)"

  tbl_rows <- list()

  for (period_name in names(subperiods)) {
    key    <- paste0(period_name, " | h=", h)
    models <- all_results[[key]]
    n_obs  <- if (is.null(models)) 0 else models$n

    tbl_rows[[paste0(period_name, "_header")]] <- data.frame(
      Variable  = paste0("── ", period_name, "  (N=", n_obs, " months)"),
      `(1) GEP` = "", `(2)+GPR` = "", `(3)+FF3` = "",
      check.names = FALSE
    )

    for (vi in seq_along(vars_display)) {
      v   <- vars_display[vi]
      row <- extract_coef(models, v)
      tbl_rows[[paste0(period_name, "_", v)]] <- data.frame(
        Variable  = paste0("   ", var_labels[vi]),
        `(1) GEP` = row[1], `(2)+GPR` = row[2], `(3)+FF3` = row[3],
        check.names = FALSE
      )
    }

    r2 <- extract_r2(models)
    tbl_rows[[paste0(period_name, "_R2")]] <- data.frame(
      Variable  = "   R²",
      `(1) GEP` = r2[1], `(2)+GPR` = r2[2], `(3)+FF3` = r2[3],
      check.names = FALSE
    )
  }

  tbl <- do.call(rbind, tbl_rows)
  rownames(tbl) <- NULL

  gt_tbl <- tbl %>%
    gt() %>%
    tab_header(
      title    = md(paste0("**Return Predictability — Monthly — ", h_label, "**")),
      subtitle = md("*GEP aggregated as monthly mean. GEP and GPR z-scored on full sample. OLS, plain SEs.*")
    ) %>%
    cols_label(
      Variable  = "Variable",
      `(1) GEP` = "(1) GEP only",
      `(2)+GPR` = "(2) + GPR",
      `(3)+FF3` = "(3) + FF3"
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
      locations = cells_body(rows = trimws(Variable) == "GEP (z)")
    ) %>%
    tab_style(
      style     = cell_fill(color = "#F5F5F5"),
      locations = cells_body(rows = trimws(Variable) == "R²")
    ) %>%
    tab_source_note("Significance: *** p<0.001  ** p<0.01  * p<0.05  . p<0.10") %>%
    tab_options(
      table.font.size           = px(12),
      column_labels.font.weight = "bold",
      data_row.padding          = px(3)
    )

  print(gt_tbl)
}
