#!/usr/bin/env Rscript
# =============================================================================
# GEP vs S&P 500 — Correlation Analysis during Crisis Periods
#
# Three complementary views:
#   1. Rolling 90-day Pearson correlation (full sample time series)
#   2. Crisis-window Pearson r + 95% CI (bar chart per episode)
#   3. Scatter plots: GEP z-score vs S&P 500 log-return, per crisis window
# =============================================================================

.libPaths(c("/home/h12429576/R_libs", .libPaths()))

suppressPackageStartupMessages({
  library(dplyr)
  library(lubridate)
  library(quantmod)
  library(ggplot2)
  library(patchwork)
  library(scales)
  library(zoo)
})

BASE <- "/Users/catepiacentini/Desktop/tesi/Master_Thesis/INDEX/index_8_revised"

# =============================================================================
# 1. Load and merge data
# =============================================================================
gep <- read.csv(file.path(BASE, "GEP_Daily_Index.csv"),
                stringsAsFactors = FALSE) %>%
  mutate(date = as.Date(date)) %>%
  filter(n_articles > 0) %>%
  select(date, GEP = score) %>%
  arrange(date)

cat(sprintf("GEP: %d days  (%s → %s)\n", nrow(gep), min(gep$date), max(gep$date)))

cat("Downloading S&P 500...\n")
getSymbols("^GSPC", from = "1995-12-29", to = "2025-12-31",
           auto.assign = TRUE, warnings = FALSE)

sp500 <- data.frame(
  date  = as.Date(index(GSPC)),
  close = as.numeric(Ad(GSPC))
) %>%
  arrange(date) %>%
  mutate(log_ret = c(NA, diff(log(close)))) %>%
  filter(!is.na(log_ret)) %>%
  select(date, log_ret)

df <- inner_join(sp500, gep, by = "date") %>%
  filter(date >= as.Date("1996-01-01")) %>%
  arrange(date) %>%
  mutate(
    GEP_z   = (GEP   - mean(GEP,   na.rm = TRUE)) / sd(GEP,   na.rm = TRUE),
    ret_z   = (log_ret - mean(log_ret, na.rm = TRUE)) / sd(log_ret, na.rm = TRUE)
  )

cat(sprintf("Merged: %d trading days  (%s → %s)\n",
            nrow(df), min(df$date), max(df$date)))

# Overall correlation
r_all <- cor(df$GEP_z, df$ret_z, use = "complete.obs")
cat(sprintf("\nOverall Pearson r(GEP, S&P500 log-ret) = %.4f\n", r_all))

# =============================================================================
# 2. Crisis window definitions
# =============================================================================
crises <- list(
  list(label = "Asian\nCrisis",      start = "1997-07-02", end = "1998-01-31",  colour = "#E05C2A"),
  list(label = "9/11",               start = "2001-09-11", end = "2001-12-31",  colour = "#C0392B"),
  list(label = "Iraq War",           start = "2003-03-20", end = "2003-09-30",  colour = "#884EA0"),
  list(label = "GFC",                start = "2008-09-01", end = "2009-06-30",  colour = "#17202A"),
  list(label = "COVID-19",           start = "2020-02-20", end = "2020-06-30",  colour = "#1A5276"),
  list(label = "Russia–\nUkraine",   start = "2022-02-24", end = "2022-12-31",  colour = "#BA4A00"),
  list(label = "Liberation\nDay",    start = "2025-04-02", end = "2025-10-31",  colour = "#148F77")
)

# =============================================================================
# 3. Rolling 90-day correlation
# =============================================================================
df <- df %>%
  mutate(
    roll_corr = rollapply(
      data.frame(GEP_z, ret_z), width = 90,
      FUN = function(m) cor(m[, 1], m[, 2], use = "complete.obs"),
      by.column = FALSE, align = "right", fill = NA
    )
  )

# =============================================================================
# 4. Plot 1 — Rolling correlation with crisis shading
# =============================================================================
crisis_rects <- bind_rows(lapply(crises, function(cr)
  data.frame(xmin = as.Date(cr$start), xmax = as.Date(cr$end),
             colour = cr$colour, label = gsub("\n", " ", cr$label),
             stringsAsFactors = FALSE)
))

p_rolling <- ggplot(df, aes(x = date, y = roll_corr)) +
  # crisis shading
  geom_rect(data = crisis_rects,
            aes(xmin = xmin, xmax = xmax, ymin = -Inf, ymax = Inf, fill = label),
            inherit.aes = FALSE, alpha = 0.12) +
  scale_fill_manual(values = setNames(crisis_rects$colour, crisis_rects$label),
                    name = "Crisis window") +
  # correlation ribbon + line
  geom_hline(yintercept = 0,     colour = "grey40", linewidth = 0.4) +
  geom_hline(yintercept = r_all, colour = "grey60", linewidth = 0.5,
             linetype = "dashed") +
  geom_ribbon(aes(ymin = pmin(roll_corr, 0), ymax = 0),
              fill = "#C0392B", alpha = 0.25, na.rm = TRUE) +
  geom_ribbon(aes(ymin = 0, ymax = pmax(roll_corr, 0)),
              fill = "#1A5276", alpha = 0.25, na.rm = TRUE) +
  geom_line(colour = "#2C3E50", linewidth = 0.65, na.rm = TRUE) +
  annotate("text", x = max(df$date), y = r_all + 0.03,
           label = sprintf("Overall r = %.3f", r_all),
           hjust = 1, size = 3, colour = "grey50") +
  scale_x_date(date_breaks = "2 years", date_labels = "%Y",
               expand = expansion(mult = 0.01)) +
  scale_y_continuous(breaks = seq(-1, 1, 0.2), limits = c(-1, 1)) +
  labs(
    title    = "Rolling 90-day Pearson Correlation: GEP vs S&P 500 Daily Log-Return",
    subtitle = "Blue fill = positive correlation  |  Red fill = negative correlation  |  Dashed = full-sample r",
    x = NULL, y = "Pearson r  (90-day window)"
  ) +
  theme_minimal(base_size = 11) +
  theme(
    panel.grid.minor  = element_blank(),
    legend.position   = "bottom",
    legend.text       = element_text(size = 8),
    legend.title      = element_text(size = 8),
    plot.title        = element_text(face = "bold", size = 12),
    plot.subtitle     = element_text(colour = "grey40", size = 9)
  ) +
  guides(fill = guide_legend(nrow = 1, override.aes = list(alpha = 0.5)))

# =============================================================================
# 5. Crisis-window Pearson r + 95 % CI
# =============================================================================
crisis_cors <- bind_rows(lapply(crises, function(cr) {
  sub <- df %>% filter(date >= as.Date(cr$start), date <= as.Date(cr$end))
  n   <- nrow(sub)
  if (n < 10) return(NULL)
  ct  <- cor.test(sub$GEP_z, sub$ret_z, method = "pearson")
  data.frame(
    label  = gsub("\n", "\n", cr$label),
    r      = ct$estimate,
    lo     = ct$conf.int[1],
    hi     = ct$conf.int[2],
    p      = ct$p.value,
    n      = n,
    colour = cr$colour,
    start  = as.Date(cr$start),
    stringsAsFactors = FALSE
  )
}))

crisis_cors <- crisis_cors %>%
  mutate(
    sig   = case_when(p < 0.001 ~ "***", p < 0.01 ~ "**",
                      p < 0.05  ~ "*",   p < 0.10 ~ ".", TRUE ~ ""),
    label = factor(label, levels = label[order(start)])
  )

p_bar <- ggplot(crisis_cors, aes(x = label, y = r, fill = colour)) +
  geom_hline(yintercept = 0,     colour = "grey40", linewidth = 0.4) +
  geom_hline(yintercept = r_all, colour = "grey60", linewidth = 0.5,
             linetype = "dashed") +
  geom_col(width = 0.6, alpha = 0.85) +
  geom_errorbar(aes(ymin = lo, ymax = hi), width = 0.25,
                colour = "grey30", linewidth = 0.7) +
  geom_text(aes(y = ifelse(r >= 0, hi + 0.04, lo - 0.04), label = sig),
            size = 4.5, colour = "grey20") +
  geom_text(aes(y = ifelse(r >= 0, lo - 0.04, hi + 0.04),
                label = paste0("n=", n)),
            size = 2.8, colour = "grey40") +
  scale_fill_identity() +
  scale_y_continuous(breaks = seq(-1, 1, 0.1), limits = c(-0.6, 0.6)) +
  annotate("text", x = Inf, y = r_all + 0.03,
           label = sprintf("Full-sample r = %.3f", r_all),
           hjust = 1.05, size = 3, colour = "grey50") +
  labs(
    title    = "Pearson r (GEP, S&P 500 log-ret) by Crisis Window",
    subtitle = "Error bars = 95% CI  |  * p<0.05  ** p<0.01  *** p<0.001  |  Dashed = full-sample r",
    x = NULL, y = "Pearson r"
  ) +
  theme_minimal(base_size = 11) +
  theme(
    panel.grid.minor  = element_blank(),
    panel.grid.major.x = element_blank(),
    plot.title        = element_text(face = "bold", size = 12),
    plot.subtitle     = element_text(colour = "grey40", size = 9)
  )

# =============================================================================
# 6. Plot 3 — Scatter: GEP_z vs log_ret per crisis (faceted)
# =============================================================================
crisis_scatter <- bind_rows(lapply(crises, function(cr) {
  df %>%
    filter(date >= as.Date(cr$start), date <= as.Date(cr$end)) %>%
    mutate(crisis = gsub("\n", " ", cr$label), colour = cr$colour)
})) %>%
  mutate(crisis = factor(crisis, levels = sapply(crises, function(cr)
    gsub("\n", " ", cr$label))))

# per-crisis r for strip labels
crisis_r <- crisis_scatter %>%
  group_by(crisis) %>%
  summarise(r = cor(GEP_z, ret_z, use = "complete.obs"),
            n = n(), .groups = "drop") %>%
  mutate(label = sprintf("r = %.3f  (n=%d)", r, n))

p_scatter <- ggplot(crisis_scatter, aes(x = GEP_z, y = log_ret)) +
  geom_hline(yintercept = 0, colour = "grey70", linewidth = 0.3) +
  geom_vline(xintercept = 0, colour = "grey70", linewidth = 0.3) +
  geom_point(aes(colour = colour), alpha = 0.45, size = 1.1) +
  geom_smooth(method = "lm", formula = y ~ x,
              colour = "#2C3E50", fill = "#AED6F1",
              linewidth = 0.8, alpha = 0.3) +
  geom_text(data = crisis_r,
            aes(label = label),
            x = -Inf, y = Inf, hjust = -0.05, vjust = 1.4,
            size = 2.8, colour = "grey30", inherit.aes = FALSE) +
  scale_colour_identity() +
  scale_y_continuous(labels = percent_format(accuracy = 1)) +
  facet_wrap(~ crisis, scales = "free", ncol = 4) +
  labs(
    title    = "GEP z-score vs S&P 500 Daily Log-Return: Scatter by Crisis Episode",
    subtitle = "OLS fit with 95% CI band shown in blue",
    x = "GEP (z-score)", y = "S&P 500 log-return"
  ) +
  theme_minimal(base_size = 10) +
  theme(
    panel.grid.minor   = element_blank(),
    strip.text         = element_text(face = "bold", size = 9),
    plot.title         = element_text(face = "bold", size = 12),
    plot.subtitle      = element_text(colour = "grey40", size = 9)
  )

# =============================================================================
# 7. Save
# =============================================================================
ggsave(file.path(BASE, "gep_sp500_rolling_corr.png"),
       p_rolling, width = 14, height = 5, dpi = 150)
cat("Saved: gep_sp500_rolling_corr.png\n")

ggsave(file.path(BASE, "gep_sp500_crisis_bar.png"),
       p_bar, width = 10, height = 5, dpi = 150)
cat("Saved: gep_sp500_crisis_bar.png\n")

ggsave(file.path(BASE, "gep_sp500_crisis_scatter.png"),
       p_scatter, width = 14, height = 7, dpi = 150)
cat("Saved: gep_sp500_crisis_scatter.png\n")

# =============================================================================
# 8. Print crisis correlation table
# =============================================================================
cat("\n", strrep("─", 65), "\n")
cat(sprintf("%-30s  %5s  %6s  %6s  %6s  %s\n",
            "Crisis", "N", "r", "95% lo", "95% hi", "p-value"))
cat(strrep("─", 65), "\n")
for (i in seq_len(nrow(crisis_cors))) {
  cr <- crisis_cors[i, ]
  cat(sprintf("%-30s  %5d  %6.3f  %6.3f  %6.3f  %.4f %s\n",
              gsub("\n", " ", as.character(cr$label)),
              cr$n, cr$r, cr$lo, cr$hi, cr$p, cr$sig))
}
cat(strrep("─", 65), "\n")
cat(sprintf("%-30s  %5d  %6.3f\n", "Full sample", nrow(df), r_all))
