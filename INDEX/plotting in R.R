install.packages(c("tidyverse", "quantmod", "patchwork"))

library(tidyverse)
library(quantmod)
library(patchwork) # Per gestire layout complessi se necessario

# 1. Caricamento e preparazione dati GEP
df <- read_csv('INDEX/GEP_Daily_Index.csv') %>%
  mutate(date = as.Date(date))

# Resampling mensile (Media) e Scalatura
monthly_gep <- df %>%
  group_by(date = floor_date(date, "month")) %>%
  summarise(score = mean(score, na.rm = TRUE) * 10000)

# 2. Scarica i dati S&P 500 (^GSPC)
getSymbols("^GSPC", src = "yahoo", from = "1996-01-01", to = "2025-12-31", auto.assign = TRUE)
sp500_raw <- data.frame(date = index(GSPC), coredata(GSPC)) %>%
  rename(Adj_Close = GSPC.Adjusted)

# Resampling mensile S&P 500 (Ultimo valore del mese)
sp500_monthly <- sp500_raw %>%
  group_by(date = floor_date(date, "month")) %>%
  summarise(Adj_Close = last(Adj_Close))

# Unione dati per il Plot 1
data_combined <- inner_join(monthly_gep, sp500_monthly, by = "date")

# ---------------------------------------------------------
# PLOT 1: GEP INDEX VS S&P 500 (Doppio Asse Y)
# ---------------------------------------------------------
# Calcolo coefficiente di scala per il secondo asse
coeff <- max(data_combined$Adj_Close) / max(data_combined$score)

p1 <- ggplot(data_combined, aes(x = date)) +
  # Linea GEP
  geom_line(aes(y = score, color = "GEP Index"), size = 1) +
  # Linea S&P 500 (scalata per stare nel grafico)
  geom_line(aes(y = Adj_Close / coeff, color = "S&P 500"), size = 1) +
  # Configurazione Doppio Asse
  scale_y_continuous(
    name = "GEP Exposure Score (Scaled x10,000)",
    sec.axis = sec_axis(~.*coeff, name = "S&P 500 (Price)")
  ) +
  scale_color_manual(values = c("GEP Index" = "#1f77b4", "S&P 500" = "#ff7f0e")) +
  labs(title = "Correlation: Monthly GEP vs S&P 500 (1996 - 2025)", x = "Anno") +
  theme_minimal() +
  theme(legend.position = "top", 
        axis.title.y.left = element_text(color = "#1f77b4", face = "bold"),
        axis.title.y.right = element_text(color = "#ff7f0e", face = "bold"))

print(p1)
