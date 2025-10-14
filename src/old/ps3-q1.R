# Provide descriptive statistics that describe the key variables

if (!require(pacman)) install.packages("pacman")
pacman::p_load(here, data.table, modelsummary, kableExtra)

# set working directory
here::i_am("src/ps3-q1.R")

# load data
data <- fread(here("input", "PS3_data.csv"))

# describe the data
datasummary(market_share + calories + organic + price ~ N + mean + sd + min + max, data = data, format = "latex", output = "kableExtra") |>
    writeLines(here("output", "tables", "summary_stats.tex"))

