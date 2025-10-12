# prepare data for random coefficient model

if (!require(pacman)) install.packages("pacman")
pacman::p_load(here, data.table)

# set working directory
here::i_am("src/ps3-q3-prep.R")

# load data
data <- fread(here("input", "PS3_data.csv"),
    colClasses = list(character = c("market", "period"))
)

# load random draw data
random_draws <- fread(here("input", "PS3_rnd.csv"),
    colClasses = list(character = c("market", "period"))
)

# create a single market id
data[, mkt_id := paste0(market, "-", period)]
random_draws[, mkt_id := paste0(market, "-", period)]

# create outside option
data[, outside_share := 1 - sum(market_share), by = mkt_id]

# create delta from log share - log outside share
data[, delta_initial := log(market_share) - log(outside_share)]


# Hausman IV regression
# average price of the product in other markets in the same period t

data[, haus_iv := {
    if (.N > 1) {
        (sum(price, na.rm = TRUE) - price) / (.N - 1)
    } else {
        NA_real_
    }
},
by = .(period, product_id)
]

# BLP IV regression

## number of competing products
data[, n_competing := .N - 1, by = "mkt_id"]

## average calories of competing products
data[, avg_calories := (sum(calories, na.rm = TRUE) - calories) / n_competing, by = "mkt_id"]

## number of organic products
data[, n_organic := sum(organic, na.rm = TRUE), by = "mkt_id"]
data[, n_organic := n_organic - organic]

## number of organic products interacted with organic dummy
data[, n_organic_interacted := paste0(as.character(organic), "-", as.character(n_organic))]

## min closest competitor in calories space
data[, closest_calories := {
    sapply(seq_len(.N), function(i) {
        if (.N == 1) {
            NA_real_
        } else {
            min(abs(calories[i] - calories[-i]), na.rm = TRUE)
        }
    })
}, by = "mkt_id"]

# separately save IV
data[, .(mkt_id, haus_iv, n_competing, avg_calories, n_organic, n_organic_interacted, closest_calories)] |>
    fwrite(here("input", "ps3-q3-iv.csv"))

# save the data for estimation
fwrite(data[, .(market, period, product_id, mkt_id, market_share, calories, organic, price, outside_share, delta_initial)], here("input", "ps3-q3-data.csv"))
fwrite(random_draws, here("input", "ps3-q3-random_draws.csv"))
