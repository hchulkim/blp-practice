# logit model estimation

if (!require(pacman)) install.packages("pacman")
pacman::p_load(here, data.table, fixest, texreg, broom, kableExtra)

# set working directory
here::i_am("src/ps3-q1.R")

# load data
data <- fread(here("input", "PS3_data.csv"),
    colClasses = list(character = c("market", "period"))
)

# create a single market id
data[, mkt_id := paste0(market, "-", period)]

# create outside option
data[, outside_share := 1 - sum(market_share), by = mkt_id]

# create delta from log share - log outside share
data[, delta := log(market_share) - log(outside_share)]

# ols regression
ols_model <- feols(delta ~ price + calories + organic, data = data)

screenreg(ols_model)

ols_model_dt <- tidy(ols_model) |> as.data.table()
ols_model_dt[, model := "OLS"]

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


haus_iv_model <- feols(delta ~ 1 | price ~ haus_iv + calories + organic, data = data)

screenreg(haus_iv_model)

haus_iv_model_dt <- tidy(haus_iv_model) |> as.data.table()
haus_iv_model_dt[, model := "hausman"]

# combine ols and hausman iv model
model_dt <- rbind(ols_model_dt, haus_iv_model_dt)



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



blp_iv_model <- feols(delta ~ 1 | price ~ n_competing + avg_calories + n_organic + n_organic_interacted + closest_calories + calories + organic, data = data)

screenreg(blp_iv_model)

blp_iv_model_dt <- tidy(blp_iv_model) |> as.data.table()
blp_iv_model_dt[, model := "blp"]

# combine ols and hausman iv model
model_dt <- rbind(model_dt, blp_iv_model_dt)

model_dt <- model_dt[term != "(Intercept)", .(model, term, estimate, std.error)]

# save model_dt
model_dt |>
    kbl(format = "latex", booktabs = TRUE, linesep = "", digits = 3) |>
    save_kable(here("output", "tables", "ps3-q2-model_dt.tex"))



vinay <- fread(here("src", "test_data.csv"))

data <- merge(data[, `:=`(market = as.numeric(market), period = as.numeric(period))], vinay, by = c("market", "period", "product_id"))

data[, `:=`(
    haus_check = haus_iv - hausman_iv,
    n_competing_check = n_competing - n_comp,
    avg_calories_check = avg_calories - avg_cal_comp,
    n_organic_check = n_organic - num_org_comp,
    closest_calories_check = closest_calories - closest_cal
)]
