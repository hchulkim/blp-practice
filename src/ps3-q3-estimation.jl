using DrWatson
@quickactivate
using DataFrames, CSV, LinearAlgebra, Statistics, Optim, NLopt

# load data
data = CSV.read(projectdir("input", "ps3-q3-data.csv"), DataFrame)
iv = CSV.read(projectdir("input", "ps3-q3-iv.csv"), DataFrame)
rnd = CSV.read(projectdir("input", "ps3-q3-random_draws.csv"), DataFrame)

# set starting parameters   
initial_theta_p = 1.0
initial_theta_b = 1.0
initial_theta_organic = 0.0

# get unique markets
mkts = unique(data.mkt_id)

# number of rows for data
data_rows = nrow(data)


#####################################################################
# BLP fixed point algorithm, inverting mkt shares to get mean utility
#####################################################################


# calculate product share for each market
# returns DataFrame with mkt_id and mkt_share
function calc_mkt_share_t(mkts::Vector{String7}, delta_t::DataFrame, data::DataFrame, rnd::DataFrame, theta_p::Float64, theta_b::Float64)
    # dimension: delta_t: (n_products in market t, 1), theta_p: (1, 1), theta_b: (1, 1), X_t: (n_products in market t, 3), Vi_t: (100, 2)

    mkt_share_t_df = DataFrame(mkt_id=String[], mkt_share=Float64[])
    for mkt in mkts
        delta_mkt_t = filter(:mkt_id => n -> n == mkt, delta_t)
        data_mkt_t = filter(:mkt_id => n -> n == mkt, data)
        rnd_mkt_t = filter(:mkt_id => n -> n == mkt, rnd)

        # Create matrices
        x_mkt_t = Matrix(select(data_mkt_t, [:price, :calories, :organic]))
        vi_mkt_t = hcat(Matrix(select(rnd_mkt_t, [:rnd_price, :rnd_calories])), zeros(size(rnd_mkt_t, 1), 1))

        # Create diagonal matrix (example with 3x3 identity matrix)
        sigma_mkt_t = Diagonal([theta_p, theta_b, 0.0])  # 3x3 diagonal matrix with ones on diagonal

        mu_mkt_t = (x_mkt_t * sigma_mkt_t * vi_mkt_t') .+ delta_mkt_t[:, 2]

        mkt_share_t = exp.(mu_mkt_t) ./ (1 .+ sum(exp.(mu_mkt_t), dims=1))

        mkt_share_t = mean(mkt_share_t, dims=2)

        # Convert matrices back to DataFrames with column names
        mkt_share_t = DataFrame(mkt_share_t, [:mkt_share])

        mkt_share_t.mkt_id = fill(mkt, nrow(mkt_share_t))

        mkt_share_t = select(mkt_share_t, :mkt_id, :mkt_share)

        # left join with mkt_share_t_df
        mkt_share_t_df = vcat(mkt_share_t_df, mkt_share_t)
    end

    return mkt_share_t_df
end

# apply contraction mapping to get mean utility
# returns DataFrame with mkt_id and delta_iter
function contraction_t(mkts::Vector{String7}, data::DataFrame, rnd::DataFrame, theta_p::Float64, theta_b::Float64, tol=1e-12, maxiter=1e5)
    # delta_new = delta_present + log(mkt_share_t) - log(mkt_share_t_predicted)
    obs_share_t = data.market_share
    delta_old = select(data, [:mkt_id, :delta_initial])
    normdiff = Inf
    iter = 0
    while normdiff > tol && iter < maxiter
        model_share_t_df = calc_mkt_share_t(mkts, delta_old, data, rnd, theta_p, theta_b)
        delta_new = delta_old[:, 2] .+ log.(obs_share_t) .- log.(model_share_t_df.mkt_share)
        normdiff = maximum(norm(delta_new .- delta_old[:, 2]))
        delta_old = DataFrame(mkt_id=model_share_t_df.mkt_id, delta_iter=delta_new)
        iter += 1
    end
    return delta_old
end

check = contraction_t(mkts, data, rnd, 1.0, 1.0)
