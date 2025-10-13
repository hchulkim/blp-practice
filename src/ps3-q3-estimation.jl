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
# Inner loop of BLP algorithm, inverting mkt shares to get mean utility
#####################################################################


# Precompute market-specific data structures for efficiency
struct MarketData
    mkt_id::String
    n_products::Int
    n_draws::Int
    x_matrix::Matrix{Float64}  # [price, calories, organic]
    vi_matrix::Matrix{Float64}  # [rnd_price, rnd_calories, zeros]
    delta_indices::Vector{Int}  # indices in full data
end

# Precompute all market data structures
function precompute_market_data(mkts::Vector{String7}, data::DataFrame, rnd::DataFrame)
    market_data = MarketData[]

    for mkt in mkts
        # Get indices for this market
        mkt_indices = findall(x -> x == mkt, data.mkt_id)
        rnd_indices = findall(x -> x == mkt, rnd.mkt_id)

        n_products = length(mkt_indices)
        n_draws = length(rnd_indices)

        # Create matrices directly from indices
        x_matrix = hcat(data.price[mkt_indices], data.calories[mkt_indices], data.organic[mkt_indices])
        vi_matrix = hcat(rnd.rnd_price[rnd_indices], rnd.rnd_calories[rnd_indices], zeros(n_draws))

        push!(market_data, MarketData(mkt, n_products, n_draws, x_matrix, vi_matrix, mkt_indices))
    end

    return market_data
end

# Optimized market share calculation using precomputed data
function calc_mkt_share_optimized(market_data::Vector{MarketData}, delta_values::Vector{Float64}, theta_p::Float64, theta_b::Float64)
    mkt_shares = Vector{Float64}()

    for mkt_data in market_data
        # Extract delta for this market
        delta_mkt = delta_values[mkt_data.delta_indices]

        # Create sigma matrix
        sigma = Diagonal([theta_p, theta_b, 0.0])

        # Compute mu = X * sigma * V' + delta
        mu = (mkt_data.x_matrix * sigma * mkt_data.vi_matrix') .+ delta_mkt

        # Compute market shares
        exp_mu = exp.(mu)
        denom = 1.0 .+ sum(exp_mu, dims=1)
        shares = exp_mu ./ denom

        # Average across draws
        avg_shares = vec(mean(shares, dims=2))

        append!(mkt_shares, avg_shares)
    end

    return mkt_shares
end

# Optimized contraction mapping using precomputed data
function contraction_optimized(market_data::Vector{MarketData}, obs_shares::Vector{Float64},
    initial_delta::Vector{Float64}, theta_p::Float64, theta_b::Float64,
    tol=1e-12, maxiter=1000)
    delta_old = copy(initial_delta)
    normdiff = Inf
    iter = 0

    while normdiff > tol && iter < maxiter
        # Calculate model shares
        model_shares = calc_mkt_share_optimized(market_data, delta_old, theta_p, theta_b)

        # Update delta: delta_new = delta_old + log(obs_share) - log(model_share)
        delta_new = delta_old .+ log.(obs_shares) .- log.(model_shares)

        # Check convergence
        normdiff = maximum(abs.(delta_new .- delta_old))
        delta_old = delta_new
        iter += 1

        # Print progress every 50 iterations
        if iter % 50 == 0
            println("Contraction iteration $iter, normdiff = $normdiff")
        end
    end

    println("Contraction converged in $iter iterations with normdiff = $normdiff")
    return delta_old
end

# Precompute all data structures once
println("Precomputing market data structures...")
market_data = precompute_market_data(mkts, data, rnd)

# Precompute IV matrix and inverse
iv_matrix = hcat(Matrix(select(iv, [:n_competing, :avg_calories, :n_organic, :closest_calories])), data.calories, data.organic)
Phi = (iv_matrix' * iv_matrix) / data_rows
inv_Phi = inv(Phi)

# Extract vectors for efficiency
obs_shares = Vector{Float64}(data.market_share)
initial_delta = Vector{Float64}(data.delta_initial)
price_vec = Vector{Float64}(data.price)
calories_vec = Vector{Float64}(data.calories)
organic_vec = Vector{Float64}(data.organic)

println("Data preprocessing complete. Starting optimization...")

# Optimized GMM objective function
function GMM_objective_optimized(theta::Vector{Float64})
    # theta = [theta_p, theta_b, alpha_0, alpha_1, alpha_2]
    theta_p, theta_b, alpha_0, alpha_1, alpha_2 = theta

    # Use optimized contraction mapping
    delta_values = contraction_optimized(market_data, obs_shares, initial_delta, theta_p, theta_b)

    # Calculate error term
    error = delta_values .- alpha_0 .* price_vec .- alpha_1 .* calories_vec .- alpha_2 .* organic_vec

    # GMM objective
    obj = error' * iv_matrix * inv_Phi * iv_matrix' * error

    return obj[1]  # Return scalar value
end

# Set up initial parameters for optimization with better starting values
# theta = [theta_p, theta_b, alpha_0, alpha_1, alpha_2]
# Use more reasonable starting values based on typical BLP estimates
initial_theta = [0.5, 0.5, -1.0, 0.1, 0.1]  # Better starting values

# Method 1: Using Optim.jl with improved settings
println("Optimizing with Optim.jl...")

# Set up optimization options with more aggressive convergence criteria
optim_options = Optim.Options(
    iterations=500,            # Reduced max iterations since we're faster now
    show_trace=true,           # Show optimization progress
    extended_trace=false,      # Don't show detailed trace
    show_every=5,              # Show progress every 5 iterations
    g_tol=1e-5,               # Gradient tolerance
    f_abstol=1e-6,             # Function tolerance
    x_abstol=1e-6              # Parameter tolerance
)

result_optim = Optim.optimize(GMM_objective_optimized,
    initial_theta,
    LBFGS(),
    optim_options)


# Extract results from Optim.jl
theta_optim = Optim.minimizer(result_optim)
gmm_value_optim = Optim.minimum(result_optim)

# Extract all parameters: theta = [theta_p, theta_b, alpha_0, alpha_1, alpha_2]
theta_p_optim = theta_optim[1]    # Price coefficient (random coefficient)
theta_b_optim = theta_optim[2]    # Calories coefficient (random coefficient)  
alpha0_optim = theta_optim[3]     # Price coefficient (mean utility)
alpha1_optim = theta_optim[4]     # Calories coefficient (mean utility)
alpha2_optim = theta_optim[5]     # Organic coefficient (mean utility)

println("\nOptim.jl Results:")
println("="^50)
println("Random Coefficients (Heterogeneity):")
println("  θ_p (price heterogeneity): ", theta_p_optim)
println("  θ_b (calories heterogeneity): ", theta_b_optim)
println()
println("Mean Utility Coefficients:")
println("  α₀ (price coefficient): ", alpha0_optim)
println("  α₁ (calories coefficient): ", alpha1_optim)
println("  α₂ (organic coefficient): ", alpha2_optim)
println()
println("GMM Objective Value: ", gmm_value_optim)
println("Converged: ", Optim.converged(result_optim))
println("Iterations: ", Optim.iterations(result_optim))
println("Reason for stopping: ", Optim.f_converged(result_optim) ? "Function converged" :
                                 Optim.g_converged(result_optim) ? "Gradient converged" :
                                 Optim.x_converged(result_optim) ? "Parameters converged" :
                                 Optim.iterations(result_optim) >= 1000 ? "Maximum iterations reached" : "Other")

# Check if optimization was successful
if !Optim.converged(result_optim)
    println("WARNING: Optimization did not converge!")
    println("Consider adjusting initial parameters or convergence tolerances.")
end