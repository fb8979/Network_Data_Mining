module FMM

using GraphRecipes, Plots, StatsBase, Distributions, Statistics, StatsPlots, SpecialFunctions, LogExpFunctions, Clustering, ColorTypes, Clustering, DataStructures, Distances, LinearAlgebra

function __init__()
    @eval using ..DPMM
end

# Performs gibbs sampling given the data, initial values and the hyperparameters
# Return MCMC samples
function gibbs_sampling(τ, init_As, init_Zs, init_αs, init_βs, init_π, init_ρ, D, K, Hu11, Hu01, Hu00, Hu10, a_star, b_star, γ, directed, M_ts)

    n = size(init_As)[1]
    n_obs = size(D)[end]

    As_samples = zeros(Bool, τ, n, n, K)
    Zs_samples = zeros(Int, τ, n_obs)
    αs_samples = zeros(Float64, τ, K)
    βs_samples = zeros(Float64, τ, K)
    π_samples = zeros(Float64, τ, K)
    ρ_samples = zeros(Float64, τ)

    As_samples[1, :, :, :] = init_As
    Zs_samples[1, :] = init_Zs
    αs_samples[1, :] = init_αs
    βs_samples[1, :] = init_βs
    π_samples[1, :] = init_π
    ρ_samples[1] = init_ρ

    gibbs_samples_likelihoods = []
    Y_tu = 0

    # In reality am tkaing τ - 1 samples

    for i in 2:τ

        X_u_ij = calc_X_u_ij(D, Zs_samples[i-1, :], K)

        # A ∼ p(A|Z, D, αs, βs, πs, ρs)
        As_samples[i, :, :, :] = draw_new_modes(X_u_ij, Zs_samples[i-1, :], ρ_samples[i-1], αs_samples[i-1, :], βs_samples[i-1, :], K, directed)

        M_us = calc_M_us(As_samples[i, :, :, :], K)
        Y_tu = calc_Y_tu(D, As_samples[i, :, :, :], directed, M_ts, M_us, K)

        # Z ∼ P(Z|D, As, αs, βs, πs)
        Zs_samples[i, :] = draw_new_assignments(D, αs_samples[i-1, :], βs_samples[i-1, :], π_samples[i-1, :], As_samples[i, :, :, :], Y_tu, directed)

        W_u = calc_W_u(Y_tu, Zs_samples[i, :], K)

        # αs ~ P(α| As, Zs)
        αs_samples[i, :] = calc_new_αs(W_u, Hu11, Hu01, K)

        # βs ~ P(β| As, Zs)
        βs_samples[i, :] = calc_new_βs(W_u, Hu00, Hu10, K)

        # π ~ P(π| Zs)
        π_samples[i, :] = calc_new_π(Zs_samples[i, :], γ, K)

        # ρ ~ P(ρ| As)
        ρ_samples[i] = calc_new_ρ(As_samples[i, :, :, :], a_star, b_star, K, directed)

    end

    return As_samples, Zs_samples, αs_samples, βs_samples, π_samples, ρ_samples

end

# Calculates the W variable
# W is a dictionary of dictionaries: first key is the mode, second key is the confusion matrix key, e.g. true poisitive : '11' and the final value is the count
function calc_W_u(Y_tu, Zs, K)
    W_u = Dict{Int,Dict{String,Int}}()
    n_obs = length(keys(Y_tu))

    for u in 1:K
        W_u[u] = Dict("11" => 0, "10" => 0, "01" => 0, "00" => 0)
    end

    for t in 1:n_obs
        u = Zs[t]
        if u != 0
            W_u[u]["11"] += Y_tu[t][u]["11"]
            W_u[u]["10"] += Y_tu[t][u]["10"]
            W_u[u]["01"] += Y_tu[t][u]["01"]
            W_u[u]["00"] += Y_tu[t][u]["00"]
        end
    end

    return W_u
end

# Calcualtes the Y variable
# Y is made up of 3 dictionaries:  the first key is the network observations number, the second key is the mode and the final key is the confusion matrix item,
# e.g.  Y_tu[1][2]["11"] is the true positive count for observation 1 for mode 2
function calc_Y_tu(D, As, directed, M_ts, M_us, K)
    Y_tu = Dict{Int,Dict{Int,Dict{String,Int}}}()

    for t in 1:size(D)[end]
        Y_tu[t] = calc_confusion_matrices(D[:, :, t], As, directed, M_ts[t], M_us, K)
    end

    return Y_tu
end

# Calculates the confusion matrices for a single measurement and all of the modes
function calc_confusion_matrices(D, As, directed, M_t, M_us, K)
    confusion_matrices = Dict{Int,Dict{String,Int}}()
    no_nodes = size(As)[1]

    for u in 1:K

        confusion_matrix = Dict{String,Int}()

        # Implementation of the accelerated confusion matrix calcs
        confusion_matrix["11"] = sum(D .& As[:, :, u])
        confusion_matrix["10"] = M_t - confusion_matrix["11"]
        confusion_matrix["01"] = M_us[u] - confusion_matrix["11"]


        if directed
            confusion_matrix["00"] = (no_nodes * (no_nodes - 1)) - (confusion_matrix["11"] + confusion_matrix["10"] + confusion_matrix["01"])
        else
            confusion_matrix["00"] = binomial(no_nodes, 2) - (confusion_matrix["11"] + confusion_matrix["10"] + confusion_matrix["01"])
        end

        confusion_matrices[u] = confusion_matrix

    end

    return confusion_matrices
end

# Calculation of the log-likelihood of the inferred model parameters given the data
function calc_likelihood(D, As, Zs, αs, βs, π, ρ, Y_tu, W_u, K, Hu11, Hu01, Hu10, Hu00, a_star, b_star, γ, directed)

    prob_D_given_ZAθ = 0.0

    # Calculates the probability of the data given Zs, As and θ
    for u in 1:K
        prob_D_given_ZAθ += calc_Nu(Zs, u) * log(π[u]) + W_u[u]["11"] * log(αs[u]) + W_u[u]["01"] * log(1 - αs[u]) + W_u[u]["10"] * log(βs[u]) + W_u[u]["00"] * log(1 - βs[u])
    end

    # Calculates the probability of As given θ
    log_prob_A_given_θ = calc_prob_As_given_θ(As, ρ, directed)
    # Calculates the probability of Zs given θ
    log_prob_Z_given_θ = calc_prob_Zs_given_θ(Zs, π, K)
    # Calculates the probability of the model parameters, θ
    log_prob_θ = calc_prob_θ(K, αs, βs, ρ, π, γ, Hu11, Hu01, Hu10, Hu00, a_star, b_star)

    # Log probabilities used, summed to provide final value
    prob = sum([prob_D_given_ZAθ, log_prob_A_given_θ, log_prob_θ])
    return prob
end

# Calculates the log probability of the model parameters
function calc_prob_θ(K, αs, βs, ρ, π, γ, Hu11, Hu01, Hu10, Hu00, a_star, b_star)
    log_prob_αs = zeros(Float64, K)
    for u in 1:K
        log_prob_αs[u] = log(calc_prob_αs(αs[u], Hu11, Hu01))
    end

    sum_α_probs = logsumexp(log_prob_αs)

    log_prob_βs = zeros(Float64, K)
    for u in 1:K
        log_prob_βs[u] = log(calc_prob_βs(βs[u], Hu10, Hu00))
    end

    sum_β_probs = logsumexp(log_prob_βs)

    log_prob_π = log(calc_prob_π(π, γ, K))

    log_prob_ρ = log(calc_prob_ρ(ρ, a_star, b_star))

    return logsumexp([sum_α_probs, sum_β_probs, log_prob_ρ, log_prob_π])
end

# Calculates the probability of the αs
function calc_prob_αs(α, Hu11, Hu01)
    return (α^(Hu11 - 1) * (1 - α)^(Hu01 - 1)) / beta_euler(Hu11, Hu01)
end

# Calculates the probability of the βs
function calc_prob_βs(β, Hu10, Hu00)
    return β^(Hu10 - 1) * (1 - β)^(Hu00 - 1) / beta_euler(Hu10, Hu00)
end

# Calculates the probability of the mixture weightings
function calc_prob_π(π, γ, K)
    prob = 1

    for u in 1:K
        prob *= π[u]^(γ[u] - 1)
    end

    return prob / beta_euler_general(γ)
end

# Calculates the probability of the network density
function calc_prob_ρ(ρ, a_star, b_star)
    return ρ^(a_star - 1) * (1 - ρ)^(b_star - 1) / beta_euler(a_star, b_star)
end

function beta_euler(x, y)
    return gamma(x) * gamma(y) / gamma(x + y)
end

function beta_euler_general(γ)
    val = 1.0
    for i in 1:length(γ)
        val *= gamma(γ[i])
    end
    val / gamma(sum(γ))
    return val
end

# Calculates the probability of the mode asignments given the model parameters
function calc_prob_Zs_given_θ(Zs, π, K)
    log_prob = 0.0
    N = calc_Nus(Zs, K)

    for u in 1:K
        log_prob += N[u] * log(π[u])
    end

    return log_prob
end

# Calculates the probability of the As given the model parameters
function calc_prob_As_given_θ(As, ρ, directed)
    M_star = calc_M_star(As)
    n = size(As)[1]
    K = size(As)[end]
    log_prob = 0

    if directed
        log_prob = M_star * log(ρ) + (K * (n * (n - 1)) - M_star) * log(1 - ρ)
    else
        log_prob = M_star * log(ρ) + (K * binomial(n, 2) - M_star) * log(1 - ρ)
    end

    return log_prob
end

# Calculation of the probability that an edge that occurs l times in the cluster u is present in the mode u given that we know the cluster assignments and parameter values
function calc_Q_u_ij(l, Nu, ρ, αu, βu)

    Q_u_ij = (ρ * αu^l * (1 - αu)^(Nu - l)) / (ρ * αu^(l) * (1 - αu)^(Nu - l) + (1 - ρ) * (βu)^l * (1 - βu)^(Nu - l))

    return Q_u_ij
end

# Calcuates the number of observations are in cluster u
function calc_Nu(Zs, u)
    return count(x -> x == u, Zs)
end

# Calculates the number of edges in each A
function calc_M_us(As, K)
    M_us = zeros(K)
    for u in 1:K
        M_us[u] = sum(As[:, :, u])
    end
    return M_us
end

# Calculate the number of observations assigned to each cluster
function calc_Nus(Zs, K)
    Nu = zeros(K)

    for t in 1:length(Zs)
        Nu[Zs[t]] += 1
    end

    return Nu
end

# Generates a new mode given the data, the mode assignments and the parameter values
function draw_new_modes(X_u_ij, Zs, ρ, αs, βs, K, directed)
    n = size(X_u_ij)[1]
    new_As = zeros(Bool, n, n, K)
    # Calculates the sets of edges that occur the same number of times in a cluster
    T_u_ls = gen_T_u_ls(X_u_ij, K, directed)

    N = calc_Nus(Zs, K)

    for u in 1:K
        for l in keys(T_u_ls[u])

            # Probability of any given edges in T_u_l occuring in new_A
            Q_u_ij = calc_Q_u_ij(l, N[u], ρ, αs[u], βs[u])

            no_edges = length(T_u_ls[u][l])
            edges_to_add = sum(rand() < Q_u_ij for _ in 1:no_edges)

            # edges from set t_u_l randomly chosen to be added to the new_A
            for _ in 1:edges_to_add
                edge = pop!(T_u_ls[u][l])
                delete!(T_u_ls[u][l], edge)
                i, j = edge
                new_As[i, j, u] = 1
            end
        end
    end

    return new_As
end

# Calculates the T_u_l dictionary: the sets of edges that occur the same number of times in their respective clusters
function gen_T_u_ls(X_u_ij, K, directed)
    T_u_ls = Dict{Int,Dict{Int,Set{Tuple}}}()

    for u in 1:K
        T_u_ls[u] = get_edge_counts(X_u_ij[:, :, u], directed)
    end

    T_u_ls
end

# Assimilates edges that occur the same number of times in a given cluster
function get_edge_counts(X_u_ij, directed)
    n = size(X_u_ij)[1]
    counts = Dict{Int,Set{Tuple}}()

    for i in 1:n
        l = directed ? 1 : i + 1
        for j in l:n
            if i != j
                if !haskey(counts, X_u_ij[i, j])
                    counts[X_u_ij[i, j]] = Set([(i, j)])
                else
                    push!(counts[X_u_ij[i, j]], (i, j))
                end
            end
        end
    end

    return counts
end

# 3D array each slice is the counts of every edge in the current clustering
function calc_X_u_ij(D, Zs, K)
    n = size(D)[1]
    X_u_ij = zeros(Int, n, n, K)

    for t in 1:size(D)[end]
        if Zs[t] != 0
            X_u_ij[:, :, Zs[t]] += D[:, :, t]
        end
    end

    return X_u_ij
end

# Calculates a set of new mode assignments
function draw_new_assignments(D, αs, βs, π, As, Y_tu, directed)
    no_measurements = size(D)[end]
    new_Zs = zeros(Int, no_measurements)

    for t in 1:no_measurements
        # Categorical probability distibution over the modes
        mode_probs = calc_R_tu(π, αs, βs, As, t, D, Y_tu, directed)
        dist = Categorical(exp.(mode_probs))
        new_Zs[t] = rand(dist)
    end

    return new_Zs
end

# Calculates the probability of a network measurement being a noisy measurement of one of each of the modes
function calc_R_tu(π, αs, βs, As, d_i, D, Y_tu, directed)
    K = length(π)
    mode_probs = zeros(Float64, K)
    log_denominator = zeros(Float64, K)

    for u in 1:K

        TP = Y_tu[d_i][u]["11"]
        FP = Y_tu[d_i][u]["10"]
        FN = Y_tu[d_i][u]["01"]
        TN = Y_tu[d_i][u]["00"]

        log_denominator[u] = log(π[u]) + TP * log(αs[u]) + FN * log(1 - αs[u]) + FP * log(βs[u]) + TN * log(1 - βs[u])
    end

    log_denominator_sum = logsumexp(log_denominator)

    for u in 1:K
        mode_probs[u] = log_denominator[u] - log_denominator_sum
    end

    return mode_probs
end

# Generates a new αs vector
function calc_new_αs(W_u, Hu11, Hu01, K)
    new_αs = zeros(K)

    for u in 1:K
        a = W_u[u]["11"] + Hu11
        b = W_u[u]["01"] + Hu01
        new_αs[u] = rand(Beta(a, b))
    end

    return new_αs
end

# Generates a new βs vector
function calc_new_βs(W_u, Hu00, Hu10, K)
    new_βs = zeros(K)

    for u in 1:K
        a = W_u[u]["10"] + Hu10
        b = W_u[u]["00"] + Hu00
        new_βs[u] = rand(Beta(a, b))
    end

    return new_βs
end

# Generates a new ρ value
function calc_new_ρ(As, a_star, b_star, K, directed)
    M_star = calc_M_star(As)
    a = M_star + a_star
    n = size(As)[1]

    if directed
        b = (K * (n * (n - 1)) - M_star + b_star)
    else
        b = (K * binomial(n, 2) - M_star + b_star)
    end

    return rand(Beta(a, b))
end

# Calculates a new mixture weighting
function calc_new_π(Zs, γ, K)
    pseudo_counts = get_mode_counts(Zs, K)
    pseudo_counts = pseudo_counts + γ
    return rand(Dirichlet(pseudo_counts))
end

function get_mode_counts(Zs, K)
    counts = zeros(Int, K)
    for i in 1:length(Zs)
        counts[Zs[i]] += 1
    end
    return counts
end

# Calculates the number of edges in each piece of data
function calc_M_ts(Ds)
    no_measurements = size(Ds)[end]
    M_ts = zeros(no_measurements)
    for t in 1:no_measurements
        M_ts[t] = sum(Ds[:, :, t])
    end
    return M_ts
end

# Calculates the number of edges in all of the modes
function calc_M_star(As)
    return sum(As)
end

# Generates a mixture of data given θ, the number of observations and the number of nodes
# As and Zs are generated here as well
function gen_mixture_D(π, n_obs, n, αs, βs, ρ, directed)
    K = length(π)
    Zs = gen_Z(n_obs, π)

    As = zeros(Bool, n, n, K)
    Ds = zeros(Bool, n, n, n_obs)

    for k in 1:K
        As[:, :, k] = DPMM.gen_rand_A(n, ρ, directed)
    end

    for i in 1:n_obs
        k = Zs[i]
        Ds[:, :, i] = gen_D(As[:, :, k], αs[k], βs[k], directed)
    end

    return Zs, As, Ds
end

# Generates a set of data
function gen_D(A, α, β, directed)
    n = size(A)[1]
    D = zeros(Bool, n, n)

    for i = 1:n
        l = directed ? 1 : i + 1
        for j in l:n
            if i != j
                if A[i, j]
                    # simulate bool with α
                    D[i, j] = rand() < α
                else
                    # simulate bool with β
                    D[i, j] = rand() < β
                end
            end
        end
    end

    return D
end

# Generates a set of mode assignments
function gen_Z(n_obs, π)
    K = length(π)
    return sample(1:K, Weights(π), n_obs)
end

# Generates a set of initial (semi) - random values
function gen_init_rand_vals(K, no_measurements, Ds, a_star, b_star, Hu11, Hu01, Hu10, Hu00)
    init_ρ = gen_rand_ρ(a_star, b_star)
    init_π = gen_rand_π(K)
    init_αs = gen_rand_αs(K, Hu11, Hu01)
    init_βs = gen_rand_βs(K, Hu10, Hu00)
    init_As = select_rand_As(Ds, K)
    init_Zs = gen_rand_Zs(Ds, K)

    return init_As, init_Zs, init_αs, init_βs, init_π, init_ρ
end

# Selects random inital values for As by simply choosing a piece of data that is assigned to an A
function select_rand_As(D, K)
    n = size(D)[1]
    As = zeros(Bool, n, n, K)
    chosen_data = []

    while length(chosen_data) < K
        i = rand(1:size(D)[end])
        if !(i in chosen_data)
            push!(chosen_data, i)
            As[:, :, length(chosen_data)] = D[:, :, i]
        end
    end
    return As
end

# Performs kmeans binary vector clustering on the data to generate a set of initial mode assignments
# v_upper = A[triu(trues(size(A)), 1)]  # upper triangle, excluding diagonal
function cluster_Zs_Kmeans(D, K)
    n = size(D)[1]
    no_measurements = size(D)[end]
    max_edges = n^2
    init_clusters = zeros(Int64, no_measurements)
    data_arrays = zeros(Int64, max_edges, no_measurements)

    for i in 1:no_measurements
        data_arrays[:, i] = vec(D[:, :, i])
    end

    clusters = kmeans(data_arrays, K)
    init_clusters = Clustering.assignments(clusters)

    return init_clusters
end

# Generates a set of inital mode assignments through k means clustering of the data
function gen_rand_Zs(Ds, K)
    return cluster_Zs_Kmeans(Ds, K)
end

# Generates a random ρ/ density value by sampling from a beta distribution
function gen_rand_ρ(a, b)
    return rand(Beta(a, b))
end

# Generates a set of random weights.
# Weights below 0.05 not used
function gen_rand_π(K)
    weights = zeros(K)
    while !all(weights .> 0.05)
        weights = rand(K)
        weights /= sum(weights)
    end
    return weights
end

# Generates random α values by sampling from beta distribution
function gen_rand_αs(K, Hu11, Hu01)
    return αs = rand(Beta(Hu11, Hu01), K)
end

# Generates random β values by sampling from beta distribution
function gen_rand_βs(K, Hu10, Hu00)
    return βs = rand(Beta(Hu10, Hu00), K)
end

# Calculates the frequency of the number of clusters in a set of gibbs samples
function count_clusters_freq(As_samples)
    cluster_count = []
    for i in 2:length(As_samples)
        push!(cluster_count, size(As_samples[i])[3])
    end
    return cluster_count
end

# Calculates point estimates for As samples from the DPMM
function calc_DPMM_As_PEs(As_samples, opt_k, burn_in, iterations, ρ, directed)
    n = size(As_samples[1])[1]
    As_PEs = zeros(Bool, n, n, opt_k)
    As_summed = zeros(Int, n, n, opt_k)
    start = round(Int, burn_in * iterations)

    for i in start:iterations
        if size(As_samples[i])[end] == opt_k
            for u in 1:opt_k
                As_summed[:, :, u] += As_samples[i][:, :, u]
            end
        end
    end

    for u in 1:opt_k
        As_PEs[:, :, u] = most_prob_edges(As_summed[:, :, u], ρ, directed)
    end

    return As_PEs
end

# Calculates point estimates for some of the model parameters for the DPMM samples
function calc_DPMM_Param_PEs(θs_samples, opt_k, burn_in, iterations)
    θs_avg = zeros(opt_k)
    count = 0
    start = round(Int, burn_in * iterations)

    for i in start:iterations
        if length(θs_samples[i]) == opt_k
            θs_avg += θs_samples[i]
            count += 1
        end
    end

    return θs_avg / count
end

# Calculates the point estimate of the density for the DPMM samples
function calc_DPMM_ρ_PEs(ρ_samples, opt_k, burn_in, τ, As_samples)
    ρ_PE = 0
    count = 0
    for i in round(Int, burn_in * τ):τ
        if size(As_samples[i])[end] == opt_k
            ρ_PE += ρ_samples[i]
            count += 1
        end
    end
    return ρ_PE / count
end

# Calculates a points estimate for the mode assignments for the DPMM samples
function calc_DPMM_Zs_PEs(Zs_samples, opt_k, burn_in, iterations)
    set_of_Zs = []
    start = round(Int, burn_in * iterations)

    for i in start:iterations
        if length(unique(Zs_samples[i])) == opt_k
            push!(set_of_Zs, Zs_samples[i])
        end
    end

    n_obs = length(Zs_samples[1])
    Zs_PEs = zeros(n_obs)

    for i in 1:n_obs
        assignments = [set_of_Zs[j][i] for j in 1:length(set_of_Zs)]
        Zs_PEs[i] = mode(assignments)
    end

    Zs_PEs = round.(Int, Zs_PEs)
    return Zs_PEs
end

# Function to execute either of the gibbs sampling algorithm and return the MCMC samples
function test_clustering(algo, D, directed, burn_in, τ, Ks, a_star, b_star, Hu11, Hu01, Hu10, Hu00, γ, alpha_val, base_dist, true_As, true_Zs, αs, βs, π, ρ)

    if algo == 1

        As_samples, Zs_samples, αs_samples, βs_samples, π_samples, ρ_samples, opt_k, mean_likelihoods = test_gibbs_FMM(D, Ks, τ, Hu11, Hu01, Hu00, Hu10, a_star, b_star, γ, burn_in, directed)

        As_PEs, Zs_PEs, αs_PEs, βs_PEs, π_PE, ρ_PE = calc_point_estimates(As_samples, Zs_samples, αs_samples, βs_samples, π_samples, ρ_samples, τ, burn_in, opt_k, directed)

        display_results(Zs_PEs, αs_PEs, βs_PEs, π_PE, ρ_PE, opt_k, true_Zs, αs, βs, π, ρ)

        return As_PEs, mean_likelihoods, [], As_samples, Zs_samples, αs_samples, βs_samples, π_samples

    else

        As_samples, Zs_samples, αs_samples, βs_samples, π_samples, ρ_samples, opt_k, cluster_freqs = DPMM.test_gibbs_DPMM(alpha_val, τ, burn_in, base_dist, D, a_star, b_star, Hu11, Hu01, Hu10, Hu00, directed)

        As_PEs, Zs_PEs, αs_PEs, βs_PEs, π_PE, ρ_PE = calc_DPMM_PEs(As_samples, Zs_samples, αs_samples, βs_samples, π_samples, ρ_samples, opt_k, burn_in, τ, directed)

        display_results(Zs_PEs, αs_PEs, βs_PEs, π_PE, ρ_PE, opt_k, true_Zs, αs, βs, π, ρ)

        return As_PEs, [], cluster_freqs, As_samples, Zs_samples, αs_samples, βs_samples, π_samples

    end

end

# Function which calculates the point estimates for the As, Zs and model parameters for the dirichlet process mixture model
function calc_DPMM_PEs(As_samples, Zs_samples, αs_samples, βs_samples, π_samples, ρ_samples, opt_k, burn_in, τ, directed)

    ρ_PE = calc_DPMM_ρ_PEs(ρ_samples, opt_k, burn_in, τ, As_samples)
    As_PEs = calc_DPMM_As_PEs(As_samples, opt_k, burn_in, τ, ρ_PE, directed)
    Zs_PEs = calc_DPMM_Zs_PEs(Zs_samples, opt_k, burn_in, τ)
    αs_PEs = calc_DPMM_Param_PEs(αs_samples, opt_k, burn_in, τ)
    βs_PEs = calc_DPMM_Param_PEs(βs_samples, opt_k, burn_in, τ)
    π_PE = calc_DPMM_Param_PEs(π_samples, opt_k, burn_in, τ)
    ρ_PE = calc_DPMM_ρ_PEs(ρ_samples, opt_k, burn_in, τ, As_samples)

    return As_PEs, Zs_PEs, αs_PEs, βs_PEs, π_PE, ρ_PE
end

# Displays the true parameters and the point estimates of the inferred parameters
function display_results(Zs_PEs, αs_PEs, βs_PEs, π_PE, ρ_PE, opt_k, true_Zs, αs, βs, π, ρ)

    println("Optimal K value:  ", opt_k)

    ari = randindex(true_Zs, Zs_PEs)
    println("\nVariation of information:  ", Clustering.varinfo(true_Zs, Zs_PEs))
    println("\nHubert and Arabie adjusted Rand Index:  ", ari[1])
    println("\nRand index ( agreement probability):  ", ari[2])
    println("\nMirkin's index ( disagreement probability):  ", ari[3])
    println("\nHuberet's index P(agree)-P(disagree):  ", ari[4])
    println("\nTrue Zs counts:     ", counts(true_Zs))
    println("Inferred Zs counts: ", counts(Zs_PEs))

    println("\nTrue αs:      ", αs)
    println("Inferred αs:  ", round.(αs_PEs, digits=4))

    println("\nTrue βs:      ", βs)
    println("Inferred βs:  ", round.(βs_PEs, digits=4))

    println("\nTrue π:       ", round.(π, digits=4))
    println("True Zs allocation %'s:  ", round.((counts(true_Zs) ./ length(true_Zs)), digits=4))
    println("Inferred π:   ", round.(π_PE, digits=4))

    println("\nTrue ρ:      ", ρ)
    println("Inferred ρ:  ", round(ρ_PE, digits=4))

end

# Performs Gibbs sampling over the specified range of k values and calculates the most likely clustering and return the MCMC samples from them
function test_gibbs_FMM(D, Ks, τ, Hu11, Hu01, Hu00, Hu10, a_star, b_star, γ, burn_in, directed)
    no_nodes = size(D)[1]
    no_measurements = size(D)[end]
    K_likelihoods = Dict{Int,Float64}()
    opt_k = 0

    # Results for most likely K
    opt_likelihood = typemin(Float64)
    As_Gs, Zs_Gs, αs_Gs, βs_Gs, π_Gs, ρ_Gs = [], [], [], [], [], [], []

    M_ts = calc_M_ts(D)

    for k in Ks
        # Gen init rand vals
        init_As, init_Zs, init_αs, init_βs, init_π, init_ρ = gen_init_rand_vals(k, no_measurements, D, a_star, b_star, Hu11, Hu01, Hu10, Hu00)

        γ = ones(k)
        # Gibbs sampling
        As_samples, Zs_samples, αs_samples, βs_samples, π_samples, ρ_samples = gibbs_sampling(τ, init_As, init_Zs, init_αs, init_βs, init_π, init_ρ, D, k, Hu11, Hu01, Hu00, Hu10, a_star, b_star, γ, directed, M_ts)

        # Calculation of point estimates
        As_PEs, Zs_PEs, αs_PEs, βs_PEs, π_PE, ρ_PE = calc_point_estimates(As_samples, Zs_samples, αs_samples, βs_samples, π_samples, ρ_samples, τ, burn_in, k, directed)

        # Auxillary variables
        M_ts = calc_M_ts(D)
        M_us = calc_M_us(As_PEs, k)
        Y_tu = calc_Y_tu(D, As_PEs, directed, M_ts, M_us, k)
        W_u = calc_W_u(Y_tu, Zs_PEs, k)

        # Calculates the likelihood of inferred model parameters given the data
        likelihood = calc_likelihood(D, As_PEs, Zs_PEs, αs_PEs, βs_PEs, π_PE, ρ_PE, Y_tu, W_u, k, Hu11, Hu01, Hu10, Hu00, a_star, b_star, γ, directed)

        K_likelihoods[k] = likelihood
        # Update most likely K
        if k == argmax(K_likelihoods)
            As_Gs, Zs_Gs, αs_Gs, βs_Gs, π_Gs, ρ_Gs = As_samples, Zs_samples, αs_samples, βs_samples, π_samples, ρ_samples
            opt_k = k
        end

    end

    # Return MCMC samples
    return As_Gs, Zs_Gs, αs_Gs, βs_Gs, π_Gs, ρ_Gs, opt_k, K_likelihoods

end

# Calculates point estimates given the gibbs samples
function calc_point_estimates(As_samples, Zs_samples, αs_samples, βs_samples, π_samples, ρ_samples, τ, burn_in, K, directed)

    start = round(Int, burn_in * τ)
    count = Int(τ - start) + 1

    Zs_PEs = calc_Zs_PEs(Zs_samples, start, τ)

    αs_PEs = sum(αs_samples[i, :] for i in start:τ) / count

    βs_PEs = sum(βs_samples[i, :] for i in start:τ) / count

    π_PE = sum(π_samples[i, :] for i in start:τ) / count

    ρ_PE = sum(ρ_samples[i] for i in start:τ) / count

    As_PEs = calc_As_PEs(As_samples, ρ_PE, start, K, τ, directed)

    return As_PEs, Zs_PEs, αs_PEs, βs_PEs, π_PE, ρ_PE
end


# Calculates the point estimates for Zs
function calc_Zs_PEs(Zs_samples, start, τ)
    n_obs = size(Zs_samples)[2]
    Zs_PE = zeros(n_obs)

    for i in 1:n_obs
        assignments = [Zs_samples[j, i] for j in start:τ]
        Zs_PE[i] = mode(assignments)
    end

    Zs_PE = round.(Int, Zs_PE)
    return Zs_PE
end

# Calculates the point estimte for the modes given the As gibbs samples
# Computes the ρ % most commonly occuring edges over all the samples for each cluster
function calc_As_PEs(As_samples, ρ, start, K, τ, directed)
    n = size(As_samples)[2]
    As_probs = zeros(Float64, n, n, K)

    for u in 1:K
        As_probs[:, :, u] = sum(As_samples[i, :, :, u] for i in start:τ) / (τ - start)
    end

    As_PEs = zeros(Bool, n, n, K)

    for u in 1:K
        As_PEs[:, :, u] = most_prob_edges(As_probs[:, :, u], ρ, directed)
    end

    return As_PEs
end

# Calculates the most probable edges occuring in a given network
# return point estimate of an A
function most_prob_edges(A, ρ, directed)
    n = size(A)[1]
    A_PE = zeros(Bool, n, n)
    edges = Dict{Tuple{Int,Int},Float64}()

    for i in 1:n
        l = directed ? 1 : i + 1
        for j in l:n
            edges[(i, j)] = A[i, j]
        end
    end

    sorted_edges = sort(collect(edges), by=x -> x[2], rev=true)
    no_edges = 0
    if directed
        no_edges = Int(round(ρ * n * (n - 1)))
    else
        no_edges = Int(round(ρ * (n * (n - 1)) / 2))
    end
    top_n_edges = sorted_edges[1:no_edges]
    top_n_edges_dict = Dict(top_n_edges)

    for (key, value) in top_n_edges_dict
        i, j = key
        A_PE[i, j] = 1
    end

    return A_PE
end

end