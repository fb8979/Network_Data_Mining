module DPMM

using GraphRecipes, Plots, StatsBase, Distributions, Statistics, StatsPlots, SpecialFunctions, LogExpFunctions, Clustering, ColorTypes, Clustering, DataStructures, Distances, LinearAlgebra

function __init__()
    @eval using ..FMM
end

function DPMM_gibbs_sampling(D, τ, alpha, base_dist, Hu11, Hu01, Hu10, Hu00, a_star, b_star, directed)

    n = size(D)[1]
    n_obs = size(D)[end]
    M_ts = FMM.calc_M_ts(D)
    # initial rand weights and cluster assignments
    init_π, init_Zs = gen_init_π_and_Zs(alpha, 100, base_dist, n_obs)
    K = length(unique(init_Zs))
    Zs_samples, As_samples, αs_samples, βs_samples, π_samples, ρ_samples = [], [], [], [], [], []

    push!(As_samples, assign_init_As(D, init_Zs, n, K))
    push!(Zs_samples, init_Zs)
    push!(αs_samples, FMM.gen_rand_αs(K, Hu11, Hu01))
    push!(βs_samples, FMM.gen_rand_βs(K, Hu10, Hu00))
    push!(π_samples, init_π)
    push!(ρ_samples, FMM.gen_rand_ρ(a_star, b_star))

    current_Zs = convert.(Int, Zs_samples[1])

    M_us = FMM.calc_M_us(As_samples[1], K)
    Y_tu = FMM.calc_Y_tu(D, As_samples[1], directed, M_ts, M_us, K)
    W_u = FMM.calc_W_u(Y_tu, current_Zs, K)
    X_u_ij = FMM.calc_X_u_ij(D, current_Zs, K)

    # given alpha & state of markov chain sample new parameters
    for i in 2:τ+1

        # Cluster assignments for current iteration
        current_Zs = convert.(Int, Zs_samples[i-1])
        current_As = As_samples[i-1]
        current_αs = αs_samples[i-1]
        current_βs = βs_samples[i-1]
        current_π = π_samples[i-1]
        current_ρ = ρ_samples[i-1]

        for t in 1:n_obs

            prev_u = current_Zs[t]

            current_Zs, current_As, current_αs, current_βs, X_u_ij, W_u = remove_obs_from_cluster(current_Zs, D, X_u_ij, Y_tu, W_u, t, current_As, current_αs, current_βs)

            # Minus 1 because of the 0 values mode assignment! check
            K = length(unique(current_Zs)) - 1

            cluster_probs = zeros(Float64, K + 1)

            current_π = gen_DPMM_π(current_Zs)

            mode_probs = draw_new_assignments_DPMM(D, t, current_αs, current_βs, current_π, current_As, Y_tu, directed)

            for k in 1:K+1
                if k != K + 1
                    # Calculation of probability of observation being in cluster k
                    cluster_probs[k] = log(cluster_prob_k(current_Zs, t, alpha)) + log(mode_probs[k])
                else
                    # Calculation of probability of observation being in a new cluster
                    cluster_probs[k] = log(new_cluster_prob(current_Zs, t, alpha)) + log(calc_prob_rand_A(gen_rand_A(n, current_ρ, directed), current_ρ, directed))
                end
            end

            # Random sample to determine new mode assignment
            cluster_probs = exp.(cluster_probs)
            cat_dist = cluster_probs ./ sum(cluster_probs)
            cluster = sample(1:length(cat_dist), Weights(cat_dist))

            # If an observation is removed from a cluster and then put back in it (assuming it is not emptied) then no new mode generation is done
            if (cluster == prev_u) && (count(x -> x == prev_u, current_Zs) > 0)
                current_Zs[t] = cluster
                K = length(unique(current_Zs))
                X_u_ij[:, :, cluster] += D[:, :, t]
                W_u = increment_W_u(Y_tu, W_u, t, cluster)
            else
                # In other cases the old and new clusters are recalculated
                new_cluster = (cluster == K + 1)
                current_Zs[t] = cluster
                K = length(unique(current_Zs))

                # Model parameters added for new cluster
                if new_cluster
                    X_u_ij = cat(X_u_ij, D[:, :, t], dims=3)
                    push!(current_αs, FMM.gen_rand_αs(1, Hu11, Hu01)[1])
                    push!(current_βs, FMM.gen_rand_βs(1, Hu10, Hu00)[1])
                else
                    X_u_ij[:, :, cluster] += D[:, :, t]
                end

                current_As = draw_new_modes_DPMM(X_u_ij, current_Zs, current_ρ, current_αs, current_βs, K, directed, [prev_u, cluster], current_As)

                M_us = update_M_us(M_us, current_As, K, [prev_u, cluster])
                Y_tu = update_Y_tu(Y_tu, D, current_As, directed, M_ts, M_us, K, [prev_u, cluster])
                W_u = FMM.calc_W_u(Y_tu, current_Zs, K)

                # New model parameters generated based off of updated counts
                current_αs = FMM.calc_new_αs(W_u, Hu11, Hu01, K)
                current_βs = FMM.calc_new_βs(W_u, Hu10, Hu00, K)
                current_π = gen_DPMM_π(current_Zs)
                current_ρ = FMM.calc_new_ρ(current_As, a_star, b_star, K, directed)

            end

        end

        push!(Zs_samples, current_Zs)
        push!(As_samples, current_As)
        push!(αs_samples, current_αs)
        push!(βs_samples, current_βs)
        push!(π_samples, current_π)
        push!(ρ_samples, current_ρ)

    end

    return Zs_samples, As_samples, αs_samples, βs_samples, π_samples, ρ_samples
end

# Removes counts of the network observation removed from it's cluster from the W_u and X_u_ij variables
function remove_obs_from_cluster(Zs, D, X_u_ij, Y_tu, W_u, t, As, αs, βs)

    if count(x -> x == Zs[t], Zs) > 1

        u = Zs[t]
        W_u[u]["11"] -= Y_tu[t][u]["11"]
        W_u[u]["10"] -= Y_tu[t][u]["10"]
        W_u[u]["01"] -= Y_tu[t][u]["01"]
        W_u[u]["00"] -= Y_tu[t][u]["00"]

        X_u_ij[:, :, u] -= D[:, :, t]

        Zs[t] = 0
    else
        u = Zs[t]
        Zs[t] = 0
        Zs = make_contiguous(Zs)
        As = remove_cluster_from_As(As, u)
        deleteat!(αs, u)
        deleteat!(βs, u)

        if u == 1
            X_u_ij = X_u_ij[:, :, 2:end]
        elseif u == (length(αs) + 1)
            X_u_ij = X_u_ij[:, :, 1:end-1]
        else
            X_u_ij_before = X_u_ij[:, :, 1:u-1]
            X_u_ij_after = X_u_ij[:, :, u+1:end]
            X_u_ij = cat(X_u_ij_before, X_u_ij_after, dims=3)
        end

    end

    return Zs, As, αs, βs, X_u_ij, W_u
end

function increment_W_u(Y_tu, W_u, t, u)

    W_u[u]["11"] += Y_tu[t][u]["11"]
    W_u[u]["10"] += Y_tu[t][u]["10"]
    W_u[u]["01"] += Y_tu[t][u]["01"]
    W_u[u]["00"] += Y_tu[t][u]["00"]

    return W_u
end

function update_M_us(M_us_old, As, K, updated)
    M_us = zeros(K)
    for u in 1:K
        if u in updated
            M_us[u] = sum(As[:, :, u])
        else
            M_us[u] = M_us_old[u]
        end
    end
    return M_us
end

function update_Y_tu(Y_tu, D, As, directed, M_ts, M_us, K, updated)

    for t in 1:size(D)[end]
        Y_tu[t] = calc_confusion_matrices_Reduced(Y_tu[t], D[:, :, t], As, directed, M_ts[t], M_us, K, updated)
    end

    return Y_tu
end

function calc_confusion_matrices_Reduced(Y_tu, D, As, directed, M_t, M_us, K, updated)
    confusion_matrices = Dict{Int,Dict{String,Int}}()
    no_nodes = size(As)[1]

    for u in 1:K

        if u in updated
            confusion_matrix = Dict{String,Int}()

            confusion_matrix["11"] = sum(D .& As[:, :, u])
            confusion_matrix["10"] = M_t - confusion_matrix["11"]
            confusion_matrix["01"] = M_us[u] - confusion_matrix["11"]


            if directed
                confusion_matrix["00"] = (no_nodes * (no_nodes - 1)) - (confusion_matrix["11"] + confusion_matrix["10"] + confusion_matrix["01"])
            else
                confusion_matrix["00"] = binomial(no_nodes, 2) - (confusion_matrix["11"] + confusion_matrix["10"] + confusion_matrix["01"])
            end

            confusion_matrices[u] = confusion_matrix
        else
            confusion_matrices[u] = Y_tu[u]
        end

    end

    return confusion_matrices
end

function draw_new_modes_DPMM(X_u_ij, Zs, ρ, αs, βs, K, directed, chosen, current_As)
    n = size(X_u_ij)[1]
    new_As = zeros(Bool, n, n, K)
    T_u_ls = FMM.gen_T_u_ls(X_u_ij, K, directed)

    N = FMM.calc_Nus(Zs, K)

    for u in 1:K
        if u in chosen

            for l in keys(T_u_ls[u])
                Q_u_ij = FMM.calc_Q_u_ij(l, N[u], ρ, αs[u], βs[u])
                no_edges = length(T_u_ls[u][l])
                edges_to_add = sum(rand() < Q_u_ij for _ in 1:no_edges)

                for _ in 1:edges_to_add
                    edge = pop!(T_u_ls[u][l])
                    delete!(T_u_ls[u][l], edge)
                    i, j = edge
                    new_As[i, j, u] = 1
                end
            end
        else
            new_As[:, :, u] = current_As[:, :, u]
        end
    end

    return new_As
end

function gen_DPMM_π(Zs)
    if 0 in Zs
        weights = counts(Zs)[2:end]
    else
        weights = counts(Zs)
    end

    weights = weights .+ 1

    return rand(Dirichlet(weights))
end

function gen_init_π_and_Zs(alpha, no_breaks, base_dist, n_obs)

    weights, assignments = stick_breaking_process(alpha, no_breaks, base_dist, n_obs)
    contiguous_assignments = make_contiguous(assignments)

    return weights, contiguous_assignments
end

function stick_breaking_process(alpha, no_breaks, base_distribution, no_obs)
    weights = Float64[]
    assignments = []
    remaining_stick = 1.0

    beta_dist = Beta(1, alpha)

    for _ in 1:no_breaks
        weight = rand(beta_dist) * remaining_stick
        push!(weights, weight)
        remaining_stick = 1 - sum(weights)
    end

    weights_summed = zeros(Float64, length(weights))
    weights_summed[1] = weights[1]

    for i in 2:length(weights)
        weights_summed[i] = weights_summed[i-1] + weights[i]
    end

    for i in 1:no_obs
        push!(assignments, rand(1:no_breaks))
    end

    assignments = convert.(Int, assignments)
    return weights, assignments
end

# Runs the dirichlet process mixture model gibbs sampling algorithm and return samples and optimal found clustering value
function test_gibbs_DPMM(alpha_val, τ, burn_in, base_dist, D, a_star, b_star, Hu11, Hu01, Hu10, Hu00, directed)

    DPMM_Zs_samples, DPMM_As_samples, DPMM_αs_samples, DPMM_βs_samples, DPMM_π_samples, DPMM_ρ_samples = DPMM.DPMM_gibbs_sampling(D, τ, alpha_val, base_dist, Hu11, Hu01, Hu10, Hu00, a_star, b_star, directed)

    cluster_freqs = FMM.count_clusters_freq(DPMM_As_samples)

    DPMM_opt_k = mode(cluster_freqs)

    return DPMM_As_samples, DPMM_Zs_samples, DPMM_αs_samples, DPMM_βs_samples, DPMM_π_samples, DPMM_ρ_samples, DPMM_opt_k, cluster_freqs
end

# Finds the values that stop a set of clusters from being contiguous
function find_missing(assignments)
    max = maximum(assignments)
    missing = []
    for i in 1:max
        flag = i in assignments
        if !flag
            push!(missing, i)
        end
    end
    return missing
end

# Calculates the probability of a network given a density value
function calc_prob_rand_A(A, ρ, directed)

    M_star = sum(A)
    n = size(A)[1]
    val = directed ? n * (n - 1) : binomial(n, 2)
    prob = M_star * log(ρ) + val * log(1 - ρ)

    return exp(prob)
end

# Generates a random network measurement given a density value, ρ
function gen_rand_A(n, ρ, directed)
    A = zeros(Bool, n, n)
    for i in 1:n
        l = directed ? 1 : i + 1
        for j in l:n
            if i != j
                A[i, j] = rand() < ρ
            end
        end
    end
    return A
end

# Generates initial As values from choosing a random piece of data for each set that is assigned to each A
function assign_init_As(D, Zs, n, K)
    As = zeros(Bool, n, n, K)

    for u in 1:K
        indices = [i for i in 1:length(Zs) if Zs[i] == u]
        index = rand(indices)
        As[:, :, u] = D[:, :, index]
    end

    return As
end

# Draws a new cluster assignment for a measurement
function draw_new_assignments_DPMM(D, index, αs, βs, π, As, Y_tu, directed)
    no_measurements = size(D)[end]
    mode_probs = FMM.calc_R_tu(π, αs, βs, As, index, D, Y_tu, directed)
    mode_probs = exp.(mode_probs)
    return mode_probs
end

# Removes a given cluster from As
function remove_cluster_from_As(As, cluster)

    if cluster == 1
        return As[:, :, 2:end]
    elseif cluster == size(As)[end]
        return As[:, :, 1:end-1]
    else
        arr1 = As[:, :, 1:cluster-1]
        arr2 = As[:, :, cluster+1:end]
        return cat(arr1, arr2, dims=3)
    end

end

# Chinese restaurant process calculation for an observation joining a existing cluster 
function cluster_prob_k(Zs, t, alpha)
    n = length(Zs)
    k_ex_t = count(x -> x == Zs[t], Zs)
    return k_ex_t / (n - 1 + alpha)
end

# Chinese restaurant process calculation for an observation joining a new cluster 
function new_cluster_prob(Zs, t, alpha)
    n = length(Zs)
    return alpha / (n - 1 + alpha)
end

# Makes a set of mode assignments contiguous
function make_contiguous(Zs)

    missing = sort(find_missing(Zs), rev=true)

    for c in missing
        for t in 1:length(Zs)
            if Zs[t] > c
                Zs[t] -= 1
            end
        end
    end

    return Zs
end

end
