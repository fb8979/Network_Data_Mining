module Testing

using GraphRecipes, Plots, StatsBase, Distributions, Statistics, StatsPlots, SpecialFunctions,
    LogExpFunctions, Clustering, ColorTypes, Clustering, DataStructures, Distances, LinearAlgebra, Hungarian

function __init__()
    # Defer the import until Testing is fully compiled
    # The '..' refers to the parent module's scope (ClusteringProject)
    @eval using ..FMM, ..DPMM
end

function kmeans_elbow_method(adjacency_matrices, K_range; directed=false, n_runs=10)
    """
    Elbow method using K-means clustering on graph adjacency matrices

    Parameters:
    - adjacency_matrices: Array of shape (no_obs, no_nodes, no_nodes)
    - K_range: Range of K values to test (e.g., 1:8)
    - directed: Boolean, whether to use full matrix (true) or upper triangle (false)
    - n_runs: Number of random initializations per K (default: 10)

    Returns:
    - wcss_values: Dictionary of WCSS for each K
    - elbow_plot: The elbow plot
    """

    no_obs, no_nodes, _ = size(adjacency_matrices)

    if directed
        n_features = no_nodes * (no_nodes - 1)
        data_matrix = zeros(n_features, no_obs)

        for i in 1:no_obs
            A = adjacency_matrices[i, :, :]
            feature_idx = 1
            for row in 1:no_nodes
                for col in 1:no_nodes
                    if row != col
                        data_matrix[feature_idx, i] = A[row, col]
                        feature_idx += 1
                    end
                end
            end
        end
    else
        n_features = div(no_nodes * (no_nodes - 1), 2)
        data_matrix = zeros(n_features, no_obs)

        for i in 1:no_obs
            A = adjacency_matrices[i, :, :]
            feature_idx = 1
            for row in 1:no_nodes
                for col in (row+1):no_nodes
                    data_matrix[feature_idx, i] = A[row, col]
                    feature_idx += 1
                end
            end
        end
    end

    wcss_values = Dict()

    for k in K_range

        if k == 1
            centroid = mean(data_matrix, dims=2)
            wcss = sum((data_matrix .- centroid) .^ 2)
            wcss_values[k] = wcss
            continue
        end

        best_wcss = Inf
        successful_runs = 0

        for run in 1:n_runs
            try
                result = kmeans(data_matrix, k; maxiter=200, display=:none)
                current_wcss = result.totalcost
                successful_runs += 1

                if current_wcss < best_wcss
                    best_wcss = current_wcss
                end
            catch e
                continue
            end
        end

        if best_wcss < Inf
            wcss_values[k] = best_wcss
        else
            wcss_values[k] = NaN
        end
    end

    ks = sort(collect(keys(wcss_values)))
    wcss = [wcss_values[k] for k in ks]

    valid_indices = .!isnan.(wcss)
    ks_valid = ks[valid_indices]
    wcss_valid = wcss[valid_indices]

    elbow_plt = plot(ks_valid, wcss_valid,
        xlabel="Number of Clusters (K)",
        ylabel="Within-Cluster Sum of Squares (WCSS)",
        title="K-means Elbow Method",
        linewidth=3,
        marker=:circle,
        markersize=10,
        legend=false,
        grid=true,
        size=(900, 600),
        linecolor=:steelblue,
        markercolor=:steelblue)

    elbow_k = nothing
    if length(ks_valid) >= 3
        improvements = -diff(wcss_valid)

        if length(improvements) >= 2
            accelerations = diff(improvements)
            elbow_idx = argmin(accelerations) + 1
            elbow_k = ks_valid[elbow_idx]

            scatter!(elbow_plt, [elbow_k], [wcss_values[elbow_k]],
                markersize=15,
                markercolor=:red,
                markerstrokewidth=3,
                label="Elbow at K=$elbow_k")

            annotate!(elbow_plt, elbow_k, wcss_values[elbow_k] * 1.05,
                text("← Optimal K=$elbow_k", :left, 10, :red, :bold))

            println("\nOPTIMAL K: $elbow_k")
        end
    end

    # Improvement rate plot
    improvement_plt = nothing
    if length(ks_valid) >= 2
        improvements = -diff(wcss_valid)
        pct_improvements = (improvements ./ wcss_valid[1:end-1]) .* 100

        improvement_plt = plot(ks_valid[2:end], pct_improvements,
            xlabel="Number of Clusters (K)",
            ylabel="% Decrease in WCSS",
            title="Rate of Improvement",
            linewidth=3,
            marker=:circle,
            markersize=10,
            legend=false,
            grid=true,
            size=(900, 600),
            linecolor=:green,
            markercolor=:green)

        hline!(improvement_plt, [5], linestyle=:dash, linewidth=2, color=:orange, alpha=0.7)
    end

    # Summary table
    println("\n" * "="^60)
    println("SUMMARY")
    println("="^60)
    println(rpad("K", 6), rpad("WCSS", 18), rpad("Improvement", 18), "% Decrease")
    println("-"^60)

    for (idx, k) in enumerate(ks_valid)
        wcss_val = round(wcss_values[k], digits=2)

        if idx > 1
            improvement = wcss_valid[idx-1] - wcss_valid[idx]
            pct = (improvement / wcss_valid[idx-1]) * 100

            marker = k == elbow_k ? " ← ELBOW" : ""

            println(rpad(string(k), 6),
                rpad(string(wcss_val), 18),
                rpad(string(round(improvement, digits=2)), 18),
                string(round(pct, digits=2)) * "%",
                marker)
        else
            println(rpad(string(k), 6), rpad(string(wcss_val), 18), "-", "-")
        end
    end
    println("="^60)

    if elbow_k !== nothing
        println("\n RECOMMENDATION: Use K=$elbow_k clusters")
    end

    return wcss_values, elbow_plt, improvement_plt
end

function generate_plots_data(τ, no_obs_vals, no_nodes, burn_in, directed, a_star, b_star, Hu11, Hu01, Hu10, Hu00, γ, alpha_val, base_dist, K, do_calc, algo, π, ρ)

    if algo == 1
        FMM_part_dist, FMM_param_dist, FMM_network_dist, FMM_network_certainty = results(1, no_obs_vals, no_nodes, τ, burn_in, directed, a_star, b_star, Hu11, Hu01, Hu10, Hu00, γ, alpha_val, base_dist, K, do_calc, π, ρ)

        return FMM_part_dist, FMM_param_dist, FMM_network_dist, FMM_network_certainty
    else
        DPMM_part_dist, DPMM_param_dist, DPMM_network_dist, DPMM_network_certainty = results(2, no_obs_vals, no_nodes, τ, burn_in, directed, a_star, b_star, Hu11, Hu01, Hu10, Hu00, γ, alpha_val, base_dist, K, do_calc, π, ρ)

        return DPMM_part_dist, DPMM_param_dist, DPMM_network_dist, DPMM_network_certainty
    end
end

function align_clusters(pred_As, pred_Zs, pred_αs, pred_βs, pred_π,
    true_As, true_Zs, true_αs, true_βs, true_π)
    """
    Align predicted cluster parameters to true cluster parameters using Hungarian algorithm.

    Parameters:
    -----------
    pred_As : Array{Bool, 3} - Predicted adjacency matrices (n_nodes × n_nodes × K_pred)
    pred_Zs : Vector{Int} - Predicted cluster assignments (n_obs,)
    pred_αs : Vector{Float64} - Predicted true positive rates (K_pred,)
    pred_βs : Vector{Float64} - Predicted false positive rates (K_pred,)
    pred_π : Vector{Float64} - Predicted mixture weights (K_pred,)
    true_As : Array{Bool, 3} - True adjacency matrices (n_nodes × n_nodes × K_true)
    true_Zs : Vector{Int} - True cluster assignments (n_obs,)
    true_αs : Vector{Float64} - True true positive rates (K_true,)
    true_βs : Vector{Float64} - True false positive rates (K_true,)
    true_π : Vector{Float64} - True mixture weights (K_true,)

    Returns:
    --------
    aligned_As : Array{Bool, 3} - Aligned adjacency matrices
    aligned_Zs : Vector{Int} - Aligned cluster assignments
    aligned_αs : Vector{Float64} - Aligned true positive rates
    aligned_βs : Vector{Float64} - Aligned false positive rates
    aligned_π : Vector{Float64} - Aligned mixture weights
    label_mapping : Dict{Int, Int} - Mapping from pred cluster → true cluster
    """

    K_true = length(true_αs)
    K_pred = length(pred_αs)
    K_max = max(K_true, K_pred)
    n_obs = length(true_Zs)
    n_nodes = size(true_As, 1)

    # Create cost matrix based on multiple criteria
    cost_matrix = zeros(Float64, K_max, K_max)

    for i in 1:K_max  # true cluster index
        for j in 1:K_max  # predicted cluster index
            cost = 0.0

            if i <= K_true && j <= K_pred
                # 1. Cost from cluster assignment mismatches
                true_in_i = (true_Zs .== i)
                pred_in_j = (pred_Zs .== j)
                assignment_cost = sum(true_in_i .!= pred_in_j)

                # 2. Cost from parameter differences (L1 distance)
                param_cost = abs(true_αs[i] - pred_αs[j]) +
                             abs(true_βs[i] - pred_βs[j]) +
                             abs(true_π[i] - pred_π[j])

                # 3. Cost from network structure differences (Hamming distance)
                network_cost = sum(true_As[:, :, i] .!= pred_As[:, :, j])

                # Combine costs (you can weight these differently)
                cost = assignment_cost + 100 * param_cost + network_cost
            else
                # Penalize matching to non-existent clusters
                cost = n_obs * 1000.0
            end

            cost_matrix[i, j] = cost
        end
    end

    # Solve assignment problem
    assignment, total_cost = hungarian(cost_matrix)

    # Create label mapping: pred_idx → true_idx
    label_mapping = Dict{Int,Int}()
    inverse_mapping = Dict{Int,Int}()

    for true_idx in 1:K_max
        pred_idx = assignment[true_idx]
        if pred_idx <= K_pred
            inverse_mapping[pred_idx] = true_idx
            label_mapping[pred_idx] = true_idx
        end
    end

    # Initialize aligned arrays with appropriate sizes
    aligned_As = zeros(Bool, n_nodes, n_nodes, K_true)
    aligned_Zs = copy(pred_Zs)  # Will modify in place
    aligned_αs = zeros(Float64, K_true)
    aligned_βs = zeros(Float64, K_true)
    aligned_π = zeros(Float64, K_true)

    # Apply mapping to cluster assignments
    for i in 1:n_obs
        if haskey(label_mapping, pred_Zs[i])
            aligned_Zs[i] = label_mapping[pred_Zs[i]]
        else
            # If predicted cluster doesn't map to anything, assign to a default
            aligned_Zs[i] = pred_Zs[i]  # or could use K_true + 1 for "unmatched"
        end
    end

    # Apply mapping to parameters and networks
    for pred_idx in 1:K_pred
        if haskey(label_mapping, pred_idx)
            true_idx = label_mapping[pred_idx]
            if true_idx <= K_true
                aligned_αs[true_idx] = pred_αs[pred_idx]
                aligned_βs[true_idx] = pred_βs[pred_idx]
                aligned_π[true_idx] = pred_π[pred_idx]
                aligned_As[:, :, true_idx] = pred_As[:, :, pred_idx]
            end
        end
    end

    return aligned_As, aligned_Zs, aligned_αs, aligned_βs, aligned_π, label_mapping
end

function results(algo, no_obs_vals, no_nodes, τ, burn_in, directed, a_star, b_star, Hu11, Hu01, Hu10, Hu00, γ, alpha_val, base_dist, K, do_calc, π, ρ)

    Ks = (algo == 1) ? [K] : []
    noise_vals = range(0.025, stop=0.5, step=0.025)

    part_dist = Dict{Int,Dict{Float64,Float64}}()
    param_dist = Dict{Int,Dict{Float64,Float64}}()
    network_dist = Dict{Int,Dict{Float64,Float64}}()
    network_certainty = Dict{Int,Dict{Float64,Float64}}()

    # Generate As
    As = zeros(Bool, no_nodes, no_nodes, K)
    for k in 1:K
        As[:, :, k] = DPMM.gen_rand_A(no_nodes, ρ, directed)
    end

    for n_obs in no_obs_vals
        part_dist[n_obs] = gen_noise_dict(noise_vals)
        param_dist[n_obs] = gen_noise_dict(noise_vals)
        network_dist[n_obs] = gen_noise_dict(noise_vals)
        network_certainty[n_obs] = gen_noise_dict(noise_vals)

        # Generate Zs
        Zs = FMM.gen_Z(n_obs, π)
        true_π = counts(Zs) ./ n_obs

        for noise in noise_vals

            αs = ones(K) .- noise
            βs = zeros(K) .+ noise

            # Generate a set of data from As and Zs 
            D = zeros(Bool, no_nodes, no_nodes, n_obs)

            for i in 1:n_obs
                D[:, :, i] = FMM.gen_D(As[:, :, Zs[i]], αs[Zs[i]], βs[Zs[i]], directed)
            end

            results = measure_recovery_performance(D, algo, directed, Ks, τ, burn_in, K, As, Zs, αs, βs, true_π, ρ, a_star, b_star, Hu11, Hu01, Hu10, Hu00, γ, alpha_val, base_dist, do_calc)

            part_dist[n_obs][noise] = results[1]
            param_dist[n_obs][noise] = results[2]
            network_dist[n_obs][noise] = results[3]
            network_certainty[n_obs][noise] = results[4]

        end

    end

    return part_dist, param_dist, network_dist, network_certainty
end

function measure_recovery_performance(D, algo, directed, Ks, τ, burn_in, K, true_As, true_Zs, true_αs, true_βs, true_π, true_ρ, a_star, b_star, Hu11, Hu01, Hu10, Hu00, γ, alpha_val, base_dist, do_calc)

    results = zeros(Float64, 4)

    if algo == 1

        As_samples, Zs_samples, αs_samples, βs_samples, π_samples, ρ_samples, opt_k, mean_likelihoods = FMM.test_gibbs_FMM(D, Ks, τ, Hu11, Hu01, Hu00, Hu10, a_star, b_star, γ, burn_in, directed)

        As_PEs, Zs_PEs, αs_PEs, βs_PEs, π_PE, ρ_PE = FMM.calc_point_estimates(As_samples, Zs_samples, αs_samples, βs_samples, π_samples, ρ_samples, τ, burn_in, opt_k, directed)

    else

        As_samples, Zs_samples, αs_samples, βs_samples, π_samples, ρ_samples, opt_k, cluster_freqs = DPMM.test_gibbs_DPMM(alpha_val, τ, burn_in, base_dist, D, a_star, b_star, Hu11, Hu01, Hu10, Hu00, directed)

        As_PEs, Zs_PEs, αs_PEs, βs_PEs, π_PE, ρ_PE = FMM.calc_DPMM_PEs(As_samples, Zs_samples, αs_samples, βs_samples, π_samples, ρ_samples, opt_k, burn_in, τ, directed)

    end

    aligned_As_PEs, aligned_Zs_PEs, aligned_αs_PEs, aligned_βs_PEs, aligned_π_PE, label_mapping = align_clusters(As_PEs, Zs_PEs, αs_PEs, βs_PEs, π_PE,
        true_As, true_Zs, true_αs, true_βs, true_π)

    # Calc Variation of information for Zs
    if do_calc[1] == 1
        results[1] = Clustering.varinfo(true_Zs, aligned_Zs_PEs)
    else
        results[1] = 0
    end

    # Calculation of the parameter distance: l1 distance
    if do_calc[2] == 1
        results[2] = calc_parameters_distance(true_αs, true_βs, true_π, true_ρ,
            aligned_αs_PEs, aligned_βs_PEs, aligned_π_PE, ρ_PE)
    else
        results[2] = 0
    end

    # Calc network distance for As
    if do_calc[3] == 1
        results[3] = compare_As(true_As, aligned_As_PEs, directed, K)
    else
        results[3] = 0
    end

    # Network certainty: KL divergence
    if do_calc[4] == 1
        if algo == 1
            results[4] = calc_KL_divergence(D, aligned_Zs_PEs, directed, aligned_αs_PEs, aligned_βs_PEs, K, ρ_PE)
        else
            results[4] = calc_KL_divergence(D, aligned_Zs_PEs, directed, aligned_αs_PEs, aligned_βs_PEs, opt_k, ρ_PE)
        end
    else
        results[4] == 0
    end

    return results
end

# Calculates the Kullbeck-Liebler divergence
function calc_KL_divergence(D, Zs, directed, αs, βs, K, ρ)
    n = size(D)[1]
    prob = 0.0
    epsilon = 1e-16

    N = FMM.calc_Nus(Zs, K)

    X_u_ij = FMM.calc_X_u_ij(D, Zs, K)

    for u in 1:K
        for i in 1:n
            a = directed ? 1 : i + 1
            for j in a:n
                if i != j
                    l = X_u_ij[i, j, u]
                    Quij = FMM.calc_Q_u_ij(l, N[u], ρ, αs[u], βs[u])
                    Quij_safe = clamp(Quij, epsilon, 1 - epsilon)
                    prob += Quij_safe * log(Quij_safe / ρ) + (1 - Quij_safe) * log((1 - Quij_safe) / (1 - ρ))
                end
            end
        end
    end

    return prob
end

function gen_noise_dict(noise_vals)
    inner_dict = Dict{Float64,Float64}()
    for val in noise_vals
        inner_dict[val] = 0
    end
    return inner_dict
end

function calc_parameters_distance(true_αs, true_βs, true_π, true_ρ, αs, βs, π, ρ)
    param_dist = 0
    param_dist += l1_dist(true_αs, αs)
    param_dist += l1_dist(true_βs, βs)
    param_dist += l1_dist(true_π, π)
    param_dist += l1_dist(true_ρ, ρ)
    return param_dist
end

function l1_dist(θ_true, θ_inferred)
    diff = length(θ_true) - length(θ_inferred)
    if diff != 0
        if diff < 0
            append!(θ_true, zeros(abs(diff)))
        else
            append!(θ_inferred, zeros(abs(diff)))
        end
    end

    dist = θ_true - θ_inferred
    abs_dist = abs.(dist)
    sum_abs_dist = sum(abs_dist)
    return sum_abs_dist
end

function compare_As(true_As, As_PEs, directed, K)
    FN = 0
    FP = 0
    n = size(true_As)[1]

    diff = size(true_As)[end] - size(As_PEs)[end]
    if diff != 0
        empty_As = zeros(Bool, n, n, abs(diff))
        if diff < 0
            true_As = cat(true_As, empty_As, dims=3)
        else
            As_PEs = cat(As_PEs, empty_As, dims=3)
        end
    end

    for u in 1:size(As_PEs)[end]
        for i in 1:n
            l = directed ? 1 : i + 1
            for j in l:n
                if i != j
                    FN += (true_As[i, j, u] == 1) && (As_PEs[i, j, u] == 0)
                    FP += (true_As[i, j, u] == 0) && (As_PEs[i, j, u] == 1)
                end
            end
        end
    end

    return FN + FP
end

function gen_x_and_y(data, measure, n_obs_values, algorithm)
    indices = [20, 50, 100, 200]
    labels = ["20", "50", "100", "200"]

    algo_used = (algorithm == 1) ? "FMM " : "DPMM "

    ylabel = measure == 1 ? "Partition distance" :
             measure == 2 ? "Parameter distance" :
             measure == 3 ? "Network distance" : "Network certainty"
    title = algo_used * "Noise vs " * ylabel

    plt = plot(xlabel="Flip Probability", ylabel=ylabel, title=title, legend=true)

    for (i, idx) in enumerate(indices)
        if haskey(data, idx)
            sorted_pairs = sort(collect(data[idx]))
            sorted_dict = OrderedDict(sorted_pairs)
            dt_keys = collect(keys(sorted_dict))
            dt_values = collect(values(sorted_dict))
            plot!(plt, dt_keys, dt_values, label=labels[i])
        else
            @warn "Index $idx not found in data"
        end
    end

    return plt
end

function count_parameters(K, no_nodes, directed)
    mixture_weights = K - 1
    tpr = K
    fpr = K

    if directed
        structures = K * no_nodes * (no_nodes - 1)
    else
        structures = K * (no_nodes * (no_nodes - 1)) / 2
    end

    density = 1

    total = mixture_weights + tpr + fpr + structures + density

    return total
end

end