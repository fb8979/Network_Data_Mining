### A Pluto.jl notebook ###
# v0.20.21

using Markdown
using InteractiveUtils

# ╔═╡ eb9a5e3a-31a5-4aec-ab69-3e68b2a95638
begin
import Pkg
Pkg.activate(@__DIR__; io=devnull)
using ClusteringProject
end

# ╔═╡ f58b4537-55e4-458f-b5f7-ff63415bbcfa
using GraphRecipes, Plots, StatsBase, Distributions, Statistics, StatsPlots, SpecialFunctions, LogExpFunctions, Clustering, ColorTypes, Clustering, DataStructures, Distances, LinearAlgebra, Hungarian

# ╔═╡ 7aa9240b-da4b-4495-89a8-246ac4af6f83
md"""
# Network Data Mining: 
### Clustering of heterogeneous populations of networks
"""

# ╔═╡ 5d1bb53f-82af-4091-ab78-10d1708aba28
# Number of subpopulations in the mixture
K = 3;

# ╔═╡ 6ef374d5-eb27-40d2-ad2c-ea09bfe6536e
# Mixture weightings
π = FMM.gen_rand_π(K)

# ╔═╡ cfa38aa6-27f5-4370-9920-f98ec8e7e273
# True positive rates
αs = ones(K) * 0.9;

# ╔═╡ 2b3adaa2-4258-4489-aa53-c750b625659f
# False positive rates
βs = ones(K) * 0.1;

# ╔═╡ 6e2c92ae-b213-4569-bbcf-89d18d95aa48
#Number of nodes in each network observation
no_nodes = 50;

# ╔═╡ 099e82cc-2bde-45e9-ab2c-792374a1976c
# Number of network observations
no_obs = 100;

# ╔═╡ 5e9467e0-c0f3-4a82-be7c-7b0d15ca5d62
# Network density
ρ = 0.01;

# ╔═╡ 836e2a16-4ebb-47ca-8ba1-011eef5243a2
# Range of values the FMM gibbs sampling algorithm is run over
Ks = 1:8;

# ╔═╡ 66ccf09c-476c-406c-b9d5-bc81c8fdf78e
# Iterations
τ = 200;

# ╔═╡ 4bfd7e61-de33-43a6-8360-2acf627a8866
# Concentration parameter for Dirichlet process 
alpha_val = 0.005;

# ╔═╡ 92f7dc15-ebc0-4fb1-b71d-a776a18488bb
# Base distribution for dirichlet process 
base_dist = Beta(1,1);

# ╔═╡ b03bd791-224e-42e7-bdfa-d580b0059b05
burn_in = 0.25;

# ╔═╡ b18e9f8c-c439-4463-8ec2-26a292d44dee
directed = true;

# ╔═╡ 23cecd75-4ff3-49f1-83dc-98b6a1964b3b
#Hyper parameters for density
a_star, b_star = 1, 20;

# ╔═╡ 8ff1cf5d-b210-4563-9bb3-937b673cfe18
#Hyperparameters for TP rate/ alpha
Hu11, Hu01 = 25, 5;

# ╔═╡ f34a1f25-919a-47c2-8db3-e8c25060819d
#Hyperparameters for FP rate/ beta
Hu10, Hu00 = 5, 25;

# ╔═╡ f724b93b-bee2-44a6-8934-d55fa6975b17
#Hyperparameters for mixture weighting
γ = [];

# ╔═╡ 9bef5431-93f3-4498-b780-0d130517cdca
# Ground truth networks, mode allocations and the data
true_Zs, true_As, D = FMM.gen_mixture_D(π, no_obs, no_nodes, αs, βs, ρ, directed);

# ╔═╡ 2dabbb7e-5cb6-4cdd-8214-d2095ad87fbe
# FMM = 1, DPMM = 2
algo = 1;

# ╔═╡ 04a8b9e3-3909-4ab8-afc8-c9888dc8aa17
As_point_estimates, k_likelihoods, cluster_counts, As_samples, Zs_samples, αs_samples, βs_samples, π_samples = FMM.test_clustering(algo, D, directed, burn_in, τ, Ks, a_star, b_star, Hu11, Hu01, Hu10, Hu00, γ, alpha_val, base_dist, true_As, true_Zs, αs, βs, π, ρ);

# ╔═╡ ad78f9ae-01a4-427c-ab2e-d331d13e579d
# True As

# ╔═╡ 6b0a5616-dcfd-4899-b0c5-6702ce7e37ed
img1 = [Gray.(true_As[:, :, i]) for i in 1:size(true_As)[end]]

# ╔═╡ 6de64707-a9d9-4afe-96d8-2d78fb74a23e
# Inferred As

# ╔═╡ 43bf0629-cae3-4c64-8557-58ac44d917dc
img2 = [Gray.(As_point_estimates[:, :, i]) for i in 1:size(As_point_estimates)[end]]

# ╔═╡ b4d87a92-1b48-4c39-8e57-0fc5613b61bc
begin
	if algo == 1
		plot(k_likelihoods, xlabel="No. of Clusters", ylabel="log-likelihood", title="Log-likelihood of Point Estimates over clusters")
	else
		histogram(cluster_counts, xlabel="No. of Clusters", ylabel="Frequency", title="DPMM: Frequency of No. of Clusters", bins=length(cluster_counts), xticks=0:1:60)
	end
end

# ╔═╡ dc822a8d-aa13-44d7-9bae-02bd78eecbf4
md"""
## Akaike Information Criterion
"""

# ╔═╡ aacc9a6e-e4fc-4c61-8f92-3ffcad960978
begin
	if algo == 1
		AICs = zeros(length(Ks))
		for k in Ks
    		num_params = Testing.count_parameters(k, no_nodes, directed)
    		AICs[k] = 2 * num_params - 2 * k_likelihoods[k]
		end
		plot(AICs, xlabel="No. of Cluster", ylabel="BIC",
			title="AIC Scores")
	end
end

# ╔═╡ 3396c362-f503-4e99-a1ba-afeee34b2a79
md"""
## Comparison with K-Means

"""

# ╔═╡ e79849f5-e657-46f6-bc67-471fe917dbd7
wcss_values, elbow_plt, improvement_plt = Testing.kmeans_elbow_method(D, Ks, directed=directed);

# ╔═╡ 23933363-cfea-4d85-817e-bc3dee4f4161
elbow_plt

# ╔═╡ 97500f39-db84-45f0-a0c7-657c71d3cdd5
md"""
## Testing
"""

# ╔═╡ b33a6856-798a-4f7b-a777-fd8abe3c63a3
fix_label_switching = true

# ╔═╡ 48cc5fca-ef6d-4ddd-b819-0905572fa5ca
begin 

	do_calc = [1,1,1,1]
	n_obs_values = [20, 50, 100, 200]

	# Includes attempt at fixing label switching problem
	if fix_label_switching
		MM_part_data, MM_param_data, MM_network_dist_data, MM_network_cert_data = Testing.generate_plots_data(τ, n_obs_values , no_nodes, burn_in, directed, a_star, b_star, Hu11, Hu01, Hu10, Hu00, γ, alpha_val, base_dist, K, do_calc, algo, π, ρ)
	end

end

# ╔═╡ 3854716c-adb5-44c6-9324-16abf2566d6e
Testing.gen_x_and_y(MM_part_data, 1, n_obs_values, 2)

# ╔═╡ ceb33617-3513-49eb-aed3-eda885b40f71
Testing.gen_x_and_y(MM_param_data, 2, n_obs_values, 2)

# ╔═╡ f771d5cf-bc87-4716-b8a8-4402c26866eb
Testing.gen_x_and_y(MM_network_dist_data, 3, n_obs_values, 2)

# ╔═╡ b347fda9-812b-48a0-8af6-33313a334f65
Testing.gen_x_and_y(MM_network_cert_data, 4, n_obs_values, 2)

# ╔═╡ Cell order:
# ╟─7aa9240b-da4b-4495-89a8-246ac4af6f83
# ╠═eb9a5e3a-31a5-4aec-ab69-3e68b2a95638
# ╠═f58b4537-55e4-458f-b5f7-ff63415bbcfa
# ╠═5d1bb53f-82af-4091-ab78-10d1708aba28
# ╠═6ef374d5-eb27-40d2-ad2c-ea09bfe6536e
# ╠═cfa38aa6-27f5-4370-9920-f98ec8e7e273
# ╠═2b3adaa2-4258-4489-aa53-c750b625659f
# ╠═6e2c92ae-b213-4569-bbcf-89d18d95aa48
# ╠═099e82cc-2bde-45e9-ab2c-792374a1976c
# ╠═5e9467e0-c0f3-4a82-be7c-7b0d15ca5d62
# ╠═836e2a16-4ebb-47ca-8ba1-011eef5243a2
# ╠═66ccf09c-476c-406c-b9d5-bc81c8fdf78e
# ╠═4bfd7e61-de33-43a6-8360-2acf627a8866
# ╠═92f7dc15-ebc0-4fb1-b71d-a776a18488bb
# ╠═b03bd791-224e-42e7-bdfa-d580b0059b05
# ╠═b18e9f8c-c439-4463-8ec2-26a292d44dee
# ╠═23cecd75-4ff3-49f1-83dc-98b6a1964b3b
# ╠═8ff1cf5d-b210-4563-9bb3-937b673cfe18
# ╠═f34a1f25-919a-47c2-8db3-e8c25060819d
# ╠═f724b93b-bee2-44a6-8934-d55fa6975b17
# ╠═9bef5431-93f3-4498-b780-0d130517cdca
# ╠═2dabbb7e-5cb6-4cdd-8214-d2095ad87fbe
# ╠═04a8b9e3-3909-4ab8-afc8-c9888dc8aa17
# ╠═ad78f9ae-01a4-427c-ab2e-d331d13e579d
# ╟─6b0a5616-dcfd-4899-b0c5-6702ce7e37ed
# ╠═6de64707-a9d9-4afe-96d8-2d78fb74a23e
# ╟─43bf0629-cae3-4c64-8557-58ac44d917dc
# ╟─b4d87a92-1b48-4c39-8e57-0fc5613b61bc
# ╟─dc822a8d-aa13-44d7-9bae-02bd78eecbf4
# ╟─aacc9a6e-e4fc-4c61-8f92-3ffcad960978
# ╟─3396c362-f503-4e99-a1ba-afeee34b2a79
# ╟─e79849f5-e657-46f6-bc67-471fe917dbd7
# ╟─23933363-cfea-4d85-817e-bc3dee4f4161
# ╟─97500f39-db84-45f0-a0c7-657c71d3cdd5
# ╠═b33a6856-798a-4f7b-a777-fd8abe3c63a3
# ╠═48cc5fca-ef6d-4ddd-b819-0905572fa5ca
# ╠═3854716c-adb5-44c6-9324-16abf2566d6e
# ╠═ceb33617-3513-49eb-aed3-eda885b40f71
# ╠═f771d5cf-bc87-4716-b8a8-4402c26866eb
# ╠═b347fda9-812b-48a0-8af6-33313a334f65
