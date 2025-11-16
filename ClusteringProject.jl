module ClusteringProject
include("FMM.jl")

# 2. Include DPMM second
include("DPMM.jl")

# 3. Include Testing last
include("Testing.jl")

export FMM, DPMM, Testing
end
