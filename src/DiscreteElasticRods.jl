module DiscreteElasticRods

include("rod_linalg.jl")
include("rod_core.jl")
include("rod_specification.jl")

export material2stiffness, straight_rod, discretize_span, discrete_rod, continuous_rod


end
