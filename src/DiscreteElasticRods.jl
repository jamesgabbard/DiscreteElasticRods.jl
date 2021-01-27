module DiscreteElasticRods

using LinearAlgebra
using SparseArrays
export DER

const DER = DiscreteElasticRods
const PTRANSPORT_TOL = 0.06

include("rod_types.jl")
include("rod_nonmutating.jl")
include("rod_mutating.jl")
include("rod_specification.jl")
include("rod_gradients.jl")
include("rod_sparsity.jl")
include("rod_plot.jl")

end
