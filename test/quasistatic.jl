# A simple quasi-static problem, based on a Newton (or Newton Krylov) method

using DiscreteElasticRods
DER = DiscreteElasticRods
using BenchmarkTools

# Parameters
ns = 40
E = 5e6
ν = 0.48
L = 0.30
R = 0.003
ρ = 1e3
g = 9.8

# Create a straight rod with NS seg
p1 = [0., 0., 0.]
p2 = [L, 0., 0.]
d0 = [0., 0., 1.]
ref_rod = DER.straight_rod(p1, p2, d0, ns)

# Reference kinematics and properties
props = DER.elliptical_stiffness(E, ν, R, R)
ref_strain = DER.full_kinematics(ref_rod)
m = ref_strain.l*(ρ*π*R^2)

# q ordering: [θ2, x3, θ3, x4, ..., x, θ]
ncore = 4*ns - 7
q = 1:ncore + 7
#Δx = zeros(3,ns+2)
#Δθ = zeros(1,ns+1)
function param_2_rod!(Δx, Δθ, q)

    # Rearrange core DOFs
    Δx[1, 3:end-2] = q[2:4:ncore]
    Δx[2, 3:end-2] = q[3:4:ncore]
    Δx[3, 3:end-2] = q[4:4:ncore]
    Δθ[2:end-1] = q[1:4:ncore]

    # Cantilever BC: do nothing
    # Free BC: fill in the end of the coords
    Δx[:,end-1] = q[ncore+1:ncore+3]
    Δθ[end] = q[ncore+4]
    Δx[:,end] = q[ncore+5:ncore+7]

end

function param_2_rod(q)
    T = eltype(q)
    Δx = zeros(T, 3,ns+2)
    Δθ = zeros(T, 1,ns+1)
    param_2_rod!(Δx, Δθ, q)
    Δx, Δθ
end

# Incorporate Elastic Energy
cache = zeros(27, ns)
Δx = zeros(3,ns+2)
Δθ = zeros(1,ns+1)
new_rod = DER.copy(ref_rod)
strain = DER.copy(ref_strain)

function total_energy(q)
    param_2_rod!(Δx, Δθ, q)
    DER.copy!(new_rod, ref_rod)
    DER.rod_update!(new_rod, Δx, Δθ, cache)
    DER.full_kinematics!(strain, new_rod, cache)
    Ee = DER.elastic_energy(ref_strain, strain, props, cache)
    Eg = DER.gravitational_energy(new_rod.x, m, g, cache)
    Ee + Eg
end


immutable DiffCache{T<:AbstractArray, S<:AbstractArray}
    du::T
    dual_du::S
end

## DiffCache - how do?
function DiffCache{chunk_size}(T, size, ::Type{Val{chunk_size}})
    DiffCache(zeros(T, size...), zeros(ForwardDiff.Dual{nothing,T,chunk_size}, size...))
end

DiffCache(u::AbstractArray) = DiffCache(eltype(u),size(u),Val{ForwardDiff.pickchunksize(length(u))})
DiffCache(u::AbstractArray, size) = DiffCache(eltype(u),size,Val{ForwardDiff.pickchunksize(length(u))})
get_tmp{T<:ForwardDiff.Dual}(dc::DiffCache, ::Type{T}) = dc.dual_du
get_tmp(dc::DiffCache, T) = dc.du

# Let's take derivatives!
using ForwardDiff
using FiniteDiff
using ReverseDiff

q = rand(ncore + 7)
grad1(q) = FiniteDiff.finite_difference_gradient(total_energy, q)
grad2(q) = ForwardDiff.gradient(total_energy, q)

const tape = ReverseDiff.GradientTape(total_energy, q)
const compiled_tape = ReverseDiff.compile(tape)
grad3!(res,q) = ReverseDiff.gradient!(res, compiled_tape, q)

# Timings for each method
q = rand(ncore + 7)
res = similar(q)
@btime total_energy(q)
#@btime grad1(q)
#@btime grad2(q)
#@btime grad3!(res,q)
