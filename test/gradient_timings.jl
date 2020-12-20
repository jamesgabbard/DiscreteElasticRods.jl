using DiscreteElasticRods
DER = DiscreteElasticRods
using Zygote
using ForwardDiff
using FiniteDiff
using ReverseDiff
using BenchmarkTools
using IterativeSolvers
using SparseDiffTools
using DelimitedFiles

struct AHesVec{F,uType}
    f::F
    u::uType
end

Base.size(L::AHesVec) = (length(L.u),length(L.u))
Base.size(L::AHesVec,i::Int) = length(L.u)
Base.:*(L::AHesVec,x::AbstractVector) =  SparseDiffTools.autoback_hesvec(L.f, L.u, x)

import LinearAlgebra
function LinearAlgebra.mul!(du::AbstractVector,L::AHesVec,v::AbstractVector)
    du .= L*v
end

# Rod sizes
n_array = [10, 20, 30, 40, 50]
grad_array = zeros(3, length(n_array))
hess_array = zeros(5, length(n_array))
minres_array = zeros(2, length(n_array))

#for k = 1:length(n_array)
k = 1
    # Setup an energy functional
n = n_array[k]
ref_rod = DER.random_rod(n)
ref_strains = DER.full_kinematics(ref_rod)
rod_props = DER.elliptical_stiffness(10, 0.3, 0.01, 0.04)

function total_energy(q)
    Δx = reshape(q[1:3*n+6], 3, n+2)
    Δθ = reshape(q[3*n+7:end], 1, n+1)
    rod_coords = DER.rod_delta(Δx, Δθ)

    new_rod = DER.rod_update(ref_rod, rod_coords)
    strains = DER.full_kinematics(new_rod)
    DER.elastic_energy(ref_strains, strains, rod_props)
end

function total_energy_array(q)
    Δx = reshape(q[1:3*n+6], 3, n+2)
    Δθ = reshape(q[3*n+7:end], 1, n+1)
    x, d = DER.rod_update(ref_rod.x, ref_rod.d, Δx, Δθ)
    l, κ, τ = DER.full_kinematics(x, d)
    strains = DER.rod_strains(l, κ, τ)
    DER.elastic_energy(ref_strains, strains, rod_props)
end

nq = 4*n+7
q = rand(nq)
# tape = ReverseDiff.GradientTape(total_energy_array, q)
# compiled_tape = ReverseDiff.compile(tape)
#
# grad_reverse!(out, q) = ReverseDiff.gradient!(out, compiled_tape, q)
# grad_forward(q) = ForwardDiff.gradient(total_energy, q)
# grad_zygote(q) = first(Zygote.gradient(total_energy, q))
#
# # Timing the Gradients
# out = similar(q)
# grad_reverse!(out,q)
# grad_forward(q)
# grad_zygote(q)
# rbench = @benchmark grad_reverse!($out, $q)
# fbench = @benchmark grad_forward($q)
# zbench = @benchmark grad_zygote($q)
#
# grad_array[1,k] = minimum(fbench).time/1e9
# grad_array[2,k] = minimum(rbench).time/1e9
# grad_array[3,k] = minimum(zbench).time/1e9

# q = zeros(nq)
# v = rand(nq)
# nh = @benchmark num_hesvec(total_energy, $q, $v)
# nah = @benchmark numauto_hesvec(total_energy, $q, $v)
# anh = @benchmark autonum_hesvec(total_energy, $q, $v)
# nbh = @benchmark numback_hesvec(total_energy, $q, $v)
# abh = @benchmark autoback_hesvec(total_energy, $q, $v)
#
# hess_array[1,k] = minimum(nh).time/1e9
# hess_array[2,k] = minimum(nah).time/1e9
# hess_array[3,k] = minimum(anh).time/1e9
# hess_array[4,k] = minimum(nbh).time/1e9
# hess_array[5,k] = minimum(abh).time/1e9

#writedlm("hessian_timing.txt", hess_array, ',')

grd = rand(nq)

Hvec1 = HesVec(total_energy_array, q; autodiff=true)
out = zeros(length(q))
mna = @benchmark minres!($out, $Hvec1, $grd)

AHVec = AHesVec(total_energy, q)
out = zeros(length(q))
mab = @benchmark minres!($out, $AHVec, $grd)

minres_array[1,k] = minimum(mna).time/1e9
minres_array[2,k] = minimum(mab).time/1e9
writedlm("minres_timing.txt", minres_array, ',')

print("Benchmarked n = $n\n")
#end

writedlm("gradient_timing.txt", grad_array, ',')
writedlm("hessian_timing.txt", hess_array, ',')
writedlm("minres_timing.txt", minres_array, ',')

# ------------------------------------------------------------------------------
# Minres Timing
# ------------------------------------------------------------------------------
