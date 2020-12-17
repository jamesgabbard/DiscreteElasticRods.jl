using DiscreteElasticRods
DER = DiscreteElasticRods
using Zygote
using ForwardDiff
using FiniteDiff
using ReverseDiff

# Rod size
n = 10

# Pullbacks through constructors
using Zygote:@adjoint
@adjoint DER.basic_rod(x, d) = DER.basic_rod(x, d), dr -> (dr.x, dr.d)
@adjoint DER.rod_delta(Δx, Δθ) = DER.rod_delta(Δx, Δθ), dΔr -> (dΔr.Δx, dΔr.Δθ)
@adjoint DER.rod_strains(l, κ, τ) = DER.rod_strains(l, κ, τ), ds -> (ds.l, ds.κ, ds.τ)

# Setup an energy functional
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
tape = ReverseDiff.GradientTape(total_energy_array, q)
compiled_tape = ReverseDiff.compile(tape)
grad_reverse!(out, q) = ReverseDiff.gradient!(out, compiled_tape, q)
grad_forward(q) = ForwardDiff.gradient(total_energy, q)
grad_zygote(q) = Zygote.gradient(total_energy, q)

# Timing the Gradients
using BenchmarkToolss
out = similar(q)
@btime grad_reverse!(out, q)
@btime grad_forward(q)
@btime grad_zygote(q)


# ------------------------------------------------------------------------------
# Hessian Timing
# ------------------------------------------------------------------------------
using SparseDiffTools
using SparsityDetection: hessian_sparsity, jacobian_sparsity

input = similar(q)
output = similar(q)

function grad_forward!(output, input)
    output .= grad_forward(input)
end

function grad_zygote!(output, input)
    output .= grad_zygote(input)
end

#hessian_pattern2 = jacobian_sparsity(grad_forward!, output, input; verbose = true)
#hessian_prototype = Float64.(hessian_pattern)
using BandedMatrices: BandedMatrix
hessian_prototype = BandedMatrix{Float64}(Ones(nq, nq), (10, 10))
colors = SparseDiffTools.matrix_colors(hessian_prototype)
