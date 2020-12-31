using DiscreteElasticRods
DER = DiscreteElasticRods
using Test

# Setup a random configuration
N = 5
rod = DER.random_rod(N)
ref_strains = DER.full_kinematics(DER.random_rod(N))
rod_props = DER.elliptical_stiffness(2.0, 3.0, 4.0, 2.0)

# Utilities
function arr2delt(q)
    dx = reshape(q[1:3*(N+2)], N+2, 3)
    dt = reshape(q[3*N+7:4*N+7], N+1, 1)
    DER.rod_delta(dx,dt)
end

function delt2arr(dr)
    q = [dr.Δx[:]; dr.Δθ]
end

# Energy trio of functions
using FiniteDiff
q = zeros(4*N+7)
dr = arr2delt(q)

function E(q)
    dr = arr2delt(q)
    newrod = DER.rod_update(rod, dr)
    strains = DER.full_kinematics(newrod)
    DER.elastic_energy(ref_strains, strains, rod_props)
end

function ∇E(q)
    dr = arr2delt(q)
    newrod = DER.rod_update(rod, dr)
    gE = DER.grad_elastic_energy(newrod, rod_props, ref_strains)
    delt2arr(gE)
end

GradE = ∇E(q)
@test ∇E(q) ≈ FiniteDiff.finite_difference_gradient(E, q)

# Hessians
# using ForwardDiff
# HessE = ForwardDiff.jacobian(∇E, q)
# @test HessE ≈ FiniteDiff.finite_difference_hessian(E, q)
# err = HessE - FiniteDiff.finite_difference_hessian(E, q)
