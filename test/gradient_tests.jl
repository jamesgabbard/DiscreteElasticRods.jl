using Test
using LinearAlgebra
using FiniteDiff
using ForwardDiff
using DiscreteElasticRods
DER = DiscreteElasticRods

# Setup a random configuration
N = 15
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
    [dr.Δx[:]; dr.Δθ]
end

function E(q)
    dr = arr2delt(q)
    newrod = DER.rod_update(rod, dr)
    strains = DER.full_kinematics(newrod)
    DER.elastic_energy(ref_strains, strains, rod_props)
end

# Compare the nonmutating gradient to FiniteDiff
@testset "gradient_nonmutating" begin

    function ∇E(q)
        dr = arr2delt(q)
        newrod, ζ = DER.gradient_update(rod, dr)
        gE = DER.elastic_forces(newrod, rod_props, ref_strains)

        gE.Δx[1:end-1,:] .-= gE.Δθ .* ζ
        gE.Δx[2:end,:] .+= gE.Δθ .* ζ

        delt2arr(gE)
    end

    q = rand(4*N+7)
    GradE = ∇E(q)
    @test ∇E(q) ≈ FiniteDiff.finite_difference_gradient(E, q)

    HessE = ForwardDiff.jacobian(∇E, q)
    err = HessE - FiniteDiff.finite_difference_hessian(E, q)
    @test norm(err)/norm(HessE) < 1e-5
end

# Compare the mutating gradient to the non-mutating one
@testset "gradient_mutating" begin
    gE = DER.allocate_delta(Float64, N)
    caches = (zeros(N+1, 7), zeros(N, 28))

    function ∇E_mutating(q)
        dr = arr2delt(q)
        newrod, ζ = DER.gradient_update(rod, dr)
        DER.elastic_forces!(gE, newrod, rod_props, ref_strains, caches)
        gE.Δx[1:end-1,:] .-= gE.Δθ .* ζ
        gE.Δx[2:end,:] .+= gE.Δθ .* ζ

        delt2arr(gE)
    end

    function ∇E_reference(q)
        dr = arr2delt(q)
        newrod, ζ = DER.gradient_update(rod, dr)
        grdE = DER.elastic_forces(newrod, rod_props, ref_strains)

        grdE.Δx[1:end-1,:] .-= grdE.Δθ .* ζ
        grdE.Δx[2:end,:] .+= grdE.Δθ .* ζ

        delt2arr(grdE)
    end

    # Gradient Acuracy
    q = rand(4*N+7)
    ∇Em = ∇E_mutating(q)
    ∇Er = ∇E_reference(q)
    @test norm(∇Em - ∇Er)/norm(∇Er) < 1e-13

    # Hessian Accuracy
    jcfg = ForwardDiff.JacobianConfig(∇E_dual, q)
    DualType = eltype(jcfg)

    gE = DER.allocate_delta(DualType, N)
    caches = (zeros(DualType, N+1, 7), zeros(DualType, N, 28))

    HessE = ForwardDiff.jacobian(∇E_dual, q, jcfg)
    err = HessE - FiniteDiff.finite_difference_hessian(E, q)
    @test norm(err)/norm(HessE) < 1e-5
end
