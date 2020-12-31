using Test
using LinearAlgebra
using DiscreteElasticRods
DER = DiscreteElasticRods

@testset "gradient_nonmutating" begin


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

    # Energy trio of functions
    function E(q)
        dr = arr2delt(q)
        newrod = DER.rod_update(rod, dr)
        strains = DER.full_kinematics(newrod)
        DER.elastic_energy(ref_strains, strains, rod_props)
    end

    function ∇E(q)
        dr = arr2delt(q)
        newrod, ζ = DER.gradient_update(rod, dr)
        gE = DER.elastic_forces(newrod, rod_props, ref_strains)

        gE.Δx[1:end-1,:] -= gE.Δθ .* ζ
        gE.Δx[2:end,:] += gE.Δθ .* ζ

        delt2arr(gE)
    end

    using FiniteDiff
    using ForwardDiff

    q = rand(4*N+7)
    GradE = ∇E(q)
    @test ∇E(q) ≈ FiniteDiff.finite_difference_gradient(E, q)

    HessE = ForwardDiff.jacobian(∇E, q)
    err = HessE - FiniteDiff.finite_difference_hessian(E, q)
    @test norm(err)/norm(HessE) < 1e-5

end
