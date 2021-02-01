using Test
using LinearAlgebra
using FiniteDiff
using ForwardDiff
using DiscreteElasticRods
using Random
Random.seed!(42)

# Setup a random configuration
N = 20
rod = DER.random_rod(N)
ref_strains = DER.full_kinematics(DER.random_rod(N))
rod_props = DER.elliptical_stiffness(2.0, 3.0, 4.0, 2.0)

function E(q)
    dr = DER.coordinate_storage(q)
    newrod = DER.rod_update(rod, dr)
    strains = DER.full_kinematics(newrod)
    DER.elastic_energy(ref_strains, strains, rod_props)
end

# Compare the nonmutating gradient to FiniteDiff
@testset "gradient_nonmutating" begin

    function ∇E(q)
        dr = DER.coordinate_storage(q)
        newrod, ζ = DER.gradient_update(rod, dr)
        gE = DER.elastic_forces(newrod, rod_props, ref_strains)

        gE.Δx[1:end-1,:] .-= gE.Δθ .* ζ
        gE.Δx[2:end,:] .+= gE.Δθ .* ζ

        DER.coordinate_storage(gE)
    end

    q = rand(4*N+7)
    GradE = ∇E(q)
    err = GradE - FiniteDiff.finite_difference_gradient(E, q)
    @test norm(err)/norm(GradE) < 1e-5

    HessE = ForwardDiff.jacobian(∇E, q)
    err = HessE - FiniteDiff.finite_difference_hessian(E, q)
    @test norm(err)/norm(HessE) < 1e-5
end


@testset "gradient_mutating" begin

    # Test gradient_update! against allocating version
    dr = DER.random_delta(N)
    nr_ref, ζ_ref = DER.gradient_update(rod, dr)

    newrod = DER.allocate_rod(Float64, N)
    ζ = zeros(N+1, 3)
    cache_gu = DER.allocate_cache(DER.gradient_update!, Float64, N)
    DER.gradient_update!(newrod, ζ, rod, dr, cache_gu)

    @test newrod.x ≈ nr_ref.x
    @test newrod.d ≈ nr_ref.d
    @test ζ ≈ ζ_ref

    # Test elastic_forces! against allocating version
    ∇E_ref = DER.elastic_forces(newrod, rod_props, ref_strains)

    ∇E = DER.allocate_delta(Float64, N)
    cache_ef = DER.allocate_cache(DER.elastic_forces!, Float64, N)
    DER.elastic_forces!(∇E, newrod, rod_props, ref_strains, cache_ef)

    @test ∇E.Δx ≈ ∇E_ref.Δx
    @test ∇E.Δθ ≈ ∇E_ref.Δθ

    # Test fused_update_gradient! against allocating sequence
    function ∇E_reference(q)
        DER.coordinate_storage!(dr, q)
        newrod, ζ = DER.gradient_update(rod, dr)
        grdE = DER.elastic_forces(newrod, rod_props, ref_strains)

        grdE.Δx[1:end-1,:] .-= grdE.Δθ .* ζ
        grdE.Δx[2:end,:] .+= grdE.Δθ .* ζ

        DER.coordinate_storage(grdE)
    end

    cache_fug = DER.allocate_cache(DER.fused_update_gradient!, Float64, N)
    function ∇E_fused(q)
        DER.coordinate_storage!(dr, q)
        DER.fused_update_gradient!(∇E, rod, dr, rod_props, ref_strains, cache_fug)
        DER.coordinate_storage(∇E)
    end

    q = rand(4*N+7)
    ∇Er = ∇E_reference(q)
    ∇Ef = ∇E_fused(q)
    @test norm(∇Ef - ∇Er)/norm(∇Er) < 1e-13

    # Hessian Accuracy
    jcfg = ForwardDiff.JacobianConfig(∇E_fused, q)
    DualType = eltype(jcfg)

    ∇E = DER.allocate_delta(DualType, N)
    dr = DER.allocate_delta(DualType, N)
    cache_fug = DER.allocate_cache(DER.fused_update_gradient!, DualType, N)

    HessE = ForwardDiff.jacobian(∇E_fused, q, jcfg)
    err = HessE - FiniteDiff.finite_difference_hessian(E, q)
    @test norm(err)/norm(HessE) < 1e-5
end

# # Time the allocating version
# function ∇E(q)
#     dr = arr2delt(q)
#     newrod, ζ = DER.gradient_update(rod, dr)
#     gE = DER.elastic_forces(newrod, rod_props, ref_strains)
#     delt2arr(gE)
# end
#
# q = rand(4*N+7)
# @btime ∇E($q)
#
#
# # Time the non-allocating version (IO is allocating, still)
# gE = DER.allocate_delta(Float64, N)
# newrod = DER.allocate_rod(Float64, N)
# dr = DER.allocate_delta(Float64, N)
# ζ = zeros(N+1, 3)
# force_caches = (zeros(N+1, 7), zeros(N, 28))
# update_cache = zeros(N+1, 15)
#
# function ∇E!(out, q)
#     arr2delt!(dr, q)
#     DER.copy!(newrod, rod)
#     DER.gradient_update!(newrod, ζ, dr, update_cache)
#     DER.elastic_forces!(gE, newrod, rod_props, ref_strains, force_caches)
#     (@view gE.Δx[1:end-1, :]) .-= gE.Δθ .* ζ
#     (@view gE.Δx[2:end,:]) .+= gE.Δθ .* ζ
#     delt2arr!(out,gE)
# end
#
# q = rand(4*N+7)
# out = similar(q)
# ∇E!(out,q)
# @btime ∇E!($out, $q)
#
# using SparseDiffTools
# sparsity = DER.coordinate_hessian_sparsity(N)
# colors = matrix_colors(sparsity)
# hess_cache = ForwardColorJacCache(∇E!, q; colorvec=colors, sparsity=sparsity)
# HessE = Float64.(sparsity)
#
# DualType = eltype(hess_cache.fx)
#
# gE = DER.allocate_delta(DualType, N)
# newrod = DER.allocate_rod(DualType, N)
# dr = DER.allocate_delta(DualType, N)
# ζ = zeros(DualType, N+1, 3)
# force_caches = (zeros(DualType, N+1, 7), zeros(DualType, N, 28))
# update_cache = zeros(DualType, N+1, 15)
#
# @btime forwarddiff_color_jacobian!(HessE, ∇E!, q, hess_cache)
# @profile for i = 1:50 forwarddiff_color_jacobian!(HessE, ∇E!, q, hess_cache) end
#
# q = zeros(DualType, 4*N+7)
# out = similar(q)
# @btime ∇E!(out, q)
