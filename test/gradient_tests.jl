using Test
using LinearAlgebra
using FiniteDiff
using ForwardDiff
using DiscreteElasticRods

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

# Compare the mutating gradient to the non-mutating one
@testset "gradient_mutating" begin

    # Dual Cache trick requires a reinterpret!

    gE = DER.allocate_delta(Float64, N)
    dr = DER.allocate_delta(Float64, N)
    caches = DER.allocate_cache(DER.elastic_forces!, Float64, N)

    function ∇E_mutating(q)
        DER.coordinate_storage!(dr, q)
        newrod, ζ = DER.gradient_update(rod, dr)
        DER.elastic_forces!(gE, newrod, rod_props, ref_strains, caches)
        gE.Δx[1:end-1,:] .-= gE.Δθ .* ζ
        gE.Δx[2:end,:] .+= gE.Δθ .* ζ

        DER.coordinate_storage(gE)
    end

    function ∇E_reference(q)
        DER.coordinate_storage!(dr, q)
        newrod, ζ = DER.gradient_update(rod, dr)
        grdE = DER.elastic_forces(newrod, rod_props, ref_strains)

        grdE.Δx[1:end-1,:] .-= grdE.Δθ .* ζ
        grdE.Δx[2:end,:] .+= grdE.Δθ .* ζ

        DER.coordinate_storage(grdE)
    end

    # Gradient Acuracy
    q = rand(4*N+7)
    ∇Em = ∇E_mutating(q)
    ∇Er = ∇E_reference(q)
    @test norm(∇Em - ∇Er)/norm(∇Er) < 1e-13

    # Hessian Accuracy
    jcfg = ForwardDiff.JacobianConfig(∇E_mutating, q)
    DualType = eltype(jcfg)

    gE = DER.allocate_delta(DualType, N)
    dr = DER.allocate_delta(DualType, N)
    caches = DER.allocate_cache(DER.elastic_forces!, DualType, N)

    HessE = ForwardDiff.jacobian(∇E_mutating, q, jcfg)
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
