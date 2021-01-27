using DiscreteElasticRods

# Contains a reference config + coordinates, as well as props/ref_strains
struct elastic_rod{T}
    r::DER.basic_rod{T}
    q::DER.rod_delta{T}
    p::DER.rod_properties
    s::DER.rod_strains{T}
end

# Define a Rod
p1 = [0, 0, 0]
p2 = [1, 0, 0]
R = 1.0
ns = 15
d = [0, 0, 1.0]

ref_rod = DER.arc_rod(p1, p2, d, R, ns)
ref_strains = DER.full_kinematics(ref_rod)
rod_props = DER.elliptical_stiffness(1e6, 0.48, 0.01, 0.01)
DER.plot(ref_rod)

er = elastic_rod(ref_rod, DER.zero_delta(Float64, ns), rod_props, ref_strains)

# Typing for dual caches
NumType = Float64
using SparseDiffTools
DualTag = SparseDiffTools.DeivVecTag
DualType = ForwardDiff.Dual{DualTag, NumType, 1}

using DiffEqBase: DiffCache, dualcache, get_tmp

using ForwardDiff
abstract type DualType{V,N} end
struct MyTag end

struct dual_rod{V,N} <: DualType{V,N}
    num::DER.basic_rod{V}
    dual::DER.basic_rod{ForwardDiff.Dual{nothing,V,N}}
end

struct dual_delta{V,N} <: DualType{V,N}
    num::DER.rod_delta{V}
    dual::DER.rod_delta{ForwardDiff.Dual{nothing,V,N}}
end

function resolve_type(dual::DualType{V,N}, ::Type{V}) where {V, N}
    return dual.num
end

function resolve_type(dual::DualType{V,N}, ::Type{ForwardDiff.Dual{T,V,N}}) where {T, V, N}
    return reinterpret(ForwardDiff.Dual{T,V,N}, dual.dual)
    return dual.dual
end

# also extend Base.reinterpret to my rods
function Base.reinterpret(T::Type, r::DER.basic_rod)
    DER.basic_rod(reinterpret(T, r.x), reinterpret(T, r.d))
end

function Base.reinterpret(T::Type, Δr::DER.rod_delta)
    DER.basic_rod(reinterpret(T, Δr.Δx), reinterpret(T, Δr.Δθ))
end



ns = 15
nd = 3
r = DER.allocate_rod(Float64, ns)
dr = DER.allocate_rod(ForwardDiff.Dual{nothing, Float64, nd}, ns)
dual_r = dual_rod(r, dr)

resolve_type(dual_r, ForwardDiff.Dual{nothing, Float64, 3})


# Allocate coordinate layer
s = zeros(NumType, 4*ns)
∇s = zeros(NumType, 4*ns)

# Allocate expanded coordinate layer
q = zeros(NumType, 4*ns+7)
∇q = zeros(NumType, 4*ns+7)
qd = zeros(DualType, 4*ns+7)
∇qd = zeros(DualType, 4*ns+7)

# Convert between the two
function s_to_q!(q, s)
    q[8:end] .= s
end

function ∇q_to_∇s!(∇s, ∇q)
    ∇s .= @view ∇q[8:end]
end

# Non-allocating gradient for NumType
gradients = DER.allocate_delta(DualType, ns)
caches = DER.allocate_cache(fused_update_gradient!, NumType, ns)

function dEdq!(∇q::Vector{NumType}, q::Vector{NumType})
   DER.banded_storage!(er.q, q)
   DER.fused_update_gradient!(gradients, er.r, er.q, er.p, er.s, caches)
   DER.banded_storage!(∇q, gradients)
end

function dEds!(∇s::Vector{NumType}, s::Vector{NumType})
    s_to_q!(q, s)
    dEdq!(∇q, q)
    ∇q_to_∇s!(∇s, ∇q)
end

# Non-allocating gradient for DualType
gradients2 = DER.allocate_delta(DualType, ns)
caches2 = DER.allocate_cache(fused_update_gradient!, DualType, ns)
er2 = (r = ref_rod,
       q = DER.allocate_delta(DualType, ns),
       p = rod_props,
       s = ref_strains)

function dEdq!(∇q::Vector{DualType}, q::Vector{DualType})
   DER.banded_storage!(er2.q, q)
   DER.fused_update_gradient!(gradients2, er2.r, er2.q, er2.p, er2.s, caches2)
   DER.banded_storage!(∇q, gradients2)
end

function dEds!(∇s::Vector{DualType}, s::Vector{DualType})
   s_to_q!(qd, s)
   dEdq!(∇qd, qd)
   ∇q_to_∇s!(∇s, ∇qd)
end

# Pullback to coordinate space




# Hessian Vector Products w/ SparseDiffTools
H = SparseDiffTools.JacVec(dEds!, s)
using LinearAlgebra
v = rand(4*ns)
Hv = similar(v)
@btime mul!(Hv, H, v)








# Specification layer for one end - Zygote help me
using Zygote
function coord_to_delta(s)
    [zeros(eltype(s), 7); s]
end

s = zeros(5)
q, back = Zygote.pullback(coord_to_delta, s)
@code_llvm(back(q))

using ForwardDiff
DualType = ForwardDiff.Dual{nothing, Float64, 12}
s = zeros(DualType, 2000)
q = zeros(DualType, 2007)
@btime back(q)




# Banded Hessian for this purpose

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
