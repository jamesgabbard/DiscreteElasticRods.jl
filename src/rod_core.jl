## rod_construction.jl
# James Gabbard, November 30th, 2020
#
# Interface for specifying a physical system composed of elastic rods

# ----------------------------------------------------------------------------------
#  Types
# ----------------------------------------------------------------------------------

struct basic_rod{T<:Real}
    x::Matrix{T} # 3 x ns + 2
    d::Matrix{T} # 1 x ns + 1
end

struct rod_update{T<:Real}
    Δx::Matrix{T} # 3 x ns + 2
    Δθ::Matrix{T} # 1 x ns + 1
end

# Strains
struct rod_strains{T<:Real}
    l::Matrix{T} # 1 x ns + 1
    κ::Matrix{T} # 2 x ns
    τ::Matrix{T} # 1 x ns
end

function copy(s::rod_strains)
    rod_strains(Base.copy(s.l), Base.copy(s.κ), Base.copy(s.τ))
end

function copy(r::basic_rod)
    basic_rod(Base.copy(r.x), Base.copy(r.d))
end

function copy!(r1::basic_rod, r2::basic_rod)
    r1.x .= r2.x
    r1.d .= r2.d
end

# ----------------------------------------------------------------------------------
#  Kinematics
# --------------------------------------------------------------------------------

function edges!(e, x)
    x1 = @view x[:,2:end]
    x2 = @view x[:,1:end-1]
    e .= x1 .- x2
end

function edges(x)
    x1 = @view x[:,2:end]
    x2 = @view x[:,1:end-1]
    x1 .- x2
end

function tangents!(t, x)
    edges!(t,x)
    norm3!(t)
end

function tangents(x)
    t = edges(x)
    norm3!(t)
    t
end

# Overload for convenience
edges(r::basic_rod) = edges(r.x)
tangents(r::basic_rod) = tangents(r.x)

function edges!(e, r::basic_rod)
    edges!(e, r.x)
end

function tangents!(e, r::basic_rod)
    tangents!(e, r.x)
end

# Update kinematics given displacement and twist
function rod_update(x0, d0, Δx, Δθ)

    x = x0 + Δx
    t = tangents(x)
    d = ptransport(d0, tangents(x0), t)
    rotate_orthogonal_unit!(d, t, Δθ)

    basic_rod(x, d)
end

rod_update(r::basic_rod, Δx, Δθ) = rod_update(r.x, r.d, Δx, Δθ)

function rod_update!(r::basic_rod, Δx, Δθ)

    t0 = tangents(r.x)
    r.x .+= Δx
    t = tangents(r.x)
    r.d .= ptransport(r.d, t0, t)
    rotate_orthogonal_unit!(r.d, t, Δθ)
end

function rod_update!(r::basic_rod, Δx, Δθ, cacheraw)
    ne = length(Δθ)
    cache = reshape(view(cacheraw, 1:18*ne), 18, ne)

    t0 = view(cache, 1:3, :)
    t = view(cache, 4:6, :)
    d0 = view(cache, 7:9, :)
    cache_pt = view(cache, 10:18, :)
    cache_rt = view(cache, 10:14, :)

    tangents!(t0, r.x)
    r.x .+= Δx
    tangents!(t, r.x)
    d0 .= r.d
    ptransport_cache!(r.d, d0, t0, t, cache_pt)
    rotate_orthogonal_unit!(r.d, t, Δθ) #TODO: cache this
end

# cache is 27 rows, nv-2 columns
function full_kinematics!(strains::rod_strains, x, d1, cache)

    # Tangents, lengths
    nv = size(x,2)
    t = edges(x)
    norm3!(strains.l,t)
    t ./= strains.l
    d2 = cross3(t, d1)

    # Caches
    t1 = view(cache, 1:3, :)
    t2 = view(cache, 4:6, :)
    d1l = view(cache, 7:9, :)
    d1r = view(cache, 10:12, :)
    d2l = view(cache, 13:15, :)
    d2r = view(cache, 16:18, :)
    tmp = view(cache, 19:21, :)
    k = view(cache, 22:24, :)

    t1 .= view(t, :, 1:nv-2)
    t2 .= view(t, :, 2:nv-1)
    d1l .= view(d1,:,1:nv-2)
    d1r .= view(d1,:,2:nv-1)
    d2l .= view(d2,:,1:nv-2)
    d2r .= view(d2,:,2:nv-1)

    # Curvature
    cross3!(k, t1, t2)
    k .*= 2.0./(1.0 .+ dot3(t1, t2))
    tmp .= k.*(d2l .+ d2r)./2
    sum!(strains.κ[1:1,:], tmp)
    tmp .= -k.*(d1l .+ d1r)./2
    sum!(strains.κ[2:2,:], tmp)

    # Twists (could move in part of ptransport to reduce intermediates)
    pt_cache = view(cache, 19:27, :) # overrides k, tmp
    d1_transport = view(cache, 13:15, :) # overrides d2l
    ptransport_cache!(d1_transport, d1l, t1, t2, pt_cache)

    cosτ = view(cache, 19:19, :)
    sinτ = view(cache, 20:20, :)

    dot3!(cosτ, d1r, d1_transport, tmp)
    dot3!(sinτ, d2r, d1_transport, tmp)
    strains.τ .= atan.(sinτ, cosτ)
end

function full_kinematics!(strains::rod_strains, r::basic_rod, cache)
    full_kinematics!(strains::rod_strains, r.x, r.d, cache)
end


# Recover Strains from Abosolute Kinematics
function full_kinematics(x, d1)

    # Tangents, lengths
    nv = size(x,2)
    t = edges(x)
    l = norm3(t)
    t ./= l

    # Directors
    d2 = cross3(t, d1)

    # Curvatures
    t1 = @view t[:,1:end-1]
    t2 = @view t[:,2:end]
    k = 2*cross3(t1, t2)./(1.0 .+ dot3(t1, t2))

    κ = Matrix{typeof(d1[1]*k[1])}(undef, 2, length(l)-1)
    κ[1,:] = dot3(k, (view(d2,:,1:nv-2) + view(d2, :, 2:nv-1))/2)
    κ[2,:] = -dot3(k, (view(d1,:,1:nv-2) + view(d1,:,2:nv-1))/2)

    # Twists (could move in part of ptransport to reduce intermediates)
    d1_transport = ptransport(view(d1,:,1:nv-2), t1, t2)
    cosτ = dot3(view(d1,:,2:nv-1), d1_transport)
    sinτ = norm3(cross3(view(d1,:,2:nv-1), d1_transport))
    τ = atan.(sinτ, cosτ)

    rod_strains(l, κ, τ)
end

function full_kinematics(r::basic_rod)
    x, d = r.x, r.d
    full_kinematics(x, d)
end

#----------------------------------------------------------------------------------
#  Reference Twist Kinematic Implementation (Lestringant 2020)
#----------------------------------------------------------------------------------
function spherical_excess(t1, t2, t3)

    ca = dot3(t2,t3)
    cb = dot3(t3,t1)
    cc = dot3(t1,t2)

    sa = norm3(cross3(t2,t3))
    sb = norm3(cross3(t3,t1))
    sc = norm3(cross3(t1,t2))

    cA = (ca - cb.*cc)./(sb.*sc)
    cB = (cb - cc.*ca)./(sc.*sa)
    cC = (cc - ca.*cb)./(sa.*sb)

    V = dot3(cross3(t1,t2),t3)
    sA = V./(sb.*sc)
    sB = V./(sc.*sa)
    sC = V./(sa.*sb)

    atan.(sA,cA) + atan.(sB,cB) + atan.(sC,cC) - π
end

function fused_update_strains_energy(r::basic_rod, Δr::rod_update, ϵr::rod_strains)

    # Unpack
    x0, d0 = r
    Δx, Δθ = Δr
    l0, κ0, τ0 = ϵr

    # Tangents (Should maybe save these?)
    t0 = x0[:,2:end] - x0[:,1:end-1]
    t = t0 + Δx[:,2:end] - Δx[:,1:end-1]
    t0 = t0./norm3(t0)
    l = norm3(t)
    t = t./l

    # Update
    d1 = ptransport(d0, t0, t)
    rotate_orthogonal_unit!(d1, t, Δθ)
    d2 = cross3(t, d1)

    # Curvatures
    k = 2*cross3(t[:,1:end-1], t[:,2:end])./(1 + dot3(t[:,1:end-1], t[:,2:end]))
    κ = Matrix{Float64}(undef, 2, length(l)-1)
    κ[1,:] = dot3(k, (d2[:,1:end-1] + d2[:,2:end])/2)
    κ[2,:] = -dot3(k, (d1[:,1:end-1] + d1[:,2:end])/2)

    # Twists (could move in part of ptransport to reduce intermediates)
    τ = (τ0 + Δθ[2:end] - Δθ[1:end-1]
            + spherical_excess(t0[:,1:end-1], t0[:,2:end], t[:,1:end-1])
            + spherical_excess(t[:,1:end-1], t0[:,2:end], t[:,2:end]))

    rod_strains(l, κ, τ)
end

# ----------------------------------------------------------------------------------
#  Elastic Energy Computation
# ----------------------------------------------------------------------------------
# Rod properties: the relevant elastic properties of the rod. Can be scalar
#   k - stretching stiffness
#   B - bending stiffness principal components
#   β - twisting stiffness
# For convenience, the director field MUST match the eigenvectors of the
# bending stiffness tensor. Thus B = (B[1] d1 ⊗ d1) + (B[2] d2 ⊗ d2).

abstract type rod_properties end

# All properties vary along the length
struct vector_rod_properties{T<:Real} <: rod_properties
    k::Vector{T}
    B::Matrix{T} # 2 x nv_int
    β::Vector{T}
end

# Constant properties, anisotropic cross section
struct scalar_rod_properties{T<:Real} <: rod_properties
    k::T # 2 x nv_int
    B::Vector{T}
    β::T
end

# Constant properties, isotropic cross section
struct cylindrical_rod_properties{T<:Real} <: rod_properties
    k::T # 2 x nv_int
    B::T
    β::T
end

# The energy functional
function elastic_energy(ref_strains, strains, rod_properties)

    # Strains, Elastic Properties, Reference Kinematis
    l0, κ0, τ0 = ref_strains.l, ref_strains.κ, ref_strains.τ
    l, κ, τ = strains.l, strains.κ, strains.τ
    k, B, β = rod_properties.k, rod_properties.B, rod_properties.β

    # Reference voronoi lengths
    vl0 = [l0[1]; l0[2:end-1]/2]' + [l0[2:end-1]/2; l0[end]]'

    # Energies
    Es = 1/2*sum(k./l0.*(l - l0).^2)
    Et = 1/2*sum(β./vl0.*(τ - τ0).^2)
    Eb = 1/2*sum(B./vl0.*(κ - κ0).^2)
    Es + Et + Eb
end

# cache length at least 3*nv
function elastic_energy(ref_strains, strains, rod_properties, cache)
    # Strains, Elastic Properties, Reference Kinematis
    l0, κ0, τ0 = ref_strains.l, ref_strains.κ, ref_strains.τ
    l, κ, τ = strains.l, strains.κ, strains.τ
    k, B, β = rod_properties.k, rod_properties.B, rod_properties.β

    ne = length(l)
    nv = ne - 1

    # Reference voronoi lengths
    vl0 = reshape(view(cache, (2*nv+1):3*nv), 1, nv)
    vl0[1] = l0[1] + l0[2]/2
    vl0[2:nv-1] .= (view(l0, 2:ne-2) .+ view(l0, 3:ne-1))/2
    vl0[nv] = l0[ne-1]/2 + l0[ne]

    # vl0 .= [l0[1]; l0[2:end-1]/2]' + [l0[2:end-1]/2; l0[end]]'
    cache_s = reshape(view(cache, 1:ne), 1, ne)
    cache_t = reshape(view(cache, 1:nv), 1, nv)
    cache_b = reshape(view(cache, 1:2*nv), 2, nv)

    cache_s .= k./l0.*(l .- l0).^2
    cache_t .= β./vl0.*(τ .- τ0).^2
    cache_b .= B./vl0.*(κ .- κ0).^2

    1/2*sum(cache_s) + 1/2*sum(cache_t) + 1/2*sum(cache_b)
end

function gravitational_energy(x, m, g)
    nv = size(x,2)
    z = (x[3:3, 1:nv-1] .+ x[3:3,2:nv])./2
    sum(g.*m.*z)
end

function gravitational_energy(x, m, g, cache)
    nv = size(x,2)
    tmp = reshape(view(cache, 1:nv-1), 1, nv-1)

    tmp .= view(x,3:3,1:nv-1).*m.*g
    E1 = sum(tmp)
    tmp .= view(x,3:3,2:nv).*m.*g
    E2 = sum(tmp)

    return (E1 + E2)/2
end
