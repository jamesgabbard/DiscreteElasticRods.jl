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
    rotate!(d, t, Δθ)

    basic_rod(x, d)
end

rod_update(r::basic_rod, Δx, Δθ) = rod_update(r.x, r.d, Δx, Δθ)

function rod_update!(r::basic_rod, Δx, Δθ)

    t0 = tangents(r.x)
    r.x += Δx
    t = tangents(r.x)
    r.d = ptransport(r.d, t0, t)
    rotate!(r.d, t, Δθ)
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

    κ = Matrix{Float64}(undef, 2, length(l)-1)
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
    rotate!(d1, t, Δθ)
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

function gravitational_energy(x, m, g)
    nv = size(x,2)
    z = (view(x,3,1:nv-1) .+ view(x,3,2:nv))/2
    sum(g*m.*z)
end
