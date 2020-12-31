## rod_types.jl
# James Gabbard, November 30th, 2020
#
# Interface for specifying a physical system composed of elastic rods

# ------------------------------------------------------------------------------
#  Types
# ------------------------------------------------------------------------------

struct basic_rod{T<:Real}
    x::Matrix{T} # (ns + 2) x 3
    d::Matrix{T} # (ns + 1) x 3
end

struct rod_delta{T<:Real}
    Δx::Matrix{T} # (ns + 2) x 3
    Δθ::Matrix{T} # (ns + 1) x 1
end

# Strains
struct rod_strains{T<:Real}
    l::Matrix{T} # ns + 1
    κ::Matrix{T} # ns x 2
    τ::Matrix{T} # ns
end

# ------------------------------------------------------------------------------
#  Allocation
# ------------------------------------------------------------------------------

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

function allocate_rod(T::Type, ns::Int)
    x = Matrix{T}(undef, ns+2, 3)
    d = Matrix{T}(undef, ns+1, 3)
    basic_rod(x,d)
end

function allocate_strain(T::Type, ns::Int)
    l = Matrix{T}(undef, ns+1)
    κ = Matrix{T}(undef, ns, 2)
    τ = Matrix{T}(undef, ns)
    rod_strains(l, κ, τ)
end

function allocate_delta(T::Type, ns::Int)
    Δx = Matrix{T}(undef, ns+2, 3)
    Δθ = Matrix{T}(undef, ns+1, 1)
    return rod_delta(Δx, Δθ)
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
    B::Matrix{T} # ns x 2
    β::Vector{T}
end

# Constant properties, anisotropic cross section
struct scalar_rod_properties{T<:Real} <: rod_properties
    k::T
    B::Matrix{T}# 1 x 2
    β::T
end

# Constant properties, isotropic cross section
struct cylindrical_rod_properties{T<:Real} <: rod_properties
    k::T # 2 x nv_int
    B::T
    β::T
end
