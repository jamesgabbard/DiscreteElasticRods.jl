## rod_types.jl
# James Gabbard, November 30th, 2020
#
# Interface for specifying a physical system composed of elastic rods

# ------------------------------------------------------------------------------
#  Types
# ------------------------------------------------------------------------------

struct basic_rod{T<:Real}
    x::Matrix{T} # 3 x ns + 2
    d::Matrix{T} # 3 x ns + 1
end

struct rod_delta{T<:Real}
    Δx::Matrix{T} # 3 x ns + 2
    Δθ::Matrix{T} # 1 x ns + 1
end

# Strains
struct rod_strains{T<:Real}
    l::Matrix{T} # 1 x ns + 1
    κ::Matrix{T} # 2 x ns
    τ::Matrix{T} # 1 x ns
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
    x = Matrix{T}(undef, 3, ns+2) # 3 x ns + 2
    d = Matrix{T}(undef, 3, ns+1)
    basic_rod(x,d)
end

function allocate_strain(T::Type, ns::Int)
    l = Matrix{T}(undef, 1, ns+1)
    κ = Matrix{T}(undef, 2, ns)
    τ = Matrix{T}(undef, 1, ns)
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
