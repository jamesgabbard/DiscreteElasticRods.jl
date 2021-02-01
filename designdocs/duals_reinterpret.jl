# ------------------------------------------------------------------------------
# This would be a convenient system to manage cacheing of dual numbers.
# Unfortunately, at least in Julia 1.5, reinterpreting an array of
# Dual numbers leads to a large performance penalty. Until this is
# resolved, a non-reinterpret-based solution is necessary

# Note that ForwardDiff2 is currently under construction, so that
# could be interesting
# ------------------------------------------------------------------------------

using DiscreteElasticRods
using ForwardDiff
abstract type DualType{V,N} end

struct dual_rod{V,N} <: DualType{V,N}
    num::DER.basic_rod{V}
    dual::DER.basic_rod{ForwardDiff.Dual{nothing,V,N}}
end

struct dual_delta{V,N} <: DualType{V,N}
    num::DER.rod_delta{V}
    dual::DER.rod_delta{ForwardDiff.Dual{nothing,V,N}}
end

struct dual_vector{V,N} <: DualType{V,N}
    num::Vector{V}
    dual::Vector{ForwardDiff.Dual{nothing,V,N}}
end

struct dual_cache{V,N} <: DualType{V,N}
    num::Tuple{Vararg{Array{V}}}
    dual::Tuple{Vararg{Array{ForwardDiff.Dual{nothing,V,N}}}}
end

# Constructors
dual_rod(V::Type, Ns::Int, Nd::Int) = dual_rod(DER.allocate_rod(V, Ns), DER.allocate_rod(ForwardDiff.Dual{nothing,V,Nd}, Ns))
dual_delta(V::Type, Ns::Int, Nd::Int) = dual_delta(DER.allocate_delta(V, Ns), DER.allocate_delta(ForwardDiff.Dual{nothing,V,Nd}, Ns))
dual_vector(V::Type, N::Int, Nd::Int) = dual_vector(Vector{V}(undef, N), Vector{ForwardDiff.Dual{nothing,V,Nd}}(undef, N))

# Get the num or dual component
function resolve_type(dual::DualType{V,N}, ::Type{V}) where {V, N}
    return dual.num
end

function resolve_type(dual::DualType{V,N}, ::Type{ForwardDiff.Dual{T,V,N}}) where {T,V,N}
    return reinterpret(ForwardDiff.Dual{T,V,N}, dual.dual)
end

function resolve_type(dual::dual_cache{V,N}, ::Type{ForwardDiff.Dual{T,V,N}}) where {T,V,N}
    return reinterpret_dual_tuple(ForwardDiff.Dual{T,V,N}, dual.dual)
end

# Extend reinterpret (renamed for tuples, to avoid type piracy)
function Base.reinterpret(T::Type, r::DER.basic_rod)
    DER.basic_rod(reinterpret(T, r.x), reinterpret(T, r.d))
end

function Base.reinterpret(T::Type, Δr::DER.rod_delta)
    DER.basic_rod(reinterpret(T, Δr.Δx), reinterpret(T, Δr.Δθ))
end

function reinterpret_dual_tuple(T::Type{ForwardDiff.Dual{T1, V, N}}, a::Tuple{Vararg{Array{ForwardDiff.Dual{T2,V,N}}}}) where {T1, T2, V, N}
    map(x->reinterpret(T, x), a)
end

# Finally, allow any cache structure in my code to be a dual
function allocate_dual_cache(f::Function, V::Type, Ns::Int, Nd::Int)
    dual_cache{V,Nd}(DER.allocate_cache(f, V, Ns), DER.allocate_cache(f, ForwardDiff.Dual{nothing, V, Nd}, Ns))
end

# Minimal functionality
Ns = 50
Nd = 3
NType = Float64
struct MyTag end
DType = ForwardDiff.Dual{MyTag, NType, Nd}

r = dual_rod(NType, Ns, Nd)
resolve_type(r, NType)
resolve_type(r, DType)

r = dual_delta(NType, Ns, Nd)
resolve_type(r, NType)
resolve_type(r, DType)

v = dual_vector(NType, Ns, Nd)
resolve_type(v, NType)
resolve_type(v, DType)

c = allocate_dual_cache(DER.elastic_forces!, Float64, 50, 3)
resolve_type(c, NType)
resolve_type(c, DType)

# How hard does reinterpret hurt timings?
struct MyTag2 end
DType2 = ForwardDiff.Dual{MyTag2, NType, Nd}

Ns = 500
a0 = rand(Ns, 3)
b0 = rand(Ns, 3)
c0 = similar(a0)
@btime DER.cross3!($c0, $a0, $b0)

a1 = DType.(rand(Ns,3))
b1 = DType.(rand(Ns,3))
c1 = similar(a)
@btime DER.cross3!($c1, $a1, $b1)

a2 = reinterpret(DType2, a1)
b2 = reinterpret(DType2, b1)
c2 = reinterpret(DType2, c1)
@btime DER.cross3!($c2, $a2, $b2)

a0 = rand(Ns, 3)
b0 = rand(Ns, 3)
c0 = rand(Ns)
@btime DER.dot3!($c0, $a0, $b0)

a1 = DType.(rand(Ns,3))
b1 = DType.(rand(Ns,3))
c1 = DType.(rand(Ns))
@btime DER.dot3!($c1, $a1, $b1)

a2 = reinterpret(DType2, a1)
b2 = reinterpret(DType2, b1)
c2 = reinterpret(DType2, c1)
@btime DER.dot3!($c2, $a2, $b2)
