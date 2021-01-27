using IntelVectorMath
using ForwardDiff
using BenchmarkTools

function IVM.atan(xd::Vector{ForwardDiff.Dual{T, V, 1}},
                  yd::Vector{ForwardDiff.Dual{T, V, 1}}) where {T, V}

    x = ForwardDiff.value.(xd)
    y = ForwardDiff.value.(yd)
    z = IVM.atan(x, y)

    δx = ForwardDiff.partials.(xd, 1)
    δy = ForwardDiff.partials.(yd, 1)
    δz = @. (δy*x - δx*y)/(x^2 + y^2)

    ForwardDiff.Dual{T}.(z, δz)
end

struct fast_atan_dual_cache{T}
    x::Vector{T}
    y::Vector{T}
    z::Vector{T}
    δ::Matrix{T}
end

function allocate_fast_atan_dual_cache(T::Type, N::Int)
    x = Vector{T}(undef, N)
    y = Vector{T}(undef, N)
    z = Vector{T}(undef, N)
    δ = Matrix{T}(undef, N, 3)
    fast_atan_dual_cache(x, y, z, δ)
end

function IVM.atan!(zd::Vector{ForwardDiff.Dual{T, V, 1}},
                   xd::Vector{ForwardDiff.Dual{T, V, 1}},
                   yd::Vector{ForwardDiff.Dual{T, V, 1}},
                   cache::fast_atan_dual_cache{V}) where {T, V}

   x, y, z = cache.x, cache.y, cache.z
   δx, δy, δz = view(cache.δ,:,1), view(cache.δ,:,2), view(cache.δ,:,3)

    x .= ForwardDiff.value.(xd)
    y .= ForwardDiff.value.(yd)
    IVM.atan!(z, x, y)

    δx .= ForwardDiff.partials.(xd, 1)
    δy .= ForwardDiff.partials.(yd, 1)
    δz .= @. (δy*x - δx*y)/(x^2 + y^2)

    zd .= ForwardDiff.Dual{T}.(z, δz)
end

N = 500
x = rand(N)
y = rand(N)
δx = rand(N)
δy = rand(N)
xd = ForwardDiff.Dual{DERTag}.(x, δx)
yd = ForwardDiff.Dual{DERTag}.(y, δy)
zd = IVM.atan(xd,yd)

cache_atan = allocate_fast_atan_dual_cache(Float64, N)
@btime IVM.atan!(zd, xd, yd, cache_atan)
@btime $zd .= atan.($xd, $yd)

z = rand(N)
@btime IVM.atan!(z, x, y)
@btime z .= atan.(x, y)

struct DERTag end
v = rand(3)
p1 = rand(3)
p2 = rand(3)
p = [p1, p2]
d = ForwardDiff.Dual{DERTag}.(v, p...)

ForwardDiff.partials.(d, 1)
ForwardDiff.partials.(d, 2)
ForwardDiff.value.(d)
