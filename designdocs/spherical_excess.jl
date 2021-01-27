# Benchmark the computation of the area of a spherical triangle
#
# Lestringant uses this to calculate twists, but the trig calls are
# cripplingly slow. Here we investigate an allocating version, a
# non-allocating version, and a version accelerated with IntelVectorMath
#
# Conclusion: the IVM non-allocating version wins hands down, and is
# almost 3x faster than its competitor. However, it doesn't permit forward-
# mode automatic differentiation because IVM functions only evaluate on
# Float64 and Float32 types. Overall twist calculation via 2 calls
# to spherical excess with IVM is no faster than twist calculations
# based on parallel transport, and the latter is type-generic.
#
# A possible way forward is to overload IVM.atan!(a, b, c) to operate
# on ForwardDiff.Dual types. That would allow at least fast HVPs,
# and is worth investigating. Note that
#
# atan(y + δy, x + δx) = atan(y, x) + (δy*x - δx*y)/(x^2 + y^2)

using DiscreteElasticRods
DER = DiscreteElasticRods
using BenchmarkTools

# Full formula form Lestringant 2020
function spherical_excess_reference(t1, t2, t3)

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

# Removing common denominators in atan calls
function spherical_excess(t1, t2, t3)

    ca = DER.dot3(t2,t3)
    cb = DER.dot3(t3,t1)
    cc = DER.dot3(t1,t2)
    V = DER.dot3(DER.cross3(t1,t2),t3)

    atan.(V, ca .- cb.*cc) .+
    atan.(V, cb .- cc.*ca) .+
    atan.(V, cc .- ca.*cb) .- π
end

# non-allocating vector triple product
function triple3!(out, a, b, c)
    @inbounds for i in 1:size(out,1)
        out[i] = c[i,1]*(a[i,2]*b[i,3] - a[i,3]*b[i,2]) +
                 c[i,2]*(a[i,3]*b[i,1] - a[i,1]*b[i,3]) +
                 c[i,3]*(a[i,1]*b[i,2] - a[i,2]*b[i,1])
   end
end

# Non-allocating spherical excess
# (ns + 1) x 4 cache
# 90% of this function's runtime is atan calls
function spherical_excess!(γ, t1, t2, t3, cache)

    ca = @view cache[:,1]
    cb = @view cache[:,2]
    cc = @view cache[:,3]
    V = @view cache[:,4]

    triple3!(V, t1, t2, t3)
    DER.dot3!(ca, t2, t3)
    DER.dot3!(cb, t3, t1)
    DER.dot3!(cc, t1, t2)

    γ .= @. atan(V, ca-cb*cc) + atan(V, cb-cc*ca) + atan(V, cc-ca*cb) - π
end

# IVM acceleration
# Note that IVM.atan!(a,b,c) can only operator on Array{Float64,N}, so
# the cache cannot use views or subarrays of any kind
using IntelVectorMath

struct spherical_excess_IVM_cache{T<:Real}
    ca::Vector{T}
    cb::Vector{T}
    cc::Vector{T}
    V::Vector{T}
    tmp1::Vector{T}
    tmp2::Vector{T}
end

function allocate_seIVM(T::Type, N::Int)
    ca = Vector{T}(undef, N)
    cb = Vector{T}(undef, N)
    cc = Vector{T}(undef, N)
    tmp1 = Vector{T}(undef, N)
    tmp2 = Vector{T}(undef, N)
    V = Vector{T}(undef, N)
    spherical_excess_IVM_cache(ca, cb, cc, V, tmp1, tmp2)
end

function spherical_excess_IVM!(γ, t1, t2, t3, cache::spherical_excess_IVM_cache)

    ca, cb, cc = cache.ca, cache.cb, cache.cc
    V, tmp1, tmp2 =  cache.V, cache.tmp1, cache.tmp2

    triple3!(V, t1, t2, t3)
    DER.dot3!(ca, t2, t3)
    DER.dot3!(cb, t3, t1)
    DER.dot3!(cc, t1, t2)

    γ .= -π
    tmp1 .= ca .- cb.*cc
    IVM.atan!(tmp2, V, tmp1)
    γ .+= tmp2

    tmp1 .= cb .- cc.*ca
    IVM.atan!(tmp2, V, tmp1)
    γ .+= tmp2

    tmp1 .= cc .- ca.*cb
    IVM.atan!(tmp2, V, tmp1)
    γ .+= tmp2
end

# Benchmarking
N = 500

# Pure atan comparison
e = rand(N)
d = rand(N)
f = similar(e)
@btime IVM.atan!($f, $e, $d)
@btime $f .= atan.($e, $d)

# Non-allocating triple product
a = rand(N,3)
b = rand(N,3)
c = rand(N,3)
out = zeros(N)
@btime triple3!($out, $a, $b, $c)

# Allocating vs non-allocating vs IVM
cache_sphere = zeros(N,6)
cache_sphere_IVM = allocate_seIVM(Float64, N)
@btime spherical_excess($a, $b, $c)
@btime spherical_excess!($out, $a, $b, $c, $cache_sphere)
@btime spherical_excess_IVM!($out, $a, $b, $c, $cache_sphere_IVM)
