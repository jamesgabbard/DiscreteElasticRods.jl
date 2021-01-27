using DiffEqBase: dualcache, get_tmp
using ForwardDiff
using BenchmarkTools
num_partials = 5
N = 500

a = rand(N)
d = dualcache(rand(N), Val{num_partials})

# Directly accessing the dual cache, no reinterpret
dual_du = d.dual_du
@btime $dual_du .+= $a

# Accessing the dual cache with a reinterpret
struct MyTag end
dual_du = get_tmp(d, ForwardDiff.Dual{MyTag, Float64, num_partials}(0.0))
@btime $dual_du .+= $a

# What if its just a reinterpret of generic memory?
b = zeros(6, 500)
c = reinterpret(reshape, ForwardDiff.Dual{MyTag, Float64, 5}, b)

@btime $c .+= $a
