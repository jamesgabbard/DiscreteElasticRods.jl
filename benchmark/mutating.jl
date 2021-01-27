# Used to check up on the performance of non-allocating rod functions,
# and quickly check that they only allocate views, if anything.

using DiscreteElasticRods
using BenchmarkTools
using Random

N = 500
suite = BenchmarkGroup()
Random.seed!(42)

# Linear ALgebra Sections
a = rand(N,3)
b = rand(N,3)
c = rand(N,3)

suite["cross3!"] = @benchmarkable DER.cross3!($c,$a,$b)

d = rand(N,1)
suite["dot3!"] = @benchmarkable DER.dot3!($d,$a,$b) samples = 50000
suite["norm3!(a,b)"] = @benchmarkable DER.norm3!($d, $a)
suite["norm3(a)!"] = @benchmarkable DER.norm3!($a)

e = rand(N-1,3)
suite["edges!"] = @benchmarkable DER.edges!($e,$b) samples = 50000
suite["tangents!"] = @benchmarkable DER.tangents!($e,$b) samples = 50000

# Parallel Transport and Rotation
v1 = rand(N,3)
v2 = similar(v1)
t1 = rand(N,3)
t2 = rand(N,3)
DER.norm3!(t1)
DER.norm3!(t2)
DER.norm3!(v1)

cache = zeros(N,6)
suite["ptransport!"] = @benchmarkable DER.ptransport!($v2,$v1,$t1,$t2,$cache)

θ = rand(N,1)
suite["rotate_orthogonal_unit!"] = @benchmarkable DER.rotate_orthogonal_unit!($v2,$t2,$θ,$cache) setup=(v2 .= $v1)

# Update and Kinematics
x0 = rand(N,3)
d0 = rand(N-1,3)
DER.norm3!(d0)
x1 = zeros(N,3)
d1 = zeros(N-1,3)
Δx = rand(N,3)
Δθ = rand(N-1,1)
cache2 = DER.allocate_cache(DER.rod_update!, Float64, N-2)
suite["rod_update!"] =
    @benchmarkable DER.rod_update!($x1,$d1,$x0,$d0,$Δx,$Δθ,$cache2)

l = zeros(N-1)
κ = zeros(N-2,2)
τ = zeros(N-2)
cache3 = DER.allocate_cache(DER.full_kinematics!, Float64, N-2)
suite["full_kinematics!"] =
    @benchmarkable DER.full_kinematics!($l,$κ,$τ,$x0,$d0,$cache3)

l0 = zeros(N-1)
κ0 = zeros(N-2,2)
τ0 = zeros(N-2)
k = rand()
B = rand(1,2)
β = rand()
suite["elastic_energy"] =
    @benchmarkable DER.elastic_energy($l,$κ,$τ,$l0,$κ0,$τ0,$k,$B,$β,$cache3[2])

m = rand(N-1)
g = 10.0
suite["gravitational_energy"] =
    @benchmarkable DER.gravitational_energy($x0,$m,$g,$cache3[1])

# Set: tune, run, and save a benchmark
# tune!(suite)
# BenchmarkTools.save("benchmark/params.json", params(suite));
# BenchmarkTools.save("benchmark/results.json", minimum(run(suite)))

# Check: load params/results, then run and compare
prior_params = BenchmarkTools.load("benchmark/params.json")[1]
loadparams!(suite, prior_params, :evals, :samples)
prior = BenchmarkTools.load("benchmark/results.json")[1]
posterior = minimum(run(suite))
print(judge(posterior, prior))
