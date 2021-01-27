# Used to check up on the performance of non-allocating rod functions,
# and quickly check that they only allocate views, if anything.

using DiscreteElasticRods
using BenchmarkTools

N = 500

# Linear ALgebra Sections
a = rand(N,3)
b = rand(N,3)
c = rand(N,3)

@btime DER.cross3!($c,$a,$b)

c = rand(N,1)
@btime DER.dot3!($c,$a,$b)
@btime DER.norm3!($c, $a)
@btime DER.norm3!($a)

e = rand(N-1,3)
@btime DER.edges!($e,$b)
@btime DER.tangents!($e,$b)

# Parallel Transport and Rotation
v1 = rand(N,3)
v2 = similar(v1)
t1 = rand(N,3)
t2 = rand(N,3)
DER.norm3!(t1)
DER.norm3!(t2)

cache = zeros(N,6)
@btime DER.ptransport!($v2,$v1,$t1,$t2,$cache)

θ = rand(N,1)
@btime DER.rotate_orthogonal_unit!($v2,$t2,$θ,$cache)

# Update and Kinematics
x = rand(N,3)
d = rand(N-1,3)
DER.norm3!(d)
Δx = rand(N,3)
Δθ = rand(N-1,1)

cache = DER.allocate_cache(DER.rod_update!, Float64, N)
@btime DER.rod_update!(x,d,Δx,Δθ,cache)

l = zeros(N-1)
κ = zeros(N-2,2)
τ = zeros(N-2)
caches = DER.allocate_cache(full_kinematics!, Float64, N)
@btime DER.full_kinematics!(l, κ, τ, x, d, caches)

l0 = zeros(N-1)
κ0 = zeros(N-2,2)
τ0 = zeros(N-2)
k = rand()
B = rand(1,2)
β = rand()

@btime DER.elastic_energy(l, κ, τ, l0, κ0, τ0, k ,B, β, caches[2])

m = rand(N-1)
g = 10.0
@btime DER.gravitational_energy(x, m, g, caches[1])

# Gradients
