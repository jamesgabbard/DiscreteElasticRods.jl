using DiscreteElasticRods
DER = DiscreteElasticRods
using BenchmarkTools

N = 500

# Linear ALgebra Sections
a = rand(3,N)
b = rand(3,N)
c = rand(3,N)

@btime DER.cross3!($c,$a,$b)

c = rand(1,N)
@btime DER.dot3!($c,$a,$b)

@btime DER.norm3!($c, $a)

@btime DER.norm3!($a)

e = rand(3,N-1)
@btime DER.edges!($e,$b)
@btime DER.tangents!($e,$b)

# Parallel Transport and Rotation
v1 = rand(3,N)
v2 = similar(v1)
t1 = rand(3,N)
t2 = rand(3,N)
DER.norm3!(t1)
DER.norm3!(t2)

cache = zeros(6,N)
@btime DER.ptransport!($v2,$v1,$t1,$t2,$cache)

θ = rand(1,N)
@btime DER.rotate_orthogonal_unit!($v2,$t2,$θ,$cache)

# Update and Kinematics
x = rand(3,N)
d = rand(3,N)
DER.norm3!(d)
Δx = rand(3,N)
Δθ = rand(1,N)

cache = zeros(15,N)
@btime DER.rod_update!(x,d,Δx,Δθ,cache)

l = zeros(1, N-1)
κ = zeros(2, N-2)
τ = zeros(1, N-2)
caches = (zeros(6, N-1), zeros(9, N-2))
@btime DER.full_kinematics!(l, κ, τ, x, d, caches)

l0 = zeros(1, N-1)
κ0 = zeros(2, N-2)
τ0 = zeros(1, N-2)
k = rand()
B = rand(2,1)
β = rand()

@btime DER.elastic_energy(l, κ, τ, l0, κ0, τ0, k ,B, β, caches[2])

m = rand(1, N-1)
g = 10.0
@btime DER.gravitational_energy(x, m, g, caches[1])
