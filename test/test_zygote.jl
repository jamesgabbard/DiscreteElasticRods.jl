using DiscreteElasticRods
DER = DiscreteElasticRods
using Zygote

# Basic linalg
n = 40
a = randn(3,n)
b = randn(3,n)
c = randn(3,n)

linalg3_test(a,b,c) = sum(DER.dot3(c, DER.cross3(a,b))) + sum(DER.norm3(b))

linalg3_test(a,b,c)
out, back = Zygote.pullback(linalg3_test, a, b, c)
back(rand())

# Parallel transport
t1 = rand(3,n); DER.norm3!(t1)
t2 = rand(3,n); DER.norm3!(t2)
v = rand(3,n);
out, back = Zygote.pullback(DER.ptransport, v, t1, t2)
back(rand(3,n))

# Rotations
θ = rand(1,n)
out, back = Zygote.pullback(DER.rotate_orthogonal_unit, t1, t2, θ)
back(rand(3,n))

# Kinematics
r = DER.random_rod(n)

DER.full_kinematics(r.x, r.d)
out, back = Zygote.pullback(DER.full_kinematics, r.x, r.d)
out, back = Zygote.pullback(DER.full_kinematics, r)
back(out)

# Test case: the entire energy function
ref_rod = DER.copy(r)
ref_strains = DER.full_kinematics(r)
rod_props = DER.elliptical_stiffness(10, 0.3, 0.01, 0.04)

function total_energy(rod_coords::DER.rod_delta)
    new_rod = DER.rod_update(ref_rod, rod_coords)
    strains = DER.full_kinematics(new_rod)
    DER.elastic_energy(ref_strains, strains, rod_props)
end

rod_coord = DER.rod_delta(rand(3,n), rand(1,n-1))
out, back = Zygote.pullback(total_energy, rod_coord)

grad_total_energy(rod_coords::DER.rod_delta) = Zygote.gradient(total_energy, rod_coords)

using BenchmarkTools
@btime grad_total_energy($rod_coord)
