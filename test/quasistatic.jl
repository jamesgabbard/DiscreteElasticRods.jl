# A simple quasi-static problem, based on a Newton (or Newton Krylov) method

using DiscreteElasticRods
DER = DiscreteElasticRods

# Parameters
ns = 20
E = 5e6
ν = 0.48
L = 0.30
R = 0.003
ρ = 1e3

# Create a straight rod with NS seg
p1 = [0., 0., 0.]
p2 = [L, 0., 0.]
d0 = [0., 0., 1.]
red_rod = DER.straight_rod(p1, p2, d0, ns)

# Reference kinematics and properties
props = DER.elliptical_stiffness(E, ν, R, R)
ref_strain = DER.full_kinematics(ref_rod)
m = ref_strain.l*(ρ*π*R^2)

# Boundary condition layer: params go to a rod
# struct fixed_cantilever
#     t0::Vector{Float64}
# end
#
# struct free_end
# end
function param_2_rod(q)
    Δx = Matrix{Float64}(3,ns+2)
    Δθ = Matrix{Float64}(1,ns+1)
end
