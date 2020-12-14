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
g = 9.8

# Create a straight rod with NS seg
p1 = [0., 0., 0.]
p2 = [L, 0., 0.]
d0 = [0., 0., 1.]
ref_rod = DER.straight_rod(p1, p2, d0, ns)

# Reference kinematics and properties
props = DER.elliptical_stiffness(E, ν, R, R)
ref_strain = DER.full_kinematics(ref_rod)
m = ref_strain.l*(ρ*π*R^2)

# q ordering: [θ2, x3, θ3, x4, ..., x, θ]
ncore = 4*ns - 7
q = 1:ncore + 7
Δx = zeros(3,ns+2)
Δθ = zeros(1,ns+1)
function param_2_rod!(Δx, Δθ, q)

    # Rearrange core DOFs
    Δx[1, 3:end-2] = q[2:4:ncore]
    Δx[2, 3:end-2] = q[3:4:ncore]
    Δx[3, 3:end-2] = q[4:4:ncore]
    Δθ[2:end-1] = q[1:4:ncore]

    # Cantilever BC: do nothing
    # Free BC: fill in the end of the coords
    Δx[:,end-1] = q[ncore+1:ncore+3]
    Δθ[end] = q[ncore+4]
    Δx[:,end] = q[ncore+5:ncore+7]

end

function param_2_rod(q)
    Δx = zeros(3,ns+2)
    Δθ = zeros(1,ns+1)
    param_2_rod!(Δx, Δθ, q)
    Δx, Δθ
end

function rod_2_energy(Δx, Δθ)
    r = DER.rod_update(ref_rod, Δx, Δθ)
    strain = DER.full_kinematics(ref_rod)
    E = (DER.elastic_energy(ref_strain, strain, props) +
         DER.gravitational_energy(r.x, m , g))
    E
end

function total_energy(q)
    Δx, Δθ = param_2_rod(q)
    rod_2_energy(Δx, Δθ)
end
