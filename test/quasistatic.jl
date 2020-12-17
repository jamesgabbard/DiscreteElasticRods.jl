# A simple quasi-static problem, based on an L-BFGS minimization
using DiscreteElasticRods
DER = DiscreteElasticRods
using BenchmarkTools
using ForwardDiff
using Optim
using LineSearches

# Parameters
ns = 40
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
nq = ncore + 7
q = 1:nq

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

# function param_2_rod(q)
#     T = eltype(q)
#     Δx = zeros(T, 3,ns+2)
#     Δθ = zeros(T, 1,ns+1)
#     param_2_rod!(Δx, Δθ, q)
#     Δx, Δθ
# end

function total_energy(q::Vector{T}) where T

    Δx, Δθ, rod, strain, cache = get_cache(T)

    param_2_rod!(Δx, Δθ, q)
    DER.copy!(rod, ref_rod)
    DER.rod_update!(rod, Δx, Δθ, cache)
    DER.full_kinematics!(strain, rod, cache)
    Ee = DER.elastic_energy(ref_strain, strain, props, cache)
    Eg = DER.gravitational_energy(rod.x, m, g, cache)
    Ee + Eg
end

# Extensive cache system
T = Float64

cache = zeros(T, 27, ns)
Δx = zeros(T, 3,ns+2)
Δθ = zeros(T, 1,ns+1)
rod = DER.allocate_rod(T, ns)
strain = DER.allocate_strain(T,ns)

p_cache = (Δx, Δθ, rod, strain, cache)

chunk = ForwardDiff.pickchunksize(nq)
#Td = ForwardDiff.Dual{nothing,T,chunk}
Td = ForwardDiff.Dual{ForwardDiff.Tag{typeof(total_energy),T},T,chunk}

cache_dual = zeros(Td, 27, ns)
Δx_dual = zeros(Td, 3,ns+2)
Δθ_dual = zeros(Td, 1,ns+1)
rod_dual = DER.allocate_rod(Td, ns)
strain_dual = DER.allocate_strain(Td,ns)

d_cache = (Δx_dual, Δθ_dual, rod_dual, strain_dual, cache_dual)

get_cache(::Type{T}) where T<:ForwardDiff.Dual = d_cache
get_cache(::Type{T}) = p_cache

# Let's take derivatives!
using FiniteDiff
# using ReverseDiff
#
grad1(q) = FiniteDiff.finite_difference_gradient(total_energy, q)
grad2(q) = ForwardDiff.gradient(total_energy, q)
#
# tape1 = ReverseDiff.GradientTape(total_energy, q)
# compiled_tape1 = ReverseDiff.compile(tape1)
# grad3!(res,q) = ReverseDiff.gradient!(res, compiled_tape1, q)

# Timings for each method
#q = rand(nq)
#res = similar(q)
#@btime total_energy(q)
#@btime grad1(q)
#@btime grad2(q)
#@btime grad3!(res,q)


# Optimization
f(x) = total_energy(x)

function g!(G,x)
    ForwardDiff.gradient!(G,total_energy,x)
end

method = Optim.BFGS(; alphaguess = LineSearches.InitialStatic(),
                linesearch = LineSearches.HagerZhang(),
                initial_invH = nothing,
                initial_stepnorm = nothing,
                manifold = Flat())

#q0 = zeros(nq)
q0 = 0.0001*rand(nq)
ops = Optim.Options(; g_tol = 1e-2,
                      time_limit = 60,
                      show_trace = true)
results = Optim.optimize(f, g!, q0, method, ops)
qf = Optim.minimizer(results)

# Visuals?
function reconstruct(q)
    Δx, Δθ, rod, strain, cache = p_cache

    param_2_rod!(Δx, Δθ, q)
    DER.copy!(rod, ref_rod)
    DER.rod_update!(rod, Δx, Δθ, cache)

    DER.plot(rod)
    ax = PyPlot.gca()
    ax.set_ylim(-L/2, L/2)
    ax.set_zlim(-L/2, L/2)

    rod
end

reconstruct(qf)


# Get optim to work
ftst(x) = 1/2*sum(abs2, x)

function gtst!(G,x)
    ForwardDiff.gradient!(G,ftst,x)
end

# function gtst!(r,x)
#     r .= -x
# end
x0 = rand(5)

ftst(x0)
tmp = zeros(5)
gtst!(tmp, x0)

results = Optim.optimize(ftst, gtst!, x0, LBFGS(), ops)

rosenbrock(x) =  (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
result = optimize(rosenbrock, zeros(2), BFGS(), ops)
