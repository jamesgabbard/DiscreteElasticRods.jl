using DiscreteElasticRods
using BenchmarkTools
using LinearAlgebra
using ForwardDiff
using SparseDiffTools
using IterativeSolvers

ns = 50

# Typing for dual caches
DualTag = SparseDiffTools.DeivVecTag
DualType = ForwardDiff.Dual{DualTag, Float64, 1}

# For now, all dual_caches are tuples
function allocate_dual_cache(F::Function, DType::Type{ForwardDiff.Dual{T,V,N}}, Ns) where{T,V,N}
    (DER.allocate_cache(F, V, Ns), DER.allocate_cache(F, DType, Ns))
end

function allocate_dual_vector(DType::Type{ForwardDiff.Dual{T,V,N}}, Ne) where{T,V,N}
    (zeros(V, Ne), zeros(DType, Ne))
end

function allocate_dual_delta(DType::Type{ForwardDiff.Dual{T,V,N}}, Ns) where{T,V,N}
    (DER.allocate_delta(V, Ns), DER.allocate_delta(DType, Ns))
end

resolve_type(x, ::Type) = x[1]
function resolve_type(x, ::Type{ForwardDiff.Dual{T,V,N}}) where{T,V,N}
    x[2]
end

# Define a Rod
p1 = [0, 0, 0]
p2 = [1, 0, 0]
R = 1.0
d = [0, 0, 1.0]

ref_rod = DER.arc_rod(p1, p2, d, R, ns)
ref_strains = DER.full_kinematics(ref_rod)
ref_strains.κ .= 0.0
ref_strains.τ .= 0.0
rod_props = DER.elliptical_stiffness(1e6, 0.48, 0.01, 0.01)
DER.plot(ref_rod)

# Allocate coordinate layer
s = zeros(Float64, 4*ns)
∇s = zeros(Float64, 4*ns)

# Allocate expanded coordinate layer
q = allocate_dual_vector(DualType, 4*ns+7)
∇q = allocate_dual_vector(DualType, 4*ns+7)

# Convert between the two
function s_to_q!(q, s)
    q[8:end] .= s
end

function ∇q_to_∇s!(∇s, ∇q)
    ∇s .= @view ∇q[8:end]
end

# Non-allocating gradient for Float64
gradients = allocate_dual_delta(DualType, ns)
coordinates = allocate_dual_delta(DualType, ns)
caches = allocate_dual_cache(DER.fused_update_gradient!, DualType, ns)

function dEdq!(∇q, q)
    T = eltype(q)
    coord = resolve_type(coordinates,T)
    grads = resolve_type(gradients,T)
    cache = resolve_type(caches,T)

    DER.banded_storage!(coord, q)
    DER.fused_update_gradient!(grads, ref_rod, coord, rod_props, ref_strains, cache)
    DER.banded_storage!(∇q, grads)
end

function dEds!(∇s, s)
    q_res = resolve_type(q, eltype(s))
    ∇q_res = resolve_type(∇q, eltype(s))

    s_to_q!(q_res, s)
    dEdq!(∇q_res, q_res)
    ∇q_to_∇s!(∇s, ∇q_res)
end

# Test that everything works
s = rand(4*ns)
∇s = zeros(4*ns)
dEds!(∇s, s)

t = DualType.(rand(4*ns))
∇t = DualType.(zeros(4*ns))
dEds!(∇t, t)

@btime dEds!(∇s, s)
@btime dEds!(∇t, t)

# Test for correctness?
using FiniteDiff
newrods = (DER.allocate_rod(Float64, ns), DER.allocate_rod(DualType, ns))
newstrains = (DER.allocate_strain(Float64, ns), DER.allocate_strain(DualType, ns))
function E(s)
    T = eltype(s)
    q_res = resolve_type(q, T)
    coord = resolve_type(coordinates, T)
    cache = resolve_type(caches, T)
    newrod = resolve_type(newrods, T)
    newstrain = resolve_type(newstrains, T)
    s_to_q!(q_res, s)

    DER.banded_storage!(coord, q_res)
    DER.rod_update!(newrod, ref_rod, coord, cache[2])
    DER.full_kinematics!(newstrain, newrod, (cache[2], cache[3]))
    DER.elastic_energy(ref_strains, newstrain, rod_props)
end

∇s0 = similar(s)
dEds!(∇s0, s)

∇s1 = similar(s)
v = zeros(length(s))
for i = 1:length(s)
    v[i] = 1.0
    t .= ForwardDiff.Dual{DualTag}.(s,v)
    Et = E(t)
    ∇s1[i] = ForwardDiff.partials(Et,1)
    v[i] = 0.0
end

using FiniteDiff
∇s2 = FiniteDiff.finite_difference_gradient(E,s)


norm(∇s0 - ∇s1)/norm(∇s1)
norm(∇s1 - ∇s2)/norm(∇s2)
# Passed, move on

# Hessian Vector Products w/ SparseDiffTools
H = SparseDiffTools.JacVec(dEds!, s)
Hdense = zeros(length(s), length(s))
v = zeros(length(s))
for i = 1:length(s)
    v[i] = 1.0
    mul!(view(Hdense, :, i), H, v)
    v[i] = 0.0
end





out1, out2 = minres!(update, H, ∇s; log = true)
