# A simple quasi-static problem, based on an L-BFGS minimization
using DiscreteElasticRods
DER = DiscreteElasticRods
using BenchmarkTools
using DelimitedFiles

# Parameters
E = 5e6
ν = 0.48
L = 0.30
R = 0.003

n_array = [10, 20, 30, 40, 50, 75, 100, 200, 500]
time_array = zeros(2, length(n_array))
mem_array = zeros(2, length(n_array))

for k = 1:length(n_array)
    ns = n_array[k]

    # Create a straight rod with NS seg
    p1 = [0., 0., 0.]
    p2 = [L, 0., 0.]
    d0 = [0., 0., 1.]
    ref_rod = DER.straight_rod(p1, p2, d0, ns)

    # Reference kinematics and properties
    props = DER.elliptical_stiffness(E, ν, R, R)
    ref_strain = DER.full_kinematics(ref_rod)

    # q ordering: [θ2, x3, θ3, x4, ..., x, θ]
    ncore = 4*ns - 7
    nq = ncore + 7

    # non - allocating
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

    T = Float64
    cache = zeros(T, 27, ns)
    Δx = zeros(T, 3,ns+2)
    Δθ = zeros(T, 1,ns+1)
    rod = DER.allocate_rod(T, ns)
    strain = DER.allocate_strain(T,ns)
    p_cache = (Δx, Δθ, rod, strain, cache)

    function total_energy_nonalloc(q::Vector{T}) where T

        Δx, Δθ, rod, strain, cache = p_cache

        param_2_rod!(Δx, Δθ, q)
        delta = DER.rod_delta(Δx, Δθ)
        DER.copy!(rod, ref_rod)
        DER.rod_update!(rod, delta, cache)
        DER.full_kinematics!(strain, rod, cache)
        DER.elastic_energy(ref_strain, strain, props, cache)
    end

    # Allocating
    function param_2_rod(q)
        T = eltype(q)
        Δx = zeros(T, 3,ns+2)
        Δθ = zeros(T, 1,ns+1)
        param_2_rod!(Δx, Δθ, q)
        Δx, Δθ
    end

    function total_energy(q)
        Δx, Δθ = param_2_rod(q)
        rod_coords = DER.rod_delta(Δx, Δθ)

        new_rod = DER.rod_update(ref_rod, rod_coords)
        strains = DER.full_kinematics(new_rod)
        DER.elastic_energy(ref_strain, strains, props)
    end

    # Benchmark Both
    q = rand(nq)
    total_energy(q)
    total_energy_nonalloc(q)
    yes_alloc = @benchmark total_energy($q)
    no_alloc = @benchmark total_energy_nonalloc($q)

    time_array[1,k] = minimum(yes_alloc).time/1e9
    mem_array[1,k] = memory(yes_alloc)
    time_array[2,k] = minimum(no_alloc).time/1e9
    mem_array[2,k] = memory(no_alloc)
end

writedlm("evaluation_timing.txt", time_array, ',')
writedlm("evaluation_memory.txt", mem_array, ',')
