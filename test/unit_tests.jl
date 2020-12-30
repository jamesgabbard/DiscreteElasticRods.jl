DER = DiscreteElasticRods
using LinearAlgebra

@testset "core_nonmutating" begin

    # Basic Linear Algebra: cross3, dot3, norm3, norm3!
    a = [1. 2.; 2. 5.; 9. 6.]
    b = [4. 3.; 5. 2.; 7. 8.]
    xab = [-31. 28.; 29. 2.; -3. -11]
    dab = [77. 64.]
    na = [sqrt(86) sqrt(65)]
    @test xab ≈ DER.cross3(a,b)
    @test dab ≈ DER.dot3(a,b)
    @test na ≈ DER.norm3(a)

    # Edges and Tangents
    ea = [1.; 3.; -3.]
    ta = ea./sqrt(19)
    @test ea ≈ DER.edges(a)
    @test ta ≈ DER.tangents(a)

    # Parallel transport
    θ = [1.0 0.01 0.0]
    ptic_θ = [(1 - cos(1.0))/sin(1.0)^2 (1 - cos(0.01))/sin(0.01)^2 0.5]

    s2 = sin.(θ).^2
    c = cos.(θ)
    @test ptic_θ ≈ DER.ptransport_inner_coefficient.(s2, c)

    t1 = [1. 0. 0.]'
    t2 = [1. 1. 1.]'/sqrt(3)
    v1 = [0. 1. 1.]'
    v2 = [-2. 1. 1.]'/sqrt(3)
    @test DER.ptransport(t1, t1, t2) ≈ t2
    @test DER.ptransport(DER.cross3(t1, t2), t1, t2) ≈ DER.cross3(t1, t2)
    @test DER.ptransport([v1 v1], [t1 t1], [t2 t2]) ≈ [v2 v2]

    # Rotations of orthogonal unit vectors
    v1 = [[0., 1., 0.,] [0., 0., 1.,]]
    v2 = [[0., 1., 1.,] [0., -1., 1.,]]/sqrt(2)
    θ = π/4
    t = [1., 0., 0.]
    @test DER.rotate_orthogonal_unit(v1, [t t], [θ θ]) ≈ v2

    # Basic Rod update
    x0 = [0. 1. 2.; 0. 0. 0.; 0. 0. 0.]
    d0 = [0. 0.; 0. 0.; 1. 1.]
    Δx = [0. 0. 0.; 0. 0. 0.; 0. 1. 0.]
    Δθ = [0. π/2]
    x, d = DER.rod_update(x0, d0, Δx, Δθ)
    @test x ≈ x0 + Δx
    @test d ≈ [-1/sqrt(2) 0; 0. -1.; 1/sqrt(2) 0.]

    # Full kinematics
    l, κ, τ = DER.full_kinematics(x, d)
    @test l ≈ [sqrt(2) sqrt(2)]
    @test κ ≈ [-1.; 1.]
    @test τ ≈ [π/2]
end

@testset "core_mutating" begin

    # Basic Linear Algebra: cross3, dot3, norm3, norm3!
    a = [1. 2.; 2. 5.; 9. 6.]
    b = [4. 3.; 5. 2.; 7. 8.]
    xab = [-31. 28.; 29. 2.; -3. -11]
    dab = [77. 64.]
    na = [sqrt(86) sqrt(65)]
    ta = a./na

    c = similar(a)
    d = zeros(1,2)
    cache = similar(a)

    DER.cross3!(c,a,b)
    @test xab ≈ c

    DER.dot3!(d,a,b)
    @test dab ≈ d

    DER.norm3!(d,a)
    @test na ≈ d

    c = copy(a)
    DER.norm3!(c)
    @test ta ≈ c

    # Edges and tangents
    eb = [-1.0; -3.0; 1.0]
    c = similar(eb)
    DER.edges!(c,b)
    @test eb ≈ c

    tb = eb./sqrt(11)
    c = similar(tb)
    DER.tangents!(c,b)
    @test tb ≈ c

    # Parallel transport inner coeff: TBD!
    θ = [1.0 0.01 0.0]
    ptic_θ = [(1 - cos(1.0))/sin(1.0)^2 (1 - cos(0.01))/sin(0.01)^2 0.5]

    s2 = sin.(θ).^2
    c = cos.(θ)
    out = similar(θ)
    DER.ptransport_inner_coefficient!(out, s2, c)
    @test ptic_θ ≈ out

    # Parallel transport
    t1 = [1. 0. 0.]'
    t2 = [1. 1. 1.]'/sqrt(3)
    v1 = [0. 1. 1.]'
    v2 = [-2. 1. 1.]'/sqrt(3)
    cache = zeros(9, 1)

    val = similar(t1)
    DER.ptransport!(val, t1, t1, t2, cache)
    @test val ≈ t2

    DER.ptransport!(val, DER.cross3(t1, t2), t1, t2, cache)
    @test val ≈ DER.cross3(t1, t2)

    cache = zeros(9, 2)
    val = [v1 v1]
    DER.ptransport!(val, [v1 v1], [t1 t1], [t2 t2], cache)
    @test val ≈ [v2 v2]

    # Rotations of orthogonal unit vectors
    v1 = [[0., 1., 0.,] [0., 0., 1.,]]
    v2 = [[0., 1., 1.,] [0., -1., 1.,]]/sqrt(2)
    θ = π/4
    t = [1., 0., 0.]

    cache = zeros(5,2)
    DER.rotate_orthogonal_unit!(v1, [t t], [θ θ], cache)
    @test  v1 ≈ v2

    # Basic Rod update
    x1 = [0. 1. 2.; 0. 0. 0.; 0. 0. 0.]
    d1 = [0. 0.; 0. 0.; 1. 1.]
    Δx = [0. 0. 0.; 0. 0. 0.; 0. 1. 0.]
    Δθ = [0. π/2]
    x2 = x0 + Δx
    d2 = [-1/sqrt(2) 0; 0. -1.; 1/sqrt(2) 0.]
    cache = zeros(15,3)
    DER.rod_update!(x1, d1, Δx, Δθ, cache)
    @test x1 ≈ x2
    @test d1 ≈ d2

    # Full Kinematics
    l = zeros(1,2)
    κ = zeros(2,1)
    τ = zeros(1,1)
    caches = (zeros(6,2), zeros(9,1))
    DER.full_kinematics!(l, κ, τ, x2, d2, caches)
    @test l ≈ [sqrt(2) sqrt(2)]
    @test κ ≈ [-1.; 1.]
    @test τ ≈ [π/2]

end

# Next: Elastic and Gravitational energy
