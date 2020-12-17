DER = DiscreteElasticRods
using LinearAlgebra

@testset "DER_nonmutating" begin

    # Basic Linear Algebra: cross3, dot3, norm3, norm3!
    a = [1. 2.; 2. 5.; 9. 6.]
    b = [4. 3.; 5. 2.; 7. 8.]
    xab = [-31. 28.; 29. 2.; -3. -11]
    dab = [77. 64.]
    na = [sqrt(86) sqrt(65)]
    ta = a./na
    @test xab ≈ DER.cross3(a,b)
    @test dab ≈ DER.dot3(a,b)
    @test na ≈ DER.norm3(a)

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

end

@testset "DER_mutating" begin

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

    DER.dot3!(d,a,b,cache)
    @test dab ≈ d

    DER.norm3!(d,a)
    @test na ≈ d

    c = copy(a)
    DER.norm3!(c)
    @test ta ≈ c

    # Parallel transport inner coeff: TBD!
    # Parallel transport
    # t1 = [1. 0. 0.]'
    # t2 = [1. 1. 1.]'/sqrt(3)
    # v1 = [0. 1. 1.]'
    # v2 = [-2. 1. 1.]'/sqrt(3)
    # @test DER.ptransport(t1, t1, t2) ≈ t2
    # @test DER.ptransport(DER.cross3(t1, t2), t1, t2) ≈ DER.cross3(t1, t2)
    # @test DER.ptransport([v1 v1], [t1 t1], [t2 t2]) ≈ [v2 v2]
    #
    # # Rotations of orthogonal unit vectors
    # v1 = [[0., 1., 0.,] [0., 0., 1.,]]
    # v2 = [[0., 1., 1.,] [0., -1., 1.,]]/sqrt(2)
    # θ = π/4
    # t = [1., 0., 0.]
    # @test DER.rotate_orthogonal_unit(v1, [t t], [θ θ]) ≈ v2
    # DER.rotate_orthogonal_unit!(v1, [t t], [θ θ])
    # @test v1 ≈ v2

end

# Next up: edges and tangents! (everything in rod_core)
