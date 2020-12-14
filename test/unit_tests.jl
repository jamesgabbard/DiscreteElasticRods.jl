DER = DiscreteElasticRods

@testset "DER_linalg" begin

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

    c = copy(a)
    DER.norm3!(c)
    @test ta ≈ c

    # Parallel transport
    θ = [1.0 0.01 0.0]
    ptic_θ = [(1 - cos(1.0))/sin(1.0)^2 (1 - cos(0.01))/sin(0.01)^2 0.5]
    @test ptic_θ ≈ DER.ptransport_inner_coefficient.(θ)

    t1 = [1., 0., 0.]
    t2 = [1., 1., 1.]/sqrt(3)
    v1 = [0., 1., 1.]
    v2 = [-2., 1., 1.]/sqrt(3)
    @test DER.ptransport(t1, t1, t2) ≈ t2
    @test DER.ptransport(cross(t1, t2), t1, t2) ≈ cross(t1, t2)
    @test DER.ptransport([v1 v1], [t1 t1], [t2 t2]) ≈ [v2 v2]

    # Rotations of orthogonal unit vectors
    v1 = [[0., 1., 0.,] [0., 0., 1.,]]
    v2 = [[0., 1., 1.,] [0., -1., 1.,]]/sqrt(2)
    θ = π/4
    t = [1., 0., 0.]
    @test DER.rotate_orthogonal_unit(v1, [t t], [θ θ]) ≈ v2
    DER.rotate_orthogonal_unit!(v1, [t t], [θ θ])
    @test v1 ≈ v2

end

# Next up: edges and tangents! (everything in rod_core)
