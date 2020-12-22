# The entire non-mutating rod sequence.
# For organization, the entire front half accepts arrays and
# returns tuples of arrays. Convenience overloads are at the
# end of the file

# Store rods as either rows or columns
# This typing is compiled away, but allows you to specify the storage
# order for rod coordiantes

"""
        cross3(a,b)

Cross products of the columns of two 3 x N arrays. Output is also 3 x N
"""
function cross3(a,b)
    vcat(a[2:2,:].*b[3:3,:] .- a[3:3,:].*b[2:2,:],
         a[3:3,:].*b[1:1,:] .- a[1:1,:].*b[3:3,:],
         a[1:1,:].*b[2:2,:] .- a[2:2,:].*b[1:1,:])
end

"""
        dot3(a,b)

Dot products of the columns of two 3 x N arrays. Output is 1 x N
"""
function dot3(a,b)
    sum(a.*b, dims=1)
end

"""
        norm3(a,b)

Norms of the columns of two 3 x N arrays. Output is 1 x N
"""
function norm3(a)
    sqrt.(sum(abs2, a, dims=1))
end

"""
        edges(x)

Edges of a centerline `x`, stored as a 3 x N array. Output is 3 x N-1
"""
function edges(x)
    x1 = @view x[:,2:end]
    x2 = @view x[:,1:end-1]
    x1 .- x2
end

"""
        tangents(x)

Tangents of a centerline `x`, stored as a 3 x N array. Output is 3 x N-1
"""
function tangents(x)
    e = edges(x)
    e./norm3(e)
end

"""
        ptransport(v, t1, t2)

Parallel transport of vectors `v` from tangents `t1` to tangents 't2'.

All inputs are (3 x N). Each column of `t1` and `t2` must have unit norm.

Uses the Rodriguez rotation formula. There is a singularity in this definition
when t1 == t2, which is handled by switching to an small-θ expansion
(see ptransport_inner_coefficient).

"""
function ptransport(v, t1, t2)
    X = cross3(t1,t2);
    s2 = dot3(X,X);
    c = dot3(t1,t2);
    ptic = ptransport_inner_coefficient.(s2,c)
    c.*v .+ cross3(X,v) .+ ptic.*dot3(X,v).*X
end

function ptransport_inner_coefficient(s2,c)
    opt1 = (1-c)/s2
    opt2 = 1/2 + (1-c)/4 + (1-c)^2/8*(1 + s2/4)
    s2 > PTRANSPORT_TOL^2 ?  opt1 : opt2
end

"""
        rotate_orthogonal_unit(v, t, θ)

Rotate unit vectors `v` around axes `t1` by angles `θ`.

`v` and `t` are (3 x N), `θ` is (1 x N).

The columns `v[:,i]` and `t[:,i]` must be orthogonal unit vectors. The function
uses the Rodriguez rotation formula speciailized to this case.
"""
function rotate_orthogonal_unit(v, t, θ)
    cos.(θ).*v .+ sin.(θ).*cross3(t,v)
end

"""
        full_kinematics(x, d)

Returns the edge lengths, curvatures, and twists of a rod.

# Arguments
- `x`: centerline vertices, size (3 x N)
- `d`: directors, size (3 x N-1). Unit length and orthogonal to `edges(x)`

# Returns
- `l`: edge lengths (1 x N-1)
- `κ`: curvature normals (2 x N-2), expressed in the director frame
- 'τ': scalar twist at each vertex (1 x N-2)

Twists are calculated by parallel transporting the directors across each
vertex, and comparing with the original. This avoids the reference
twist calculations used in Lestringant, Audoly, and Kochmann (2020)
"""
function full_kinematics(x, d1)

    # Tangents and directors
    nv = size(x,2)
    #e =x[:, 2:nv] .- x[:, 1:nv-1]
    e = edges(x)
    l = norm3(e)
    t = e./l
    d2 = cross3(t, d1)

    # Split into halves of the rod
    t1 = @view t[:,1:nv-2]
    t2 = @view t[:,2:nv-1]
    d1l = @view d1[:,1:nv-2]
    d1r = @view d1[:,2:nv-1]
    d2l = @view d2[:,1:nv-2]
    d2r = @view d2[:,2:nv-1]

    # Curvatures
    k = 2 .*cross3(t1, t2)./(1.0 .+ dot3(t1, t2))
    κ  = vcat(dot3(k, (d2l + d2r)./2),
             -dot3(k, (d1l + d1r)./2))

    # Twists
    d1_transport = ptransport(d1l, t1, t2)
    cosτ = dot3(d1_transport, d1r)
    sinτ = dot3(d1_transport, d2r)
    τ = atan.(sinτ, cosτ)

    l, κ, τ
end
"""
        rod_update(x0, d0, Δx, Δθ)

Update the centerline and directors of a rod.

The vertices `x0` are translated `Δx`. The directors `d0` are parallel
transported to the new centerline, then rotated by `Δθ`

# Arguments
- `x0`: centerline vertices, size (3 x N)
- `d0`: directors, size (3 x N-1). Unit length and orthogonal to `edges(x0)`
- `Δx`: displacements, size (3 x N)
- `Δ0`: rotations, size (1 x N-1).

# Returns
- `x`, `d`: new centerline and directors.

"""
function rod_update(x0, d0, Δx, Δθ)

    x = x0 + Δx
    t0 = tangents(x0)
    t = tangents(x)
    d = ptransport(d0, t0, t)
    d = rotate_orthogonal_unit(d, t, Δθ)

    x, d
end

"""
        elastic_energy(l0,κ0,τ0, l,κ,τ, k,B,β)

Elastic energy, calculated from stretching, bending, and twisting strains.

# Arguments
- `l0`, `κ0`, `τ0`: reference lengths, curvatures, and twists
- `l`, `κ`, `τ`: current lengths, curvatures, and twists
- `k`, `B`, `β`: stretching, bending, and twisting stiffness.

Lengths, curvatures, and twists should be (1 x N-1), (2 x N-2), and (1 x N-2),
respectively.

 `k` and `β` may be scalar or similar to `l` and `τ`. The bending stiffness
 `B` may constant isotropic (scalar), constant anisotropic (2 x 1),
 or non-constant anisotropic (2 x N-2).
"""
function elastic_energy(l0,κ0,τ0, l,κ,τ, k,B,β)

    # Reference voronoi lengths
    vl0 = [l0[1]; l0[2:end-1]/2]' + [l0[2:end-1]/2; l0[end]]'

    # Energies
    Es = 1/2*sum(k./l0.*(l - l0).^2)
    Et = 1/2*sum(β./vl0.*(τ - τ0).^2)
    Eb = 1/2*sum(B./vl0.*(κ - κ0).^2)
    Es + Et + Eb
end

"""
        gravitational_energy(x, m, g)

Gravitational energy, calculated from centerline.

# Arguments
- `x`: centerline vertices, 3 x N
- `m`: edges masses, 1 x N-1. Typically `ρ*A*l0`
- `g`: magnitude of gravitational acceleration (positive).

Gravity is assumed to act in the negative `z` direction.
"""
function gravitational_energy(x, m, g)
    nv = size(x,2)
    z = (x[3:3, 1:nv-1] .+ x[3:3,2:nv])./2
    sum(g.*m.*z)
end

# ------------------------------------------------------------------------------
# Convenience type overloads
# ------------------------------------------------------------------------------
edges(r::basic_rod) = edges(r.x)
tangents(r::basic_rod) = tangents(r.x)

function full_kinematics(r::basic_rod)
    l, κ, τ = full_kinematics(r.x, r.d)
    rod_strains(l, κ, τ)
end

function rod_update(r::basic_rod, dr::rod_delta)
    x, d = rod_update(r.x, r.d, dr.Δx, dr.Δθ)
    basic_rod(x,d)
end

function elastic_energy(ref_strains::rod_strains, strains::rod_strains, rod_props::rod_properties)
    l0, κ0, τ0 = ref_strains.l, ref_strains.κ, ref_strains.τ
    l, κ, τ = strains.l, strains.κ, strains.τ
    k, B, β = rod_props.k, rod_props.B, rod_props.β
    elastic_energy(l0,κ0,τ0, l,κ,τ, k,B,β)
end
