# The entire non-mutating rod sequence.
# For organization, the entire front half accepts arrays and
# returns tuples of arrays. Convenience overloads are at the
# end of the file

function cross3(a,b)
    vcat(a[2:2,:].*b[3:3,:] .- a[3:3,:].*b[2:2,:],
         a[3:3,:].*b[1:1,:] .- a[1:1,:].*b[3:3,:],
         a[1:1,:].*b[2:2,:] .- a[2:2,:].*b[1:1,:])
end

function dot3(a,b)
    sum(a.*b, dims=1)
end

function norm3(a)
    sqrt.(sum(abs2, a, dims=1))
end

function edges(x)
    x1 = @view x[:,2:end]
    x2 = @view x[:,1:end-1]
    x1 .- x2
end

function tangents(x)
    e = edges(x)
    e./norm3(e)
end

PTRANSPORT_TOL = 0.06

function ptransport_inner_coefficient(s2,c)
    s2 > PTRANSPORT_TOL^2 ? (1-c)/s2 : 1/2 + (1-c)/4 + (1-c)^2/8*(1 + s2/4)
end

function ptransport(v, t1, t2)
    X = cross3(t1,t2);
    s2 = dot3(X,X);
    c = dot3(t1,t2);
    ptic = ptransport_inner_coefficient.(s2,c)
    c.*v .+ cross3(X,v) .+ ptic.*dot3(X,v).*X
end

function rotate_orthogonal_unit(v, t, θ)
    cos.(θ).*v .+ sin.(θ).*cross3(t,v)
end

function full_kinematics(x, d1)

    # Tangents and directors
    nv = size(x,2)
    e =x[:, 2:nv] .- x[:, 1:nv-1]
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

function rod_update(x0, d0, Δx, Δθ)

    x = x0 + Δx
    t0 = tangents(x0)
    t = tangents(x)
    d = ptransport(d0, t0, t)
    d = rotate_orthogonal_unit(d, t, Δθ)

    x, d
end

function elastic_energy(l0,κ0,τ0, l,κ,τ, k,B,β)

    # Reference voronoi lengths
    vl0 = [l0[1]; l0[2:end-1]/2]' + [l0[2:end-1]/2; l0[end]]'

    # Energies
    Es = 1/2*sum(k./l0.*(l - l0).^2)
    Et = 1/2*sum(β./vl0.*(τ - τ0).^2)
    Eb = 1/2*sum(B./vl0.*(κ - κ0).^2)
    Es + Et + Eb
end

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
