"""
        elastic_forces(rod::basic_rod, rod_props::rod_properties, ref_strains::rod_strains)

Calculate elastic forces acting on a rod. Output is a rod_delta type, containing
the (positive) gradient of elastic energy with respect to each degree of freedom.
"""
function elastic_forces(rod::basic_rod, rod_props::rod_properties, ref_strains::rod_strains)

    # unpack
    l0, κ0, τ0 =  ref_strains.l, ref_strains.κ, ref_strains.τ
    k, B, β = rod_props.k, rod_props.B, rod_props.β
    x, d1 = rod.x, rod.d

    # Tangents and directors
    T = eltype(x)
    nv = size(x,1)
    e = x[2:nv,:] .- x[1:nv-1,:]
    l = norm3(e)
    t = e./l
    d2 = cross3(t, d1)

    # Split into halves of the rod
    l1 = @view l[1:nv-2]
    l2 = @view l[2:nv-1]
    t1 = @view t[1:nv-2,:]
    t2 = @view t[2:nv-1,:]
    d1l = @view d1[1:nv-2,:]
    d1r = @view d1[2:nv-1,:]
    d2l = @view d2[1:nv-2,:]
    d2r = @view d2[2:nv-1,:]

    chi = 1 .+ dot3(t1, t2)
    ttild = (t1 .+ t2)./chi
    d1tild = (d1l .+ d1r)./chi
    d2tild = (d2l .+ d2r)./chi
    kb = 2 .*cross3(t1, t2)./chi

    κ1l =  dot3(kb, d2l)
    κ1r =  dot3(kb, d2r)
    κ2l = -dot3(kb, d1l)
    κ2r = -dot3(kb, d1r)
    κ1 = (κ1l + κ1r)./2
    κ2 = (κ2l + κ2r)./2
    κ = hcat(κ1, κ2)

    # Curvature Derivatives
    dκ1de1 = (-κ1.*ttild .+ cross3(t2, d2tild))./l1
    dκ1de2 = (-κ1.*ttild .- cross3(t1, d2tild))./l2

    dκ2de1 = (-κ2.*ttild .- cross3(t2, d1tild))./l1
    dκ2de2 = (-κ2.*ttild .+ cross3(t1, d1tild))./l2

    # Twists
    d1_transport = ptransport(d1l, t1, t2)
    cosτ = dot3(d1_transport, d1r)
    sinτ = dot3(d1_transport, d2r)
    τ = -atan.(sinτ, cosτ)

    # Twist Derivatives
    dτde1 = kb./(2 .*l1);
    dτde2 = kb./(2 .*l2);

    # Energy derivatives
    vl0 = [l0[1]; l0[2:end-1]/2] + [l0[2:end-1]/2; l0[end]]
    dEdl = k.*(l - l0)./l0
    dEdκ = B.*(κ .- κ0)./vl0
    dEdτ = β.*(τ .- τ0)./vl0

    dEsde = dEdl.*t
    dEde1 = dEdκ[:,1].*dκ1de1 .+ dEdκ[:,2].*dκ2de1 .+ dEdτ.*dτde1
    dEde2 = dEdκ[:,1].*dκ1de2 .+ dEdκ[:,2].*dκ2de2 .+ dEdτ.*dτde2

    dEdx = [-dEde1; zeros(T,2,3)] .+ [zeros(T,1,3); dEde1 .- dEde2; zeros(T,1,3)] .+ [zeros(T,2,3); dEde2]
    dEdx += [-dEsde; zeros(T,1,3)] .+ [zeros(T,1,3); dEsde]

    dEtdθ = [0; dEdτ] .- [dEdτ; 0]
    dEbdθ = ([(κ2l.*dEdκ[:,1] .- κ1l.*dEdκ[:,2]); 0]./2
          .+ [0; (κ2r.*dEdκ[:,1] .- κ1r.*dEdκ[:,2])]./2)
    dEdθ = dEtdθ .+ dEbdθ

    rod_delta(dEdx, dEdθ)
end

function gradient_update(x0, d0, Δx, Δθ)

    x1 = x0 + Δx
    e1 = edges(x1)
    l1 = norm3(e1)
    t1 = e1./l1
    t0 = tangents(x0)
    d1 = ptransport(d0, t0, t1)
    d1 = rotate_orthogonal_unit(d1, t1, Δθ)

    ζ = cross3(t1, t0)./(l1.*(1 .+ dot3(t1, t0)))

    x1, d1, ζ
end

function gradient_update(ref::basic_rod, dr::rod_delta)
    x, d, ζ = gradient_update(ref.x, ref.d, dr.Δx, dr.Δθ)
    basic_rod(x, d), ζ
end

# Cache should be ((ns + 1 x 7), (ns x 28))
function elastic_forces!(forces::rod_delta, rod::basic_rod,
    rod_props::rod_properties, ref_strains::rod_strains, caches)

    # unpack
    l0, κ0, τ0 =  ref_strains.l, ref_strains.κ, ref_strains.τ
    k, B, β = rod_props.k, rod_props.B, rod_props.β
    x, d1 = rod.x, rod.d
    dEdx, dEdθ = forces.Δx, forces.Δθ

    elastic_forces!(dEdx, dEdθ, x, d1, k, B, β, l0, κ0, τ0, caches)
end

function elastic_forces!(dEdx, dEdθ, x, d1, k, B, β, l0, κ0, τ0, caches)

    nv = size(x,1)
    ne = nv - 1

    # Unpack and divy up cache1
    cache1, cache2 = caches
    t = view(cache1, :, 1:3)
    l = view(cache1, :, 4)
    dEsde = view(cache1, :, 5:7) # Overwritten early
    d2 = view(cache1, :, 5:7)

    # Permanent cache2 space (never overwritten)
    vl0 = view(cache2, :, 1)
    kb = view(cache2, :, 2:4)
    dEde1 = view(cache2, :, 5:7)
    dEde2 = view(cache2, :, 8:10)

    # Bending cache2 space (overwritten for twist)
    chi = view(cache2, :, 11)
    ttild = view(cache2, :, 12:14)
    dtild = view(cache2, :, 15:17)
    κ1l = view(cache2, :, 18)
    κ1r = view(cache2, :, 19)
    κ2l = view(cache2, :, 20)
    κ2r = view(cache2, :, 21)
    κ1 = view(cache2, :, 22)
    κ2 = view(cache2, :, 23)
    dEdκ1 = view(cache2, :, 24)
    dEdκ2 = view(cache2, :, 25)
    dκde = view(cache2, :, 26:28)

    # Edges and lengths
    edges!(t, x)
    norm3!(l, t)
    t ./= l

    vl0[1] = l0[1] + l0[2]/2
    vl0[2:ne-2] .= (view(l0, 2:ne-2) .+ view(l0, 3:ne-1))./2
    vl0[ne-1] = l0[ne-1]/2 + l0[ne]

    # Stretching Forces
    dEsde .= (k.*(l .- l0)./l0).*t
    dEdx .= 0.0
    view(dEdx, 1:nv-1, :) .-= dEsde
    view(dEdx, 2:nv, :) .+= dEsde

    # Break tangents and directors into sets
    cross3!(d2, t, d1)
    l1 = @view l[1:nv-2]
    l2 = @view l[2:nv-1]
    t1 = @view t[1:nv-2,:]
    t2 = @view t[2:nv-1,:]
    d1l = @view d1[1:nv-2,:]
    d1r = @view d1[2:nv-1,:]
    d2l = @view d2[1:nv-2,:]
    d2r = @view d2[2:nv-1,:]

    # Curvature
    dot3!(chi, t1, t2)
    chi .+= 1.0
    cross3!(kb, t1, t2)
    kb .*= 2.0./chi

    dot3!(κ1l, kb, d2l)
    dot3!(κ1r, kb, d2r)
    dot3!(κ2l, kb, d1l)
    dot3!(κ2r, kb, d1r)
    κ2l .*= -1.0
    κ2r .*= -1.0
    κ1 .= (κ1l .+ κ1r)./2
    κ2 .= (κ2l .+ κ2r)./2
    dEdκ1 .= view(B, :, 1).*(κ1 .- view(κ0, :, 1))./vl0
    dEdκ2 .= view(B, :, 2).*(κ2 .- view(κ0, :, 2))./vl0

    # Curvature Derivatives
    ttild .= (t1 .+ t2)./chi
    dtild .= (d2l .+ d2r)./chi

    # dκ1de1
    dtild .= (d2l .+ d2r)./chi # d2tild
    cross3!(dκde, t2, dtild)
    dκde .-= κ1.*ttild
    dκde ./= l1
    dEde1 .= dEdκ1.*dκde

    # dκ1de2
    cross3!(dκde, t1, dtild)
    dκde .+= κ1.*ttild
    dκde ./= -1.0.*l2
    dEde2 .= dEdκ1.*dκde

    # dκ2de1
    dtild .= (d1l .+ d1r)./chi # d1tild
    cross3!(dκde, t2, dtild)
    dκde .+= κ2.*ttild
    dκde ./= -1.0.*l1
    dEde1 .+= dEdκ2.*dκde

    # dκ2de2
    cross3!(dκde, t1, dtild)
    dκde .-= κ2.*ttild
    dκde ./= l2
    dEde2 .+= dEdκ2.*dκde

    # Bending forces and moments
    dEdθ .= 0.0
    view(dEdθ, 1:nv-2) .+= (κ2l.*dEdκ1 .- κ1l.*dEdκ2)./2
    view(dEdθ, 2:nv-1) .+= (κ2r.*dEdκ1 .- κ1r.*dEdκ2)./2

    # Reallocate Cache Space for Twists
    pt_cache = view(cache2, :, 11:16)
    d1trans = view(cache2, :, 17:19)
    cosτ = view(cache2, :, 20)
    sinτ = view(cache2, :, 21)
    τ = view(cache2, :, 22)
    dEdτ = view(cache2, :, 23)

    # Twists
    ptransport!(d1trans, d1l, t1, t2, pt_cache)
    dot3!(cosτ, d1r, d1trans)
    dot3!(sinτ, d2r, d1trans)
    τ .= -1.0.*atan.(sinτ, cosτ)
    #
    # # Twisting Moments
    dEdτ .= β.*(τ .- view(τ0, :))./vl0
    view(dEdθ, 1:nv-2) .-= dEdτ
    view(dEdθ, 2:nv-1) .+= dEdτ
    dEde1 .+= dEdτ.*kb./(2 .*l1)
    dEde2 .+= dEdτ.*kb./(2 .*l2)

    # Assemble into output array
    view(dEdx, 1:nv-2, :) .-= dEde1
    view(dEdx, 2:nv-1, :) .+= dEde1 .- dEde2
    view(dEdx, 3:nv, :) .+= dEde2

    return nothing
end

allocate_cache(f::typeof(elastic_forces!), T::Type, ns) =
    (zeros(T, ns+1, 7), zeros(T, ns, 28))

function gradient_update!(x1, d1, ζ, x0, d0, Δx, Δθ, cache)
    t0 = view(cache, :, 1:3)
    t1 = view(cache, :, 4:6)
    l1 = view(cache, :, 7)
    t1_dot_t0 = view(cache, :, 8)
    cache_pt = view(cache, :, 8:13)
    cache_rt = view(cache, :, 8:12)

    tangents!(t0, x0)

    x1 .= x0 .+ Δx
    edges!(t1, x1)
    norm3!(l1, t1)
    t1 ./= l1

    ptransport!(d1, d0, t0, t1, cache_pt)
    rotate_orthogonal_unit!(d1, t1, Δθ, cache_pt)

    cross3!(ζ, t1, t0)
    dot3!(t1_dot_t0, t1, t0)
    ζ ./= (l1.*(1.0 .+ t1_dot_t0))
end

function gradient_update!(new::basic_rod, ζ, old::basic_rod, dr::rod_delta, cache)
    gradient_update!(new.x, new.d, ζ, old.x, old.d, dr.Δx, dr.Δθ, cache)
end

allocate_cache(f::typeof(gradient_update!), T::Type, ns) = zeros(T, ns+1, 13)

#--------------------------------------------------------
# Fusing the gradient update and forces step
#--------------------------------------------------------

function fused_update_gradient!(∇E, r, Δr, p, s, caches)

    cache_vertices, cache_edges, cache_interior = caches

    gu_cache = cache_edges
    ef_cache = (cache_edges, cache_interior)
    x = cache_vertices
    d = @view cache_edges[:, 14:16]
    ζ = @view cache_edges[:, 17:19]

    gradient_update!(x, d, ζ, r.x, r.d, Δr.Δx, Δr.Δθ, gu_cache)
    elastic_forces!(∇E.Δx, ∇E.Δθ, x, d, p.k, p.B, p.β, s.l, s.κ, s.τ, ef_cache)
    (@view ∇E.Δx[1:end-1, :]) .-= ∇E.Δθ .* ζ
    (@view ∇E.Δx[2:end,:]) .+= ∇E.Δθ .* ζ
end

allocate_cache(f::typeof(fused_update_gradient!), T::Type, ns) =
    (zeros(T, ns+2, 3), zeros(T, ns+1, 19), zeros(T, ns, 28))
