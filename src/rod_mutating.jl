# Reducing allocations as much as possible.
# Included after the non-mutating versions, so that we can
# fall back to a less efficient version where needed

function cross3!(c,a,b)
    a1 = @view a[1,:]
    a2 = @view a[2,:]
    a3 = @view a[3,:]

    b1 = @view b[1,:]
    b2 = @view b[2,:]
    b3 = @view b[3,:]

    c[1,:] .= @. a2*b3 - a3*b2
    c[2,:] .= @. a3*b1 - a1*b3
    c[3,:] .= @. a1*b2 - a2*b1
end

function dot3!(c,a,b)
    sum!(c, a.*b)
end

function dot3!(c,a,b,cache)
    cache .= a.*b
    sum!(c, cache)
end

function norm3!(a)
    a ./= sqrt.(sum(abs2, a, dims=1))
end

function norm3!(out,a)
    out .= sqrt.(sum(abs2, a, dims=1))
end

function edges!(e, x)
    x1 = @view x[:,2:end]
    x2 = @view x[:,1:end-1]
    e .= x1 .- x2
end

function tangents!(t, x)
    edges!(t,x)
    norm3!(t)
end

# ----------------------------------------------------------------------------------
# Parallel Transport (Vectorized)
# ----------------------------------------------------------------------------------
#
# Uses the Rodriguez rotation formula. There is a singularity in
# this parametrization of rotation matrices, for t1 and t2 almost
# aligned. This singularity is handled in ptransport_inner_coefficient,
# by switching to a small-θ expansion.
#
# The conditional does not affect the "true" value of the function,
# it simply switches to a more stable method of computing the same
# quantity. Consequently, it is hidden from reverse-mode AD by defining
# a custom pullback for the function 'ptransport_inner_coefficient'
#
# There is some inneficiency in re-computing sin(θ) and cos(θ) in this function,
# but I'm not into the micro-optimization at the moment!

function ptransport_inner_coefficient!(out,s2,c)
    out = s2 > PTRANSPORT_TOL^2 ? (1-c)/s2 : 1/2 + (1-c)/4 + (1-c)^2/8*(1 + s2/4)
end

# Cache needs to be 9 x ne
function ptransport_cache!(v2, v1, t1, t2, cache)
    X = view(cache, 1:3, :)
    tmp = view(cache, 4:6, :)
    s2 = view(cache, 7:7, :)
    c = view(cache, 8:8, :)
    ptic = view(cache, 9:9, :)

    cross3!(X, t1, t2)
    dot3!(s2, X, X, tmp)
    dot3!(c, t1, t2, tmp)
    ptransport_inner_coefficient!.(ptic,s2,c)

    # s2, c, no longer useful. Overwriting with new temporary
    dXv = view(cache, 7:7, :)
    dot3!(dXv,X,v1,tmp)
    cross3!(v2,X,v1)
    v2 .+= c.*v1 .+ ptic.*dXv.*X
end

# ------------------------------------------------------------------------------
# Rotate about Unit Vector (Vectorized)
# ------------------------------------------------------------------------------
# Rodriguez rotation formula, specialized to the case where the vector
# and the axis are unit length and orthogonal.

# cache is 5 x ne
function rotate_orthogonal_unit!(v, t, θ, cache)
    #ne = length(θ)
    #cache = reshape(view(cacheraw,1:5*ne), 5, ne)

    tmp = view(cache, 1:3, :)
    c = view(cache, 4:4, :)
    s = view(cache, 5:5, :)

    c .= cos.(θ)
    s .= sin.(θ)
    cross3!(tmp, t, v)
    v .= c.*v .+ s.*tmp
end

# Update kinematics given displacement and twist
# Also TODO: get rid of this
function rod_update!(r::basic_rod, Δr::rod_delta)

    t0 = tangents(r.x)
    r.x .+= Δr.Δx
    t = tangents(r.x)
    r.d .= ptransport(r.d, t0, t)
    rotate_orthogonal_unit!(r.d, t, Δr.Δθ)
end

function rod_update!(x, d, Δx, Δθ, cacheraw)
    ne = length(Δθ)
    cache = reshape(view(cacheraw, 1:18*ne), 18, ne)

    t0 = view(cache, 1:3, :)
    t = view(cache, 4:6, :)
    d0 = view(cache, 7:9, :)
    cache_pt = view(cache, 10:18, :)
    cache_rt = view(cache, 10:14, :)

    tangents!(t0, x)
    x .+= Δx
    tangents!(t, x)
    d0 .= d
    ptransport_cache!(d, d0, t0, t, cache_pt)
    rotate_orthogonal_unit!(d, t, Δθ, cache) #TODO: cache this
end

# cache is 27 rows, nv-2 columns
function full_kinematics!(l, κ, τ, x, d1, cache)

    # Tangents, lengths
    nv = size(x,2)
    t = edges(x)
    norm3!(l,t)
    t ./= l
    d2 = cross3(t, d1)

    # Caches
    t1 = view(cache, 1:3, :)
    t2 = view(cache, 4:6, :)
    d1l = view(cache, 7:9, :)
    d1r = view(cache, 10:12, :)
    d2l = view(cache, 13:15, :)
    d2r = view(cache, 16:18, :)
    tmp = view(cache, 19:21, :)
    k = view(cache, 22:24, :)

    t1 .= view(t, :, 1:nv-2)
    t2 .= view(t, :, 2:nv-1)
    d1l .= view(d1,:,1:nv-2)
    d1r .= view(d1,:,2:nv-1)
    d2l .= view(d2,:,1:nv-2)
    d2r .= view(d2,:,2:nv-1)

    # Curvature
    cross3!(k, t1, t2)
    k .*= 2.0./(1.0 .+ dot3(t1, t2))
    tmp .= k.*(d2l .+ d2r)./2
    sum!(view(κ, 1:1, :), tmp)
    tmp .= -k.*(d1l .+ d1r)./2
    sum!(view(κ, 2:2, :), tmp)

    # Twists (could move in part of ptransport to reduce intermediates)
    pt_cache = view(cache, 19:27, :) # overrides k, tmp
    d1_transport = view(cache, 13:15, :) # overrides d2l
    ptransport_cache!(d1_transport, d1l, t1, t2, pt_cache)

    cosτ = view(cache, 19:19, :)
    sinτ = view(cache, 20:20, :)

    dot3!(cosτ, d1r, d1_transport, tmp)
    dot3!(sinτ, d2r, d1_transport, tmp)
    τ .= atan.(sinτ, cosτ)
end

# cache length at least 3*nv
function elastic_energy(l0,κ0,τ0, l,κ,τ, k,B,β, cache)

    ne = length(l)
    nv = ne - 1

    # Reference voronoi lengths
    vl0 = reshape(view(cache, (2*nv+1):3*nv), 1, nv)
    vl0[1] = l0[1] + l0[2]/2
    vl0[2:nv-1] .= (view(l0, 2:ne-2) .+ view(l0, 3:ne-1))/2
    vl0[nv] = l0[ne-1]/2 + l0[ne]

    # vl0 .= [l0[1]; l0[2:end-1]/2]' + [l0[2:end-1]/2; l0[end]]'
    cache_s = reshape(view(cache, 1:ne), 1, ne)
    cache_t = reshape(view(cache, 1:nv), 1, nv)
    cache_b = reshape(view(cache, 1:2*nv), 2, nv)

    cache_s .= k./l0.*(l .- l0).^2
    Es = 1/2*sum(cache_s)
    cache_t .= β./vl0.*(τ .- τ0).^2
    Et = 1/2*sum(cache_t)
    cache_b .= B./vl0.*(κ .- κ0).^2
    Eb = 1/2*sum(cache_b)

    Es + Et + Eb
end

function gravitational_energy(x, m, g, cache)
    nv = size(x,2)
    tmp = reshape(view(cache, 1:nv-1), 1, nv-1)

    tmp .= view(x,3:3,1:nv-1).*m.*g
    E1 = sum(tmp)
    tmp .= view(x,3:3,2:nv).*m.*g
    E2 = sum(tmp)

    return (E1 + E2)/2
end

# ------------------------------------------------------------------------------
# Overload for convenience
# ------------------------------------------------------------------------------
function edges!(e, r::basic_rod)
    edges!(e, r.x)
end

function tangents!(e, r::basic_rod)
    tangents!(e, r.x)
end

function rod_update!(r::basic_rod, Δr::rod_delta, cache)
    rod_update!(r.x, r.d, Δr.Δx, Δr.Δθ, cache)
end

function full_kinematics!(s::rod_strains, r::basic_rod, cache)
    full_kinematics!(s.l, s.κ, s.τ, r.x, r.d, cache)
end

function elastic_energy(ref_strains::rod_strains, strains::rod_strains,
                        rod_props::rod_properties, cache)
    l0, κ0, τ0 = ref_strains.l, ref_strains.κ, ref_strains.τ
    l, κ, τ = strains.l, strains.κ, strains.τ
    k, B, β = rod_props.k, rod_props.B, rod_props.β
    elastic_energy(l0,κ0,τ0, l,κ,τ, k,B,β, cache)
end
