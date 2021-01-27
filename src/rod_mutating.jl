# Reducing allocations as much as possible.
# Included after the non-mutating versions, so that we can
# fall back to a less efficient version where needed

function cross3!(c,a,b)
    a1 = @view a[:,1]
    a2 = @view a[:,2]
    a3 = @view a[:,3]

    b1 = @view b[:,1]
    b2 = @view b[:,2]
    b3 = @view b[:,3]

    c[:,1] .= @. a2*b3 - a3*b2
    c[:,2] .= @. a3*b1 - a1*b3
    c[:,3] .= @. a1*b2 - a2*b1
end

function dot3!(c,a,b)
    @inbounds for i = 1:length(c)
        c[i] = a[i,1]*b[i,1] + a[i,2]*b[i,2] + a[i,3]*b[i,3]
    end
end

function norm3!(a)
    @inbounds for i = 1:size(a,1)
        tmp = sqrt(a[i,1]*a[i,1] + a[i,2]*a[i,2] + a[i,3]*a[i,3])
        a[i,1] /= tmp
        a[i,2] /= tmp
        a[i,3] /= tmp
    end
end

function norm3!(out, a)
    @inbounds for i = 1:length(out)
        out[i] = sqrt(a[i,1]*a[i,1] + a[i,2]*a[i,2] + a[i,3]*a[i,3])
    end
end

function edges!(e, x)
    nv = size(x, 1)
    e .= view(x, 2:nv, :) .- view(x, 1:nv-1, :)
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

const PTRANSPORT_TOL2 = PTRANSPORT_TOL^2
function ptransport_inner_coefficient!(out, s2, c)
    out .= 1/2 .+ (1 .- c)./4 .+ (1 .- c).^2 .*(1 .+ s2./4)./8
    @inbounds for i = 1:length(out)
        if s2[i] > PTRANSPORT_TOL2 #ind[i]
            out[i] = (1 - c[i])/s2[i]
        end
    end
end

# Cache needs to be ne x 6
function ptransport!(v2, v1, t1, t2, cache)
    X = view(cache, :, 1:3)
    s2 = view(cache, :, 4)
    c = view(cache, :, 5)
    ptic = view(cache, :, 6)

    cross3!(X, t1, t2)
    dot3!(s2, X, X)
    dot3!(c, t1, t2)
    ptransport_inner_coefficient!(ptic,s2,c)

    dXv = view(cache, :, 4) # Overwrites s2
    dot3!(dXv,X,v1)
    cross3!(v2,X,v1)
    v2 .+= c.*v1 .+ ptic.*dXv.*X
end

# ------------------------------------------------------------------------------
# Rotate about Unit Vector (Vectorized)
# ------------------------------------------------------------------------------
# Rodriguez rotation formula, specialized to the case where the vector
# and the axis are unit length and orthogonal.

# cache is ne x 5
function rotate_orthogonal_unit!(v, t, θ, cache)
    tmp = view(cache, :, 1:3)
    c = view(cache, :, 4:4)
    s = view(cache, :, 5:5)

    c .= cos.(θ)
    s .= sin.(θ)
    cross3!(tmp, t, v)
    v .= c.*v .+ s.*tmp
end

# Update kinematics given displacement and twist
function rod_update!(x1, d1, x0, d0, Δx, Δθ, cache)
    t0 = view(cache, :, 1:3)
    t1 = view(cache, :, 4:6)
    cache_pt = view(cache, :, 7:12)
    cache_rt = view(cache, :, 7:11)

    tangents!(t0, x0)
    x1 .= x0 .+ Δx
    tangents!(t1, x1)

    ptransport!(d1, d0, t0, t1, cache_pt)
    rotate_orthogonal_unit!(d1, t1, Δθ, cache_pt)
end

allocate_cache(f::typeof(rod_update!), T::Type, ns) = zeros(T, ns+1, 12)

function full_kinematics!(l, κ, τ, x, d1, caches)

    # Unpack and divy up caches
    cache1, cache2 = caches
    t = view(cache1, :, 1:3)
    d2 = view(cache1, :, 4:6)
    k = view(cache2, :, 1:3)
    tmp1 = view(cache2, :, 4:4)
    tmp3 = view(cache2, :, 4:6)
    pt_cache = view(cache2, :, 1:6) # overrides k, tmp
    d1trans = view(cache2, :, 7:9)
    cosτ = view(cache2, :, 1)
    sinτ = view(cache2, :, 2)

    # Lengths
    edges!(t, x)
    norm3!(l, t)
    t ./= l
    cross3!(d2, t, d1)

    # Break tangents and directors into sets
    nv = size(x,1)
    t1 = view(t, 1:nv-2, :)
    t2 = view(t, 2:nv-1, :)
    d1l = view(d1, 1:nv-2, :)
    d1r = view(d1, 2:nv-1, :)
    d2l = view(d2, 1:nv-2, :)
    d2r = view(d2, 2:nv-1, :)

    # Curvature
    cross3!(k, t1, t2)
    dot3!(tmp1, t1, t2)
    k .*= 2.0./(1.0 .+ tmp1)
    tmp3 .= k.*(d2l .+ d2r)./2
    sum!(view(κ, :, 1:1), tmp3)
    tmp3 .= -1.0.*k.*(d1l .+ d1r)./2
    sum!(view(κ, :, 2:2), tmp3)

    # # Twists (could move in part of ptransport to reduce intermediates)
    ptransport!(d1trans, d1l, t1, t2, pt_cache)
    dot3!(cosτ, d1r, d1trans)
    dot3!(sinτ, d2r, d1trans)
    τ .= -1.0.*atan.(sinτ, cosτ)
end

allocate_cache(f::typeof(full_kinematics!), T::Type, ns) =
    (zeros(T, ns+1, 6), zeros(T, ns, 9))

# Elastic Energy
function elastic_energy(l0,κ0,τ0, l,κ,τ, k,B,β, cache)
    ne = length(l)
    nv = ne - 1
    cache_s = view(cache, 1:ne)
    cache_s .= k./l0.*(l .- l0).^2
    Es = 1/2*sum(cache_s)

    # Reference voronoi lengths
    vl0 = view(cache, 1:nv, 1)
    vl0[1] = l0[1] + l0[2]/2
    vl0[2:nv-1] .= (view(l0, 2:ne-2) .+ view(l0, 3:ne-1))./2
    vl0[nv] = l0[ne-1]/2 + l0[ne]

    cache_t = view(cache, 1:nv, 2)
    cache_t .= β./vl0.*(τ .- τ0).^2
    Et = 1/2*sum(cache_t)

    cache_b = view(cache, 1:nv, 2:3)
    cache_b .= B./vl0.*(κ .- κ0).^2
    Eb = 1/2*sum(cache_b)

    Es + Et + Eb
end

allocate_cache(f::typeof(elastic_energy), T::Type, ns) = zeros(T, ns+1, 3)

# cache is at least ne
function gravitational_energy(x, m, g, cache)
    nv = size(x,1)
    tmp = view(cache, 1:nv-1)
    tmp .= (view(x, 1:nv-1, 3) .+ view(x, 2:nv, 3)).*m
    return sum(tmp)*g/2
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

function rod_update!(new::basic_rod, old::basic_rod, Δr::rod_delta, cache)
    rod_update!(new.x, new.d, old.x, old.d, Δr.Δx, Δr.Δθ, cache)
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

# ------------------------------------------------------------------------------
# Overload for convenience
# ------------------------------------------------------------------------------

# immutable DiffCache{T<:AbstractArray, S<:AbstractArray}
#     du::T
#     dual_du::S
# end
# #
# function DiffCache{chunk_size}(T, size, ::Type{Val{chunk_size}})
#     DiffCache(zeros(T, size...), zeros(ForwardDiff.Dual{nothing,T,chunk_size}, size...))
# end
#
# DiffCache(u::AbstractArray) = DiffCache(eltype(u),size(u),Val{ForwardDiff.pickchunksize(length(u))})
# DiffCache(u::AbstractArray, size) = DiffCache(eltype(u),size,Val{ForwardDiff.pickchunksize(length(u))})
# get_tmp{T<:ForwardDiff.Dual}(dc::DiffCache, ::Type{T}) = dc.dual_du
# get_tmp(dc::DiffCache, T) = dc.du
