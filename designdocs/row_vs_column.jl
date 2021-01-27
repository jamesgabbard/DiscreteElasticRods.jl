# A script that compares row storage to column storage for rods
# Also, a good reference for the most highly optimized kinematics
# routines I could come up with for each storage pattern.
# Unsurprisingly, row storage is never slower, and often 2-3x faster
# due to better memory access patterns.

using BenchmarkTools
N = 500

# ------------------------------------------------------------------------------
# Cross Products (Rows Win)
# ------------------------------------------------------------------------------
function cross3_col!(c,a,b)
    @inbounds for i = 1:size(c,2)
        c[1,i] = a[2,i]*b[3,i] - a[3,i]*b[2,i]
        c[2,i] = a[3,i]*b[1,i] - a[1,i]*b[3,i]
        c[3,i] = a[1,i]*b[2,i] - a[2,i]*b[1,i]
    end
end

function cross3_row!(c,a,b)
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

a = rand(3,N)
b = rand(3,N)
c = rand(3,N)
@btime cross3_col!($c,$a,$b)

a = rand(N,3)
b = rand(N,3)
c = rand(N,3)
@btime cross3_row!($c,$a,$b)

# ------------------------------------------------------------------------------
# Dot Products (Rows Win)
# ------------------------------------------------------------------------------
function dot3_col!(c,a,b)
    @inbounds for i = 1:length(c)
        c[i] = a[1,i]*b[1,i] + a[2,i]*b[2,i] + a[3,i]*b[3,i]
    end
end

function dot3_row!(c,a,b)
    @inbounds for i = 1:length(c)
        c[i] = a[i,1]*b[i,1] + a[i,2]*b[i,2] + a[i,3]*b[i,3]
    end
end

a = rand(3,N)
b = rand(3,N)
c = rand(1,N)
@btime dot3_col!($c,$a,$b)

a = rand(N,3)
b = rand(N,3)
c = rand(N,1)
@btime dot3_row!($c,$a,$b)

# ------------------------------------------------------------------------------
# Normalizing In Place (rows win, by less)
# ------------------------------------------------------------------------------
function norm3_col!(a)
    @inbounds for i = 1:size(a,2)
        tmp = sqrt(a[1,i]*a[1,i] + a[2,i]*a[2,i] + a[3,i]*a[3,i])
        a[1,i] /= tmp
        a[2,i] /= tmp
        a[3,i] /= tmp
    end
end

function norm3_row!(a)
    @inbounds for i = 1:size(a,1)
        tmp = sqrt(a[i,1]*a[i,1] + a[i,2]*a[i,2] + a[i,3]*a[i,3])
        a[i,1] /= tmp
        a[i,2] /= tmp
        a[i,3] /= tmp
    end
end

a = rand(3,N)
@btime norm3_col!($a)

a = rand(N,3)
@btime norm3_row!($a)

# ------------------------------------------------------------------------------
# Norms out of Place (rows win, by alot)
# ------------------------------------------------------------------------------
function norm3_col!(out, a)
    @inbounds for i = 1:length(out)
        out[i] = sqrt(a[1,i]*a[1,i] + a[2,i]*a[2,i] + a[3,i]*a[3,i])
    end
end

function norm3_row!(out, a)
    @inbounds for i = 1:length(out)
        out[i] = sqrt(a[i,1]*a[i,1] + a[i,2]*a[i,2] + a[i,3]*a[i,3])
    end
end

a = rand(3,N)
out = rand(1,N)
@btime norm3_col!($out,$a)

a = rand(N,3)
out = rand(N,1)
@btime norm3_row!($out, $a)

# ------------------------------------------------------------------------------
# Edges (rows win, off the charts)
# ------------------------------------------------------------------------------
function edges_col!(e, x)
    @inbounds for i = 1:size(e,2)
        e[1,i] = x[1,i+1] - x[1,i]
        e[2,i] = x[2,i+1] - x[2,i]
        e[3,i] = x[3,i+1] - x[3,i]
    end
end

function edges_row!(e, x)
    nv = size(x, 1)
    e .= view(x, 2:nv, :) .- view(x, 1:nv-1, :)
end


function tangents_col!(t, x)
    edges_col!(t,x)
    norm3_col!(t)
end

function tangents_row!(t, x)
    edges_row!(t,x)
    norm3_row!(t)
end

a = rand(3,N)
e = rand(3,N-1)
@btime edges_col!($e,$a)
@btime tangents_col!($e,$a)

a = rand(N,3)
e = rand(N-1,3)
@btime edges_row!($e,$a)
@btime tangents_row!($e,$a)

# ------------------------------------------------------------------------------
# Parallel Transport (Example, 3x speedup with row storage)
# ------------------------------------------------------------------------------
const PTRANSPORT_TOL2 = 0.00036
function ptransport_inner_coefficient!(out, s2, c)
    out .= 1/2 .+ (1 .- c)./4 .+ (1 .- c).^2 .*(1 .+ s2./4)./8
    @inbounds for i = 1:length(out)
        if s2[i] > PTRANSPORT_TOL2
            out[i] = (1 - c[i])/s2[i]
        end
    end
end

function ptransport_col!(v2, v1, t1, t2, cache)
    X = view(cache, 1:3, :)
    s2 = view(cache, 4:4, :)
    c = view(cache, 5:5, :)
    ptic = view(cache, 6:6, :)

    cross3_col!(X, t1, t2)
    dot3_col!(s2, X, X)
    dot3_col!(c, t1, t2)
    ptransport_inner_coefficient!(ptic,s2,c)

    dXv = view(cache, 4:4, :) # Overwrites s2
    dot3_col!(dXv,X,v1)
    cross3_col!(v2,X,v1)
    v2 .+= c.*v1 .+ ptic.*dXv.*X
end

function ptransport_row!(v2, v1, t1, t2, cache)
    X = view(cache, :, 1:3)
    s2 = view(cache, :, 4)
    c = view(cache, :, 5)
    ptic = view(cache, :, 6)

    cross3_row!(X, t1, t2)
    dot3_row!(s2, X, X)
    dot3_row!(c, t1, t2)
    ptransport_inner_coefficient!(ptic,s2,c)

    dXv = view(cache, :, 4) # Overwrites s2
    dot3_row!(dXv,X,v1)
    cross3_row!(v2,X,v1)
    v2 .+= c.*v1 .+ ptic.*dXv.*X
end

v1 = rand(3,N)
v2 = similar(v1)
t1 = rand(3,N)
t2 = rand(3,N)
cache = zeros(6,N)
@btime ptransport_col!($v2,$v1,$t1,$t2,$cache)

v1 = rand(N,3)
v2 = similar(v1)
t1 = rand(N,3)
t2 = rand(N,3)
cache = zeros(N,6)
@btime ptransport_row!($v2,$v1,$t1,$t2,$cache)

# ------------------------------------------------------------------------------
# Rotation (Example, only minor speedup)
# ------------------------------------------------------------------------------
function rotate_orthogonal_unit_col!(v, t, θ, cache)
    #ne = length(θ)
    #cache = reshape(view(cacheraw,1:5*ne), 5, ne)

    tmp = view(cache, 1:3, :)
    c = view(cache, 4:4, :)
    s = view(cache, 5:5, :)

    c .= cos.(θ)
    s .= sin.(θ)
    cross3_col!(tmp, t, v)
    v .= c.*v .+ s.*tmp
end

function rotate_orthogonal_unit_row!(v, t, θ, cache)
    tmp = view(cache, :, 1:3)
    c = view(cache, :, 4:4)
    s = view(cache, :, 5:5)

    c .= cos.(θ)
    s .= sin.(θ)
    cross3_row!(tmp, t, v)
    v .= c.*v .+ s.*tmp
end

v = rand(3,N)
t = rand(3,N)
norm3_col!(t)
θ = rand(1,N)
cache = zeros(5,N)
@btime rotate_orthogonal_unit_col!($v, $t, $θ, $cache)

v = rand(N,3)
t = rand(N,3)
norm3_row!(t)
θ = rand(N,1)
cache = zeros(N,5)
@btime rotate_orthogonal_unit_row!($v, $t, $θ, $cache)

# ------------------------------------------------------------------------------
# Rod Update (2x Speedup for Rows)
# ------------------------------------------------------------------------------
function rod_update_col!(x, d, Δx, Δθ, cache)
    t0 = view(cache, 1:3, :)
    t = view(cache, 4:6, :)
    d0 = view(cache, 7:9, :)
    cache_pt = view(cache, 10:15, :)
    cache_rt = view(cache, 10:14, :)

    tangents_col!(t0, x)
    x .+= Δx
    tangents_col!(t, x)

    d0 .= d
    ptransport_col!(d, d0, t0, t, cache_pt)
    rotate_orthogonal_unit_col!(d, t, Δθ, cache_pt)
end

function rod_update_row!(x, d, Δx, Δθ, cache)
    t0 = view(cache, :, 1:3)
    t = view(cache, :, 4:6)
    d0 = view(cache, :, 7:9)
    cache_pt = view(cache, :, 10:15)
    cache_rt = view(cache, :, 10:14)

    tangents_row!(t0, x)
    x .+= Δx
    tangents_row!(t, x)

    d0 .= d
    ptransport_row!(d, d0, t0, t, cache_pt)
    rotate_orthogonal_unit_row!(d, t, Δθ, cache_pt)
end

x = rand(3,N)
d = rand(3,N-1)
norm3_col!(d)
Δx = rand(3,N)
Δθ = rand(1,N-1)
cache = zeros(15,N-1)
@btime rod_update_col!($x,$d,$Δx,$Δθ,$cache)

x = rand(N,3)
d = rand(N-1,3)
norm3_row!(d)
Δx = rand(N,3)
Δθ = rand(N-1,1)
cache = zeros(N-1,15)
@btime rod_update_row!($x,$d,$Δx,$Δθ,$cache)

# ------------------------------------------------------------------------------
# Full Kinematics Update (Factor of 2.5 for Row Storage)
# ------------------------------------------------------------------------------
function full_kinematics_col!(l, κ, τ, x, d1, caches)

    # Unpack and divy up caches
    cache1, cache2 = caches
    t = view(cache1, 1:3, :)
    d2 = view(cache1, 4:6, :)
    k = view(cache2, 1:3, :)
    tmp1 = view(cache2, 4:4, :)
    tmp3 = view(cache2, 4:6, :)
    pt_cache = view(cache2, 1:6, :) # overrides k, tmp
    d1trans = view(cache2, 7:9, :)
    cosτ = view(cache2, 1:1, :)
    sinτ = view(cache2, 2:2, :)

    # Lengths
    edges_col!(t, x)
    norm3_col!(l, t)
    t ./= l
    cross3_col!(d2, t, d1)

    # Break tangents and directors into sets
    nv = size(x,2)
    t1 = view(t,:,1:nv-2)
    t2 = view(t,:,2:nv-1)
    d1l = view(d1,:,1:nv-2)
    d1r = view(d1,:,2:nv-1)
    d2l = view(d2,:,1:nv-2)
    d2r = view(d2,:,2:nv-1)

    # Curvature
    cross3_col!(k, t1, t2)
    dot3_col!(tmp1, t1, t2)
    k .*= 2.0./(1.0 .+ tmp1)
    tmp3 .= k.*(d2l .+ d2r)./2
    sum!(view(κ, 1:1, :), tmp3)
    tmp3 .= -1.0.*k.*(d1l .+ d1r)./2
    sum!(view(κ, 2:2, :), tmp3)

    # # Twists (could move in part of ptransport to reduce intermediates)
    ptransport_col!(d1trans, d1l, t1, t2, pt_cache)
    dot3_col!(cosτ, d1r, d1trans)
    dot3_col!(sinτ, d2r, d1trans)
    τ .= -1.0.*atan.(sinτ, cosτ)
end

function full_kinematics_row!(l, κ, τ, x, d1, caches)

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
    edges_row!(t, x)
    norm3_row!(l, t)
    t ./= l
    cross3_row!(d2, t, d1)

    # Break tangents and directors into sets
    nv = size(x,1)
    t1 = view(t, 1:nv-2, :)
    t2 = view(t, 2:nv-1, :)
    d1l = view(d1, 1:nv-2, :)
    d1r = view(d1, 2:nv-1, :)
    d2l = view(d2, 1:nv-2, :)
    d2r = view(d2, 2:nv-1, :)

    # Curvature
    cross3_row!(k, t1, t2)
    dot3_row!(tmp1, t1, t2)
    k .*= 2.0./(1.0 .+ tmp1)
    tmp3 .= k.*(d2l .+ d2r)./2
    sum!(view(κ, :, 1:1), tmp3)
    tmp3 .= -1.0.*k.*(d1l .+ d1r)./2
    sum!(view(κ, :, 2:2), tmp3)

    # # Twists (could move in part of ptransport to reduce intermediates)
    ptransport_row!(d1trans, d1l, t1, t2, pt_cache)
    dot3_row!(cosτ, d1r, d1trans)
    dot3_row!(sinτ, d2r, d1trans)
    τ .= -1.0.*atan.(sinτ, cosτ)
end

x = rand(3,N)
d = rand(3,N-1)
norm3_col!(d)
l = zeros(1, N-1)
κ = zeros(2, N-2)
τ = zeros(1, N-2)
caches = (zeros(6, N-1), zeros(9, N-2))
@btime full_kinematics_col!(l, κ, τ, x, d, caches)

x = rand(N,3)
d = rand(N-1,3)
norm3_row!(d)
l = zeros(N-1)
κ = zeros(N-2, 2)
τ = zeros(N-2)
caches = (zeros(N-1,6), zeros(N-2,9))
@btime full_kinematics_row!(l, κ, τ, x, d, caches)

# ------------------------------------------------------------------------------
# Elastic Energy
# ------------------------------------------------------------------------------
# cache is (3 x ne) or larger
function elastic_energy_col(l0,κ0,τ0, l,κ,τ, k,B,β, cache)

    ne = length(l)
    nv = ne - 1
    cache_s = reshape(view(cache, 1:ne), 1, ne)
    cache_s .= k./l0.*(l .- l0).^2
    Es = 1/2*sum(cache_s)

    # Reference voronoi lengths
    vl0 = view(cache, 1:1, 1:nv)
    vl0[1] = l0[1] + l0[2]/2
    vl0[2:nv-1] .= (view(l0, 2:ne-2) .+ view(l0, 3:ne-1))./2
    vl0[nv] = l0[ne-1]/2 + l0[ne]

    cache_t = view(cache, 2:2, 1:nv)
    cache_t .= β./vl0.*(τ .- τ0).^2
    Et = 1/2*sum(cache_t)

    cache_b = view(cache, 2:3, 1:nv)
    cache_b .= B./vl0.*(κ .- κ0).^2
    Eb = 1/2*sum(cache_b)

    Es + Et + Eb
end

# cache is (ne x 3) or larger
function elastic_energy_row(l0,κ0,τ0, l,κ,τ, k,B,β, cache)

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

l0 = rand(1, N-1)
κ0 = rand(2, N-2)
τ0 = rand(1, N-2)
l = rand(1, N-1)
κ = rand(2, N-2)
τ = rand(1, N-2)
k = rand()
B = rand(2,1)
β = rand()
cache = zeros(3,N-1)

@btime elastic_energy_col(l, κ, τ, l0, κ0, τ0, k ,B, β, cache)

l0 = rand(N-1)
κ0 = rand(N-2,2)
τ0 = rand(N-2)
l = rand(N-1)
κ = rand(N-2,2)
τ = rand(N-2)
k = rand()
B = rand(1, 2)
β = rand()
cache = zeros(N-1,3)

@btime elastic_energy_row(l, κ, τ, l0, κ0, τ0, k ,B, β, cache)

# ------------------------------------------------------------------------------
# Gravitational Energy
# ------------------------------------------------------------------------------
# cache is at least 1xne
function gravitational_energy_col(x, m, g, cache)
    nv = size(x,2)
    tmp = reshape(view(cache, 1, 1:nv-1), 1, nv-1)
    tmp .= (view(x,3:3,1:nv-1) .+ view(x,3:3,2:nv)).*m
    return sum(tmp)*g/2
end

# cache is at least 1xne
function gravitational_energy_row(x, m, g, cache)
    nv = size(x,1)
    tmp = view(cache, 1:nv-1)
    tmp .= (view(x, 1:nv-1, 3) .+ view(x, 2:nv, 3)).*m
    return sum(tmp)*g/2
end

x = rand(3, N)
m = rand(1, N-1)
g = 10.0
cache = zeros(1, N-1)
@btime gravitational_energy_col(x, m, g, cache)

x = rand(N, 3)
m = rand(N-1)
cache = zeros(N-1)
@btime gravitational_energy_row(x, m, g, cache)
