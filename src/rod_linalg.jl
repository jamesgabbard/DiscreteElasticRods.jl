using LinearAlgebra
using StaticArrays

# ----------------------------------------------------------------------------------
# Overloads for sets of 3-vectors stored as 3xN matrices
# ----------------------------------------------------------------------------------
# Note that cross only works on matrices. To cross 2 vectors, they need
# to be stored as 3x1 matrices.


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

function cross3(a,b)
    c = Matrix{typeof(a[1,1]*b[1,1])}(undef, size(a))
    cross3!(c,a,b)
    c
end

function dot3(a,b)
    sum(a.*b, dims=1)
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

function norm3(a)
    sqrt.(sum(abs2, a, dims=1))
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

PTRANSPORT_TOL = 0.06

function ptransport_inner_coefficient(θ)
    c = cos(θ)
    s = sin(θ)
    θ > PTRANSPORT_TOL ? (1-c)/s^2 : 1/2 + (1-c)/4 + (1-c)^2/8*(1 + s^2/4)
end

function ptransport_inner_coefficient(s2,c)
    s2 > PTRANSPORT_TOL^2 ? (1-c)/s2 : 1/2 + (1-c)/4 + (1-c)^2/8*(1 + s2/4)
end

function ptransport_inner_coefficient!(out,s2,c)
    out = s2 > PTRANSPORT_TOL^2 ? (1-c)/s2 : 1/2 + (1-c)/4 + (1-c)^2/8*(1 + s2/4)
end

function ptransport(v, t1, t2)
    X = cross3(t1,t2);
    s2 = dot3(X,X);
    c = dot3(t1,t2);
    ptic = ptransport_inner_coefficient.(s2,c)
    c.*v .+ cross3(X,v) .+ ptic.*dot3(X,v).*X
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
function rotate_orthogonal_unit(v, t, θ)
    cos.(θ).*v .+ sin.(θ).*cross3(t,v)
end

function rotate_orthogonal_unit!(v, t, θ)
    c = cos.(θ)
    v .= cos.(θ).*v .+ sin.(θ).*cross3(t,v)
end

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
