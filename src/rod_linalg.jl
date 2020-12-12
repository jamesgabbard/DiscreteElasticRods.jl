using LinearAlgebra

#----------------------------------------------------------------------------------
# Overloads for sets of 3-vectors stored as 3xN matrices
#----------------------------------------------------------------------------------
function cross3!(a::AbstractArray, b::AbstractArray)

    # Size Checks
    @assert size(a,2) == size(b,2)
    @assert size(a,1) == 3
    @assert size(b,1) == 3

    # Action
    @inbounds for i = 1:size(a,2)
        a[:,i] = cross(a[:,i], b[:,i])
    end   
end

function cross3(a::AbstractArray, b::AbstractArray)

    # Size Checks
    @assert size(a,2) == size(b,2)
    @assert size(a,1) == 3
    @assert size(b,1) == 3

    # Action
    c = similar(a)
    @inbounds for i = 1:size(a,2)
        c[:,i] = cross(a[:,i], b[:,i])
    end
    c   
end

function dot3(a,b)
    sum(a.*b, dims=1)
end

function norm3(a)
    sqrt.(sum(a.*a, dims=1))
end

#----------------------------------------------------------------------------------
# Parallel Transport (Vectorized)
#----------------------------------------------------------------------------------
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

using ChainRulesCore
@scalar_rule(ptransport_inner_coefficient(θ),
             ptransport_inner_coefficient(θ)^2*sin(θ))

function ptransport(v, t1, t2)
    X = cross3(t1,t2);
    s = norm3(X);
    c = dot3(t1,t2);
    θ = atan.(s, c)
    ptic = ptransport_inner_coefficient.(θ)
    c.*v + cross3(X,v) + ptic.*dot3(X,v).*X
end


#----------------------------------------------------------------------------------
# Rotate about Unit Vector (Vectorized)
#----------------------------------------------------------------------------------
# Again, Rodriguez rotation formula. No numerical instability.
function rotate(v, t, θ)
    cos.(θ).*v + sin.(θ).*cross3(t,v) + (1 .- cos.(θ)).*dot3(t,v).*v 
end

function rotate!(v, t, θ)
    v = rotate(v, t, θ) 
end