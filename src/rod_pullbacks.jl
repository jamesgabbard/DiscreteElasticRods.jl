using ChainRulesCore
using ReverseDiff
ChainRulesCore.@scalar_rule(ptransport_inner_coefficient(θ),
             ptransport_inner_coefficient(θ)^2*sin(θ))


ReverseDiff.@grad
f(x::ReverseDiff.TrackedVector) = ReverseDiff.track(f, x)
ReverseDiff.@grad function f(x)
    xv = ReverseDiff.value(x)
    return dot(xv, xv), Δ -> (Δ * 2 * xv,)
end
