# ReverseDiff pullbacks for tracing AD

# using ChainRulesCore
# using ReverseDiff
# ChainRulesCore.@scalar_rule(ptransport_inner_coefficient(θ),
#              ptransport_inner_coefficient(θ)^2*sin(θ))
#
#
# ReverseDiff.@grad
# f(x::ReverseDiff.TrackedVector) = ReverseDiff.track(f, x)
# ReverseDiff.@grad function f(x)
#     xv = ReverseDiff.value(x)
#     return dot(xv, xv), Δ -> (Δ * 2 * xv,)
# end

# DiffCache to reduce cache craziness

# DiffCache - how do?
# immutable DiffCache{T<:AbstractArray, S<:AbstractArray}
#     du::T
#     dual_du::S
# end
#
#
# function DiffCache{chunk_size}(T, size, ::Type{Val{chunk_size}})
#     DiffCache(zeros(T, size...), zeros(ForwardDiff.Dual{nothing,T,chunk_size}, size...))
# end
#
# DiffCache(u::AbstractArray) = DiffCache(eltype(u),size(u),Val{ForwardDiff.pickchunksize(length(u))})
# DiffCache(u::AbstractArray, size) = DiffCache(eltype(u),size,Val{ForwardDiff.pickchunksize(length(u))})
# get_tmp{T<:ForwardDiff.Dual}(dc::DiffCache, ::Type{T}) = dc.dual_du
# get_tmp(dc::DiffCache, T) = dc.du

# function fused_update_strains_energy(r::basic_rod, Δr::rod_update, ϵr::rod_strains)
#
#     # Unpack
#     x0, d0 = r
#     Δx, Δθ = Δr
#     l0, κ0, τ0 = ϵr
#
#     # Tangents (Should maybe save these?)
#     t0 = x0[:,2:end] - x0[:,1:end-1]
#     t = t0 + Δx[:,2:end] - Δx[:,1:end-1]
#     t0 = t0./norm3(t0)
#     l = norm3(t)
#     t = t./l
#
#     # Update
#     d1 = ptransport(d0, t0, t)
#     rotate_orthogonal_unit!(d1, t, Δθ)
#     d2 = cross3(t, d1)
#
#     # Curvatures
#     k = 2*cross3(t[:,1:end-1], t[:,2:end])./(1 + dot3(t[:,1:end-1], t[:,2:end]))
#     κ = Matrix{Float64}(undef, 2, length(l)-1)
#     κ[1,:] = dot3(k, (d2[:,1:end-1] + d2[:,2:end])/2)
#     κ[2,:] = -dot3(k, (d1[:,1:end-1] + d1[:,2:end])/2)
#
#     # Twists (could move in part of ptransport to reduce intermediates)
#     τ = (τ0 + Δθ[2:end] - Δθ[1:end-1]
#             + spherical_excess(t0[:,1:end-1], t0[:,2:end], t[:,1:end-1])
#             + spherical_excess(t[:,1:end-1], t0[:,2:end], t[:,2:end]))
#
#     rod_strains(l, κ, τ)
# end
