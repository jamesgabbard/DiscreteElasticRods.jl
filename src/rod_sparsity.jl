# Sparsity Patterns for the Hessian Matrix of rod
function sparse_append!(I, J, rows::AbstractVector{T}, cols::AbstractVector{T}) where T
    oner = ones(T, size(rows))
    onec = ones(T, size(cols))
    append!(I, kron(onec, rows))
    append!(J, kron(cols, oner))
end

function hessian_sparsity(xind, yind, zind, θind) where Ti

    T = eltype(xind)
    nv = length(xind)
    ne = nv-1

    vind = hcat(xind, yind, zind)
    tind = reshape(θind, length(θind), 1)
    I = Vector{T}()
    J = Vector{T}()

    # x - x interactions
    for i = 1:nv, j = max(i-2, 1):min(i+2, nv)
        sparse_append!(I, J, vind[i,:], vind[j,:])
    end

    # x - θ interactions
    for i = 1:nv, j = max(i-2, 1):min(i+1, ne)
        sparse_append!(I, J, vind[i,:], tind[j,:])
    end

    # θ - x interactions
    for i = 1:ne, j = max(i-1, 1):min(i+2, nv)
        sparse_append!(I, J, tind[i,:], vind[j,:])
    end

    # θ - θ interactions
    for i = 1:ne, j = max(i-1, 1):min(i+1, ne)
        sparse_append!(I, J, tind[i,:], tind[j,:])
    end

    V = ones(T, length(I))
    sparse(I, J ,V)
end

function banded_hessian_sparsity(ns)
    xind = 1 .+ 4*(0:ns+1)
    yind = xind .+ 1
    zind = xind .+ 2
    θind = 4 .+ 4*(0:ns)
    hessian_sparsity(xind, yind, zind, θind)
end

function block_hessian_sparsity(ns)
    xind = 1 .+ 3*(0:ns+1)
    yind = xind .+ 1
    zind = xind .+ 2
    θind = 3*(ns+2) .+ (1:ns+1)
    hessian_sparsity(xind, yind, zind, θind)
end

function coordinate_hessian_sparsity(ns)
    nv = ns + 2
    xind = 1:nv
    yind = xind .+ nv
    zind = yind .+ nv
    θind = 3*nv .+ (1:ns+1)
    hessian_sparsity(xind, yind, zind, θind)
end

# using Plots
# spy(banded_hessian_sparsity(ns))
# spy(block_hessian_sparsity(ns))
# spy(coordinate_hessian_sparsity(ns))

# ------------------------------------------------------------------------------
# Convert rod_delta to and from a vector representation
# ------------------------------------------------------------------------------
function banded_storage!(q, dr::rod_delta)
    N = length(dr.Δθ) - 1
    q[1:4:4*N+5] .= @view dr.Δx[:,1]
    q[2:4:4*N+6] .= @view dr.Δx[:,2]
    q[3:4:4*N+7] .= @view dr.Δx[:,3]
    q[4:4:4*N+4] .= @view dr.Δθ[:]
end

function banded_storage!(dr::rod_delta, q)
    N = length(dr.Δθ) - 1
    dr.Δx[:,1] .= @view q[1:4:4*N+5]
    dr.Δx[:,2] .= @view q[2:4:4*N+6]
    dr.Δx[:,3] .= @view q[3:4:4*N+7]
    dr.Δθ[:] .= @view q[4:4:4*N+4]
end

function block_storage!(q, dr::rod_delta)
    N = length(dr.Δθ) - 1
    q[1:3:3*N+4] .= @view dr.Δx[:,1]
    q[2:3:3*N+5] .= @view dr.Δx[:,2]
    q[3:3:3*N+6] .= @view dr.Δx[:,3]
    q[3*N+7:4*N+7] .= @view dr.Δθ[:]
end

function block_storage!(dr::rod_delta, q)
    N = length(dr.Δθ) - 1
    dr.Δx[:,1] .= @view q[1:3:3*N+4]
    dr.Δx[:,2] .= @view q[2:3:3*N+5]
    dr.Δx[:,3] .= @view q[3:3:3*N+6]
    dr.Δθ[:] .= @view q[3*N+7:4*N+7]
end

function coordinate_storage!(q, dr::rod_delta)
    N = length(dr.Δθ) - 1
    q[1:3*(N+2)] .= @view dr.Δx[:]
    q[3*N+7:4*N+7] .= view(dr.Δθ, :, 1)
end

function coordinate_storage!(dr::rod_delta, q)
    N = length(dr.Δθ) - 1
    dr.Δx[:,1] .= @view q[1:(N+2)]
    dr.Δx[:,2] .= @view q[N+3:2*(N+2)]
    dr.Δx[:,3] .= @view q[2*N+5:3*(N+2)]
    dr.Δθ .= @view q[3*N+7:4*N+7]
end

# ------------------------------------------------------------------------------
# Allocating versions of the above
# ------------------------------------------------------------------------------
for _storage = (:banded_storage, :block_storage, :coordinate_storage)
    _storage! = Symbol(_storage, "!")
    @eval begin
        function $(_storage)(dr::rod_delta{T}) where T
            N = length(dr.Δθ) - 1
            q = Vector{T}(undef, 4*N+7)
            $(_storage!)(q, dr)
            return q
        end

        function $(_storage)(q::Vector{T}) where T
            N = div(length(q) - 7, 4)
            dr = allocate_delta(T, N)
            $(_storage!)(dr, q)
            return dr
        end
    end
end

# Check on storage orders
# using BenchmarkTools
# ns = 500
# q = Float64.(1:4*ns+7)
# dr = DER.allocate_delta(Float64, ns)
# @btime DER.banded_storage!($dr, $q)
# @btime DER.banded_storage!($q, $dr)
# @btime DER.block_storage!($dr, $q)
# @btime DER.block_storage!($q, $dr)
# @btime DER.coordinate_storage!($dr, $q)
# @btime DER.coordinate_storage!($q, $dr)
