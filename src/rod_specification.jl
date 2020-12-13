## A class to hold them all!
# struct rod_problem
#     ns::Vector{Int}
#     bc::Vector{rod_bc}
# end

"""
    discretize_span(sspan, ns)

Return the vertex and edge locations (`sv` and `se`) for `sspan` divided into `ns` segments.

# Examples
```julia-repl
julia> sv, se = discretize_span([0,1], 10)
```
"""
function discretize_span(sspan, ns)
    ds = (sspan[2] - sspan[1])/ns
    sv = [sspan[1]; LinRange(sspan[1]+ds/2, sspan[2]-ds/2, ns); sspan[2]]
    se = LinRange(sspan[1], sspan[2], ns+1)
    sv, se
end

"""
    straight_rod(p1, p2, d, N)

Returns a rod with endpoints `p1`, `p2` and director `d` with `N` segments.

If `d` has a component parallel to `p2 - p1`, it is projected out.

"""
function straight_rod(p1::Array{T,1}, p2::Array{T,1}, d::Array{T,1}, N::Int) where T

    # X coordinates
    L = norm(p2-p1)
    t = (p2-p1)/L
    sv, ~ = discretize_span([0,L], N)
    xmat = p1 .+ t*sv'

    # Directors
    d_projected = d - dot(d, t)*t
    d_projected /= norm(d_projected)
    dmat = repeat(d_projected, 1, N+1)

    basic_rod(xmat, dmat)
end

# Helper for continuous_rod
using OrdinaryDiffEq
function bishop_system(vec, p, s)
    # κ1_fun, κ2_fun, τ_fun = p
    #κ1, κ2, τ = κ1_fun(s), κ2_fun(s), τ_fun(s)
    κ1, κ2, τ = p[1](s), p[2](s), p[3](s)

    x = @view vec[1:3]
    t = @view vec[4:6]
    d1 = @view vec[7:9]
    d2 = cross(t, d1)

    [t; κ1*d1 + κ2*d2; -κ1*t + τ*d2]
end

"""
    continuous_rod(x0, t0, d0, κ1, κ2, τ, s)

Returns two functions x and d defining the centerline and directors of a rod.

These are determined by solving the ODE which defines a Bishop frame with
curvatures `κ1(s)`, `κ2(s)`, and `τ(s)`. The integration starts from initial conditions
`x0`, `t0`, and `d0`, and covers length `s = [s1, s2]`


# Examples
```julia-repl
julia> xfun, dfun = continuous_rod([0,0,0], [1,0,0], [0,0,1],
                                      s -> 2, s -> 1, s -> s*(1-s), [0,1])
```
"""
 function continuous_rod(x0, t0, d0, κ1::Function, κ2::Function, τ::Function, s)
    p = (κ1, κ2, τ)
    prob = ODEProblem(bishop_system, [x0;t0;d0], s, p)
    sol = solve(prob, Tsit5(), abs_tol = 1e-8, rel_tol = 1e-8)

    xfun(s) = sol(s)[1:3]
    dfun(s) = sol(s)[7:9]
    xfun, dfun
 end

 """
    discrete_rod(xfun, dfun, s, N)

Return a rod with centerline and directors approximating xfun(s) and dfun(s)

Directors and centerline are sampled from the continous curve, and then
any tangential component of the directors is removed by projection.

# Examples
```julia-repl
julia> r = discrete_rod([0,0,0], [1,0,0], [0,0,1], s -> 2, s -> 1, s -> s*(1-s), [0,1])
```
"""
function discrete_rod(xfun::Function, dfun::Function, s, N::Int)

    # Sample Directors
    sv, se = discretize_span(s, N)
    x = Matrix{Float64}(undef, 3, N+2)
    d = Matrix{Float64}(undef, 3, N+1)
    for i in 1:N+2; x[:,i] = xfun(sv[i]); end
    for i in 1:N+1; d[:,i] = dfun(se[i]); end

    # Project and normalize directors
    t = x[:,2:end] - x[:,1:end-1]
    t = t./norm3(t)
    d = d .- dot3(d,t).*t
    d = d./norm3(d)

    basic_rod(x,d)
 end

"""
    elliptical_stiffness(E, ν, a, b)

Return the elastic properties of a rod with elliptical section.

# Arguments
- `E`: Elastic Modulus
- `ν`: Poisson Ratio
- `a`, `b`: Radii of the cross section
"""
function elliptical_stiffness(E, ν, a, b)
    G = E/(2*(1 + ν))
    A = π*a*b
    k = E*A
    B = E*A/4*[a^2, b^2]
    β = G*A*(a^2 + b^2)/4
    scalar_rod_properties(k, B, β)
end
