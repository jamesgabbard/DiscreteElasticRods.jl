# Kinematics

Julia's main source-to-source reverse mode automatic differentiation package
(Zygote.jl) does not allow for mutation of array elements. At the same time,
the dual-number based forward-mode automatic differentiation in ForwardDiff.jl
is most efficient when used with non-allocating code. As a compromise, the
kinematics functions in DiscreteElasticRods.jl come in both non-mutating and
non-allocating forms. The latter are distinguished by a trailing `!`, and
usually require a cache argument.

# Basic 3-Vector Operations
```@docs
DiscreteElasticRods.cross3
DiscreteElasticRods.dot3
DiscreteElasticRods.norm3
```

# Low-Level Rod Operations
```@docs
DiscreteElasticRods.edges
DiscreteElasticRods.tangents
DiscreteElasticRods.rotate_orthogonal_unit
DiscreteElasticRods.ptransport
```

# High-Level Rod Operations
```@docs
DiscreteElasticRods.full_kinematics
DiscreteElasticRods.elastic_energy
DiscreteElasticRods.gravitational_energy
```
