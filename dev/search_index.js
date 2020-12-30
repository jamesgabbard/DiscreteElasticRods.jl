var documenterSearchIndex = {"docs":
[{"location":"specification/#Problem-Specification","page":"Problem Specification","title":"Problem Specification","text":"","category":"section"},{"location":"specification/","page":"Problem Specification","title":"Problem Specification","text":"These functions are used for constructing rods, specifying boundary conditions, and specifying material properties.","category":"page"},{"location":"specification/","page":"Problem Specification","title":"Problem Specification","text":"DiscreteElasticRods.elliptical_stiffness\nDiscreteElasticRods.straight_rod\nDiscreteElasticRods.discretize_span\nDiscreteElasticRods.discrete_rod\nDiscreteElasticRods.continuous_rod","category":"page"},{"location":"specification/#DiscreteElasticRods.elliptical_stiffness","page":"Problem Specification","title":"DiscreteElasticRods.elliptical_stiffness","text":"elliptical_stiffness(E, ν, a, b)\n\nReturn the elastic properties of a rod with elliptical section.\n\nArguments\n\nE: Elastic Modulus\nν: Poisson Ratio\na, b: Radii of the cross section\n\n\n\n\n\n","category":"function"},{"location":"specification/#DiscreteElasticRods.straight_rod","page":"Problem Specification","title":"DiscreteElasticRods.straight_rod","text":"straight_rod(p1, p2, d, N)\n\nReturns a rod with endpoints p1, p2 and director d with N segments.\n\nIf d has a component parallel to p2 - p1, it is projected out.\n\n\n\n\n\n","category":"function"},{"location":"specification/#DiscreteElasticRods.discretize_span","page":"Problem Specification","title":"DiscreteElasticRods.discretize_span","text":"discretize_span(sspan, ns)\n\nReturn the vertex and edge locations (sv and se) for sspan divided into ns segments.\n\nExamples\n\njulia> sv, se = discretize_span([0,1], 10)\n\n\n\n\n\n","category":"function"},{"location":"specification/#DiscreteElasticRods.discrete_rod","page":"Problem Specification","title":"DiscreteElasticRods.discrete_rod","text":"discrete_rod(xfun, dfun, s, N)\n\nReturn a rod with centerline and directors approximating xfun(s) and dfun(s)\n\nDirectors and centerline are sampled from the continous curve, and then any tangential component of the directors is removed by projection.\n\nExamples\n\njulia> r = discrete_rod([0,0,0], [1,0,0], [0,0,1], s -> 2, s -> 1, s -> s*(1-s), [0,1])\n\n\n\n\n\n","category":"function"},{"location":"specification/#DiscreteElasticRods.continuous_rod","page":"Problem Specification","title":"DiscreteElasticRods.continuous_rod","text":"continuous_rod(x0, t0, d0, κ1, κ2, τ, s)\n\nReturns two functions x and d defining the centerline and directors of a rod.\n\nThese are determined by solving the ODE which defines a Bishop frame with curvatures κ1(s), κ2(s), and τ(s). The integration starts from initial conditions x0, t0, and d0, and covers length s = [s1, s2]\n\nExamples\n\njulia> xfun, dfun = continuous_rod([0,0,0], [1,0,0], [0,0,1],\n                                      s -> 2, s -> 1, s -> s*(1-s), [0,1])\n\n\n\n\n\n","category":"function"},{"location":"#DiscreteElasticRods.jl","page":"Home","title":"DiscreteElasticRods.jl","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for DiscreteElasticRods.jl","category":"page"},{"location":"kinematics/#Kinematics","page":"Kinematics","title":"Kinematics","text":"","category":"section"},{"location":"kinematics/","page":"Kinematics","title":"Kinematics","text":"Julia's main source-to-source reverse mode automatic differentiation package (Zygote.jl) does not allow for mutation of array elements. At the same time, the dual-number based forward-mode automatic differentiation in ForwardDiff.jl is most efficient when used with non-allocating code. As a compromise, the kinematics functions in DiscreteElasticRods.jl come in both non-mutating and non-allocating forms. The latter are distinguished by a trailing !, and usually require a cache argument.","category":"page"},{"location":"kinematics/#Basic-3-Vector-Operations","page":"Kinematics","title":"Basic 3-Vector Operations","text":"","category":"section"},{"location":"kinematics/","page":"Kinematics","title":"Kinematics","text":"DiscreteElasticRods.cross3\nDiscreteElasticRods.dot3\nDiscreteElasticRods.norm3","category":"page"},{"location":"kinematics/#DiscreteElasticRods.cross3","page":"Kinematics","title":"DiscreteElasticRods.cross3","text":"    cross3(a,b)\n\nCross products of the columns of two 3 x N arrays. Output is also 3 x N\n\n\n\n\n\n","category":"function"},{"location":"kinematics/#DiscreteElasticRods.dot3","page":"Kinematics","title":"DiscreteElasticRods.dot3","text":"    dot3(a,b)\n\nDot products of the columns of two 3 x N arrays. Output is 1 x N\n\n\n\n\n\n","category":"function"},{"location":"kinematics/#DiscreteElasticRods.norm3","page":"Kinematics","title":"DiscreteElasticRods.norm3","text":"    norm3(a,b)\n\nNorms of the columns of two 3 x N arrays. Output is 1 x N\n\n\n\n\n\n","category":"function"},{"location":"kinematics/#Low-Level-Rod-Operations","page":"Kinematics","title":"Low-Level Rod Operations","text":"","category":"section"},{"location":"kinematics/","page":"Kinematics","title":"Kinematics","text":"DiscreteElasticRods.edges\nDiscreteElasticRods.tangents\nDiscreteElasticRods.rotate_orthogonal_unit\nDiscreteElasticRods.ptransport","category":"page"},{"location":"kinematics/#DiscreteElasticRods.edges","page":"Kinematics","title":"DiscreteElasticRods.edges","text":"    edges(x)\n\nEdges of a centerline x, stored as a 3 x N array. Output is 3 x N-1\n\n\n\n\n\n","category":"function"},{"location":"kinematics/#DiscreteElasticRods.tangents","page":"Kinematics","title":"DiscreteElasticRods.tangents","text":"    tangents(x)\n\nTangents of a centerline x, stored as a 3 x N array. Output is 3 x N-1\n\n\n\n\n\n","category":"function"},{"location":"kinematics/#DiscreteElasticRods.rotate_orthogonal_unit","page":"Kinematics","title":"DiscreteElasticRods.rotate_orthogonal_unit","text":"    rotate_orthogonal_unit(v, t, θ)\n\nRotate unit vectors v around axes t1 by angles θ.\n\nv and t are (3 x N), θ is (1 x N).\n\nThe columns v[:,i] and t[:,i] must be orthogonal unit vectors. The function uses the Rodriguez rotation formula speciailized to this case.\n\n\n\n\n\n","category":"function"},{"location":"kinematics/#DiscreteElasticRods.ptransport","page":"Kinematics","title":"DiscreteElasticRods.ptransport","text":"    ptransport(v, t1, t2)\n\nParallel transport of vectors v from tangents t1 to tangents 't2'.\n\nAll inputs are (3 x N). Each column of t1 and t2 must have unit norm.\n\nUses the Rodriguez rotation formula. There is a singularity in this definition when t1 == t2, which is handled by switching to an small-θ expansion (see ptransportinnercoefficient).\n\n\n\n\n\n","category":"function"},{"location":"kinematics/#High-Level-Rod-Operations","page":"Kinematics","title":"High-Level Rod Operations","text":"","category":"section"},{"location":"kinematics/","page":"Kinematics","title":"Kinematics","text":"DiscreteElasticRods.full_kinematics\nDiscreteElasticRods.elastic_energy\nDiscreteElasticRods.gravitational_energy","category":"page"},{"location":"kinematics/#DiscreteElasticRods.full_kinematics","page":"Kinematics","title":"DiscreteElasticRods.full_kinematics","text":"    full_kinematics(x, d)\n\nReturns the edge lengths, curvatures, and twists of a rod.\n\nArguments\n\nx: centerline vertices, size (3 x N)\nd: directors, size (3 x N-1). Unit length and orthogonal to edges(x)\n\nReturns\n\nl: edge lengths (1 x N-1)\nκ: curvature normals (2 x N-2), expressed in the director frame\n'τ': scalar twist at each vertex (1 x N-2)\n\nTwists are calculated by parallel transporting the directors across each vertex, and comparing with the original. This avoids the reference twist calculations used in Lestringant, Audoly, and Kochmann (2020)\n\n\n\n\n\n","category":"function"},{"location":"kinematics/#DiscreteElasticRods.elastic_energy","page":"Kinematics","title":"DiscreteElasticRods.elastic_energy","text":"    elastic_energy(l0,κ0,τ0, l,κ,τ, k,B,β)\n\nElastic energy, calculated from stretching, bending, and twisting strains.\n\nArguments\n\nl0, κ0, τ0: reference lengths, curvatures, and twists\nl, κ, τ: current lengths, curvatures, and twists\nk, B, β: stretching, bending, and twisting stiffness.\n\nLengths, curvatures, and twists should be (1 x N-1), (2 x N-2), and (1 x N-2), respectively.\n\nk and β may be scalar or similar to l and τ. The bending stiffness  B may constant isotropic (scalar), constant anisotropic (2 x 1),  or non-constant anisotropic (2 x N-2).\n\n\n\n\n\n","category":"function"},{"location":"kinematics/#DiscreteElasticRods.gravitational_energy","page":"Kinematics","title":"DiscreteElasticRods.gravitational_energy","text":"    gravitational_energy(x, m, g)\n\nGravitational energy, calculated from centerline.\n\nArguments\n\nx: centerline vertices, 3 x N\nm: edges masses, 1 x N-1. Typically ρ*A*l0\ng: magnitude of gravitational acceleration (positive).\n\nGravity is assumed to act in the negative z direction.\n\n\n\n\n\n","category":"function"}]
}