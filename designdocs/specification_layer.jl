using DiscreteElasticRods
DER = DiscreteElasticRods

segments = [10, 10]
bcs = Vector{DER.rod_bc}()
push!(bcs, DER.free_bc(1,0))
push!(bcs, DER.static_clamp_bc(1,1))
push!(bcs, DER.free_bc(2,0))
push!(bcs, DER.static_clamp_bc(2,1))

nq, param_2_rod = DER.get_specification_layer(segments, bcs)
q = rand(80)

param_2_rod(q)

using Zygote
out, back = Zygote.pullback(param_2_rod, rand(nq))

back(out)
