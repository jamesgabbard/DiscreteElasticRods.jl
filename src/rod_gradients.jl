function grad_elastic_energy(rod::basic_rod, rod_props::rod_properties, ref_strains::rod_strains)

    # unpack
    l0, κ0, τ0 =  ref_strains.l, ref_strains.κ, ref_strains.τ
    k, B, β = rod_props.k, rod_props.B, rod_props.β
    x, d1 = rod.x, rod.d

    # Tangents and directors
    T = eltype(x)
    nv = size(x,1)
    e = x[2:nv,:] .- x[1:nv-1,:]
    l = norm3(e)
    t = e./l
    d2 = cross3(t, d1)

    # Split into halves of the rod
    l1 = @view l[1:nv-2]
    l2 = @view l[2:nv-1]
    t1 = @view t[1:nv-2,:]
    t2 = @view t[2:nv-1,:]
    d1l = @view d1[1:nv-2,:]
    d1r = @view d1[2:nv-1,:]
    d2l = @view d2[1:nv-2,:]
    d2r = @view d2[2:nv-1,:]

    chi = 1 .+ dot3(t1, t2)
    ttild = (t1 .+ t2)./chi
    d1tild = (d1l .+ d1r)./chi
    d2tild = (d2l .+ d2r)./chi
    kb = 2 .*cross3(t1, t2)./chi

    κ1l =  dot3(kb, d2l)
    κ1r =  dot3(kb, d2r)
    κ2l = -dot3(kb, d1l)
    κ2r = -dot3(kb, d1r)
    κ1 = (κ1l + κ1r)./2
    κ2 = (κ2l + κ2r)./2
    κ = hcat(κ1, κ2)

    # Curvature Derivatives
    dκ1de1 = (-κ1.*ttild .+ cross3(t2, d2tild))./l1
    dκ1de2 = (-κ1.*ttild .- cross3(t1, d2tild))./l2

    dκ2de1 = (-κ2.*ttild .- cross3(t2, d1tild))./l1
    dκ2de2 = (-κ2.*ttild .+ cross3(t1, d1tild))./l2

    # Twists
    d1_transport = ptransport(d1l, t1, t2)
    cosτ = dot3(d1_transport, d1r)
    sinτ = dot3(d1_transport, d2r)
    τ = -atan.(sinτ, cosτ)

    # Twist Derivatives
    dτde1 = kb./(2 .*l1);
    dτde2 = kb./(2 .*l2);

    # Energy derivatives
    vl0 = [l0[1]; l0[2:end-1]/2] + [l0[2:end-1]/2; l0[end]]
    dEdl = k.*(l - l0)./l0
    dEdκ = B.*(κ .- κ0)./vl0
    dEdτ = β.*(τ .- τ0)./vl0

    dEsde = dEdl.*t
    dEde1 = dEdκ[:,1].*dκ1de1 .+ dEdκ[:,2].*dκ2de1 .+ dEdτ.*dτde1
    dEde2 = dEdκ[:,1].*dκ1de2 .+ dEdκ[:,2].*dκ2de2 .+ dEdτ.*dτde2

    dEdx = [-dEde1; zeros(T,2,3)] .+ [zeros(T,1,3); dEde1 .- dEde2; zeros(T,1,3)] .+ [zeros(T,2,3); dEde2]
    dEdx += [-dEsde; zeros(T,1,3)] .+ [zeros(T,1,3); dEsde]

    dEtdθ = [0; dEdτ] .- [dEdτ; 0]
    dEbdθ = ([(κ2l.*dEdκ[:,1] .- κ1l.*dEdκ[:,2]); 0]./2
          .+ [0; (κ2r.*dEdκ[:,1] .- κ1r.*dEdκ[:,2])]./2)
    dEdθ = dEtdθ .+ dEbdθ

    rod_delta(dEdx, dEdθ)
end
