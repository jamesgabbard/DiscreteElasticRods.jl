
function forces(x, d1), (vl0, k0, B, beta)

    # Tangents and directors
    nv = size(x,2)
    e =x[:, 2:nv] .- x[:, 1:nv-1]
    l = norm3(e)
    t = e./l
    d2 = cross3(t, d1)

    # Split into halves of the rod
    l1 = @view l[1:1, 1:nv-2]
    l2 = @view l[1:1, 2:nv-1]
    t1 = @view t[:,1:nv-2]
    t2 = @view t[:,2:nv-1]
    d1l = @view d1[:,1:nv-2]
    d1r = @view d1[:,2:nv-1]
    d2l = @view d2[:,1:nv-2]
    d2r = @view d2[:,2:nv-1]

    d1v = (d1l .+ d1r)./2
    d2v = (d1l .+ d1r)./2

    chi = 1 + dot3(t1, t2)
    ttild = (t1 .+ t2)./chi
    d1tild = 2 .*d1v./chi
    d2tild = 2 .*d2v./chi

    k = 2 .*cross3(t1, t2)./chi
    κ1 =  dot3(k, d2v)
    κ2 = -dot3(k, d1v)
    κ = vcat(κ1, κ2)

    # Curvature Derivatives
    dκ1de1 = (-κ1.*ttild .+ cross3(t2, d2tild))./l1
    dκ1de2 = (-κ1.*ttild .- cross3(t1, d2tild))./l2

    dκ2de1 = (-κ2.*ttild .- cross3(t2, d1tild))./l1
    dκ2de2 = (-κ2.*ttild .+ cross3(t1, d1tild))./l2

    # Twists
    d1_transport = ptransport(d1l, t1, t2)
    cosτ = dot3(d1_transport, d1r)
    sinτ = dot3(d1_transport, d2r)
    τ = atan.(sinτ, cosτ)



    # Twist Derivatives
    dτde1 = k./(2 .*l1);
    dτde2 = k./(2 .*l2);

    # Derivative wrt edge i-1
    dEbde1 = ([dκ1de1; dκ2de1]' * B * ([κ1; κ2] - k0(i-1,:)') +

    dEtde1 = β.*τ.*dτde1./vl0;
    dEtde2 = β.*τ.*dτde2./vl0;

    # Derivative wrt edge i
    de2 = ([dκ1de2; dκ2de2]' * B * ([κ1; κ2] - k0(i-1,:)') + beta*m*dmde2')./vl0;


    F(i-1,:) = F(i-1,:) + de1';
    F(i,:) = F(i,:) - de1' + de2';
    F(i+1,:) = F(i+1,:) - de2';

end


J = [0 -1; 1 0];

% Curvature Vector in Material Frame


% Twist Derivatives
gradTwist = β.*([0; τ./vl0] - [τ./vl0; 0])

% Bending Derivatives
rhs = (J*B*(κ-κ0)')';
gradBend = [0.5*dot(omegaLeft./vl0, rhs, 2); 0] + ...
           [0; 0.5*dot(omegaRight./vl0, rhs, 2)];

% Moments (aka the negative Gradient of Elastic energy wrt Thetas)
moment = -(gradTwist + gradBend);
