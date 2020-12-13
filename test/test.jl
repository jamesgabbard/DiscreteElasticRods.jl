using DiscreteElasticRods
DER = DiscreteElasticRods

# A nice silicone rod
# Between 1 and 50 MPa is apparently typical E for silicone
E = 5e6
ν = 0.48
L = 0.30
R = 0.003
props = DER.elliptical_stiffness(E, ν, R, R)

# A nice straight IC
# Rod on x-axis, vertical director
p1 = [0., 0., 0.]
p2 = [1., 0., 0.]
d0 = [0., 0., 1.]
ns = 12
r = DER.straight_rod(p1, p2, d0, ns)

strain = DER.full_kinematics(r.x, r.d)

# ----------------------------------------------------------------------------------
#  Some test code
# ----------------------------------------------------------------------------------
κ1 = s -> 1.0
κ2 = s -> 0.5
τ  = s -> 15*s*(1-s)
sspan = [0,1]
x0 = [0,0,0]
t0 = [1,0,0]
d0 = [0,0,1]
N = 20

xfun, dfun = DER.continuous_rod(x0, t0, d0, κ1, κ2, τ, sspan)
r = DER.discrete_rod(xfun, dfun, sspan, N)
DER.plot(r)
