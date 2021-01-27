import numpy as np
import matplotlib.pyplot as plt

# Function Timings
# n = [10, 20, 30, 40, 50, 75, 100, 200, 500]
# t_all = [2.1301e-5,  2.9001e-5,  3.62e-5,  4.31e-5,  5.0e-5,  6.69e-5,  8.6201e-5,  0.0001536,  0.0003693]
# t_nall = [1.33e-5,1.83e-5,2.3301e-5,2.79e-5,3.25e-5,4.44e-5,5.6799e-5,0.0001053,0.0002534]
# m_all =[39952.0,  61376.0,  84208.0,  106288.0,  127328.0,  188080.0,  233392.0,  460272.0,  1.08712e6]
# m_nall = [11984.0,	15936.0,	20000.0,	24496.0,	28096.0,	38960.0,	47696.0,	90224.0,	206432.0]
#
# plt.figure()
# f = plt.loglog(n, t_all, '-ok')
# r = plt.loglog(n, t_nall, '-or')
# plt.legend(("Allocating", "Non-Allocating"))
# plt.xlabel("Segments")
# plt.ylabel("Time [s]")
#
# plt.figure()
# f = plt.loglog(n, m_all, '-ok')
# r = plt.loglog(n, m_nall, '-or')
# plt.legend(("Allocating", "Non-Allocating"))
# plt.xlabel("Segments")
# plt.ylabel("Memory Allocated [bytes]")
# plt.show()

# Gradient Timings
# fwd = [0.000176199,  0.000641,     0.0011952,    0.0019804,    0.0031863,  0.0072633,  0.0142603,  0.107809,   0.694797]
# rev = [0.000140801,  0.000264299,  0.000447899,  0.000632399,  0.0007319,  0.0014311,  0.0020132,  0.0047546,  0.0120519]
# zyg = [0.000581401,  0.000861,     0.0011396,    0.0014208,    0.0017008,  0.0023889,  0.0031288,  0.0059794,  0.0145427]
# n = [10, 20, 30, 40, 50, 75, 100, 200, 500]
#
# plt.figure()
# f = plt.loglog(n, fwd, '-ok')
# r = plt.loglog(n, rev, '-or')
# z = plt.loglog(n, zyg, '-ob')
# plt.legend(("ForwardDiff", "ReverseDiff", "Zygote"))
# plt.xlabel("Segments")
# plt.ylabel("Gradient Evaluation [s]")
# plt.show()

# Hessvec Timings
n = 4*np.array([10, 20, 30, 40, 50])+7
num = [0.0039019,0.0101131,0.0184881,0.039217699,0.0664047]
numauto = [0.0003498,0.0012155,0.002467601,0.004103899,0.0066309]
autonum = [0.002347499,0.0061786,0.011535599,0.0190285,0.0334111]
numback = [0.001183599,0.001789599,0.0023833,0.002978201,0.003565201]
autoback = [0.000639599,0.0009512,0.001261599,0.0015935,0.001915999]

plt.figure()
plt.loglog(n, num, '-o')
plt.loglog(n, numauto, '-o')
plt.loglog(n, autonum, '-o')
plt.loglog(n, numback, '-o')
plt.loglog(n, autoback, '-o')
plt.legend(("Numerical", "NumAuto", "AutoNum", "NumBack", "AutoBack"))
plt.xlabel("Degrees of Freedom")
plt.ylabel("HVP [s]")

# MinRes Timings
numauto = [0.0237681,0.1509678,0.530635801,1.1698535,2.095379199]
autoback = [0.031669499,0.090868999,0.1995156,0.3195482,0.4514346]

plt.figure()
plt.loglog(n, numauto, '-or')
plt.loglog(n, autoback, '-ob')
plt.legend(("NumAuto", "AutoBack"))
plt.xlabel("Degrees of Freedom")
plt.ylabel("Minres Solution [s]")
plt.show()
