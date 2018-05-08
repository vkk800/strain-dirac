"""This example computes self-consistent values of the inhomogenous
superconducting order parameter in a system with the Tang-Fu potential.
We use parallel python to parallelize the computations.

The results are plotted in two files: example-deltamaxvsg.pdf
and example-deltaxvsg.pdf.
"""

import pp
from scipy import *
import matplotlib.pyplot as plt

NUMCPU = 4

# Values of the superconducting coupling strength and \beta for the computation
g = concatenate((linspace(1e-4, 1e-2, 20), linspace(1.1e-2, 3e-2, 10)))
betas = array([20, 30])

x = linspace(0, 1, 200) # Real space grid for inverse Fourier transform in the plot


def compute_delta(beta, gcoupl):
    newd = tools.constd(1e-4*beta, beta)

    # For reasons lost to time, different solving strategies are needed for
    # different parameter regimes
    if gcoupl > 2 * numpy.sqrt(beta / 2.0) * numpy.exp(-beta / numpy.pi):
        newd = diracp.newtonitern(tools.TFpot(beta, beta), newd, 1e-6, gcoupl,
                                  beta, nbands=1, verb=True, cache=True, beta=beta)
    else:
        newd = diracp.krylovsolversymmetric(tools.TFpot(beta, beta), newd, 1e-6, gcoupl,
                                            beta, nbands=1, verb=True, cache=True, beta=beta)
    x = numpy.linspace(0, 1, 200)
    deltax = tools.finverse(newd[0], x)
    maxd = sum(abs(deltax))*(x[2]-x[1])
    return (deltax, maxd)


job_server = pp.Server(NUMCPU, ppservers=(), secret="")
jobs = [[job_server.submit(compute_delta, (bb, gg), (), ("diracp", "tools", "numpy"))
         for gg in g] for bb in betas]

# Extract the results and plot them
maxd = zeros((len(betas), len(g)))
deltax = zeros((len(betas), len(g), len(x)))
for bb in range(len(betas)):
    for gg in range(len(g)):
        deltax[bb, gg, :], maxd[bb, gg] = jobs[bb][gg]()

plt.figure()
plt.plot(g, transpose(maxd))
plt.xlabel(r"$g$")
plt.ylabel(r"$\Delta$")
plt.savefig("example-deltamaxvsg.pdf")

plt.figure()
for i in range(len(betas)):
    plt.subplot(len(betas), 1, i+1)
    plt.plot(x, transpose(deltax[i, :, :]))
    plt.xlabel(r"$x / L$")
    plt.ylabel(r"$\Delta$")
plt.savefig("example-deltaxvsg.pdf")

plt.show()
