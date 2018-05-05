"""Set of somewhat generic utility functions used in other parts of the code.
Many of the functions are related to doing regular NumPy/SciPy operations on
complex valued inputs.
"""

from scipy import *
from scipy.integrate import quad, dblquad

pauli0 = array([[1, 0], [0, 1]])
paulix = array([[0, 1.], [1., 0]])
pauliy = array([[0, -1.j], [1.j, 0]])
pauliz = array([[1, 0], [0, -1]])
mat1 = [[1, 0], [0, 0]]
mat2 = [[0, 0], [0, 1]]
mat3 = [[0, 1], [0, 0]]
mat4 = [[0, 0], [1, 0]]


def complex_quad(func, a, b, **kwargs):
    """Calculate integral of a complex valued function over real domain."""
    def func_r(x):
        return real(func(x))

    def func_i(x):
        return imag(func(x))
    int_r = quad(func_r, a, b, **kwargs)
    int_i = quad(func_i, a, b, **kwargs)
    return (int_r[0] + int_i[0] * 1j, int_r[1] + int_i[1])


def complex_dblquad(func, a, b, gfun, hfun, **kwargs):
    """Calculate double integral of a complex valued function over real domain.
    """
    def func_r(x, y):
        return real(func(x, y))

    def func_i(x, y):
        return imag(func(x, y))

    int_r = dblquad(func_r, a, b, gfun, hfun, **kwargs)
    int_i = dblquad(func_i, a, b, gfun, hfun, **kwargs)
    return (int_r[0] + int_i[0] * 1j, int_r[1] + int_i[1])


def ftransform(f, nspace):
    """Fourier transform of periodic function f with period [0,1].
    Fourier components from nspace. Not the most efficient way to
    do this, but good enough.
    """
    fn = array([complex_quad(lambda x: exp(1j * 2 * pi * x * nn) *
                             f(x), 0., 1.)[0]
                for nn in nspace])
    return fn


def finverse(fn, xspace):
    """The inverse fourier transform of periodic function with Fourier components
    fn. Again not the most efficient way to do this, but fine when it is not
    needed too much.
    """
    nspace = arange(-(len(fn) - 1) / 2, (len(fn) - 1) / 2 + 1)
    f = array([sum(exp(-1j * 2 * pi * nspace * xx) * fn) for xx in xspace])
    return f


def unpackE(eigs):
    """For an array of eigenvalues and vectors, as returned by diracp.solveDirac(),
    extract the eigenvalues. Returns an array of size (len(eigs), n_bands)
    """
    Es = []
    for i in range(len(eigs)):
        Es.append(array([v['E'] for v in eigs[i]]))
    return Es


def savecomplexv(fname, vec):
    """Save a complex valued vector into a file. Works like scipy.savetxt().
    """
    vecr = real(vec)
    veci = imag(vec)
    out = asarray((vecr, veci))
    savetxt(fname, out)


def loadcomplexv(fname):
    """Load a complex valued vector from a file written using savecomplexv().
    """
    input_data = loadtxt(fname)
    return input_data[0] + 1j * input_data[1]


def TFpot(beta, nm):
    """A function that returns the Fourier transform of a Tang-Fu style periodic
    potential.
    """
    v0 = zeros(2 * nm + 1)

    def v3(x):
        return 1j * cos(2 * pi * x) * beta
    v3 = ftransform(v3, arange(-nm, nm + 1))
    return [v0, v0, v3]


def zerod(nm):
    """Returns a tuple of zeros of shape (2*nm+1, 2*nm+1).
    """
    return (zeros(2 * nm + 1, dtype=complex), zeros(2 * nm + 1, dtype=complex))


def constd(m, nm):
    """Returns the Fourier transform of constant real space potential.
    """
    d = zerod(nm)
    d[0][nm] = m
    d[1][nm] = m
    return d
