"""Functions needed to calculate the spectrum of 2D dirac particles
in a periodic potential with superconductivity.
"""

from scipy import *
from matplotlib.pyplot import *
from scipy.linalg import eigh
from scipy.integrate import odeint
from scipy.integrate import quad
from scipy.integrate import dblquad
from scipy.optimize import newton_krylov

import tools as t
from tools import pauli0, paulix, pauliy, pauliz, mat1, mat2, mat3, mat4
import diracpc2 as dc


def constructVDmat(Vn, dn):
    """Constructs and returns the potential and \Delta
    matrices from given input.

    Arguments:
    Vn: Array containing the Fourier representation
        of the potential with dimensions (3, 2*n_max+1).
    dn: Array containing the Fourier representation
        of \Delta with dimensions (2, 2*n_max+1).
    """

    if len(Vn[0]) is not len(dn[0]):
        print "Error: potential and \Delta matrices don't match!"
        return None

    numn = len(Vn[0])
    nmax = (numn-1)/2

    V1m = zeros((numn, numn))
    V2m = zeros((numn, numn))
    V3m = zeros((numn, numn))
    V3mc = zeros((numn, numn))
    dm1 = zeros((numn, numn))
    dm1c = zeros((numn, numn))
    dm2 = zeros((numn, numn))
    dm2c = zeros((numn, numn))
    for nn in range(-nmax, nmax+1):
        V1m = V1m + diag((numn-abs(nn))*[Vn[0][nn+nmax]], nn)
        V2m = V2m + diag((numn-abs(nn))*[Vn[1][nn+nmax]], nn)
        V3m = V3m + diag((numn-abs(nn))*[Vn[2][nn+nmax]], nn)
        V3mc = V3mc + diag((numn-abs(nn))*[conjugate(Vn[2][-nn+nmax])], nn)

        dm1 = dm1 + diag((numn-abs(nn))*[dn[0][nn+nmax]], nn)
        dm2 = dm2 + diag((numn-abs(nn))*[dn[1][nn+nmax]], nn)
        dm1c = dm1c + diag((numn-abs(nn))*[conjugate(dn[0][nn+nmax])], nn)
        dm2c = dm2c + diag((numn-abs(nn))*[conjugate(dn[1][nn+nmax])], nn)

    Vmat = kron(V1m, kron(pauliz, mat1)) + \
        kron(V2m, kron(pauliz, mat2)) + \
        kron(V3m, kron(pauliz, mat3)) + \
        kron(V3mc, kron(pauliz, mat4))

    dmat = kron(transpose(dm1), kron(mat3, mat1)) + \
        kron(transpose(dm2), kron(mat3, mat2)) + \
        kron(dm1c, kron(mat4, mat1)) + \
        kron(dm2c, kron(mat4, mat2))

    return [Vmat, dmat]


def constructBdG(Vn, dn, kx, ky, verb=False, VDmat=None, psx=0., psy=0.):
    """Constructs and returns the BdG matrix.

    Arguments:
    Vn -- Array containing the Fourier representation
    of the potential with dimensions (3, 2*n_max+1).
    Not used if VDmat is provided.
    dn -- Array containing the Fourier representation
    of \Delta with dimensions (2, 2*n_max+1).
    Not used if VDmat is provided.
    kx -- x component of the momentum
    ky -- y component of the momentum
    verb -- Whether to print information about what is being
    calculated.
    VDmat -- Pre-constructed potential and \Delta matrix
    if available. If not provided, it will be constructed.
    psx -- x component of the superfluid momentum
    psy -- y component of the superfluid momentum
    """

    if VDmat is None:
        numn = len(Vn[0])
        nmax = (numn-1)/2
    else:
        numn = len(VDmat[0])/4
        nmax = (numn-1)/2

    if verb:
        print "Constructing BdG Hamiltonian with kx = " + str(kx) + \
            ", ky = " + str(ky) + " and nmax = " + str(nmax) + "."

    # momentum dependency, diagonal in Bloch space
    kmat = kron(diag(arange(-nmax, nmax+1)*2*pi+kx), kron(pauliz, paulix)) + \
        kron(diag(numn*[ky]), kron(pauliz, pauliy)) + \
        kron(diag(numn*[psx]), kron(pauli0, paulix)) + \
        kron(diag(numn*[psy]), kron(pauli0, pauliy))

    if VDmat is None:
        VDmat = constructVDmat(Vn, dn)

    m = kmat + VDmat[0] + VDmat[1]

    if not (conjugate(transpose(m)) - m < 1e-10).all():
        print "Warning: Hamiltonian is not hermitian!"

    return m


def solveDirac(Vn, dn, kx, ky, VDmat=None, psx=0., psy=0.):
    """Find the eigenvectors and eigenvalues of the BdG equation.
    Arguments are the same as for constructBdG()

    Returns:
    Array of dictionaries with indices 'E', 'u1', 'u2', 'v1, 'v2'
    corresponding to eigenenergies and eigenvectors of the Hamiltonian.
    Sorted by eigenenergies.
    """

    e, v = eigh(constructBdG(Vn, dn, kx, ky, VDmat=VDmat, psx=psx, psy=psy))

    eigvals = [{'E': e[i],
                'u1': v[:, i][::4],
                'u2': v[:, i][1::4],
                'v1': v[:, i][2::4],
                'v2': v[:, i][3::4]} for i in range(len(e))]

    return sorted(eigvals, key=lambda v: v['E'])


def solveDiraci(Vn, dn, kx, ky, i0=0, i1=3, VDmat=None, psx=0., psy=0.):
    """Solve the eigenvalues and eigenvectors of the BdG equation
    but return only some of them. Not more efficient than solveDirac,
    exists only for convenience.

    Arguments:
    i0 -- index of the first band to return. 0 corresponds to eigenvalue
    number 4*n_max
    i1 -- index of the last band to return.
    """

    if VDmat is None:
        numn = len(Vn[0])
        nmax = (numn-1)/2
    else:
        numn = len(VDmat[0])/4
        nmax = (numn-1)/2

    return solveDirac(Vn, dn, kx, ky,
                      VDmat, psx, psy)[(4*nmax+i0):(4*nmax+i1+1)]


def deltaintegrandnormal(kx, ky, Vn, dn, T, nbands=1,
                         VDmat=None, p=[0.0, 0.0]):
    if VDmat is None:
        numn = len(Vn[0])
        nmax = (numn-1)/2
    else:
        numn = len(VDmat[0])/4
        nmax = (numn-1)/2
    nr = arange(-nmax, nmax+1)
    delta1 = zeros(2*nmax+1, dtype=complex)
    delta2 = zeros(2*nmax+1, dtype=complex)
    bands = solveDiraci(Vn, dn, kx, ky, 2*(1-nbands),
                        2*(1+nbands), VDmat, p[0], p[1])

    for i in range(4*nbands):
        u1 = bands[i]['u1']
        v1 = bands[i]['v1']
        u2 = bands[i]['u2']
        v2 = bands[i]['v2']
        th = tanh(bands[i]['E']/(2*T))

        for n in nr:
            if n == 0:
                delta1[n+nmax] = delta1[n+nmax] + sum(conjugate(v1)*u1)*th
                delta2[n+nmax] = delta2[n+nmax] + sum(conjugate(v2)*u2)*th
            elif n > 0:
                delta1[n+nmax] = delta1[n+nmax] + \
                    sum(conjugate(v1[n:])*u1[:-n])*th
                delta2[n+nmax] = delta2[n+nmax] + \
                    sum(conjugate(v2[n:])*u2[:-n])*th
            else:
                delta1[n+nmax] = delta1[n+nmax] + \
                    sum(conjugate(v1[:n])*u1[-n:])*th
                delta2[n+nmax] = delta2[n+nmax] + \
                    sum(conjugate(v2[:n])*u2[-n:])*th

    bands = solveDiraci(Vn2, dn, kx, ky, 2*(1-nbands),
                        2*(1+nbands), VDmat, p[0], p[1])

    return (delta1, delta2)


def deltaintegrandvalley(kx, ky, Vn, dn, T, nbands=1,
                         VDmat=None, p=[0.0, 0.0]):
    if VDmat is None:
        numn = len(Vn[0])
        nmax = (numn-1)/2
    else:
        numn = len(VDmat[0])/4
        nmax = (numn-1)/2
    Vn = t.TFpot(30, 30)
    Vn2 = t.TFpot(-30, 30)
    nr = arange(-nmax, nmax+1)
    delta1 = zeros(2*nmax+1, dtype=complex)
    delta2 = zeros(2*nmax+1, dtype=complex)
    bands = solveDiraci(Vn, dn, kx, ky, 2*(1-nbands),
                        2*(1+nbands), VDmat, p[0], p[1])

    for i in range(4*nbands):
        u1 = bands[i]['u1']
        v1 = bands[i]['v1']
        u2 = bands[i]['u2']
        v2 = bands[i]['v2']
        th = tanh(bands[i]['E']/(2*T))

        for n in nr:
            if n == 0:
                delta1[n+nmax] = delta1[n+nmax] + sum(conjugate(v1)*u1)*th
                delta2[n+nmax] = delta2[n+nmax] + sum(conjugate(v2)*u2)*th
            elif n > 0:
                delta1[n+nmax] = delta1[n+nmax] + \
                    sum(conjugate(v1[n:])*u1[:-n])*th
                delta2[n+nmax] = delta2[n+nmax] + \
                    sum(conjugate(v2[n:])*u2[:-n])*th
            else:
                delta1[n+nmax] = delta1[n+nmax] + \
                    sum(conjugate(v1[:n])*u1[-n:])*th
                delta2[n+nmax] = delta2[n+nmax] + \
                    sum(conjugate(v2[:n])*u2[-n:])*th

    bands = solveDiraci(Vn2, dn, kx, ky, 2*(1-nbands),
                        2*(1+nbands), VDmat, p[0], -p[1])

    for i in range(4*nbands):
        u1 = bands[i]['u1']
        v1 = bands[i]['v1']
        u2 = bands[i]['u2']
        v2 = bands[i]['v2']
        th = tanh(bands[i]['E']/(2*T))

        for n in nr:
            if n == 0:
                delta1[n+nmax] = delta1[n+nmax] + sum(conjugate(v1)*u1)*th
                delta2[n+nmax] = delta2[n+nmax] + sum(conjugate(v2)*u2)*th
            elif n > 0:
                delta1[n+nmax] = delta1[n+nmax] + \
                    sum(conjugate(v1[n:])*u1[:-n])*th
                delta2[n+nmax] = delta2[n+nmax] + \
                    sum(conjugate(v2[n:])*u2[:-n])*th
            else:
                delta1[n+nmax] = delta1[n+nmax] + \
                    sum(conjugate(v1[:n])*u1[-n:])*th
                delta2[n+nmax] = delta2[n+nmax] + \
                    sum(conjugate(v2[:n])*u2[-n:])*th

    return (delta1, delta2)


def deltaintegrandsymmetric(kx, ky, Vn, dn, T, nbands=1, VDmat=None, p=[0.0, 0.0]):
    if VDmat is None:
        numn = len(Vn[0])
        nmax = (numn-1)/2
    else:
        numn = len(VDmat[0])/4
        nmax = (numn-1)/2
    Vn = t.TFpot(30, 30)
    Vn2 = t.TFpot(-30, 30)
    nr = arange(-nmax, nmax+1)
    delta = zeros(2*nmax+1, dtype=complex)
    bands = solveDiraci(Vn, dn, kx, ky, 2*(1-nbands),
                        2*(1+nbands), VDmat, p[0], p[1])

    for i in range(4*nbands):
        u1 = bands[i]['u1']
        v1 = bands[i]['v1']
        u2 = bands[i]['u2']
        v2 = bands[i]['v2']
        th = tanh(bands[i]['E']/(2*T))

        for n in nr:
            if n == 0:
                delta[n+nmax] = delta[n+nmax] + \
                    sum(conjugate(v1)*u1+conjugate(v2)*u2)*th
            elif n > 0:
                delta[n+nmax] = delta[n+nmax] + \
                                sum(conjugate(v1[n:])*u1[:-n] +
                                    conjugate(v2[n:])*u2[:-n])*th
            else:
                delta[n+nmax] = delta[n+nmax] + \
                                sum(conjugate(v1[:n])*u1[-n:] +
                                    conjugate(v2[:n])*u2[-n:])*th
    return delta


def deltaintegralvalley(Vn, dn, T, g, pmax, nbands=1,
                        kx=1e-5, VDmat=None, p=[0.0, 0.0]):
    if VDmat is None:
        numn = len(Vn[0])
        nmax = (numn-1)/2
    else:
        numn = len(VDmat[0])/4
        nmax = (numn-1)/2

    def integrand(v, ky):
        (d1, d2) = deltaintegrandvalley(kx, ky, Vn, dn, T, nbands, VDmat, p)
        a = concatenate((real(d1), imag(d1), real(d2), imag(d2)))
        return concatenate((real(d1), imag(d1), real(d2), imag(d2)))

    res = odeint(integrand, zeros((2*nmax+1)*4),
                 [0., pmax], rtol=1e-5, atol=1e-5)[1] - \
        odeint(integrand, zeros((2*nmax+1)*4),
               [0., -pmax], rtol=1e-5, atol=1e-5)[1]

    return (g*(res[:(2*nmax+1)]+1j*res[(2*nmax+1):(4*nmax+2)])/(2*pi),
            g*(res[(4*nmax+2):(6*nmax+3)]+1j*res[(6*nmax+3):])/(2*pi))


def deltaintegral(Vn, dn, T, g, pmax, nbands=1, kx=1e-5, VDmat=None, p=[0.0, 0.0]):
    if VDmat is None:
        numn = len(Vn[0])
        nmax = (numn-1)/2
    else:
        numn = len(VDmat[0])/4
        nmax = (numn-1)/2

    def integrand(v, ky):
        (d1, d2) = deltaintegrand(kx, ky, Vn, dn, T, nbands, VDmat, p)
        a = concatenate((real(d1), imag(d1), real(d2), imag(d2)))
        return concatenate((real(d1), imag(d1), real(d2), imag(d2)))

    res = odeint(integrand, zeros((2*nmax+1)*4), [0., pmax], rtol=1e-5, atol=1e-5)[1] - \
        odeint(integrand, zeros((2*nmax+1)*4),
               [0., -pmax], rtol=1e-5, atol=1e-5)[1]

    return (g*(res[:(2*nmax+1)]+1j*res[(2*nmax+1):(4*nmax+2)])/(2*pi),
            g*(res[(4*nmax+2):(6*nmax+3)]+1j*res[(6*nmax+3):])/(2*pi))


def deltaintegralsymmetric(Vn, dn, T, g, pmax, nbands=1,
                           kx=1e-5, VDmat=None, p=[0.0, 0.0]):
    if VDmat is None:
        numn = len(Vn[0])
        nmax = (numn-1)/2
    else:
        numn = len(VDmat[0])/4
        nmax = (numn-1)/2

    def integrand(v, ky):
        d1 = deltaintegrandsymmetric(kx, ky, Vn, dn, T, nbands, VDmat, p)
        return concatenate((real(d1), imag(d1)))

    res = odeint(integrand, zeros((2*nmax+1)*2), [0., pmax], rtol=1e-5, atol=1e-5)[1] - \
        odeint(integrand, zeros((2*nmax+1)*2),
               [0., -pmax], rtol=1e-5, atol=1e-5)[1]

    return (g*(res[:(2*nmax+1)]+1j*res[(2*nmax+1):(4*nmax+2)])/(2*pi),
            g*(res[:(2*nmax+1)]+1j*res[(2*nmax+1):(4*nmax+2)])/(2*pi))


def newtonitern(Vn, dn0, T, g, pmax, nbands=1, kx=1e-5, tol=1e-5, maxloops=30,
                p=[0.0, 0.0], verb=False, cache=True, beta=-1):

    numn = len(Vn[0])
    nmax = (numn-1)/2
    if cache:
        if not (beta < 0):
            delta = loaddelta(g, T, beta, nmax, p)
            if delta is not None:
                return delta

    if verb:
        print "Calculating delta with Newton iteration with g=", g

    loopN = 0
    dn = dn0
    while True:
        if verb:
            print "We're at loop nr", loopN
        VDmat = constructVDmat(Vn, dn)
        dnnew = deltaintegralsymmetric(
            Vn, dn, T, g, pmax, nbands, kx, VDmat, p)
        err1 = sum(abs(dn[0]-dnnew[0]))/sum(abs(dnnew[0]))
        err2 = sum(abs(dn[1]-dnnew[1]))/sum(abs(dnnew[1]))
        if verb:
            print "Max error is", max(err1, err2), "Tolerance is", tol
        if max(err1, err2) < tol:
            if cache:
                if not (beta < 0):
                    cachedelta(dnnew, g, T, beta, p)
            return dnnew

        elif loopN >= maxloops:
            print "Maximum number of loops reached and solution was not found!"
            return dnnew
        else:
            dn = dnnew
        loopN = loopN+1


def krylovsolversymmetric(Vn, dn0, T, g, pmax, nbands=1, kx=1e-5, tol=1e-5,
                          p=[0.0, 0.0], verb=False, cache=True, beta=-1.):

    numn = len(Vn[0])
    nmax = (numn-1)/2

    if cache:
        if not (beta < 0):
            delta = loaddelta(g, T, beta, nmax, p)
            if delta is not None:
                return delta

    if verb:
        print "Finding self-consistent delta with g =", g

    def minfun(dng):
        if verb:
            print "Iteration..."
        dn = dng[:(2*nmax+1)]+1j*dng[(2*nmax+1):]
        VDmat = constructVDmat(Vn, [dn, dn])
        integral = deltaintegralsymmetric(
            None, None, T, g, pmax, nbands, kx, VDmat, p)
        res = dn-integral[0]
        return concatenate((real(res), imag(res)))

    dn0 = concatenate((real(dn0[0]), imag(dn0[0])))
    dnconv = newton_krylov(minfun, dn0, verbose=verb, f_tol=tol)
    newdelta = (dnconv[:(2*nmax+1)]+1j*dnconv[(2*nmax+1):],
                dnconv[:(2*nmax+1)]+1j*dnconv[(2*nmax+1):])
    if cache:
        if not (beta < 0):
            cachedelta(newdelta, g, T, beta, p)

    return newdelta


def scurrentintegrandn(n, ky, Vn, dn, T, nbands=1, VDmat=None, kx=1e-5, p=[0.0, 0.0],
                       calcx=True, calcy=True):
    if VDmat is None:
        numn = len(Vn[0])
        nmax = (numn-1)/2
    else:
        numn = len(VDmat[0])/4
        nmax = (numn-1)/2
    nr = arange(-nmax, nmax+1)
    bands = solveDiraci(Vn, dn, kx, ky, 2*(1-nbands),
                        2*(1+nbands), VDmat, p[0], p[1])
    j0x = 0.
    j0y = 0.
    jcurrx = 0.
    jcurry = 0.
    for i in range(nbands*4):
        u1 = bands[i]['u1']
        v1 = bands[i]['v1']
        u2 = bands[i]['u2']
        v2 = bands[i]['v2']
        Eg = bands[i]['E']
        if n == 0:
            if calcx:
                jcurrx = jcurrx + sum((conjugate(u2[:])*u1[:]+conjugate(u1[:])*u2[:])*(
                    Eg < 0.)-(conjugate(v2[:])*v1[:]+conjugate(v1[:])*v2[:])*(Eg > 0.))
            if calcy:
                jcurry = jcurry + 1j*sum((conjugate(u2[:])*u1[:]-conjugate(u1[:])*u2[:])*(
                    Eg < 0.)-(conjugate(v2[:])*v1[:]-conjugate(v1[:])*v2[:])*(Eg > 0.))
        elif n > 0:
            if calcx:
                jcurrx = jcurrx + sum((conjugate(u2[n:])*u1[:-n]+conjugate(u1[n:])*u2[:-n])*(
                    Eg < 0.)-(conjugate(v2[n:])*v1[:-n]+conjugate(v1[n:])*v2[:-n])*(Eg > 0.))
            if calcy:
                jcurry = jcurry + 1j*sum((conjugate(u2[n:])*u1[:-n]-conjugate(u1[n:])*u2[:-n])*(
                    Eg < 0.)-(conjugate(v2[n:])*v1[:-n]-conjugate(v1[n:])*v2[:-n])*(Eg > 0.))
        else:
            nn = abs(n)
            if calcx:
                jcurrx = jcurrx + sum((conjugate(u2[:-nn])*u1[nn:]+conjugate(u1[:-nn])*u2[nn:])*(
                    Eg < 0.)-(conjugate(v2[:-nn])*v1[nn:]+conjugate(v1[:-nn])*v2[nn:])*(Eg > 0.))
            if calcy:
                jcurry = jcurry + 1j*sum((conjugate(u2[:-nn])*u1[nn:]-conjugate(u1[:-nn])*u2[nn:])*(
                    Eg < 0.)-(conjugate(v2[:-nn])*v1[nn:]-conjugate(v1[:-nn])*v2[nn:])*(Eg > 0.))

    sol = zeros(calcx + calcy)
    if calcx:
        sol[0] = jcurrx
    if calcy:
        sol[-1] = jcurry
    return sol


def scurrentintn(n, Vn, dn, T, pmax, nbands=1, VDmat=None, kx=1e-5, p=[0.0, 0.0],
                 calcx=False, calcy=False):
    currx = 0.0
    if calcx:
        currx = quad(lambda k: scurrentintegrandn(n, k, Vn, dn, T, nbands=1, VDmat=None, kx=kx, p=p,
                                                  calcx=True, calcy=False),
                     -pmax, pmax)[0]
    if calcy:
        curry = quad(lambda k: scurrentintegrandn(n, k, Vn, dn, T,
                                                  nbands=1, VDmat=None, kx=kx, p=p,
                                                  calcx=False, calcy=True),
                     -pmax, pmax)[0]
    curr = zeros(calcx + calcy)
    if calcx:
        curr[0] = currx
    if calcy:
        curr[-1] = curry
    return curr


def cachedelta(delta, g, T, beta, p=[0.0, 0.0]):
    nm = (len(delta[0])-1)/2
    fname = "delta"+str(float(g))+str(float(T))+str(float(beta)) + \
        str(float(p[0]))+str(float(p[1]))+str(int(nm))+".dat"
    savetxt("deltacache/"+fname, array([real(delta[0]), imag(delta[0]),
                                        real(delta[1]), imag(delta[1])]))
    return


def loaddelta(g, T, beta, nm, p=[0.0, 0.0]):
    fname = "delta" + str(float(g)) + str(float(T)) + str(float(beta)) \
            + str(float(p[0])) + str(float(p[1])) + str(int(nm)) + ".dat"
    delta0 = None
    try:
        delta0 = loadtxt("deltacache/"+fname)
        delta = array([zeros(2*nm+1, dtype=complex),
                       zeros(2*nm+1, dtype=complex)])
        delta[0] = delta0[0]+delta0[1]*1j
        delta[1] = delta0[2]+delta0[3]*1j
    except IOError:
        print "not found", fname
        delta = delta0
        pass
    return delta
