"""
"""

from scipy import *
from numpy import ctypeslib


bdglib = ctypeslib.load_library(
    'bdgeq.so', "./")
bdglib.constructBdG.restype = None
bdglib.constructBdG.argtypes = [ctypeslib.ndpointer(int, flags='aligned, contiguous'),
                                ctypeslib.ndpointer(
                                    complex, flags='aligned, contiguous'),
                                ctypeslib.ndpointer(
                                    complex, flags='aligned, contiguous'),
                                ctypeslib.ndpointer(
                                    complex, flags='aligned, contiguous, writeable'),
                                ctypeslib.ndpointer(
                                    double, flags='aligned, contiguous'),
                                ctypeslib.ndpointer(double, flags='aligned, contiguous')]

bdglib.solvedirac.restype = None
bdglib.solvedirac.argtypes = [ctypeslib.ndpointer(int, flags='aligned, contiguous'),
                              ctypeslib.ndpointer(
                                  complex, flags='aligned, contiguous'),
                              ctypeslib.ndpointer(
                                  complex, flags='aligned, contiguous'),
                              ctypeslib.ndpointer(
                                  complex, flags='aligned, contiguous, writeable'),
                              ctypeslib.ndpointer(
                                  float, flags='aligned, contiguous'),
                              ctypeslib.ndpointer(
                                  float, flags='aligned, contiguous'),
                              ctypeslib.ndpointer(double, flags='aligned, contiguous, writeable')]

bdglib.selfcintegrand.restype = None
bdglib.selfcintegrand.argtypes = [ctypeslib.ndpointer(int, flags='aligned, contiguous'),
                                  ctypeslib.ndpointer(
                                      int, flags='aligned, contiguous'),
                                  ctypeslib.ndpointer(
                                      float, flags='aligned, contiguous'),
                                  ctypeslib.ndpointer(
                                      complex, flags='aligned, contiguous'),
                                  ctypeslib.ndpointer(
                                      complex, flags='aligned, contiguous'),
                                  ctypeslib.ndpointer(
                                      complex, flags='aligned, contiguous, writeable'),
                                  ctypeslib.ndpointer(
                                      float, flags='aligned, contiguous'),
                                  ctypeslib.ndpointer(
                                      float, flags='aligned, contiguous'),
                                  ctypeslib.ndpointer(
                                      double, flags='aligned, contiguous, writeable'),
                                  ctypeslib.ndpointer(complex, flags='aligned, contiguous, writeable')]


def constructBdGc(vn, dn, kx, ky):
    vn3 = asarray(vn[2], dtype=complex)
    dn = concatenate(dn)
    dn = asarray(dn, dtype=complex)
    nm = (len(vn)-1)/2
    res = zeros((4*(2*nm+1))**2, dtype=complex)

    bdglib.constructBdG(array(nm, dtype='int', ndmin=1), vn3, dn, res,
                        array(kx, dtype='double', ndmin=1),
                        array(ky, dtype='double', ndmin=1))

    return reshape(res, (4*(2*nm+1), 4*(2*nm+1)))


def solveDiracc(vn, dn, kx, ky):
    resv = zeros((4*(2*nm+1))**2, dtype=complex)
    rese = zeros(4*(2*nm+1), dtype=double)
    vn3 = asarray(vn[2], dtype=complex)
    dn = asarray(concatenate(dn), dtype=complex)
    nm = (len(vn3)-1)/2

    bdglib.solvedirac(array(nm, dtype='int', ndmin=1), vn3, dn, resv,
                      array(kx, dtype='double', ndmin=1), array(
                          ky, dtype='double', ndmin=1),
                      rese)
    resv = transpose(reshape(resv, (4*(2*nm+1), 4*(2*nm+1))))

    eigvals = [{'E': rese[i],
                'u1': resv[:, i][::4],
                'u2': resv[:, i][1::4],
                'v1': resv[:, i][2::4],
                'v2': resv[:, i][3::4]} for i in range(len(rese))]

    return eigvals


def deltaintegrandnc(kx, ky, vn, dn, T, nbands=1):
    print ky
    vn3 = asarray(vn[2], dtype=complex)
    dn = asarray(concatenate(dn), dtype=complex)
    nm = (len(vn3)-1)/2
    newd = zeros(2*(2*nm+1), dtype=complex)
    resv = zeros((4*(2*nm+1))**2, dtype=complex)
    rese = zeros(4*(2*nm+1), dtype=double)

    bdglib.selfcintegrand(array(nm, dtype='int', ndmin=1), array(nbands, dtype='int', ndmin=1),
                          array(T, dtype='double', ndmin=1), vn3, dn, resv,
                          array(kx, dtype='double', ndmin=1), array(
                              ky, dtype='double', ndmin=1),
                          rese, newd)

    return (real(newd[:(2*nm+1)]), imag(newd[:(2*nm+1)]),
            real(newd[(2*nm+1):]), imag(newd[(2*nm+1):]))
