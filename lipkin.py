# coding: utf-8
#/usr/bin/env python3
__doc__ = """
This Python program calculates the time-periodically driven mean field dynamics 
of the quantum transverse field Ising Model (Lipkin - Meshkov- Glick model) as detailed in [1,2]. 
It also evaluates fft of sigma^z.

Refs:

[1] Lipkin, H. J., N. Meshkov, and A. J. Glick, Nuclear Physics 62, no. 2 (February 1, 1965): 188–98. 
    https://doi.org/10/fpqf4q .
[2] Mori, T. Accessed October 30, 2019. https://arxiv.org/abs/1810.01584.
"""

import numpy as np
from odeintw import odeintw
from numpy.linalg import multi_dot, eig
from multiprocessing import Pool
from scipy.signal import find_peaks

verbose = False
pheightmin = 100


#Pauli Matrices
sx = np.array([[0, 1],[ 1, 0]])
sy = np.array([[0, -1j],[1j, 0]])
sz = np.array([[1, 0],[0, -1]])

def mf_jac (psi, t, h0, h, omega):
    sx_mf = multi_dot([psi.conjugate(), sx, psi])
    drive = h0 + h * np.cos(omega * t)
    jac = (1j) * (sx_mf * sx + drive * sz)
    return jac

def mf_func (psi, t, h0, h, omega):
    return np.dot(mf_jac(psi, t, h0, h, omega), psi)

def tls_jac (psi, t, h0, h, omega):
    drive = h0 + h * np.cos(omega * t)
    jac = (1j) * (sx + drive * sz)
    return jac

def tls_func (psi, t, h0, h, omega):
    return np.dot(tls_jac(psi, t, h0, h, omega), psi)

def sd_mag_mf(times, psi0, h0, h, omega):
    """
    Returns Mean Field Standard Deviaion of Sz magnetization for a
    particular drive frequency and amplitude
    """
    if verbose:
        print("h = %2.3lf, w = %2.3lf" % (h, omega))
    sol_mf = odeintw(mf_func, psi0, times, args=(h0, h, omega), Dfun=mf_jac)
    #calculate expectation values
    mz_mf = np.einsum("ij,jk,ik->i", sol_mf.conjugate(), sz, sol_mf)
    return np.std(mz_mf.real)

def sd_mag_tls(times, psi0, h0, h, omega):
    """
    Returns Mean Field Standard Deviaion of Sz magnetization for a
    particular drive frequency and amplitude
    """
    if verbose:
        print("h = %2.3lf, w = %2.3lf" % (h0, omega))
    sol_tls = odeintw(tls_func, psi0, times, args=(h0, h, omega), Dfun=tls_jac)
    #calculate expectation values
    mz_tls = np.einsum("ij,jk,ik->i", sol_tls.conjugate(), sz, sol_tls)
    return np.std(mz_tls.real)

def maxfft_mag_mf(t, psi0, h0, h, omega):
    if verbose:
        print("h = %2.3lf, w = %2.3lf" % (h, omega))
    sol_mf = odeintw(mf_func, psi0, t, args=(h0, h, omega), Dfun=mf_jac)
    #calculate expectation values
    mz_mf = np.einsum("ij,jk,ik->i", sol_mf.conjugate(), sz, sol_mf)
    spectrum_mf = np.fft.fftshift(np.fft.fft(mz_mf - np.average(mz_mf)))
    peaks, _ = find_peaks(np.abs(spectrum_mf), height=pheightmin)
    return np.amax(np.abs(spectrum_mf[peaks]))
    
def maxfft_mag_tls(t, psi0, h0, h, omega):
    if verbose:
        print("h = %2.3lf, w = %2.3lf" % (h, omega))
    sol_tls = odeintw(tls_func, psi0, t, args=(h0, h, omega), Dfun=tls_jac)
    #calculate expectation values
    mz_tls = np.einsum("ij,jk,ik->i", sol_tls.conjugate(), sz, sol_tls)
    spectrum_tls = np.fft.fftshift(np.fft.fft(mz_tls - np.average(mz_tls)))
    peaks, _ = find_peaks(np.abs(spectrum_tls), height=pheightmin)
    return np.amax(np.abs(spectrum_tls[peaks]))


if __name__ == '__main__':
    nprocs = 13
    evals, evecs = eig((1j) * tls_jac(None, 0.0, 0.0, 0.0, 0.0))
    psi0 = evecs[:,np.argmin(evals)].copy()
    h = 25.0
    h0 = 0.1
    omegas = np.linspace(20.0, 21.0,  3 * nprocs)
    t = np.linspace(0, 500, 20000)
    p = Pool(processes = nprocs)
    dc = h0
    print("TLS dynamics for h0 = %f, w with %d processes ..." % (dc, nprocs))
    dq_tls = p.starmap(maxfft_mag_tls,[(t, psi0, dc, h, w) for w in omegas])
    print("MF dynamics for h0 = %f, w with %d processes ..." % (dc, nprocs))
    dq_mf = p.starmap(maxfft_mag_mf,[(t, psi0, dc, h, w) for w in omegas])
    fname = "mf_tls_driven_hd_{}".format(h) + "_h0_{}".format(h0) + "_pheight.csv"
    np.savetxt(fname,np.vstack((omegas, dq_mf, dq_tls)).T,delimiter=',')
    print("Data saved in file: " + fname)
