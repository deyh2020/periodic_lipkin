# coding: utf-8
#/usr/bin/env python3
__doc__ = """
This Python program calculates and plots the time-periodically driven mean field dynamics 
of the quantum transverse field Ising Model (Lipkin - Meshkov- Glick model) as detailed in [1,2]. 
It also evaluates long time averages and sd of sigma^z and plots against drive frequency omega.

Refs:

[1] Lipkin, H. J., N. Meshkov, and A. J. Glick, Nuclear Physics 62, no. 2 (February 1, 1965): 188–98. 
    https://doi.org/10/fpqf4q .
[2] Mori, T. Accessed October 30, 2019. https://arxiv.org/abs/1810.01584.
"""

import numpy as np
from odeintw import odeintw
from numpy.linalg import multi_dot, eig
from multiprocessing import Pool
import time
from scipy.interpolate import InterpolatedUnivariateSpline
import json

#Problem Parameters
h0 = 25.0
omegas = np.linspace(0., 3.5 * h0, 5)
maxtime = 500
t = np.linspace(0, maxtime, 20000)
fname = "minimum_sz_variance_time_" + str(maxtime) + "_jsonfile.json"
nvals_amps = 3
nvals_omegas = 6
threshold = 0.015
h0 = 25.0
nprocs = 3

amps = np.linspace(h0/2, h0, nvals_amps)
omegas = np.linspace(0., 3.5 * h0, nvals_omegas)

verbose = True
#Pauli Matrices
sx = np.array([[0, 1],[ 1, 0]])
sy = np.array([[0, -1j],[1j, 0]])
sz = np.array([[1, 0],[0, -1]])

def mf_jac (psi, t, h0, omega):
    """
    The Jacobian of the Schrodinger Equation for Mean field dynamics
    """
    sx_mf = multi_dot([psi.conjugate(), sx, psi])
    jac = (1j) * (sx_mf * sx + h0 * np.cos(omega *t)* sz)
    return jac

def mf_func (psi, t, h0, omega):
    return np.dot(mf_jac(psi, t, h0, omega), psi)

def tls_jac (psi, t, h0, omega):
    """
    The Jacobian of the Schrodinger Equation for Two-Level dynamics
    """
    jac = (1j) * (sx + h0 * np.cos(omega *t)* sz)
    return jac

def tls_func (psi, t, h0, omega):
    return np.dot(tls_jac(psi, t, h0, omega), psi)

def avg_mag_mf(times, psi0, h0, omega):
    """
    Returns Mean Field Average of Sz magnetization for a
    particular drive frequency and amplitude
    """
    sol_mf = odeintw(mf_func, psi0, times, args=(h0, omega), Dfun=mf_jac)
    #calculate expectation values
    mz_mf = np.einsum("ij,jk,ik->i", sol_mf.conjugate(), sz, sol_mf)
    return np.average(mz_mf.real)

def sd_mag_mf(times, psi0, h0, omega):
    """
    Returns Mean Field Standard Deviaion of Sz magnetization for a
    particular drive frequency and amplitude
    """
    if verbose:
        print("h = %2.3lf, w = %2.3lf" % (h0, omega))
    sol_mf = odeintw(mf_func, psi0, times, args=(h0, omega), Dfun=mf_jac)
    #calculate expectation values
    mz_mf = np.einsum("ij,jk,ik->i", sol_mf.conjugate(), sz, sol_mf)
    return np.std(mz_mf.real)


if __name__ == '__main__':
    evals, evecs = eig((1j) * tls_jac(None, 0.0, 0.0, 0.0))
    psi0 = evecs[:,np.argmin(evals)].copy()
        
    t = np.linspace(0, 500, 20000)
    start = time.time()
    p = Pool(processes = nprocs)
    frozen_omegas = {}
    frozen_omegas["ampls"] = amps.tolist()
    
    wherefrozen = []
    for h in amps:
        if verbose:
            print("Amplitude:", h, "Exec time:", time.time() - start)
        dq_mf = p.starmap_async(sd_mag_mf,[(t, psi0, h, w) for w in omegas]).get()
        if verbose:
            print(dq_mf)
        #Interpolate the stdev and find the minima
        #SciPy has a built-in method to find the roots of a cubic spline.
        #So, Use a 4th degree spline for interpolation, so that the roots of its derivative can be found easily.
        f = InterpolatedUnivariateSpline(omegas, dq_mf, k=4)
        cr_pts = f.derivative().roots() #Get the extrema
        cr_vals = f(cr_pts)
        dderiv = f.derivative(n=2)
        mins = cr_pts[(dderiv(cr_pts) > 0) & (cr_vals <= threshold)] #Select only minima near 0
        if verbose:
            print("Minimum sz variance at omega vals:", mins)            
        wherefrozen.append(mins.tolist())
    
    frozen_omegas["minfreqs"] = wherefrozen
    json = json.dumps(frozen_omegas)
    f = open(fname,"w")
    f.write(json)
    f.close()
    
    elapsed = (time.time() - start)  
    print("Time Elapsed = ", elapsed)
    
    

    
    