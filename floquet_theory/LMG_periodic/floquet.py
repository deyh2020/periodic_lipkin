#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 17 22:32:38 2021
Copyright 2021 Analabha Roy (daneel@utexas.edu):
Released under the MIT license @ https://opensource.org/licenses/MIT
"""
from scipy.special import jn_zeros
import numpy as np
import matplotlib.pyplot as plt
from qutip import destroy, floquet_modes
from qutip.parallel import parallel_map
from qutip.solver import Options
import qutip.settings as qset

options = Options()
options.nsteps = 1e4
plt.rcParams.update({
    "figure.figsize": (10, 10),
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.size": 22,
    "font.sans-serif": ["Helvetica"]})

# Define paramters
N = 30  # number of basis states to consider
a = destroy(N)
time_period = 0.2
h0 = 0.1
# Check these!!!
q = (a.dag() + a)/N

p = (1j) * (a.dag() - a)
omega = 2 * np.pi/time_period
args = {'w': omega}

freezing_pts = jn_zeros(0, 4)

H0 = -2.0 * q**2 + h0 * (1.-4.*q**2).sqrtm() * p.cosm()


def floquet_esys(ampl):
    H1 = ampl * (1.-4.*q**2).sqrtm() * p.cosm()
    H = [H0, [H1, lambda t, args: np.cos(args["w"]*t)]]
    f_modes_0, f_energies = floquet_modes(H, time_period, args, True)
    return f_energies


ampls = np.linspace(1.0, 250.0, 300)
quasi_evals = parallel_map(floquet_esys, ampls, progress_bar=True)
quasi_evals = np.array(quasi_evals)

np.savetxt("N_%d_w_%1.3lf.csv" % (N, omega),
           np.vstack((ampls, quasi_evals.T)).T, delimiter=',')

for i in range(N):
    # plt.scatter(ampls/args["w"], quasi_evals[:, i], s=0.1, c='b')
    plt.plot(4.0 * ampls/args["w"], quasi_evals[:, i], c='b')

for pt in freezing_pts:
    plt.axvline(x=pt, color='gray', linestyle="--")

plt.xlabel(r'$4h\;N/\omega$')

print("Number of processors = ", qset.num_cpus)
plt.show()
#plt.savefig("N_%d_w_%1.3lf.png" % (N, omega))
