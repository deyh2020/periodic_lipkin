import numpy as np
from scipy.integrate import solve_ivp
from numpy.linalg import eig

def floq_jac(t, psi0, h, h0, H1, H0, w):  
    n,m = H0.shape    
    drive = h0 + h * np.cos(w * t)
    jac = 1j * (H0  + drive * H1)             
    return jac
        
def floq_func(t,psi0, h, h0, H1, H0, w):
    floq_h = np.dot(floq_jac(t,psi0, h, h0, H1, H0, w), psi0)
    return floq_h

def floq_evolv(H0,H1,h,h0,w):
    n,m = H0.shape
    psi = np.eye(n) + (1j)* np.eye(n)
    T = 2 * np.pi/w                                  
    floqEvolution_mat = np.zeros((n,n)) + 1j * np.zeros((n,n))
    for m in np.arange(n):
        psi0 = psi[m]/(psi[m].T.conj() @ psi[m])
        sol = solve_ivp(floq_func,(0,T),psi0,t_eval=[T], method='RK45',args=(h, h0, H1, H0, w), dense_output=True)
        psi_T = sol.y[:,0]
        floqEvolution_mat[m] = psi_T/(psi_T.T.conj() @ psi_T)
    evals, evecs = eig(floqEvolution_mat)
    phasefunc = 1j * np.log(evals + 1j * 0) /T           
    return [h,phasefunc.real,evecs]

def get_hamiltonians(N):
    s = N/2.0
    ms = np.arange(-s,s+1)
    KacNorm = 2.0/(N-1)    
    H0 = KacNorm * np.diagflat(ms**2)
    ms_p = ms[0:N]
    ms_m = ms[1:]
    H1 = (np.diagflat(np.sqrt(s * (s+1) - ms_p * (ms_p+1)), k=1) + np.diagflat(np.sqrt(s * (s+1) - ms_m * (ms_m-1)), k=-1))
    return H0, H1