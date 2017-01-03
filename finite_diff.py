#-*- coding:utf-8 -*-
from math import log
import numpy as np 
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve 
from misc import *

class FDM1D:
    __theta__ = 0.5 # 0.5 for Crank-Nicolson
    """PDE: dU/dt = a * d2U/dx2 + b * dU/dx + c * U"""
    def __init__(self, stdev, fwd=1.0, nt=300, nx=300, ns=4.5, fn_abc=None, bound=(-np.inf, np.inf)):
        #nt, nx, ns: timesteps per year, spotsteps, times of stdev 
        self.nt = nt        
        self.bound = bound
        xl = max(bound[0], log(fwd) - stdev * ns)
        xu = min(bound[1], log(fwd) + stdev * ns) 
        self.xv = HashableArray(np.linspace(xl, xu, max(nx,50) + 1))
        self.dx = (self.xv[-1] - self.xv[0]) / (self.xv.size - 1)
        self.fn_abc = fn_abc # pde coefficient function          
        self.eye = sparse.eye(self.xv.size) # identity matrix

    def set_boundary_functions(self, fn_bc):
        self.fn_bc = fn_bc

    def __set_vector_and_matrix(self, t, yv):
        #                | * *   |                  | 0 0   |
        # Linearity: M = | * * * |,  Dirichlet: M = | * * * |
        #                |   * * |                  |   0 0 | 
        a, b, c = self.fn_abc(t, self.xv)
        a /= self.dx ** 2
        b /= self.dx * 2
        dm = (a - b)[1:]
        d0 = c - a * 2                
        dp = (a + b)[:-1]
        if self.bound[0] == -np.inf: # Linearity BC, unimportant, because at 5 sigmas the probility is negligible
            d0[0] = c[0] - b[0] * 2 
            dp[0] = b[0] * 2   
        else: # Dirichlet BC 
            d0[0] = dp[0] = 0 
            yv[0] = self.fn_bc[0](t)    
        if self.bound[1] == np.inf: # Linearity BC
            d0[-1] = c[-1] + b[-1] * 2
            dm[-1] = -b[-1] * 2
        else: # Dirichlet BC
            d0[-1] = dm[-1] = 0
            yv[-1] = self.fn_bc[1](t) 
        return yv, sparse.diags((dm,d0,dp), (-1,0,1))

    def evolve(self, start, end, yv, x=None): # forward PDE if dt > 0 otherwise backward PDE 
        if start > end: # tiny number is added to minimize the error due to piecewise constant vol function
            start *= 1 - 1e-14
            end *= 1 + 1e-14
        else:
            start *= 1 + 1e-14
            end *= 1 - 1e-14
        tv = np.linspace(start, end, max(abs(start - end) * self.nt, 25) + 1)
        dt = (tv[-1] - tv[0]) / (tv.size - 1) # dt can be positive or negative
        wp = dt * FDM1D.__theta__   
        wn = dt - wp
        yv, mm = self.__set_vector_and_matrix(start, yv)         
        if x is None:            
            for t in tv[1:]:
                b, mm = self.__set_vector_and_matrix(t, yv + mm * yv * wp)                               
                yv = spsolve(self.eye - mm * wn, b)
            return yv
        else:
            fv = np.empty((tv.size, x.size)) # initiate a vector
            fv[0,:] = self.functionize(yv)(x)
            for t, f in zip(tv[1:], fv[1:]):
                b, mm = self.__set_vector_and_matrix(t, yv + mm * yv * wp)                               
                yv = spsolve(self.eye - mm * wn, b)
                f[:] = self.functionize(yv)(x)
            return yv, [LinearFlat(tv, f) for f in fv.T]

    def functionize(self, yv):
        #return CubicSplineFlat(self.xv, yv)
        return LinearFlat(self.xv, yv)

if __name__ == '__main__':
    pass

