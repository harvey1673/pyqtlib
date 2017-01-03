#-*- coding:utf-8 -*-
from math import log, exp
from repoze.lru import lru_cache
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq as solver, minimize_scalar 
import QuantLib
from curve import DiscountCurve
from misc import *
from finite_diff import FDM1D
from rate_options import BlackLognormalModel

class Volatility:
    def __init__(self, startdate, sigma=0.0, kappa=0.0):
        self.kappa = kappa if abs(kappa) > 5e-5 else 0.0 # this is about the optimal cutoff value
        self.startdate = startdate # startdate >= 0
        # t[0]=0     t[1]       t[2]       t[n-1]     t[n]=999
        #  |----------|----------|--- ... --|----------|     
        #  |---v[1]---|          .          .          .
        # v[0]         ---v[2]---|          .          .
        #                         --- ... --|          .
        #                                    ---v[n]---|
        self.tenors = startdate.t + np.array([0, 999.]) # tenors[0] can be greater than 0
        self.vars = np.array([sigma ** 2] * 2) # variances
        self.__update()
    
    def __update(self,):           
        self.__nodes.cache_clear()
        self.phi.cache_clear()
        self.chi.cache_clear()
    
    def __setitem__(self, t, sigma):
        """update piecewise constant volatility of the longest maturity"""
        if t == self.tenors[-2]: # updating, note that tenors[-1] = startdate + 999
            self.vars[-2] = sigma ** 2.
        elif t > self.tenors[-2]: # bootstrapping, insert one more entry
            self.tenors = np.insert(self.tenors, -1, t)  
            self.vars = np.insert(self.vars, -1, sigma ** 2.)        
        else:
            raise BaseException('tenor must be >= current tenor structure ...')        
        self.vars[0] = self.vars[1]
        self.vars[-1] = self.vars[-2]
        self.__update()
         
    def sig_sq(self, t):
        return self.vars[np.searchsorted(self.tenors, t)]

    def bond_exp(self, s, t, T, x):
        """
        bond exponential under risk neutral measure: x ~ N(0, xi2(s,t,t,T))
        P(t,T) = P(s,t,T) * exp(-xi2(s,t,t,T)/2 - B(t,T)*chi(s,t,t,t) - B(t,T)*x)
        s0 < s < t < T, s & t must be scalar
        """
        return np.exp(-self.xi2(s,t,t,T) / 2 - self._b(t,T) * (self.chi(s,t,t,t) + x)) 

    def xi2(self, s, t, T, V): 
        """
        variance of forward bond P(t,T,V):
        xi2(s,t,T,V) = integral{(B(u,V)-B(u,T))^2*sigma(u)^2 * du} / 2 from s to t
                     = B(T,V)^2 * phi(s,t,T,T)
        s0 < s < t < T < V, s & t must be scalar
        """
        return self._b(T,V) ** 2 * self.phi(s,t,T,T)

    @lru_cache(maxsize=100)
    def __nodes(self, t): # t must be scalar
        i = np.searchsorted(self.tenors, t) + 1
        tnodes = self.tenors[:i] * 1 # make a copy, faster than numpy.copy()
        tnodes[-1] = t
        vnodes = self.vars[1:i]
        return tnodes, vnodes

    def __gamma(self, t, n):
        if n == 0.0:
            x = t
        elif self.kappa == 0.0:
            x = t ** (n + 1) / (n + 1)
        else:
            x = np.exp(n * self.kappa * t) / (n * self.kappa) 
        return np.diff(x)

    @lru_cache(maxsize=100)
    def phi(self, s, t, T, V):
        """
        phi(s,t,T,V) = integral{E(u,T)*E(u,V)*sigma(u)^2*du} from s to t
        s0 < s < t < T < V, s & t must be scalar
        """  
        def fn(u):
            a, b = self.__nodes(u)
            if self.kappa == 0.0:
                return b.dot(self.__gamma(a,0))
            else:
                return b.dot(self.__gamma(a,2)) * exp(-self.kappa * (T + V))
        return fn(t) if s is None else fn(t) - fn(s) # s is None means s = tenor[0]

    @lru_cache(maxsize=100)
    def chi(self, s, t, T, V):
        """
        chi(s,t,T,V) = integral{E(u,T)*B(u,V)*sigma(u)^2*du} from s to t
        s0 < s < t < T < V, s & t must be scalar
        """
        k = self.kappa
        g = self.__gamma
        def fn(u):
            a, b = self.__nodes(u)
            if k == 0.0:
                return b.dot(V * g(a,0) - g(a,1))
            else:
                return b.dot(exp(-k * T) * g(a,1) - exp(-k * (T + V)) * g(a,2)) / k
        return fn(t) if s is None else fn(t) - fn(s) # s is None means s = tenor[0]

    def psi(self, s, t, T, V): 
        """
        psi(s,t,T,V) = integral{B(u,T)*B(u,V)*sigma(u)^2*du} from s to t 
        s0 < s < t < T < V, s & t must be scalar
        """
        k = self.kappa
        g = self.__gamma
        def fn(u):
            a, b = self.__nodes(u)
            if k == 0.0:
                return b.dot(T * V * g(a,0) - (T + V) * g(a,1) + g(a,2))
            else:
                return b.dot(g(a,0) - (exp(-k * T) + exp(-k * V)) * g(a,1) + exp(-k * (T + V)) * g(a,2)) / k ** 2
        return fn(t) if s is None else fn(t) - fn(s) # s is None means s = tenor[0]

    def xvol(self, s=None, t=None): # total volatility of x
        if t is None: t = self.tenors[-2]  # = maturity 
        return self.phi(s,t,t,t) ** 0.5 

    def _b(self, t, T):
        """ B(t,T) = (1 - E(t,T)) / kappa """
        return T - t if self.kappa == 0.0 else (1 - np.exp(-self.kappa * (T - t))) / self.kappa

    def _e(self, t, T):
        """ E(t,T) = exp(-kappa * (T-t)) """
        return 1.0 if self.kappa == 0.0 else exp(-self.kappa * (T - t)) 

    def __repr__(self):
        info = [str(self.startdate.origin + round(t * 365)) + ":\n    " + str(v ** 0.5) for t, v in zip(self.tenors[:-1], self.vars[1:])]
        return '\n'.join(info)

    def test_volatility(self,s=.1,t=.3,T=.5,V=.7):
        a = self.psi(s,t,V,V) - self.psi(s,t,T,T)
        b = self.xi2(s,t,T,V) + 2 * self._b(T,V) * self.chi(s,t,T,T) 
        print(a, b, log(a / b))


class HullWhite1F:
    def __init__(self, proj, disc, spotdate, kappa=0, z_fwd_date=None):
        self.disc = disc
        self.proj = proj
        self.spotdate = spotdate
        self.vol = Volatility(startdate=spotdate, kappa=kappa)
        self.z_fwd_date = z_fwd_date

    # minimization function
    def __minimize(self, fn, bounds):
        return minimize_scalar(fn, bounds=bounds, method='bounded').x
    
    def curves(self, t, x):
        # x adjustment from Z-forward to risk neutral: x_zf = x_rn + chi(s,t,t,Z)
        if self.z_fwd_date is not None:
            x -= self.vol.chi(None,t,t,self.z_fwd_date.t)
        def bump(curve):
            curve_t = curve(t)
            def newcurve(T): # T can be float or Numpy.Array 
                return curve(T) / curve_t * self.vol.bond_exp(None,t,T,x)
            return Curve.from_fn(t, newcurve) 
        return bump(self.proj), bump(self.disc)
    
    #@np.vectorize
    def fn_abc(self, t, x):
        """PDE: dU/dt = a * d2U/dx2 + b * dU/dx + c * U"""
        if self.z_fwd_date is None:
            a = self.vol.sig_sq(t) / -2.
            b = self.vol.kappa * x
            c = self.disc.forward(t) + self.vol.chi(None,t,t,t) + x
        else:
            s2 = self.vol.sig_sq(t)
            B_tZ = self.vol._b(t, self.z_fwd_date.t)
            a = s2 / -2.
            b = -B_tZ * s2 + self.vol.kappa * x
            c = self.disc.forward(t) - B_tZ * self.vol.phi(None,t,t,t) + x
        return a, b, c  
    
    def __caplet_value(self, libor, strike): 
        ts = libor.effdate.t # offset 2 days, so start from startdate, this makes tf = ts
        te = libor.matdate.t
        xi = self.vol.xi2(None,ts,ts,te) ** 0.5 # == xi2(s0,ts,ts,te), and xi2(s0,ts,ts,ts) = 0
        p = self.proj(ts) / self.proj(te)
        q = (1 + strike * libor.coverage)        
        zstar = log(q / p) / xi - xi / 2.
        return self.disc(te) * (p * norm.cdf(-zstar) - q * norm.cdf(-zstar - xi))
    
    def __payer_swpn_value(self, swap, strike):
        def eta(x): #constant multiplicative spread between proj and disc curve
            return self.disc(x.matdate.t) / self.disc(x.effdate.t) * self.proj(x.effdate.t) / self.proj(x.matdate.t) 
        # floating leg cashflows
        dateflows = [(swap.rleg.effdate, 1.), (swap.rleg.matdate, -1.)] # P(T,te ) - P(T,tm )
        dateflows += [(p.index.effdate, eta(p.index) - 1.) for p in swap.rleg.cp]
        # fixed leg cashflows
        dateflows += [(p.paydate, -strike * p.accr_cov) for p in swap.pleg.cp]

        t = swap.effdate.t # option expiry (should have been swap.fixdate, but has been offset to swap.effdate) 
        def disc_tf(T, z): # bond price under t-forward measure, a martingale 
            # bond price: P(t,T)/P(s,t,T) = exp(-xi2(s,t,t,T)/2 - xi(s,t,t,T)*z),  z ~ N(0,1)
            xi2 = self.vol.xi2(None,t,t,T)
            return self.disc(T) / self.disc(t) * np.exp(-xi2 / 2 - xi2 ** 0.5 * z)

        def find_zstar(z):            
            return sum(cf * disc_tf(d.t,z) for d, cf in dateflows) # sum of cashflows = 0
        bound = 5.0
        while True:
            try:                
                zstar = solver(find_zstar, -bound, bound) # z can be regarded as a standard normal
                break
            except ValueError:
                print("bound exception: z=%s" % bound)
                bound *= 2 
        return sum(cf * self.disc(d.t) * norm.cdf(-zstar - self.vol.xi2(None,t,t,d.t) ** 0.5) for d, cf in dateflows)
    
    @classmethod
    def from_calibration(cls, proj, disc, volsurf, kappa=0.0, z_fwd_date=None, tradedate=None, tenor=None, lastdate=None):
        vanillas = volsurf.get_calibration_instruments(tradedate=tradedate, tenor=tenor, lastdate=lastdate)        
        if PRINT:
            print('effdate\t matdate\t rate\t atm_value\t blackvol')
            for prod in vanillas:
                print('%s\t %s\t %s\t %s\t %s' % (prod.underlier.effdate, prod.underlier.matdate, 
                                                  prod.forward, prod.value(), prod.stdev / prod.sqr_term))
        hw = cls(proj, disc, spotdate=volsurf.spotdate, kappa=kappa, z_fwd_date =z_fwd_date) # 
        hw.calibrate_to_vanillas(volsurf.mode, vanillas)
        if PRINT: print(hw.vol)
        return hw 

    def calibrate_to_vanillas(self, mode, vanillas, strike=None):
        if mode == 'capfloor':
            hw_model_value = self.__caplet_value 
        elif mode == 'swaption': 
            hw_model_value = self.__payer_swpn_value
        else:
            raise BaseException('invalid calibration instruments ...') 

        for opt in vanillas:
            if isinstance(opt, BlackLognormalModel) and opt.forward <= 0: 
                continue # ignore negative rates for lognormal process
            t = opt.underlier.effdate.t # shifted by spotlag, rather than using fixdate
            k = opt.forward if strike is None else strike
            black_model_value = opt.value(strike=k) #* 1.003
            def find_sigma(sigma):
                self.vol[t] = sigma
                return (hw_model_value(opt.underlier, strike=k) - black_model_value) ** 2
            self.vol[t] = self.__minimize(find_sigma, bounds=(1e-6, 0.5))  # set to solved sigma    

    def finite_difference(self, nt, nx, ns):
        return FDM1D(self.vol.xvol(), nt=nt, nx=nx, ns=ns, fn_abc=self.fn_abc)
       
    def monte_carlo(self, timeline, npaths, rng='sobol', seed=None):
        np.random.seed(seed)
        def sample_generator(dim): # generate normal samples        
            if rng == 'sobol':
                sobol = QuantLib.SobolRsg(dim) # not repeatable if dim > 16 or so (no idea why)  
                zmat = np.array([sobol.nextSequence().value() for i in range(npaths)])
                zmat = norm.ppf(np.mod(zmat + np.random.uniform(size=dim), 1)).T # shifted by uniform(0,1)
                for row in zmat: # yield row by row, each row is a time slice
                    yield row
            elif rng == 'basic':
                for i in range(dim):
                    yield np.random.standard_normal(npaths)
            else:
                raise BaseException('invalid random number generator ...')

        tline = [d.t for d in timeline] # convert from Date's to float's
        steps = len(tline) - 1 # excluding spotdate
        vol = self.vol 
        if self.z_fwd_date is None: # risk neutral measure
            sample = sample_generator(2 * steps) 
            x = np.empty((steps, npaths)) #each row is a time slice, each column is a path  
            y = np.empty_like(x) 
            xv = yv = 0
            for i in range(steps): # generate paths
                v, t = tline[i:i + 2]
                phi = vol.phi(v,t,t,t) ** 0.5          
                psi = vol.psi(v,t,t,t) ** 0.5  
                r0 = vol.chi(v,t,t,t) / phi / psi # correlation
                r1 = (1 - r0 ** 2) ** 0.5
                z0 = next(sample) 
                z1 = next(sample)                     
                yv += vol._b(v,t) * xv + psi * (r0 * z0 + r1 * z1) # update yv using old xv
                xv = vol._e(v,t) * xv + phi * z0 # update xv                 
                x[i,:] = xv
                y[i,:] = yv
        
            def fetch_state(date): # fetch the state vector @ date
                if date == timeline[0]: # date is spotdate 
                    return 0.0, 1.0
                else:
                    s, t = timeline[0].t, date.t
                    i = np.searchsorted(timeline, date) - 1
                    spot = self.disc(s) / self.disc(t)  
                    return x[i,:], spot * np.exp(vol.psi(s,t,t,t) / 2 + y[i,:]) # stochastic factor and numeraire 
        else: # z-forward measure
            sample = sample_generator(steps)
            x = np.empty((steps, npaths)) #each row is a time slice, each column is a path
            xv = 0  
            for i in range(steps): # generate paths
                v, t = tline[i:i + 2]                                      
                xv = vol._e(v,t) * xv + next(sample) * vol.phi(v,t,t,t) ** 0.5 
                x[i,:] = xv
                
            def fetch_state(date): # fetch the state vector @ date
                s, t, z = timeline[0].t, date.t, self.z_fwd_date.t
                spot = self.disc(z) / self.disc(t)
                if date == timeline[0]: # date is spotdate 
                    return 0.0, spot
                else:                      
                    i = np.searchsorted(timeline, date) - 1
                    return x[i,:], spot * np.exp(vol.xi2(s,t,t,z) / 2 - vol._b(t,z) * x[i,:]) # stochastic factor and numeraire 

        return fetch_state 

if __name__ == '__main__':
    today = Date.set_origin(16,7,2014)
    d = today + 5
    dd = d
    vol = Volatility(startdate=d, kappa=0.0)
    d += 50
    vol[d] = 0.01
    d += 50
    vol[d] = 0.02
    d += 50
    vol[d] = 0.03
    d += 50
    vol[d] = 0.04
    d += 50
    vol[d] = 0.05
    d += 50
    vol[d] = 0.06

    vol.test_volatility()
    pass

     