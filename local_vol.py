#-*- coding:utf-8 -*-
from math import log, exp, sqrt
import numpy as np
from scipy.optimize import minimize 
from scipy.stats import norm 
from misc import * 
from curve import DiscountCurve
from finite_diff import *
import pandas  

from repoze.lru import lru_cache

def plot(t, x, y):
    import matplotlib.pyplot as pyplot     
    pyplot.plot(x, y, 'r.')  
    pyplot.title('t = %s' % t)   
    pyplot.show()  

def ascending_pairs(*args):
    steps = np.sort(np.unique(np.hstack(args)))
    return zip(steps[:-1], steps[1:])

def descending_pairs(*args):
    steps = np.sort(np.unique(np.hstack(args)))[::-1]
    steps = np.hstack((steps, steps[-1])) # a redundent t0 pillar to ensure construction of full solution grid at spot time
    return zip(steps[:-1], steps[1:])

class Equity:
    class Dividend:
        def __init__(self, kind, ex=None, pv=None, cont=None):
            self.kind = kind # kind of dividends
            self._ex = np.array([]) if ex is None else ex 
            self._fixed = np.array([]) if pv is None else pv 
            self._cumu = np.cumsum(self._fixed)
            self.cont = cont

        def ex(self, t=-999, T=999):
            if self.is_fixed:
                return self._ex[(t < self._ex) & (self._ex <= T)]
            else:
                return np.array([])
             
        def fixed(self, t):
            return self._fixed[self._ex == t]
          
        def cumu(self, t):
            cumu = self._cumu[self._ex <= t]
            return cumu[-1] if cumu.size > 0 else 0
        
        @property
        def is_fixed(self):
            return self.kind == 'fixed' # else 'continuous'
        
        @property
        def is_cont(self):
            return self.kind == 'continuous'

        def rate(self, t):
            if self.is_cont:
                return self.cont.forward(t)
            else:
                return 0.0

    def __init__(self, tradedate, spot, disc, div, calendar, spotlag='3D', paylag='3D'):
        self.tradedate = tradedate
        self.t0 = tradedate.t
        self.spot = spot
        self.disc = disc
        self.div = div
        self.calendar = calendar
        self.spotlag = Period(spotlag)
        self.paylag = Period(paylag)

    @classmethod
    def load(cls, filename):        
        basic = pandas.read_excel(filename, 'basic')
        today = Date.from_timestamp(basic.value['today'])
        calendar = Calendar.fetch(basic.value['calendar'])
        spot = basic.value['spot']
        spotlag = basic.value['spotlag']
        paylag = basic.value['paylag']        
        disc = DiscountCurve.load(filename, 'curve')

        dividend = basic.value['dividend']
        sheet = pandas.read_excel(filename, 'div')
        if dividend == 'fixed':              
            ex = np.array([Date.from_timestamp(d).t for d in sheet['ex.div']])
            amt = sheet['dividend'].tolist()
            pay = [Date.from_timestamp(d).t for d in sheet['payment']]
            pv = np.array([c * disc(t) for c, t in zip(amt, pay)])
            div = cls.Dividend(kind='fixed', ex=ex, pv=pv) 
        elif dividend == 'continuous':
            tenors = np.array([Date.from_timestamp(d).t for d in sheet['tenor']])
            rates = np.array(sheet['yield'].tolist())
            dfs = np.exp(-rates * (tenors - tenors[0]))
            cont = DiscountCurve.from_array(tenors, dfs, interp_mode=DiscountCurve.InterpolationMode.LinearZero)
            div = cls.Dividend(kind='continuous', cont=cont)
        else:
            div = None 
        return cls(today, spot, disc=disc, div=div, calendar=calendar, spotlag=spotlag, paylag=paylag)

    def roll_date(self, date, period):
        return self.calendar.advance(date, period, DayRoll.ModifiedFollowing)

    def forward_npv(self, date):
        t1 = self.roll_date(self.tradedate, self.spotlag).t
        t2 = self.roll_date(date, self.paylag).t
        if self.div.is_fixed:
            return (self.spot * self.disc(t1) - self.div.cumu(date.t)) / self.disc(t2)
        elif self.div.is_cont:
            return self.spot * self.div.cont(t2) / self.disc(t2) * self.disc(t1) / self.div.cont(t1)

    @lru_cache(maxsize=None)        
    def forward(self, t): # excluding dividend ex-ed at t
        if self.div.is_fixed:
            return (self.spot - self.div.cumu(t)) / self.disc(t)
        elif self.div.is_cont:
            return self.spot * self.div.cont(t) / self.disc(t) * self.disc(self.t0) / self.div.cont(self.t0)

    def fixed_div(self, t): # excluding dividend ex-ed at t
        return self.div.fixed(t) / self.disc(t)


class ImpliedVol:
    class Smile(FlatCubicSpline):          
        def __init__(self, fwd, strikes, vols):
            super(ImpliedVol.Smile, self).__init__(strikes, vols)
            self.fwd = fwd
            
        def __call__(self, k): # k = log(strike/forward), log-moneyness
            return super(ImpliedVol.Smile, self).__call__(self.fwd * np.exp(k))

    def __init__(self, equity):
        self.equity = equity                         
        self.tenors = None
        self.smiles = None
            
    def set(self, t, strikes, vols):
        """update implied volatility smile of the longest maturity""" 
        smile = self.Smile(self.equity.forward(t), strikes, vols)
        if self.tenors is None:
            self.tenors = np.array([t])
            self.smiles = np.array([smile])
        elif t == self.tenors[-1]: # updating
            self.smiles[-1] = smile
        elif t > self.tenors[-1]: # bootstrapping, insert one more entry
            self.tenors = np.append(self.tenors, t)  
            self.smiles = np.append(self.smiles, smile)        
        else:
            raise BaseException('tenor must be >= current tenor structure ...')        
        return self 
        
    @classmethod 
    def load(cls, equity, filename, sheetname): # load implied vol from file
        sheet = pandas.read_excel(filename, sheetname)
        strikes = np.array(sheet.index) * equity.spot
        ivol = cls(equity)
        for expiry in sheet.columns:
            matdate = equity.roll_date(equity.tradedate, Period(expiry))
            ivol.set(matdate.t, strikes, sheet[expiry].values / 100.)
        return ivol

    def __call__(self, t, z): # interpolate implied vol; not (yet) interpolated in time dimension
        i = np.searchsorted(self.tenors[:-1], t) 
        return self.smiles[i](z)

    def stdev(self, t, k=0): # standard deviation given expiry and log-moneyness
        return self(t, k) * (t - self.equity.t0) ** 0.5  

    def expiries(self, start, end): # tenors between start and end time
        return self.tenors[(start < self.tenors) & (self.tenors < end)]

    def norm_forward_call(self, t, k): # normalized forward Black-Scholes call
        stdev = self.stdev(t, k)
        dp = -k / stdev + stdev / 2 
        return norm.cdf(dp) - np.exp(k) * norm.cdf(dp - stdev)

class LocalVol:
    class Smile(LinearFlat):
        grid = np.linspace(-1.5,1.5,7) # 7 strike pillers + 2 extended
        def __init__(self, stdev, vol):
            k = stdev * self.grid
            v = k * 0 + vol # make a copy
            super(LocalVol.Smile, self).__init__(k, v)

    def __init__(self, ivol):
        self.ivol = ivol                        
        self.tenors = None
        self.smiles = None
    
    def set(self, t, vol):
        """update volatility smile of the longest maturity""" 
        smile = self.Smile(self.ivol.stdev(t), vol)
        if self.tenors is None: # initiating
            self.tenors = np.array([t])
            self.smiles = np.array([smile])
        elif t == self.tenors[-1]: # updating
            self.smiles[-1] = smile
        elif t > self.tenors[-1]: # appending
            self.tenors = np.append(self.tenors, t)  
            self.smiles = np.append(self.smiles, smile)        
        else:
            raise BaseException('tenor must be >= current tenor structure ...')
                
        self.__fetch__.cache_clear()       
        return self

    @lru_cache(maxsize=None) 
    def __fetch__(self, i, z):
        return self.smiles[i](z)            
    def __call__(self, t, z):
        i = np.searchsorted(self.tenors[:-1], t)
        return self.__fetch__(i, z) 
    
    @classmethod
    def from_file(cls, ivol, filename, sheetname): # load local vol from file
        lvol = ImpliedVol.load(ivol.equity, filename, sheetname) # load file by ImpliedVol class
        localvol = cls(ivol)
        for t, smile in zip(lvol.tenors, lvol.smiles):
            lv_smile = smile._data[1][1:-1] # remove two extended flat edges
            localvol.set(t, lv_smile) # update localvol
        return localvol
    
    @classmethod
    def from_calibration(cls, ivol): # calibrate from implied vol surface 
        localvol = cls(ivol)
        s = ivol.equity.t0
        n = 7 # number of strike pillers
        soln = None
        for v, t in ascending_pairs(s, ivol.tenors):                         
            ks = ivol.stdev(t) * cls.Smile.grid # 7 strikes 
            @np.vectorize                
            def flat_vol_pde(k): # must assume proportional div (e.g. disable fixed dividends)
                return cls(ivol).set(t, ivol(t,k)).__pde_fwd_call(s, t, hide_div=True)(k) 
            calls_iv = ivol.norm_forward_call(t, ks) # flat_vol_pde(ks) # ivol.norm_forward_call(t, ks)  
                                  
            def find_localvol(lv):
                calls_lv = localvol.set(t, lv).__pde_fwd_call(v, t, soln)(ks)       
                return sqrt(np.mean((calls_lv - calls_iv) ** 2))

            lv = minimize(find_localvol, x0=ivol(t,ks), bounds=[(0,1)] * n, method='L-BFGS-B').x #options = {'gtol': 1e-9,'ftol': 1e-7}
            soln = localvol.set(t, lv).__pde_fwd_call(v, t, soln) # update local vol

            print('%s:' % round(t,4), lv)             
        return localvol 

    def __pde_fwd_call(self, start, end, fn=None, hide_div=False): # normalized forward (i.e. undiscounted) call                      
        def fn_abc(u, k): # k = ln(K/F)
            a = self(u, k) ** 2 / 2
            return a, -a, np.array([0]) # represent 0 by 1X1 vector
        stdev = max(self.ivol.stdev(self.ivol.tenors[-1], -99), self.ivol.stdev(self.ivol.tenors[-1], 99))
        fdm = FDM1D(stdev, nt=80, nx=120, ns=5, fn_abc=fn_abc)
        yv = np.maximum(1 - np.exp(fdm.xv), 0) if fn is None else fn(fdm.xv)
        expiries = self.ivol.expiries(start, end)
        exdates = [] if hide_div else self.ivol.equity.div.ex(start, end)
        for p, q in ascending_pairs(start, exdates, expiries, end):       
            yv = fdm.evolve(p, q, yv)
            if q in exdates:
                f = self.ivol.equity.forward(q)
                d = self.ivol.equity.fixed_div(q) 
                xv_hat = np.log((np.exp(fdm.xv) * f + d) / (f + d))
                yv = fdm.functionize(yv)(xv_hat) * (f + d) / f
        return fdm.functionize(yv)
     
    def __pde_bkwd_call(self, start, end, z): # forward call
        def fn_abc(u, z): # z = log(x/F)
            a = self(u, z) ** 2 / -2.
            return a, -a, np.array([0]) # represent 0 by 1X1 vector 
        stdev = max(self.ivol.stdev(end, -99), self.ivol.stdev(end, 99))
        fdm = FDM1D(stdev, nt=120, nx=220, ns=5, fn_abc=fn_abc)
        yv = np.maximum(np.exp(fdm.xv) - exp(z), 0) * self.ivol.equity.forward(end)
        expiries = self.ivol.expiries(start, end)
        exdates = self.ivol.equity.div.ex(start, end)
        for p, q in descending_pairs(start, exdates, expiries, end):       
            yv = fdm.evolve(p, q, yv)
            if q in exdates:
                f = self.ivol.equity.forward(q)
                d = self.ivol.equity.fixed_div(q) 
                xv_tmp = np.maximum(1e-10, np.exp(fdm.xv) * (f + d) - d)
                xv_hat = np.log(xv_tmp / f)
                yv = fdm.functionize(yv)(xv_hat) 
        return fdm.functionize(yv)  

    def check(self, index):
        start = self.ivol.equity.t0  
        end = self.tenors[index]
        df = self.ivol.equity.disc(end)          
        fwd = self.ivol.equity.forward(end)
        k = fwd
        z = log(k / fwd)
        print('strike =', k) 

        # 1. Black-Scholes
        call_iv = self.ivol.norm_forward_call(end, z) * fwd * df 
        # 2. Forward PDE  
        call_fwd = self.__pde_fwd_call(start, end)(z) * fwd * df 
        # 3. Backward PDE        
        call_bkwd = self.__pde_bkwd_call(start, end, z)(0) * df  
        print('end, fwd, k, call_iv, call_fwd, call_bkwd') 
        print(end, fwd, k, call_iv, call_fwd, call_bkwd)


class BarrierOption:
    def __init__(self,
                 dealdate=None,        # the date the trade is initiated    
                 maturity='1Y',        # deal tenor, e.g. '5Y'
                 option='call',        # option type, 'call' or 'put'
                 strike=None,          # option strike
                 spotlag='3D',
                 paylag='3D',
                 barrier='out',
                 window=(None,None),  
                 bound=(None,None),    # absolute spot bound
                 rebate=(0,0,'maturity')): # rebate at 'knock' or 'maturity' time      
        self.maturity = Period(maturity)        
        self.option = 1 if option == 'call' else -1
        self.strike = strike
        self.spotlag = Period(spotlag)
        self.paylag = Period(paylag)

        self.barrier = barrier
        self.window = np.array([Date(d) if d is not None else d for d in window])
        self.bound = np.array([(2 * i - 1) * np.inf if b is None else log(b) for i, b in enumerate(bound)])
        self.rebate = np.array(rebate[:2])
        self.rebate_settle = rebate[-1]
    
    def calculate_present_value(self, localvol):
        eq = localvol.ivol.equity
        df = eq.disc
        today = eq.tradedate       

        spotdate = eq.roll_date(today, self.spotlag)
        matdate = eq.roll_date(today, self.maturity)
        paydate = eq.roll_date(matdate, self.paylag)         
        fwd = eq.forward(matdate.t)
        stdev = localvol.ivol.stdev(matdate.t, -99) 
        w_stt = today.t if self.window[0] is None else self.window[0].t
        w_end = matdate.t if self.window[1] is None else self.window[1].t

        def fn_abc(u, z): # z = log(x)
            variance = localvol(u, z - log(eq.forward(u))) ** 2 / 2
            riskfree = df.forward(u)
            return -variance, variance - riskfree + eq.div.rate(u), np.array([riskfree])

        if self.rebate_settle == 'knock':
            fn_rebate = [lambda u, r=r: r * df(u + 3. / 365) / df(u) for r in self.rebate]
        else: # settled at maturity
            fn_rebate = [lambda u, r=r: r * df(paydate.t) / df(u) for r in self.rebate]

        m, n = 2.5, 1
        fd_full = FDM1D(stdev, fwd=fwd, nt=500 * m, nx=400 * m, fn_abc=fn_abc)
        fd_diri = FDM1D(stdev, fwd=fwd, nt=500 * n, nx=400 * n, fn_abc=fn_abc, bound=self.bound) 
        
        lower = fd_full.xv <= self.bound[0]
        upper = fd_full.xv >= self.bound[1]
        outer = lower | upper

        exdates = eq.div.ex(today.t, matdate.t)
        time_pairs = descending_pairs(today.t, w_stt, w_end, exdates, matdate.t)
        fdm = fd_full
        yv = np.maximum(self.option * (np.exp(fdm.xv) - self.strike), 0) * df(paydate.t) / df(matdate.t)
        if self.barrier == 'in': # Knock-In barrier option
            for p, q in time_pairs:
                if p == w_end:# moves into barrier window, converts to Dirichlet BC
                    yb = yv # yb for boundaries, make a copy
                    yv = fd_diri.xv * 0 + fn_rebate[0](p)
                    fdm = fd_diri   
                if w_end >= p and p > w_stt: # in barrier window
                    yb, bc_fn = fd_full.evolve(p, q, yb, self.bound)
                    fd_diri.set_boundary_functions(bc_fn)
                if p == w_stt:# moves out of barrier window, converts to Linearity BC
                    yv = fd_diri.functionize(yv)(fd_full.xv)
                    yv[outer] = yb[outer]
                    fdm = fd_full
                yv = fdm.evolve(p, q, yv)                        
                if q in exdates: # fixed dividend 
                    div = eq.fixed_div(q) 
                    if w_end >= p and p > w_stt: # in barrier window
                        yv = fd_diri.functionize(yv)(fd_full.xv)
                        yv[outer] = yb[outer]
                        xv_tmp = np.log(np.maximum(1e-10, np.exp(fd_diri.xv) - div))
                        yv = fd_full.functionize(yv)(xv_tmp) 
                        xv_tmp = np.log(np.maximum(1e-10, np.exp(fd_full.xv) - div)) 
                        yb = fd_full.functionize(yb)(xv_tmp)
                    else:
                        xv_tmp = np.log(np.maximum(1e-10, np.exp(fd_full.xv) - div))
                        yv = fd_full.functionize(yv)(xv_tmp)
        else: # Knock-Out barrier option
            fd_diri.set_boundary_functions(fn_rebate)            
            for p, q in time_pairs:   
                if p == w_end:# moves into barrier window, converts to Dirichlet BC
                    yv = fd_full.functionize(yv)(fd_diri.xv)
                    fdm = fd_diri                                                                                                          
                if p == w_stt:# moves out of barrier window, converts to Linearity BC
                    yv = fd_diri.functionize(yv)(fd_full.xv)
                    yv[lower] = fn_rebate[0](p)
                    yv[upper] = fn_rebate[1](p)
                    fdm = fd_full                          
                yv = fdm.evolve(p, q, yv)                
                if q in exdates: # fixed dividend
                    xv_tmp = np.log(np.maximum(1e-10, np.exp(fdm.xv) - eq.fixed_div(q)))
                    yv = fdm.functionize(yv)(xv_tmp)
                    
        return fdm.functionize(yv)(log(eq.spot)) * df(today.t) / df(spotdate.t)

if __name__ == '__main__': 
    pass

