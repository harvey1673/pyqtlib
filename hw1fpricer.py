#-*- coding:utf-8 -*-
from misc import *
from utilities import convert_to_interpolator
import numpy as np
#import matplotlib.pyplot as plt

class CallableInstrument(object):
    def __init__(self, hw, swap, # underlying swap and calibrated Hull-White model
                 calldates,  # exercise schedule
                 right,      # 'long' or 'short' the option right
                 **kwargs):
        super(CallableInstrument, self).__init__(**kwargs)
        self.hw = hw    
        self.swap = swap    
        self.spotdate = hw.spotdate  # spotdate = today (+) spotlag
        self.calldates = calldates 
        self.right = 1 if right == 'long' else -1  # 'long' or 'short' of option right
        try:
            assert set(calldates) < (set(swap.rleg.get_schedule()) & set(swap.pleg.get_schedule()))
        except:
            raise BaseException('call schedule does not follow payment schedule ...') 

    def hw_pde_value(self, nt=400, nx=400, ns=4.5):
        self.subswap_matdate = self.swap.matdate # reset subswap maturity date for cancellable swap
        timeline = ([self.spotdate, self.spotdate] + self.calldates)[::-1] # padding with one more spotdate; reversed it in descending
        fdm = self.hw.finite_difference(nt=nt, nx=nx, ns=ns)
        val_exer = 0 # exercise value 
        val_cont = 0 # continuation value
        for calldate, startdate in zip(timeline[:-1], timeline[1:]): # backward evolution, diffuse from calldate to startdate
            if PRINT: print('evolves %s --> %s' % (calldate, startdate)) 
            subswap = self.swap.subswap(calldate, self.subswap_matdate)
            @np.vectorize
            def subswap_value(x):
                newproj, newdisc = self.hw.curves(calldate.t, x) 
                return subswap.value(newproj, newproj, newdisc)      
            val_cont, val_exer = self.calculate_one_period_pde(calldate, val_cont, val_exer, subswap_value(fdm.xv))
            if calldate in self.calldates: # compare continuation value and exercise value before evolving
                val_cont = np.maximum(val_exer, val_cont) if self.right == 1 else np.minimum(val_exer, val_cont)                     
            if calldate <= self.spotdate:  # final value at spot date
                break 
            else:  # diffuse backward from ed to st (i.e. ed is later than st)    
                val_cont = fdm.evolve(calldate.t, startdate.t, val_cont) # backward diffusion                            
        return fdm.functionize(val_cont)(0) # extract value at x(0)=0  

    @classmethod
    def calculate_one_period_pde(self):
        pass

    def hw_mc_value(self,                  
                    rng='sobol',       # random number generator, e.g. 'basic'/'sobol'
                    seed=None,         # random number generator seed ('None' for no (fixed) seeding)
                    npaths=5e5,        # mc # of paths
                    interp_size=300,   # size of interpolator; interpolator is used to evaluate payoffs, greatly improve efficiency in MC
                    poly_order=8       # order of polynomials used in least square MC):
                    ):
        self.interp_size = interp_size
        self.subswap_matdate = self.swap.matdate # reset subswap maturity date for cancellable swap
        timeline = [self.spotdate] + self.calldates
        npaths = int(npaths)
        fetch_state = self.hw.monte_carlo(timeline, npaths=npaths, rng=rng, seed=seed)
        val_cont = np.zeros(npaths) # continuation value 
        val_exer = np.zeros(npaths) # immediate exercise value 
        for calldate in timeline[::-1]:
            if PRINT: print('subleg %s --> %s' % (calldate, self.subswap_matdate))  
            xv, numer = fetch_state(calldate) # x vector and corresponding numeraire vector 
            subswap = self.swap.subswap(calldate, self.subswap_matdate) # val_cont is always zero 
            @convert_to_interpolator(size=self.interp_size)
            def subswap_value(x):
                newproj, newdisc = self.hw.curves(calldate.t, x) 
                return subswap.value(newproj, newproj, newdisc)  
            val_cont, val_exer, itm = self.calculate_one_period_mc(calldate, val_cont, val_exer, subswap_value(xv) / numer)
            if calldate in self.calldates: # there exists optionality 
                if calldate == self.calldates[-1]: # at last calldate, this is an European option
                    expect_cont = val_cont
                elif np.any(itm): # there are in-the-money paths, regress on polynomials: [x^0, x^1, x^2, x^3, ...]
                    xm = np.vstack(xv ** i for i in range(poly_order))
                    # val_cont for regression must be NON-Denominated U(t) rather than U(t)/N(t) !!!
                    coef = np.linalg.lstsq(xm.T[itm], (val_cont * numer)[itm])[0] 
                    expect_cont = coef.dot(xm) / numer # convert to numeraire denominated
                else: # there are NO in-the-money paths
                    expect_cont = 0.0
                itm_and_exer = itm & (val_exer > expect_cont) if self.right == 1 else itm & (val_exer < expect_cont) 
                val_cont[itm_and_exer] = val_exer[itm_and_exer] 
        return np.mean(val_cont) * numer

    @classmethod
    def calculate_one_period_mc(self):
        pass    


class BermudanSwaption(CallableInstrument):
    def __init__(self, *args, **kwargs): # Bermudan swaption strike rate 
        super(BermudanSwaption, self).__init__(*args, **kwargs)
    
    def calculate_one_period_pde(self, calldate, val_cont, val_exer, subswap_value):      
        val_exer = subswap_value
        return val_cont, val_exer  

    def calculate_one_period_mc(self, calldate, val_cont, val_exer, subswap_value):  
        val_exer = subswap_value
        itm = self.right * val_exer > 0.0
        return val_cont, val_exer, itm


class CancellableSwap(CallableInstrument):
    def __init__(self, *args, **kwargs): # cancellable irs fixed leg rate
        super(CancellableSwap, self).__init__( *args, **kwargs)

    def calculate_one_period_pde(self, calldate, val_cont, val_exer, subswap_value):        
        self.subswap_matdate = calldate
        val_cont += subswap_value 
        return val_cont, val_exer   

    def calculate_one_period_mc(self, calldate, val_cont, val_exer, subswap_value):  
        self.subswap_matdate = calldate
        val_cont += subswap_value
        itm = self.right * subswap_value < 0.0         
        return val_cont, val_exer, itm


class TargetRedemptionNote:
    def __init__(self, hw,         # calibrated Hull-White model
                 rleg_tgt=None,       # receive target leg 
                 pleg_vnl=None,       # pay vanilla leg (can be floating/fixed or even range leg)                 
                 target=None,         # target of total accrue            
                 target_hit='full',   # if hit tarn_target: last coupon = 'full' or 'truncated'
                 target_miss='last'): # if not hit tarn_target: last coupon = 'only last' or 'remaining'                   
        self.hw = hw 
        assert set(pleg_vnl.get_schedule()) == set(rleg_tgt.get_schedule()) # both legs must have the same payment schedule
        self.rleg_tgt = rleg_tgt
        self.pleg_vnl = pleg_vnl  
        self.target = target
        self.target_hit = target_hit
        self.target_miss = target_miss        

    def hw_mc_value(self,
                    rng='sobol',       # random number generator, e.g. 'basic'/'sobol'
                    seed=None,         # random number generator seed ('None' for no (fixed) seeding)
                    npaths=5e5,        # mc # of paths
                    interp_size=300,   # size of interpolator; interpolator is used to evaluate payoffs, greatly improve efficiency in MC
                    poly_order=8):     # order of polynomials used in least square MC                         
        spotdate = self.hw.spotdate
        tgt_past = [p for p in self.rleg_tgt.cp if p.accr_end <= spotdate]      
        tgt_live = [p for p in self.rleg_tgt.cp if p.accr_end > spotdate] # self.rleg_tgt.subleg(spotdate)
        vnl_live = [p for p in self.pleg_vnl.cp if p.accr_end > spotdate] # self.pleg_vnl.subleg(spotdate)       
        timeline = [spotdate] + [p.accr_start for p in tgt_live if p.accr_start > spotdate] 

        npaths = int(npaths)
        fetch_state = self.hw.monte_carlo(timeline, npaths=npaths, rng=rng, seed=seed)            
        value = 0.0 
        couponsum = sum(p.index.forward(self.hw.proj) * p.accr_cov for p in tgt_past)
        alive = np.ones(npaths, dtype=bool)               
        for tp, vp in zip(tgt_live, vnl_live):
            if PRINT: print('period %s --> %s' % (tp.accr_start, tp.accr_end))  
            alive &= couponsum < self.target # update alive flag
            date = max(spotdate, tp.accr_start)
            @ convert_to_interpolator(size=interp_size)
            def disc_factor(x): # to discount cashflow from payment date to accr_start date
                newproj, newdisc = self.hw.curves(date.t, x) 
                return newdisc(tp.paydate.t) 
            @convert_to_interpolator(size=interp_size) # convert to interpolator
            def tgt_spot_rate(x):
                newproj, newdisc = self.hw.curves(date.t, x)
                return tp.index.forward(newproj) * self.rleg_tgt.factory.rate_leverage + self.rleg_tgt.factory.rate_spread
            @convert_to_interpolator(size=interp_size) # convert to interpolator
            def vnl_spot_rate(x):
                newproj, newdisc = self.hw.curves(date.t, x)  
                return vp.index.forward(newproj) * self.pleg_vnl.factory.rate_leverage + self.pleg_vnl.factory.rate_spread
                        
            xv, numer = fetch_state(date) # stochastic factor, and numer (w.r.t. spotdate or timeline[0])
            coupon = tgt_spot_rate(xv) * tp.accr_cov        
            if self.target_hit == 'truncated': # no action if tarn_hit == 'full'
                coupon = np.minimum(coupon, self.target - couponsum)            
            if tp is self.rleg_tgt.cp[-1] and self.target_miss == 'remaining': # no action if tarn_miss == 'last'
                coupon = np.maximum(coupon, self.target - couponsum)                                                
            value += alive * (coupon * tp.notional - vnl_spot_rate(xv) * vp.accr_cov * vp.notional) \
                           * disc_factor(xv) / numer # accumulates those still alive
            couponsum = couponsum + coupon # couponsum += coupon                                                              
        return np.mean(value)


if __name__ == '__main__':
    pass