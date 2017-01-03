#-*- coding:utf-8 -*-
from hw1fpricer import *
from misc import *
from utilities import *
from curve import *
from rate_options import *
from cashflow import *
from hw1fmodel import HullWhite1F
import os

#np.set_printoptions(precision=8) 

@time_this
def bermudan_swaption(today, proj, disc, volsurf, days_aged): # bermudan swaption
    tradedate = today - days_aged # days_aged： 0 for spot started trade; positive integer for aged trade
    tenor = Period('5Y')

    calldates = ExerciseSchedule('6M', Calendar.US_UK).dates(tradedate, start='1Y', end='3Y') # both ends inclusive 
    hw = HullWhite1F.from_calibration(proj, disc, volsurf, tradedate=tradedate, tenor=tenor, lastdate=calldates[-1], z_fwd_date=today + 3000) 

    # payer's leg is vanilla fixed leg; receiver's leg is vanilla floating leg on libor
    rleg_fac = LegFactory('3M', Calendar.US_UK, DayCount.ACT360, index_fac=libor_fac,
                          notl_base=1e6, notl_amort_freq=1, notl_amort_abs=-1e3) # amortizing, -1000 notl per quarter
    pleg_fac = LegFactory('6M', Calendar.US_UK, DayCount._30E360, index_fac=FixedIndexFactory(0.01),
                          notl_base=1e6, notl_amort_freq=1, notl_amort_abs=-1e3)     
    swap = InterestRateSwapFactory(rleg_fac=rleg_fac, pleg_fac=pleg_fac, tenor=tenor).create(tradedate=tradedate)        
          
    trade = BermudanSwaption(hw, swap, calldates=calldates, right='long')
    pv1 = trade.hw_pde_value(nt=400, nx=400, ns=4.5)
    pv2 = trade.hw_mc_value(npaths=1e6, seed=0)   
    return print_results(pv1, pv2)


@time_this
def bermudan_cancellable_irs(today, proj, disc, volsurf, days_aged): # bermudan cancellable interest rate swap
    tradedate = today - days_aged # days_aged： 0 for spot started trade; positive integer for aged trade
    tenor = Period('5Y')

    calldates = ExerciseSchedule('6M', Calendar.US_UK).dates(tradedate, start='1Y', end='4Y6M') # both ends inclusive 
    hw = HullWhite1F.from_calibration(proj, disc, volsurf, tradedate=tradedate, tenor=tenor, lastdate=calldates[-1], kappa=0.05) 

    # payer's leg is vanilla fixed leg; receiver's leg is vanilla floating leg on libor
    rleg_fac = LegFactory('3M', Calendar.US_UK, DayCount.ACT360, index_fac=libor_fac)
    pleg_fac = LegFactory('6M', Calendar.US_UK, DayCount._30E360, index_fac=FixedIndexFactory(0.01))     
    swap = InterestRateSwapFactory(rleg_fac=rleg_fac, pleg_fac=pleg_fac, tenor=tenor).create(tradedate=tradedate)
       
    trade = CancellableSwap(hw, swap, calldates=calldates, right='short') 
    pv1 = trade.hw_pde_value(nt=400, nx=400, ns=4.5)
    pv2 = trade.hw_mc_value(npaths=1e6, seed=0)    
    return print_results(pv1, pv2)


@time_this
def bermudan_cancellable_ras(today, proj, disc, volsurf, days_aged): # bermudan cancellable range accrual swap
    tradedate = today - days_aged # days_aged： 0 for spot started trade; positive integer for aged trade
    tenor = Period('5Y')

    # payer's leg is range accrual leg on libor; receiver's leg is vanilla floating leg on libor
    calldates = ExerciseSchedule('12M', Calendar.US_UK).dates(tradedate, start='1Y', end='4Y') # both ends inclusive 
    hw = HullWhite1F.from_calibration(proj, disc, volsurf, tradedate=tradedate, tenor=tenor, lastdate=calldates[-1]) 
    rleg_fac = LegFactory('3M', Calendar.US_UK, DayCount.ACT360, index_fac=libor_fac)
    range_index_fac = RangeIndexFactory(hw=hw, baserate=0.02, index_fac=libor_fac, rng_def=('between',0.005,0.015), 
                                        num_days='calendar', two_ends=(None,-1), lockout=('crystallized',-10))
    pleg_fac = LegFactory('6M', Calendar.US_UK, DayCount._30E360, index_fac=range_index_fac) 
    swap = InterestRateSwapFactory(rleg_fac=rleg_fac, pleg_fac=pleg_fac, tenor=tenor).create(tradedate=tradedate)    
    trade = CancellableSwap(hw, swap, calldates=calldates, right='short')
    pv1 = trade.hw_pde_value(nt=400, nx=400, ns=4.5)
    pv2 = trade.hw_mc_value(npaths=1e6, seed=0)     
    return print_results(pv1, pv2)


@time_this
def ir_tarn(today, proj, disc, volsurf, days_aged): # interest rate target redemption notes 
    tradedate = today - days_aged # days_aged： 0 for spot started trade; positive integer for aged trade
    tenor = Period('10Y')

    # payer's leg is vanilla fixed leg; receiver's leg is target redemption leg on libor
    rleg_tgt = LegFactory('3M', Calendar.US_UK, DayCount.ACT360, index_fac=libor_fac).create(tradedate=tradedate, tenor=tenor)
    pleg_vnl = LegFactory('3M', Calendar.US_UK, DayCount.ACT360, index_fac=FixedIndexFactory(0.01)).create(tradedate=tradedate, tenor=tenor) 
    hw = HullWhite1F.from_calibration(proj, disc, volsurf, tradedate=tradedate, tenor=tenor, lastdate=rleg_tgt.matdate)      
    trade = TargetRedemptionNote(hw, rleg_tgt=rleg_tgt, pleg_vnl=pleg_vnl, target=0.10, target_hit='truncated', target_miss='remaining') 
    pv = trade.hw_mc_value(npaths=5e5, seed=0)
    return print_results(pv_mc=pv) # For display only, there's NO PDE reult


@time_this
def range_tarn(today, proj, disc, volsurf, days_aged): # range accrual target redemption notes
    tradedate = today - days_aged # days_aged： 0 for spot started trade; positive integer for aged trade
    tenor = Period('5Y')

    # payer's leg is vanilla floating leg on libor; receiver's leg is target redemption leg on range accrual of libor
    pleg_vnl = LegFactory('3M', Calendar.US_UK, DayCount.ACT360, index_fac=libor_fac).create(tradedate=tradedate, tenor=tenor) 
    hw = HullWhite1F.from_calibration(proj, disc, volsurf, tradedate=tradedate, tenor=tenor, lastdate=pleg_vnl.matdate)
     
    range_index_fac = RangeIndexFactory(hw=hw, baserate=0.02, index_fac=libor_fac, rng_def=('between',0.005,0.015), 
                                        num_days='calendar', two_ends=(None,-1), lockout=('crystallized',-10))
    rleg_tgt = LegFactory('3M', Calendar.US_UK, DayCount.ACT360, index_fac=range_index_fac).create(tradedate=tradedate, tenor=tenor)
     
    hw = HullWhite1F.from_calibration(proj, disc, volsurf, tradedate=tradedate, tenor=tenor, lastdate=rleg_tgt.matdate)      
    trade = TargetRedemptionNote(hw, rleg_tgt=rleg_tgt, pleg_vnl=pleg_vnl, target=0.05, target_hit='truncated', target_miss='remaining') 
    pv = trade.hw_mc_value(npaths=5e5, seed=0)
    return print_results(pv_mc=pv) # For display only, there's NO PDE reult


@time_this
def constant_maturity_swap(today, proj, disc, swpnsurf, days_aged): # constant maturity swap
    tradedate = today - days_aged # days_aged： 0 for spot started trade; positive integer for aged trade
    tenor = Period('5Y')
    rleg_fac = LegFactory('3M', Calendar.US_UK, DayCount.ACT360, index_fac=libor_fac)          
    pleg_fac = LegFactory('6M', Calendar.US_UK, DayCount._30360US, index_fac=FixedIndexFactory())
    swap_index_fac = InterestRateSwapFactory(rleg_fac=rleg_fac, pleg_fac=pleg_fac, tenor=Period('2Y')) 
    cms_index_fac = CMSIndexFactory(today, proj, disc, swpnsurf, index_fac=swap_index_fac) 
    cms_index_fac.set_fixing('2016-3-14', 0.01) # for aged deal only
    cms_index_fac.set_fixing('2016-7-6', 0.01) # for aged deal only

    # payer's leg is vanilla fixed leg; receiver's leg is semi-annual floating leg on 2Y tenor par swap rate
    rleg_fac = LegFactory('6M', Calendar.US_UK, DayCount._30360US, index_fac=cms_index_fac)
    pleg_fac = LegFactory('6M', Calendar.US_UK, DayCount._30360US, index_fac=FixedIndexFactory(0.01))

    trade = InterestRateSwapFactory(rleg_fac=rleg_fac, pleg_fac=pleg_fac, tenor=tenor).create(tradedate=tradedate) 
    pv = trade.value(proj, proj, disc)
    return print_results(pv) # For display only, this is NOT PDE result



if __name__ == '__main__':
    # 1. mean reversion rate, kappa
    # 2. curve interpolation scheme
    # 3. specify coterm-swaptions / caplets
    # 4. assign proj and disc curves

    mode = 'lognormal' # 'normal' or 'lognormal' volatility (on two different days)
    print('Using %s volatility:' % mode)

    libor_fac = LiborIndexFactory('3M', Calendar.US_UK, DayCount.ACT360)
    if mode == 'lognormal':
        today = Date.set_origin(21,3,2016)    
        xls_file = os.path.dirname(os.path.realpath(__file__)) + '/MarketData/usd_market_data_2016-03-21.xls'
        libor_fac.set_fixing('2015-8-14', 0.0032445)  # fixed
        libor_fac.set_fixing('2015-11-16', 0.0036410) # fixed
        libor_fac.set_fixing('2016-2-16', 0.0061820)  # fixed
        libor_fac.set_fixing('2016-3-14', 0.0063955)  # fixed, for vanilla floating leg
        libor_fac.set_fixing('2016-3-16', 0.006390)   # fixed
        libor_fac.set_fixing('2016-3-17', 0.006234)   # fixed
        libor_fac.set_fixing('2016-3-18', 0.006243)   # fixed
        libor_fac.set_fixing('2016-3-19', 0.006243)   # not fixed, holiday, equal to last fixing 
        libor_fac.set_fixing('2016-3-20', 0.006243)   # not fixed, holiday, equal to last fixing           
    else: # 'normal'
        today = Date.set_origin(13,7,2016)
        xls_file = os.path.dirname(os.path.realpath(__file__)) + '/MarketData/usd_market_data_2016-07-13.xls'
        libor_fac.set_fixing('2015-12-4', 0.003891)  # fixed
        libor_fac.set_fixing('2016-3-4', 0.006231)   # fixed
        libor_fac.set_fixing('2016-6-6', 0.006610)   # fixed 
        libor_fac.set_fixing('2016-7-6', 0.006610)   # fixed
        libor_fac.set_fixing('2016-7-8', 0.006671)   # fixed
        libor_fac.set_fixing('2016-7-9', 0.006671)   # not fixed, holiday, equal to last fixing 
        libor_fac.set_fixing('2016-7-10', 0.006671)  # not fixed, holiday, equal to last fixing
        libor_fac.set_fixing('2016-7-11', 0.006671)  # not fixed, holiday, equal to last fixing
        libor_fac.set_fixing('2016-7-12', 0.006671)  # not fixed, holiday, equal to last fixing
        libor_fac.set_fixing('2016-7-13', 0.006801)  # fixed

    spotdate=libor_fac.roll_date(today,'spot')
    proj = DiscountCurve.load(filename=xls_file, sheetname='usdstd', t0=spotdate.t)
    disc = DiscountCurve.load(filename=xls_file, sheetname='usdois', t0=spotdate.t)

    capsurf = CapletVolSurface.load(today, proj, disc, libor_fac, mode, filename=xls_file, sheetname='cap_' + mode) # load forward volatilities

    #simple_index_fac = FloatIndexFactory() # Simply follows accrual schedule
    flt_fac = LegFactory('3M', Calendar.US_UK, DayCount.ACT360, index_fac=libor_fac) 
    fix_fac = LegFactory('6M', Calendar.US_UK, DayCount._30E360, index_fac=FixedIndexFactory())     
    swap_fac = InterestRateSwapFactory(rleg_fac=flt_fac, pleg_fac=fix_fac) 
    swpnsurf = SwaptionVolSurface.load(today, proj, disc, swap_fac, mode, filename=xls_file, sheetname='swpn_' + mode) # load swaption vol matrix (w/o smile)    

    # HW PDE and MC    
    bermudan_swaption(today, proj, disc, swpnsurf, 0) # spot started trade                                   
    bermudan_swaption(today, proj, disc, swpnsurf, 7) # 7-day aged trade                                  
    
    bermudan_cancellable_irs(today, proj, disc, swpnsurf, 0) # spot started trade                      
    bermudan_cancellable_irs(today, proj, disc, swpnsurf, 7) # 7-day aged trade   
                                      
    bermudan_cancellable_ras(today, proj, disc, capsurf, 0) # spot started trade     
    bermudan_cancellable_ras(today, proj, disc, capsurf, 7) # 7-day aged trade                                  

    # HW MC                                          
    ir_tarn(today, proj, disc, capsurf, 0) # spot started trade                
    ir_tarn(today, proj, disc, capsurf, 220) # 220-day aged trade                 
                                        
    range_tarn(today, proj, disc, capsurf, 0) # spot started trade 
    range_tarn(today, proj, disc, capsurf, 7) # 7-day aged trade   

    # CMS by Replication                                                                
    constant_maturity_swap(today, proj, disc, swpnsurf, 0) # spot started trade                                    
    constant_maturity_swap(today, proj, disc, swpnsurf, 7) # 7-day aged trade    

#output results for comparison:
#normal vol of Date(13,7,2016)                        lognormal vol of Date(21,3,2016)                     
#
#Using normal volatility:                             Using lognormal volatility:
#bermudan_swaption starts ...                         bermudan_swaption starts ...
#PV(PDE): 16885.830155941392                          PV(PDE): 27338.306668202884
#PV(MC ): 16884.3558772                               PV(MC ): 27342.4909328
#diff.% : 0.008731%                                   diff.% : -0.015304%
#bermudan_swaption finished in 10.24 seconds          bermudan_swaption finished in 10.57 seconds
                                                     
#bermudan_swaption starts ...                         bermudan_swaption starts ...
#PV(PDE): 16729.513288287424                          PV(PDE): 27147.095224684464
#PV(MC ): 16727.3068416                               PV(MC ): 27149.5498242
#diff.% : 0.013190%                                   diff.% : -0.009041%
#bermudan_swaption finished in 11.30 seconds          bermudan_swaption finished in 10.74 seconds
                                                     
#bermudan_cancellable_irs starts ...                  bermudan_cancellable_irs starts ...
#PV(PDE): -20764.133452546666                         PV(PDE): -14725.954631035765
#PV(MC ): -20763.9870057                              PV(MC ): -14727.169157
#diff.% : 0.000705%                                   diff.% : -0.008247%
#bermudan_cancellable_irs finished in 19.10 seconds   bermudan_cancellable_irs finished in 19.45 seconds
                                                     
#bermudan_cancellable_irs starts ...                  bermudan_cancellable_irs starts ...
#PV(PDE): -20778.29668950881                          PV(PDE): -14719.656567668811
#PV(MC ): -20776.5339288                              PV(MC ): -14719.9292385
#diff.% : 0.008484%                                   diff.% : -0.001852%
#bermudan_cancellable_irs finished in 19.08 seconds   bermudan_cancellable_irs finished in 19.78 seconds
                                                     
#bermudan_cancellable_ras starts ...                  bermudan_cancellable_ras starts ...
#PV(PDE): -19631.584941101202                         PV(PDE): -15357.278366134724
#PV(MC ): -19627.5417593                              PV(MC ): -15355.2806066
#diff.% : 0.020597%                                   diff.% : 0.013009%
#bermudan_cancellable_ras finished in 32.65 seconds   bermudan_cancellable_ras finished in 39.68 seconds
                                                     
#bermudan_cancellable_ras starts ...                  bermudan_cancellable_ras starts ...
#PV(PDE): -19892.330183499565                         PV(PDE): -15549.868571912933
#PV(MC ): -19888.8850022                              PV(MC ): -15548.2741885
#diff.% : 0.017321%                                   diff.% : 0.010254%
#bermudan_cancellable_ras finished in 35.96 seconds   bermudan_cancellable_ras finished in 34.56 seconds
                                                     
#ir_tarn starts ...                                   ir_tarn starts ...
#PV(MC ): 20571.1040572                               PV(MC ): 27327.2556806
#ir_tarn finished in 25.31 seconds                    ir_tarn finished in 26.09 seconds
                                                     
#ir_tarn starts ...                                   ir_tarn starts ...
#PV(MC ): 19929.5343642                               PV(MC ): 26853.8594929
#ir_tarn finished in 22.51 seconds                    ir_tarn finished in 20.75 seconds
                                                     
#range_tarn starts ...                                range_tarn starts ...
#PV(MC ): 8054.121722145579                           PV(MC ): -10825.506387996138
#range_tarn finished in 33.64 seconds                 range_tarn finished in 29.00 seconds
                                                     
#range_tarn starts ...                                range_tarn starts ...
#PV(MC ): 8400.669300076324                           PV(MC ): -10492.159792741933
#range_tarn finished in 32.96 seconds                 range_tarn finished in 25.14 seconds
                                                     
#constant_maturity_swap starts ...                    constant_maturity_swap starts ...
#PV(PDE): 3328.31005775                               PV(PDE): 23668.6195446
#constant_maturity_swap finished in 0.49 seconds      constant_maturity_swap finished in 0.39 seconds
                                                     
#constant_maturity_swap starts ...                    constant_maturity_swap starts ...
#PV(PDE): 4343.68476468                               PV(PDE): 23853.1874281
#constant_maturity_swap finished in 0.48 seconds      constant_maturity_swap finished in 0.40 seconds














