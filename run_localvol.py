#-*- coding:utf-8 -*-
import os
from misc import *  
from local_vol import *
from curve import *
from utilities import *
from finite_diff import *

import numpy as np

@time_this
def test_localvol(localvol):     
    fwddate = Date(31,8,2017)
    equity = localvol.ivol.equity
    print('fwd value:', equity.forward_npv(fwddate), equity.forward(fwddate.t)) # 1.23495610347
    spotdate = equity.roll_date(equity.tradedate, equity.spotlag)
    matdate = equity.roll_date(equity.tradedate, Period('1Y'))
    paydate = equity.roll_date(matdate, equity.paylag)
    print('fwd pv:', equity.forward_npv(matdate) * equity.disc(paydate.t) / equity.disc(spotdate.t)) # 1

    localvol.check(0)
    localvol.check(1)
    localvol.check(2)
    localvol.check(3)

    bar = equity.forward(fwddate.t) + 0.1 
    bo = BarrierOption(maturity='1Y', option='call', strike=0, barrier='out', 
                       window=('2016-8-31','2016-8-31'), bound=(None, None))
    a = bo.calculate_present_value(localvol)
    bo = BarrierOption(maturity='1Y', option='call', strike=0, barrier='out', 
                       window=('2016-8-31','2016-8-31'), bound=(bar, None))
    b = bo.calculate_present_value(localvol)
    bo = BarrierOption(maturity='1Y', option='call', strike=0, barrier='out', 
                       window=('2016-8-31','2016-8-31'), bound=(None, bar))
    c = bo.calculate_present_value(localvol)
    print('full, sum, high, low', a, b + c, b, c)

@time_this
def load_localvol(filename):
    equity = Equity.load(filename)
    ivol = ImpliedVol.load(equity, filename, 'ivol')    
    localvol = LocalVol.from_calibration(ivol)
    #localvol = LocalVol.from_file(ivol, filename, 'lvol') 
    return localvol

if __name__ == '__main__':
    today = Date.set_origin(31,8,2015)
    np.set_printoptions(precision=6)

    script_path = os.path.dirname(os.path.realpath(__file__))
    localvol_equity_index = load_localvol(script_path + '/MarketData/market_data_equity_index.xls') 
    localvol_equity = load_localvol(script_path + '/MarketData/market_data_equity.xls')    
    #test_localvol(localvol_equity_index)

    np.set_printoptions(precision=6)

    ## examples of pricing:
    # upper bound touch option on equity index with American barrier window, continuous dividend
    print('1. touch option')
    bo = BarrierOption(maturity='2y', option='call', strike=23000, barrier='out', bound=(None,23000), rebate=(0,0.69,'knock'))
    print(" - price: %s\n" % bo.calculate_present_value(localvol_equity_index))

    # single barrier up-and-in call with European barrier window at maturity, fixed dividends
    print('2. single barrier up-and-in call')
    bo = BarrierOption(maturity='2y', option='call', strike=43.92, barrier='in', window=('2017-8-31','2017-8-31'), bound=(None,47), rebate=(1.5,0,'maturity'))
    print(" - price: %s\n" % bo.calculate_present_value(localvol_equity))

    # double barrier up-in-and-down-in put with partial American barrier window, fixed dividends
    print('3. double barrier up-in-and-down-in put')
    bo = BarrierOption(maturity='2y', option='put', strike=40, barrier='in', window=('2015-11-30', '2016-1-29'),
                       bound=(44, 54), rebate=(1.2,0,'maturity'))
    print(" - price: %s\n" % bo.calculate_present_value(localvol_equity))

    # double barrier up-out-and-down-out put with American barrier window, fixed dividends
    print('4. double barrier up-out-and-down-out put')
    bo = BarrierOption(maturity='2y', option='call', strike=40, barrier='out', bound=(44, 54), rebate=(1,1.5,'knock'))
    print(" - price: %s\n" % bo.calculate_present_value(localvol_equity))



    print("5. option parity: vanilla call(put) = knock-in call(put) + knock-out call(put)")
    print('vanilla call')
    bo = BarrierOption(maturity='1y', option='call', strike=48.8, barrier='out', bound=(None, None))
    v1 = bo.calculate_present_value(localvol_equity)
    print(" - price: %s" % v1) 

    # double barrier up-out-and-down-out put with American barrier window, fixed dividends
    print('knock-out call')
    bo = BarrierOption(maturity='1y', option='call', strike=48.8, barrier='out', bound=(30, 70))
    v2 = bo.calculate_present_value(localvol_equity)
    print(" - price: %s" % v2)  

    # double barrier up-out-and-down-out put with American barrier window, fixed dividends
    print('knock-in call')
    bo = BarrierOption(maturity='1y', option='call', strike=48.8, barrier='in', bound=(30, 70))
    v3 = bo.calculate_present_value(localvol_equity)
    print(" - price: %s" % v3) 

    print("option parity:", v1, 'vs', v2 + v3, '=', v2, '+', v3)


    #Start load_localvol
    #0.0822: [ 0.398379  0.295268  0.181928  0.236996  0.170443  0.186821  0.248946]
    #0.2493: [ 0.184933  0.339194  0.203099  0.207544  0.17775   0.185102  0.187782]
    #0.4986: [ 0.214783  0.197719  0.330413  0.151043  0.193652  0.190411  0.193905]
    #1.0027: [ 0.211729  0.186783  0.277393  0.190923  0.161832  0.194373  0.182729]
    #load_localvol finished in 45.20 seconds

    #Start load_localvol
    #0.0822: [ 0.456924  0.580026  0.513179  0.416714  0.422559  0.431554  0.423993]
    #0.2493: [ 0.409836  0.351161  0.550851  0.328616  0.352877  0.374306  0.362119]
    #0.4986: [ 0.406383  0.359972  0.51984   0.33832   0.358742  0.381228  0.374582]
    #1.0027: [ 0.333763  0.310623  0.388154  0.312389  0.306961  0.333684  0.324025]
    #load_localvol finished in 46.91 seconds

    #1. touch option
    # - price: 0.632859063884

    #2. single barrier up-and-in call
    # - price: 10.8487150637

    #3. double barrier up-in-and-down-in put
    # - price: 6.22951472356

    #4. double barrier up-out-and-down-out put
    # - price: 1.23990379036

    #5. option parity: vanilla call(put) = knock-in call(put) + knock-out call(put)
    #vanilla call
    # - price: 6.62912262477
    #knock-out call
    # - price: 1.32064568976
    #knock-in call
    # - price: 5.30869863572
    #option parity: 6.62912262477 vs 6.62934432548 = 1.32064568976 + 5.30869863572



