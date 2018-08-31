#!/usr/bin/env python3
#-*- coding=utf-8 -*-

from argparse import ArgumentParser
from random import choice
from math import floor, log, sqrt, fabs

skyStar={1:.9, 2:.87, 3:.84, 4:.8, 5:.7}
class skyStock:
    def __init__(self, args):
        self.remain=self.total=args.total
        self.sold=args.sold
        self.γ0, self.dγ, self.eGamma=args.start_gamma, args.delta_gamma, args.end_gamma
        self.sPrice, self.dPrice, self.ePrice=args.start_price, args.delta_price, args.end_price
        self.cPrice=self.sPrice
        self.Ascended, self.Info=args.Ascended, args.Info
        self.nmin, self.γmax, self.starRatio=args.nmin, args.γmax, skyStar[args.star]
        self.ratio=self.Ratio=self.sold/self.total
        self.avgPrice=self.tMoney=self.amount=self.amounts=0
        self.portforlio, self.Δ, self.eps={}, 1e-7, 1e-7
        #if self.ascended: self.cPrice=self.sPrice   #cPrice=sPrice+n*dPrice
        #else: self.cPrice=self.ePrice     #cPrice=ePrice-(nmin-n-1)*dPrice
    def maxProfit(self):
        remain, total, Ratio, sold, γ0, γmax=self.remain, self.total, self.Ratio, self.sold, self.γ0, self.γmax
        eps, Δ=self.eps, self.Δ 
        print('ratio, γ0=', Ratio, γ0)
        def iterGamma(γ0, ratio=Ratio):
            n=0
            while ratio:
                γ=γ0 if not n else .1
                ratio-=γ
                n+=1
                #if 0<ratio<eps: print(ratio); return n
                if fabs(ratio)<eps: print(ratio); return n
                elif ratio<0: return None
                ratio/=1-γ     #1-.1
            print('outside=', ratio)
            return n
        while γ0<γmax:
            nmin=iterGamma(γ0)
            if nmin:
                print(nmin, γ0)
                self.nmin=nmin
                self.portforlio[nmin]=γ0
                self.calcPortforlio(nmin, γ0)
            #if nmin<8: print('γ0, nmin=', γ0, nmin)
            γ0+=Δ
        #print(self.portforlio)
    def portforlioVerbose(self, γ=0, n=0):
        Amount=self.remain*γ
        self.amount=Amount*self.starRatio
        if self.Ascended: self.cPrice=self.sPrice+n*self.dPrice
        else: self.cPrice=self.ePrice-(self.nmin-n-1)*self.dPrice     #cPrice=ePrice-(nmin-n-1)*dPrice
        self.tMoney+=Amount*self.cPrice
        self.amounts+=self.amount
        self.avgPrice=self.tMoney/self.amounts
        self.RATIO=self.amounts/self.total
        self.ratio-=γ
        print('%4d %3d %3d %3d %.5f %.5f %.3f %.3f'%(self.remain, Amount, self.amount, self.amounts, self.RATIO, γ, self.cPrice, self.avgPrice))
        self.remain-=self.amount
        self.ratio/=1-γ     #1-.1
    def calcPortforlio(self, nmin, γ0):
        remain, total, sold, ratio=self.total, self.total, self.sold, self.Ratio
        sPrice, dPrice, ePrice=self.sPrice, self.dPrice, self.ePrice
        tMoney, amount, amounts, starRatio=self.tMoney, self.amount, self.amounts, self.starRatio
        #skyStar[self.star]
        n=0
        while n<nmin:
            γ=γ0 if not n else .1
            self.portforlioVerbose(γ=γ, n=n)
            n+=1
        print()
    def profitVerbose(self):
        total, sold, γ0, γmax=self.total, self.sold, self.γ0, self.γmax
        sPrice, dPrice, ePrice=self.sPrice, self.dPrice, self.ePrice
        remain, Ratio, cPrice=self.remain, self.Ratio, self.cPrice
        #tMoney, amounts, nmin=0, 0, True
        #sPrice, dPrice, ePrice=self.start_price, self.delta_price, self.end_price
        #profit={6: 0.09999009999999393, 7: 0.03245009999999927}, {8: 0.059160100000007446, 7: 0.09999009999999393}
        for nmin, γ0 in self.portforlio.items():
            print('nmin, γ0=%3d %.9f'%(nmin, γ0))
            calcPortforlio(nmin, γ0)
    def calcGamma(self):
        γ, dγ, eGamma=self.γ0, self.dγ, self.eGamma
        sPrice, dPrice, ePrice=self.sPrice, self.dPrice, self.ePrice
        total, sold, nmin, γmax=self.total, self.sold, self.nmin, self.γmax
        ratio, remain, tMoney, starRatio, amounts, avgPrice=sold/total, total, self.tMoney, self.starRatio, 0, 0     #1./110 .00909090
        cPrice, optimumPrice=sPrice, 0
        next_gmm=γ+dγ
        #delta=ratio/n   #dMax*.99999
        #cur_gmm=.07    #delta/2
        #next_gmm=cur_gmm+delta
        #cur_gam=gamma
        print('ratio γ dγ next_gmm remain amount amounts cPrice avgPrice=%.7f %.7f %.7f %.7f %s %3d %3d %.3f'\
                %(self.ratio, self.γ0, self.dγ, next_gmm, γ*next_gmm<dγ, self.remain, self.amounts, self.cPrice))
        n=0
        while self.amounts<self.sold:
            #print(ratio, end=', ')
            #next_gmm=cur_gmm+delta
            if γ*next_gmm<dγ and dγ<1/γ:
                if next_gmm<γmax:
                #if cur_gmm<γmax and next_gmm<γmax:
                    #print('ratio', ratio, end=', ')
                    #Ratio=ratio
                    #amount=floor(remain*γ*starRatio)  #amount=floor(remain*γ*.84)
                    #cPrice+=dPrice
                    #tMoney+=amount*cPrice
                    #amounts+=amount
                    #avgPrice=tMoney/amounts
                    #ratio-=γ
                    #ratio/=1-γ
                    #print('%f %f %.3f %.3f %.3f %.3f %.3f'%(Δ, cur_gmm, remain, amount, amounts, cPrice, avgPrice))
                    #print(cur_gmm)
                    #print(next_gmm*(1-cur_gmm)*remain)
                    #tmp=cur_gmm
                    self.portforlioVerbose(γ=γ, n=n)
                    #print('n=', n)
                    n+=1
                    γ=next_gmm
                    #next_gmm=tmp+delta   #(sqrt(1+4./delta)+1)*delta/2.*.9
                    next_gmm+=dγ   #(sqrt(1+4./delta)+1)*delta/2.*.9
                    #if self.Info:
                    #    cmd='%.5f %.7f %.7f %.7f %5d %3d %4d %.3f %.3f'%(Ratio, γ, dγ, next_gmm, remain, amount, amounts, cPrice, avgPrice)
                    #    print(cmd)
                    #assert γ<γmax, 'γ>γmax'
                    #assert next_gmm<γmax, 'next_gmm>γmax'
                    #assert cPrice<ePrice, 'cPrice>ePrice'
                    if γ>γmax or next_gmm>γmax or cPrice>ePrice: return avgPrice
                    #remain-=amount
                else:
                    print('next_gmm>γmax', next_gmm)
                    break
            else:
                #oldΔ=Δ
                dγ*=(1+γ)/(1-γ)
                if γ>γmax or next_gmm>γmax: break
                #Δ/=(1-2*cur_gmm)
                #Δ=2*Δ/(1-2*cur_gmm)
                #tmp=cur_gmm
                #cur_gmm=next_gmm
                #cur_gmm=next_gmm   #(sqrt(1+4./delta)+1)*delta/2.*.9
                #next_gmm+=Δ   #(sqrt(1+4./delta)+1)*delta/2.*.9
                #print('oldΔ<Δ', oldΔ<Δ)
                #print('cur_gmm, Δ, next_gmm= %.7f %.7f %.7f'%(cur_gmm, Δ, next_gmm), next_gmm*cur_gmm<Δ)  
                #tmp, cur_gmm=cur_gmm, next_gmm
                #next_gmm=tmp+delta
                #cur_gmm/=2
                #next_gmm/=2  #(sqrt(1+4./delta)+1)*delta/2.*.9
                #print('next_gmm, Δ= %5.7f %5.7f'%(next_gmm, Δ), end='\n')
        print()
def nStock(args):
    total, sold, sGmm, eGmm=args.total, args.sold, args.start_gamma, args.end_gamma
    sPrice, dPrice, ePrice=args.start_price, args.delta_price, args.end_price
    remain, ratio, tMoney, amounts, tGmm, n=total, sold/total, 0, 0, 0, 0
    #nMin=#log(sold/total, .9)
    N=floor(args.nmin)
    while n<=N:
        gamma=sGmm+(eGmm-sGmm)*n/N
        #gmm=iterStock(n, gamma=gamma)
        #gmm=iterStock(iTer, gamma=.09)
        #tGmm+=gmm
        amount=floor(gamma*remain)
        amounts+=amount
        cPrice=sPrice+n*dPrice
        #cPrice=ePrice-(N-n)*dPrice
        tMoney+=cPrice*amount
        avgPrice=tMoney/amounts
        Ratio=amounts/total
        if cPrice>ePrice: break
        print('%.5f %4d %3d %3d %.3f %.3f %.3f'%(gamma, remain, amount, amounts, Ratio, cPrice, avgPrice), end='\n')
        #print('%.5f %4d %3d %3d %.3f %.3f'%(gamma, remain, amount, amounts, cPrice, avgPrice), end='\n')
        #print('%.5f %.5f %.5f %3d %3d %3d'%(gamma, gmm, tGmm, remain, amount, amounts), end='\n')
        remain-=amount
        n+=1
        if amounts>sold:
            #print('%.5f %.5f %.3f %.3f %.3f'%(gamma, remain, amount, amounts, avgPrice), end='\n')
            break
        #ratio-=gamma
    #print(remain)    #0.03144099999999991, 377.29199999999895
    '708.5880000000002+787.3200000000002+874.8000000000003+972.0000000000002+1080.0000000000002+1200'
def cGamma(total=0, stock=0, nmin=0, γmax=0, γ=0, Δ=0, eGamma=0, sPrice=0, dPrice=0, ePrice=0, INFO=False):
    ratio, remain, amounts, tMoney=stock/total, total, 0, 0
    cPrice=sPrice
    next_gmm=γ+Δ
    #print(amounts, stock, γ, next_gmm, Δ, γ*next_gmm<Δ)
    while amounts<stock:
        if γ*next_gmm<Δ:
            if next_gmm<γmax:
                Ratio=ratio
                ratio-=γ
                ratio/=1-γ
                amount=floor(remain*γ*.84)
                cPrice+=dPrice
                tMoney+=amount*cPrice
                amounts+=amount
                avgPrice=tMoney/amounts
                γ=next_gmm
                next_gmm+=Δ
                if INFO:
                    cmd='%.5f %.5f %.7f %.7f %3d %3d %3d %.3f %.3f'%(Ratio, γ, Δ, next_gmm, remain, amount, amounts, cPrice, avgPrice)
                    print(cmd)
                if γ>γmax or next_gmm>γmax or cPrice>ePrice: return avgPrice
                remain-=amount
        else:
            #oldΔ=Δ
            Δ*=(1+γ)/(1-γ)
            if γ>γmax or next_gmm>γmax: return None
def optimumPrice(args):
    total, stock, nmin, γmax=args.total, args.sold, args.nmin, args.γmax
    γ, Δ, eGamma=args.start_gamma, args.delta_gamma, args.end_gamma
    sPrice, dPrice, ePrice=args.start_price, args.delta_price, args.end_price
    INFO, Del=args.Info, 1e-3
    estPrice=cGamma(total=total, stock=stock, nmin=nmin, γmax=γmax, γ=γ, Δ=Δ, eGamma=eGamma, \
            sPrice=sPrice, dPrice=dPrice, ePrice=ePrice, INFO=INFO)
    γ+=Del
    optimumPrice=cGamma(total=total, stock=stock, nmin=nmin, γmax=γmax, γ=γ, Δ=Δ, eGamma=eGamma, \
        sPrice=sPrice, dPrice=dPrice, ePrice=ePrice, INFO=INFO)
    print(estPrice, optimumPrice, estPrice==optimumPrice)
    while optimumPrice<estPrice:
        #estPrice=cGamma(total=total, stock=stock, nmin=nmin, γmax=γmax, sGamma=sGamma, Δ=Δ, eGamma=eGamma, sPrice=sPrice, dPrice=dPrice, ePrice=ePrice)
        #if optimumPrice<estPrice:
        if optimumPrice<estPrice:
            Δ-=Del
            estPrice=cGamma(total=total, stock=stock, nmin=nmin, γmax=γmax, γ=γ, Δ=Δ, eGamma=eGamma, \
                sPrice=sPrice, dPrice=dPrice, ePrice=ePrice, INFO=INFO)
            if estPrice: print('γ, estPrice=%.5f %.7f'%(γ, estPrice))
            else: return
        else:
            Δ+=Del
            optimumPrice=cGamma(total=total, stock=stock, nmin=nmin, γmax=γmax, γ=γ, Δ=Δ, eGamma=eGamma, \
                sPrice=sPrice, dPrice=dPrice, ePrice=ePrice, INFO=INFO)
            if optimumPrice: print('γ, optimumPrice=%.5f %.7f'%(γ, optimumPrice))
            else: return
    print(optimumPrice)
def iterStock(n, gamma=.1):
    if n==1: return gamma
    else: return iterStock(n-1)*(1-gamma)
def calc_stock(args):
    remain_stock, sell_stock, delta, start_price, end_price=args.remain, args.sold, args.delta, args.start_price, args.end_price
    print('remain_stock', 'sell_stock', 'delta=', remain_stock, sell_stock, delta)
    n, amount, amounts, money, price, gamma, sell_remain=0, .0, .0, .0, start_price, 0.1, sell_stock
    sky_info=[]
    print('n', 'gamma', 'cur_stock', 'amount', 'remain_stock', 'amounts', 'price', 'money', end='\n')
    while amounts<sell_stock:
        n+=1
        if args.rand: gamma=choice([.1, .09, .08, .07])
        else: gamma=.1
        cur_stock, max_amount=remain_stock, remain_stock*.1
        amount=sell_remain if max_amount>sell_remain else remain_stock*gamma
        if price>end_price or amount==sell_remain:
            amounts+=amount
            sell_remain=sell_stock-amounts
            money+=amount*price
            price+=delta
            remain_stock-=amount
            print('%s,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f'%(n, gamma, cur_stock, amount, remain_stock, amounts, price, money))
            break
        else:
            amounts+=amount
            sell_remain=sell_stock-amounts
            money+=amount*price
            price+=delta
            remain_stock-=amount
        if args.detail: print('%s,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f'%(n, gamma, cur_stock, amount, remain_stock, amounts, price, money))
        #if price>end_price or amount>sell_remain or remain_stock<0:break
        #sky_info.append([stock, gamma, price, money])
        #p, q, w, x, y, z=map(lambda val: repr(val).rjust(3), [n, gamma, cur_stock, amounts, amount, sell_remain])
        #rslt=map(lambda val: repr(val).rjust(3), 
        #print(p, q, w, x, y, z)
        #print(','.join(rslt), end='\n')
        #if remain_stock<0:break
if __name__=='__main__':
    parser = ArgumentParser(description='calculate stock to the total of SKY')
    parser.add_argument('--total', '-l', type=int, default=12000, help='the total stock')
    parser.add_argument('--sold', '-s', type=int, default=6000, help='the stock sold')
    parser.add_argument('--nmin', '-n', type=int, default=6, help='the minimal iterations')
    parser.add_argument('--γmax', '-m', type=float, default=.1, help='the maximum selling percentage')
    parser.add_argument('--start_gamma', '-a', type=float, default=.05, help='the start gamma')
    parser.add_argument('--delta_gamma', '-d', type=float, default=1e-3, help='the delta gamma')
    parser.add_argument('--end_gamma', '-g', type=float, default=.1, help='the end gamma')
    parser.add_argument('--start_price', '-p', type=float, default=2.9, help='the start price')
    parser.add_argument('--delta_price', '-t', type=float, default=.03, help='the delta price')
    parser.add_argument('--end_price', '-e', type=float, default=3.3, help='the end price')
    parser.add_argument('--star', '-r', type=int, default=3, help='the star rank')
    parser.add_argument('--Calc', '-C', action='store_true', default=False, help='calculation of sky')
    parser.add_argument('--Rand', '-R', action='store_true', default=False, help='random choice of gamma')
    parser.add_argument('--Detail', '-D', action='store_true', default=False, help='detailed gamma listed')
    parser.add_argument('--Iter', '-I', action='store_true', default=False, help='iteration of gamma')
    parser.add_argument('--Gamma', '-G', action='store_true', default=False, help='iteration of gamma')
    parser.add_argument('--Sky', '-K', action='store_true', default=False, help='iteration of gamma')
    parser.add_argument('--Optimum', '-O', action='store_true', default=False, help='iteration of gamma')
    parser.add_argument('--Info', '-N', action='store_true', default=True, help='verbose of optimum')
    parser.add_argument('--Profit', '-P', action='store_true', default=False, help='maximum profit')
    parser.add_argument('--Portforlio', '-F', action='store_true', default=False, help='portforlio details')
    parser.add_argument('--Ascended', '-A', action='store_true', help='ascended price listings')
    args = parser.parse_args()
    if args.Calc: calc_stock(args)
    elif args.Iter: nStock(args)
    elif args.Gamma:
        sky=skyStock(args)
        sky.calcGamma()
    elif args.Optimum: optimumPrice(args)
    elif args.Profit:
        sky=skyStock(args)
        sky.maxProfit()
    elif args.Portforlio:
        sky=skyStock(args)
        sky.maxProfit()
        #sky.profitVerbose()
    elif args.Sky:
        sky=skyStock(args)
        sky.fn(10)
    #group = parser.add_mutually_exclusive_group()
    #group.add_argument('-v', '--verbose', action='store_true')
    #group.add_argument('-q', '--quiet', action='store_true')
    #parser.add_argument('-b', '--delta', action='store', dest='b')
    #print(args)
    #print parser.parse_args(['-a', '-bval', '-c', '3'])
    '''
    ./sky.py -F -t 980 -s 200 -a .06 -r 3 -A
    ./sky.py -G -l 4575 -s 1600 -a .06 -p 2.9 -d .0001 -A
    def getCache(self, n):
        self.portforlio[n]=self.fn(n)
        return self.portforlio[n]
    def fn(self, n, gmm=.1):
        if n in self.portforlio: return self.portforlio[n]
        else: return self.portforlio[n-1]
        if n==1: return gmm
        else: 
            self.portforlio[n]=self.fn(n-1)
            return self.fn(n-1)
        [samuel@arch ~]$ sky.py -G -l 4575 -s 1600 -a .06 -d .0001 -A
        ratio γ dγ next_gmm remain amount amounts cPrice avgPrice=0.3497268 0.0600000 0.0001000 0.0601000 False 4575   0 2.900
        4575 274 230 230 0.05040 0.06000 2.900 3.452
        4344 261 219 449 0.09834 0.06010 2.930 3.470
        4125 263 220 670 0.14664 0.06378 2.960 3.488
        3904 265 222 893 0.19533 0.06792 2.990 3.506
        3681 267 224 1118 0.24442 0.07263 3.020 3.524
        3456 269 226 1344 0.29395 0.07803 3.050 3.542
        3230 272 228 1573 0.34393 0.08427 3.080 3.560




      12000 6000 0.01
      n gamma cur_stock amount remain_stock amounts price money
      1 0.07 12000 840.0000000000001 840.0000000000001 5160.0
      2 0.1 11160.0 1956.0 1116.0 4044.0
      3 0.1 9204.0 2876.4 920.4000000000001 3123.6
      4 0.07 6327.6 3319.3320000000003 442.9320000000001 2680.6679999999997
      5 0.08 3008.268 3559.99344 240.66144 2440.00656
      6 0.1 -551.7254400000002 3504.820896 -55.172544000000016 2495.179104
      7 0.09 -4056.5463360000003 3139.7317257600002 -365.08917024000004 2860.2682742399998
      8 0.08 -7196.27806176 2564.0294808192 -575.7022449408 3435.9705191808
      9 0.08 -9760.307542579201 1783.204877412864 -780.8246034063361 4216.795122587136
def maxProfit(args):
    total, sold, γ0, γmax=args.total, args.sold, args.start_gamma, args.γmax
    Δ, eps=1e-5, 1e-6
    sPrice, dPrice, ePrice=args.start_price, args.delta_price, args.end_price
    remain, Ratio=total, sold/total
    tMoney, amounts, nmin=0, 0, True
    print('ratio=', Ratio, γ0)
    def init_gamma(γ0, ratio=Ratio):
        n=0
        while ratio:
            γ=γ0 if not n else .1
            ratio-=γ
            if ratio<eps: return n+1
            ratio/=1-γ     #1-.1
            n+=1
        return n
    profit={}
    while γ0<γmax:
        nmin=init_gamma(γ0)
        profit[nmin]=γ0
        #if nmin<8: print('γ0, nmin=', γ0, nmin)
        γ0+=Δ
    print(profit)
    '''
