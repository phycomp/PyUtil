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
        self.n, self.γmax, self.starRatio=args.nmin, args.γmax, skyStar[args.star]
        self.ratio=self.Ratio=self.sold/self.total
        self.avgPrice=self.tMoney=self.amount=self.amounts=0
        self.portforlio, self.Δ, self.eps={}, 1e-7, 1e-7

