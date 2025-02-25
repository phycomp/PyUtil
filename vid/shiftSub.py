#!/usr/bin/env python
from pysubs2 import load as sbLoad, SSAFile
from argparse import ArgumentParser
from sys import argv
from os.path import splitext
from io import StringIO
def shiftDur(args):
    sbttl=args.subttl
    base, ext=splitext(sbttl)
    tmShft=args.timeshift
    subs = sbLoad(sbttl, encoding="utf-8")
    subs.shift(s=tmShft)
    for line in subs:
        line.text = "{\\be1}" + line.text
    subs.save(f'new{base}{ext}')

def shiftMem(args):
    from sys import stdin
    print('rawSub: ')
    rawSub=stdin.read()
    sbttl=StringIO(rawSub)
    subs=SSAFile.from_string(rawSub)
    tmShft=args.timeshift
    subs.shift(s=tmShft)
    print(subs.to_string('ass'))
if __name__=='__main__':
    parser = ArgumentParser(description='calculate stock to the total of SKY')
    parser.add_argument('--subttl', '-f', type=str, help='the total stock')
    parser.add_argument('--timeshift', '-s', type=int, default=6000, help='the stock sold')
    parser.add_argument('--Timeshift', '-T', action='store_true', help='Timeshift')
    parser.add_argument('--Memory', '-M', action='store_true', help='Memory')
    args = parser.parse_args()
    if args.Memory: shiftMem(args)
    elif args.Timeshift: shiftDur(args)
