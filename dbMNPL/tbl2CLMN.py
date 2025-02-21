#!/usr/bin/env python3
from sys import stdin
from re import findall
from argparse import ArgumentParser
from stUtil import rndrCode

def DDL(args):
    data=stdin.read()
    regExp=r'"?(\w+)"?\s(.*?)\s(\w+),? ?-?-? ?(.*)'
    for clmn_name, val_type, defaultVal,comment in findall(regExp, data):
        rndrCode('@'.join([clmn_name, val_type, comment.replace(' ','')]))
def delimiter(args):
    delimiter=args.delimiter
    data=stdin.read()
    for line in data.split('\n')[:-1]:
        cols=line.replace(' ','').split(delimiter)
        rndrCode('","'.join(cols))

if __name__=='__main__':
        parser = ArgumentParser(description='html parser')
        parser.add_argument('--delimiter', '-d', action='store', default='|', help='the default delimiter')
        parser.add_argument('--DDL', '-D', action='store_true', default=False, help='the default DDL')
        parser.add_argument('--Column', '-C', action='store_true', default=False, help='Column')
        args = parser.parse_args()
        if args.Column: delimiter(args)
        if args.DDL: DDL(args)
