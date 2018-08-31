#!/usr/bin/env python
from os import system
from sys import argv

try:
	folder, regExp=argv[1:]
	cmd='ls %s|grep -i %s'%(folder, regExp)
except:
	folder=argv[1]
	cmd='ls|grep -i %s'%folder
print(cmd)
system(cmd)
