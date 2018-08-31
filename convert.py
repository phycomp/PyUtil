#!/usr/bin/env python

from glob import glob
from os.path import splitext
from os import system

Files=glob('*.jpg')
for fname in Files:
	base, ext=splitext(fname)
	cmd='convert %s -resize 50%% PNGs/%s-shrink.png'%(fname, base)
	#print(cmd)
	system(cmd)
