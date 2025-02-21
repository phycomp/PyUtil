#!/usr/bin/env python

from PIL import Image
from sys import argv
from re import search
def crop(Files):
	for fname in Files:
		original = Image.open(fname)
		cropped=original.crop((0, 240, 718, 1200))
		cropFname=search('-(\d+.png)', fname).group(1)
		print(cropFname)
		cropped.save('/tmp/%s'%cropFname)
Files = argv[1:]
crop(Files)
