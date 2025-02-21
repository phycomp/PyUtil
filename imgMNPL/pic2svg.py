#!/usr/bin/env python
from os.path import splitext
from os import system

JPGs=['7829116694213.jpg', '7829116705258.jpg']
for jpg in JPGs:
	base, ext=splitext(jpg)
	cmd='convert %s %s.svg'%(jpg, base)
	print(cmd)
	system(cmd)
