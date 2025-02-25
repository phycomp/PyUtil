#!/usr/bin/env python
from random import choice as rndmChc, choices as rndmChoices
from glob import glob
FILEs=glob('*.jpg')
loopArg, argAVTB, xfadeArg='ffmpeg -vsync 0 ', '-filter_complex "', ''
fLen, trnsDur, vidDur, offset=len(FILEs), 1, 5, 0

XFADEs=['custom', 'fade', 'wipeleft', 'wiperight', 'wipeup', 'wipedown', 'slideleft', 'slideright', 'slideup', 'slidedown', 'circlecrop', 'rectcrop', 'distance', 'fadeblack', 'fadewhite', 'radial', 'smoothleft', 'smoothright', 'smoothup', 'smoothdown', 'circleopen', 'circleclose', 'vertopen', 'vertclose', 'horzopen', 'horzclose', 'dissolve', 'pixelize', 'diagtl', 'diagtr', 'diagbl', 'diagbr', 'hlslice', 'hrslice', 'vuslice', 'vdslice', 'hblur', 'fadegrays', 'wipetl', 'wipetr', 'wipebl', 'wipebr', 'squeezeh', 'squeezev', 'zoomin', 'fadefast', 'fadeslow', 'hlwind', 'hrwind', 'vuwind', 'vdwind', 'coverleft', 'coverright', 'coverup', 'coverdown', 'revealleft', 'revealright', 'revealup', 'revealdown']

xFADE=rndmChoices(XFADEs, k=fLen)
for idx, file in enumerate(FILEs):
  loopArg+=f'-loop 1 -t {vidDur} -i {file} '
  argAVTB+=f'[{idx}]settb=AVTB[{idx}:v]; '
  offset+=vidDur-trnsDur
  xfd=xFADE[idx]
  if not idx:
    xfadeArg+=f"[0:v][1:v]xfade=transition={xfd}:duration={trnsDur}:offset={offset}[v1]; "
  elif idx==fLen-1:
    pass
  elif idx==fLen-2:
    xfadeArg+=f'''[v{idx}][{idx+1}:v]xfade=transition={xfd}:duration={trnsDur}:offset={offset},format=yuv420p[video];"'''
  else:
    xfadeArg+=f"[v{idx}][{idx+1}:v]xfade=transition={xfd}:duration={trnsDur}:offset={offset}[{idx+1}x]; "
extraARG=f""" -map "[video]" output.mp4"""
cmd=f"""{loopArg} {argAVTB} {xfadeArg} {extraARG}"""
print(cmd)

"""
ffmpeg  \
-vsync 0 \
-i 1.jpg \
-i 2.jpg \
-i 3.jpg \
-i 4.jpg \
-filter_complex "[0]settb=AVTB[0:v]; \
[1]settb=AVTB[1:v]; \
[2]settb=AVTB[2:v]; \
[3]settb=AVTB[3:v]; \
[0:v][1:v]xfade=transition=fade:duration=4:offset=1[v1]; \
[v1][2:v]xfade=transition=fade:duration=4:offset=0[v2]; \
[v2][3:v]xfade=transition=fade:duration=4:offset=2,format=yuv420p[video]" \
-b:v 10M \
-map "[video]" \
temp.mp4
"""
