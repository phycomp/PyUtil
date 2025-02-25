#!/usr/bin/env python
from random import choice as rndmChc, choices as rndmChoices
from glob import glob
FILEs=glob('*.jpg')
loopArg, xfadeArg='ffmpeg ', '-filter_complex "'
fLen, trnsDur, vidDur, offset=len(FILEs), 1, 5, 0

XFADEs=['custom', 'fade', 'wipeleft', 'wiperight', 'wipeup', 'wipedown', 'slideleft', 'slideright', 'slideup', 'slidedown', 'circlecrop', 'rectcrop', 'distance', 'fadeblack', 'fadewhite', 'radial', 'smoothleft', 'smoothright', 'smoothup', 'smoothdown', 'circleopen', 'circleclose', 'vertopen', 'vertclose', 'horzopen', 'horzclose', 'dissolve', 'pixelize', 'diagtl', 'diagtr', 'diagbl', 'diagbr', 'hlslice', 'hrslice', 'vuslice', 'vdslice', 'hblur', 'fadegrays', 'wipetl', 'wipetr', 'wipebl', 'wipebr', 'squeezeh', 'squeezev', 'zoomin', 'fadefast', 'fadeslow', 'hlwind', 'hrwind', 'vuwind', 'vdwind', 'coverleft', 'coverright', 'coverup', 'coverdown', 'revealleft', 'revealright', 'revealup', 'revealdown']

xFADE=rndmChoices(XFADEs, k=fLen)
for idx, file in enumerate(FILEs):
  loopArg+=f'-loop 1 -t {vidDur} -i {file} '
  #ffct=rndmChce(EFFECTs)
  #xffct=rndmChce(XFADEs)
  xfd=xFADE[idx]
  offset+=vidDur-trnsDur
  xfd=xFADE[idx]
  if not idx:
    xfadeArg+=f"[0p][1p]xfade=transition={xfd}:duration={trnsDur}:offset={offset}[1x]; "
  elif idx==fLen-1:
      pass
  else:
    xfadeArg+=f"[{idx}x][{idx+1}p]xfade=transition={xfd}:duration={trnsDur}:offset={offset}[{idx+1}x]; "
xfadeArg+='"'
extraARG=f""" -map "[{idx}x]" -c:v libx264 -crf 17 -y output.mp4"""
cmd=f"{loopArg} {xfadeArg} {extraARG}"
print(cmd)
"""
-filter_complex "
[0]scale=-2:720,pad=1280:720:(ow-iw)/2:(oh-ih)/2,setsar=1[0p];
[1]scale=-2:720,pad=1280:720:(ow-iw)/2:(oh-ih)/2,setsar=1[1p];
[2]scale=-2:720,pad=1280:720:(ow-iw)/2:(oh-ih)/2,setsar=1[2p];
[3]scale=-2:720,pad=1280:720:(ow-iw)/2:(oh-ih)/2,setsar=1[3p];
[0p][1p]xfade=transition=slideright:duration=1:offset=4[1x];
[1x][2p]xfade=transition=circleopen:duration=1:offset=8[2x];
[2x][3p]xfade=transition=slideup:duration=1:offset=12[3x]
" -map [3x] -c:v libx264 -crf 17 output.mp4
ffmpeg -loop 1 -t 5 -i 3.jpg -loop 1 -t 5 -i 2.jpg -loop 1 -t 5 -i 4.jpg -loop 1 -t 5 -i 1.jpg -loop 1 -t 5 -i 5.jpg  -filter_complex 
"[0p][1p]xfade=transition=coverleft:duration=1:offset=4[1x];
[1x][2p]xfade=transition=wipebr:duration=1:offset=8[2x];
[2x][3p]xfade=transition=dissolve:duration=1:offset=12[3x];
[3x][4p]xfade=transition=distance:duration=1:offset=16[4x];"  -map "[4x]" -c:v libx264 -crf 17 -y output.mp4

ffmpeg -loop 1 -t 5 -i 3.jpg -loop 1 -t 5 -i 2.jpg -loop 1 -t 5 -i 4.jpg -loop 1 -t 5 -i 1.jpg -loop 1 -t 5 -i 5.jpg  -filter_complex 
"[0p][1p]xfade=transition=smoothleft:duration=1:offset=4[1x];
[1x][2p]xfade=transition=wipebl:duration=1:offset=8[2x];
[2x][3p]xfade=transition=vuslice:duration=1:offset=12[3x];
[3x][4p]xfade=transition=slidedown:duration=1:offset=16[4x];
[4x][5p]xfade=transition=coverup:duration=1:offset=20[5x];" 
-map "[4x]" -c:v libx264 -crf 17 -y output.mp4

ffmpeg -loop 1 -t 5 -i 3.jpg -loop 1 -t 5 -i 2.jpg -loop 1 -t 5 -i 4.jpg -loop 1 -t 5 -i 1.jpg -loop 1 -t 5 -i 5.jpg  "[0][1]xfade=transition=diagtl:duration=1:offset=4[f1];[f0][1]xfade=transition=fadeblack:duration=1:offset=8[f1];[f1][2]xfade=transition=fadewhite:duration=1:offset=12[f2];[f2][3]xfade=transition=revealup:duration=1:offset=16[f3];[f3][4]xfade=transition=fade:duration=1:offset=20[f4];" -map "[f4]" -r 25 -pix_fmt yuv420p -vcodec libx264 -y output.mp4
"""
