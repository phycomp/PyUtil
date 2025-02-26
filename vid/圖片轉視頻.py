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

#!/usr/bin/env python
def png2vid():
  from random import choices as rndmChoices  #choice as rndmChc,
  from glob import glob
  FILEs=glob('*.jpg')
  loopArg, imgARG, vidArg, argFinal='', '', '', 'final'   #argAVTB, , ''
  cnctName, cnvntnName, fLen, trnsDur, vidDur, offset='', 'trns', len(FILEs), 1, 5, 0

  XFADEs=['custom', 'fade', 'wipeleft', 'wiperight', 'wipeup', 'wipedown', 'slideleft', 'slideright', 'slideup', 'slidedown', 'circlecrop', 'rectcrop', 'distance', 'fadeblack', 'fadewhite', 'radial', 'smoothleft', 'smoothright', 'smoothup', 'smoothdown', 'circleopen', 'circleclose', 'vertopen', 'vertclose', 'horzopen', 'horzclose', 'dissolve', 'pixelize', 'diagtl', 'diagtr', 'diagbl', 'diagbr', 'hlslice', 'hrslice', 'vuslice', 'vdslice', 'hblur', 'fadegrays', 'wipetl', 'wipetr', 'wipebl', 'wipebr', 'squeezeh', 'squeezev', 'zoomin', 'fadefast', 'fadeslow', 'hlwind', 'hrwind', 'vuwind', 'vdwind', 'coverleft', 'coverright', 'coverup', 'coverdown', 'revealleft', 'revealright', 'revealup', 'revealdown']

  #ffmpeg -i mergeAll.mp4 -stream_loop -1 -i lotusCourage.mp3 -c:v copy  -shortest -map 0:v -map 1:a  -y mergeFull6.mp4

  xFADE=rndmChoices(XFADEs, k=fLen)
  for idx, file in enumerate(FILEs):
    loopArg+=f'-loop 1 -t {vidDur} -i {file} '
    imgARG+=f'[{idx}]scale=-2:720,pad=1280:720:(ow-iw)/2:(oh-ih)/2,setsar=1[{idx}newPic];'
    offset+=vidDur-trnsDur
    xfd=xFADE[idx]
    if not idx:
      vidArg+=f'''[0newPic][1newPic]xfade=transition={xfd}:duration={trnsDur}:offset={offset}[1newVid]; '''
    else:
      vidArg+=f"[{idx}newPic][{idx+1}newPic]xfade=transition={xfd}:duration={trnsDur}:offset={offset}[{idx+1}newVid];"
    print(idx)
  #[va0][va1]overlay[outv]" -c:v libx264 -map [outv] -y out.mp4
  cmd=f"""ffmpeg {loopArg} -filter_complex "{imgARG} {vidArg}" -map "[{idx}newVid]" -c:v libx264 -crf 17 -y output.mp4"""  #{argAVTB}
  print(cmd)

  """
  ffmpeg -loop 1 -t 5 -i 3.jpg -loop 1 -t 5 -i 2.jpg -loop 1 -t 5 -i 4.jpg -loop 1 -t 5 -i 1.jpg -loop 1 -t 5 -i 5.jpg  -filter_complex "[0][1]xfade=transition=diagbl:duration=1:offset=4[trns0];
  [trns0][2]xfade=transition=smoothleft:duration=1:offset=8[trns1];
  [trns1][3]xfade=transition=radial:duration=1:offset=12[trns2];
  [trns2][4]xfade=transition=vertclose:duration=1:offset=16,format=yuv420p[final];"
  -map "[final]" -y output.mp4
  """
