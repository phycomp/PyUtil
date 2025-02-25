#可以運作的script手稿
ffmpeg -loop 1 -t 3 -i 1.jpg -loop 1 -t 3 -i 2.jpg -loop 1 -t 3 -i 3.jpg -loop 1 -t 3 -i 4.jpg -loop 1 -t 3 -i 5.jpg -filter_complex "[0][1]xfade=transition=circlecrop:duration=0.5:offset=2.5[f0]; \
[f0][2]xfade=transition=smoothleft:duration=0.5:offset=5[f1]; \
[f1][3]xfade=transition=pixelize:duration=0.5:offset=7.5[f2]; \
[f2][4]xfade=transition=hblur:duration=0.5:offset=10[f3],format=yuv444p,scale=375:500" -map "[f3]" -r 25 -pix_fmt yuv420p -vcodec libx264 output-swipe-custom.mp4
[0][1]scale2ref[bg][gif];[bg]setsar=1[bg];[bg][gif]overlay=shortest=1'  /tmp/outBlack2.mp4")

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
#!/usr/bin/env python
from argparse import ArgumentParser
from random import choices as rndmChoices
from glob import glob

#XFADEs=['custom', 'fade', 'wipeleft', 'wiperight', 'wipeup', 'wipedown', 'slideleft', 'slideright', 'slideup', 'slidedown', 'circlecrop', 'rectcrop', 'distance', 'fadeblack', 'fadewhite', 'radial', 'smoothleft', 'smoothright', 'smoothup', 'smoothdown', 'circleopen', 'circleclose', 'vertopen', 'vertclose', 'horzopen', 'horzclose', 'dissolve', 'pixelize', 'diagtl', 'diagtr', 'diagbl', 'diagbr', 'hlslice', 'hrslice', 'vuslice', 'vdslice', 'hblur', 'fadegrays', 'wipetl', 'wipetr', 'wipebl', 'wipebr', 'squeezeh', 'squeezev', 'zoomin', 'fadefast', 'fadeslow', 'hlwind', 'hrwind', 'vuwind', 'vdwind', 'coverleft', 'coverright', 'coverup', 'coverdown', 'revealleft', 'revealright', 'revealup', 'revealdown']
FILTERs=['fade', 'wipeleft', 'wiperight', 'wipeup', 'wipedown', 'slideleft', 'slideright', 'slideup', 'slidedown', 'circlecrop', 'rectcrop', 'distance', 'fadeblack', 'fadewhite', 'radial', 'smoothleft', 'smoothright', 'smoothup', 'smoothdown', 'circleopen', 'circleclose', 'vertopen', 'vertclose', 'horzopen', 'horzclose', 'dissolve', 'pixelize', 'diagtl', 'diagtr', 'diagbl', 'diagbr', 'hlslice', 'hrslice', 'vuslice', 'vdslice', 'hblur', 'fadegrays', 'wipetl', 'wipetr', 'wipebl', 'wipebr', 'squeezeh', 'squeezev', 'zoomin']
def mergeVid(args):
    vidMerge, vext, cnctARG='', args.vext, ''
    VIDs=glob(f'*.{vext}')
    for v in args.xclsv:
        print('eleVid=', v)
        VIDs.remove(v)
    print( VIDs)
    vLen=len(VIDs)
    for ndx, vid in enumerate(VIDs):
        vidMerge+=f'-i {vid} '
        cnctARG+=f'[{ndx}:v]'   #[{ndx}:a]
    cnctARG+=f'concat=n={vLen}:v=1:a=0[v]'  # [a]
#ffmpeg -i input1.mp4 -i input2.wmv -filter_complex "[0:0][0:1][1:0][1:1]concat=n=2:v=1:a=1[outv][outa]" -map "[outv]" -map "[outa]" output.mp4
#ffmpeg -i 1.mp4 -i 2.mp4 -filter_complex "[0:v][1:v]concat=n=2:v=1:a=0[outv]" -map "[outv]" output.mp4
    #ffmpeg -i opening.mkv -i episode.mkv -i ending.mkv -filter_complex "[0:v] [0:a] [1:v] [1:a] [2:v] [2:a] concat=n=3:v=1:a=1 [v] [a]"  -map "[v]" -map "[a]" output.mkv
    print(f'''ffmpeg {vidMerge} -filter_complex "{cnctARG}" -map "[v]" -y output.{vext}''') #-map "[a]" 

def vid2Vid(args):
    FILEs=glob(f'*.{args.vid}')
    ffCMD='''ffmpeg -i video1.mp4 \
     -i video2.mp4 \
     -i video3.mp4 \
    -filter_complex \
    "[0:v]trim=start=0:end=7,setpts=PTS-STARTPTS,scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:-1:-1,setsar=1,fps=24,format=yuv420p[v0]; \
    [1:v]trim=start=0:end=10,setpts=PTS-STARTPTS,scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:-1:-1,setsar=1,fps=24,format=yuv420p[v1]; \
    [2:v]trim=start=0:end=7,setpts=PTS-STARTPTS,scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:-1:-1,setsar=1,fps=24,format=yuv420p[v2]; \
    [0:a]atrim=start=0:end=7,asetpts=PTS-STARTPTS[a0]; \
    [1:a]atrim=start=0:end=10,asetpts=PTS-STARTPTS[a1]; \
    [2:a]atrim=start=0:end=7,asetpts=PTS-STARTPTS[a2]; \
    [v0][a0][v1][a1][v2][a2]concat=n=3:v=1:a=1[v][a]" \
    -map "[v]" -map "[a]" -c:v libx264 -c:a aac -movflags +faststart output.mp4
'''

def pic2Vid(args):
    FILEs=glob(f'*.{args.pic}')
    fLen=len(FILEs)
    FILEs=rndmChoices(FILEs, k=fLen)
    loopArg, imgARG, vidArg='', '', ''
    cnctName, cnvntnName, trnsDur, vidDur, offset='', 'trns', 1, args.dur, 0
    xFilter=rndmChoices(FILTERs, k=fLen)
    for ndx, file in enumerate(FILEs):
      loopArg+=f'-loop 1 -t {vidDur} -i {file} '
      imgARG+=f'[{ndx}]scale=-2:720,pad=1280:720:(ow-iw)/2:(oh-ih)/2,setsar=1[{ndx}newPic];'
      offset+=vidDur-trnsDur
      xfd=xFilter[ndx]
      if not ndx:
        vidArg+=f'''[0newPic][1newPic]xfade=transition={xfd}:duration={trnsDur}:offset={offset}[1newVid]; '''
      elif ndx==fLen-1:
        extraARG=f'''-map "[{ndx}newVid]" -c:v libx264 -crf 17'''
      else:
        vidArg+=f"[{ndx}newVid][{ndx+1}newPic]xfade=transition={xfd}:duration={trnsDur}:offset={offset}[{ndx+1}newVid];"
    cmd=f"""ffmpeg {loopArg} -filter_complex "{imgARG} {vidArg}" {extraARG} -y output.mp4"""  #{argAVTB}
    print(cmd)
#ffmpeg -f concat -safe 0 -i join_video.txt -c copy output_demuxer.mp4
#ffmpeg -i "concat:input1.ts|input2.ts" -c copy output_protocol.ts
#ffmpeg -i "concat:input1.mp4|input2.mp4" -c copy output_protocol.mp4
#ffmpeg -i input1.mp4 -i input2.wmv -filter_complex "[0:0][0:1][1:0][1:1]concat=n=2:v=1:a=1[outv][outa]" -map "[outv]" -map "[outa]" output.mp4



if __name__=='__main__':
    parser = ArgumentParser(description='calculate stock to the total of SKY')
    #parser.add_argument('--Upload', '-U', action='store_true', default=False, help='uploadFiles')
    parser.add_argument('--Pic', '-P', action='store_true', default=False, help='execCmd')
    parser.add_argument('--Merge', '-M', action='store_true', default=False, help='execCmd')
    parser.add_argument('--Vid', '-V', action='store_true', default=False, help='execCmd')
    parser.add_argument('--vext', '-v', type=str, default='mp4', help='nodes')
    parser.add_argument('--dur', '-d', type=int, default=5, help='duration')
    parser.add_argument('--pic', '-p', type=str, default='png', help='nodes')
    parser.add_argument('--xclsv', '-x', default=['output.mp4'], nargs='*', help='files') #type=str, 
    args = parser.parse_args()
    if args.Pic: pic2Vid(args)
    if args.Vid: vid2Vid(args)
    if args.Merge: mergeVid(args)
'''
可以運作的 ffmpeg -loop 1 -t 5 -i 3.jpg -loop 1 -t 5 -i 2.jpg -loop 1 -t 5 -i 0.jpg -loop 1 -t 5 -i 4.jpg -loop 1 -t 5 -i 1.jpg -loop 1 -t 5 -i 5.jpg  -filter_complex "[0][1]xfade=transition=diagtr:duration=1:offset=4[trns0]; [trns0][2]xfade=transition=vertopen:duration=1:offset=8[trns1]; [trns1][3]xfade=transition=wipebr:duration=1:offset=12[trns2]; [trns2][4]xfade=transition=wipedown:duration=1:offset=16[trns3]; [trns3][5]xfade=transition=smoothright:duration=1:offset=20,format=yuv420p[final]; " -c:v libx264 -map "[final]" output.mp4

ffmpeg -y -progress .progressinfo.dat
       -i "MERGED.Test0.mp4"
       -i "f3.ts"
       -filter_complex "[0:v]settb=AVTB,setpts=PTS-STARTPTS[v0];
                        [1:v]settb=AVTB,setpts=PTS-STARTPTS[v1];
                        [v0][v1]xfade=transition=fade:duration=1:offset=1.645,format=yuv420p;
                        [0:a]asettb=AVTB,asetpts=PTS-STARTPTS[a0];
                        [1:a]asettb=AVTB,asetpts=PTS-STARTPTS[a1];
                        [a0][a1]acrossfade=d=1"
        -movflags +faststart out_MERGED.Test.mp4
'''
