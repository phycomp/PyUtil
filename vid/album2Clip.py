#!/usr/bin/env python
from os import listdir, chdir, system
from shlex import split as shlexSplit
from os.path import isdir, curdir, abspath, splitext
from argparse import ArgumentParser
from random import choices as rndmChoices
from glob import glob
from subprocess import Popen, PIPE, check_output, STDOUT, check_output as sbprcssChkoutput

FILTERs=['fade', 'wipeleft', 'wiperight', 'wipeup', 'wipedown', 'slideleft', 'slideright', 'slideup', 'slidedown', 'circlecrop', 'rectcrop', 'distance', 'fadeblack', 'fadewhite', 'radial', 'smoothleft', 'smoothright', 'smoothup', 'smoothdown', 'circleopen', 'circleclose', 'vertopen', 'vertclose', 'horzopen', 'horzclose', 'dissolve', 'pixelize', 'diagtl', 'diagtr', 'diagbl', 'diagbr', 'hlslice', 'hrslice', 'vuslice', 'vdslice', 'hblur', 'fadegrays', 'wipetl', 'wipetr', 'wipebl', 'wipebr', 'squeezeh', 'squeezev', 'zoomin']

def annttPic(args):
    from glob import glob
    from PIL.Image import open as imgOpen
    from os.path import splitext
    from os import system
    FILEs=args.files
    #FILEs=glob('*.jpg')
    #convert -background lightblue -fill blue  -font Candice -size 165x70 -pointsize 24 -gravity south label:Anthony b.gif
    #convert -pointsize 40 -fill blue -draw 'text 600,600 "Love You Mom"' a.jpg b.jpg
    #magick jpg1.jpg -gravity southwest -background white -splice 0x5% -background none -bordercolor none -size %[fx:w*0.25]x ( label:"TEXT HERE" -border 5% ) -composite out.jpg
    #magick jpg1.jpg -size %[w]x%[fx:h*0.05] label:"TEXT HERE" -append out2.jpg
    #The OP also asked about -pointsize N. The units of pointsize is points, where 1 point is 1/72 of an inch. So if you need a pointsize of, say, 20 pixels then you need to know the image density in dpi (dots per inch). Suppose the image dpi is 144, then 20 pixels is 20*72/144 = 10 points, so you need -pointsize 10
    for fname in FILEs:
      #if fname.find('New')!=-1:continue
      base, ext=splitext(fname)
      im = imgOpen(fname)
      x, y=im.size
      ratio=50 if x>4000 else 100
      pntSz=100 if x>4000 else 60
      annttSz=int(x/ratio)
      base, ext=splitext(fname)
      cmd=f'''convert -pointsize {pntSz} -font 華康魏碑體 -gravity south -stroke 'blue' -strokewidth 4 -annotate +{annttSz}+{annttSz} "{base}" -stroke none  -fill yellow  -annotate +{annttSz}+{annttSz} "{base}" {fname}  {base}New{ext}'''
      system(cmd)

def mkPicVid(args, FILEs):
    loopArg, imgARG, vidArg, cnctArg='', '', '', ''
    cnctName, cnvntnName, trnsDur, vidDur, offset='', 'trns', 1, args.dur, 0
    fLen=len(FILEs)
    xFilter=rndmChoices(FILTERs, k=fLen)
    if FILEs:
        for ndx, file in enumerate(FILEs):
          loopArg+=f'-loop 1 -t {vidDur} -i {file} '
          #imgARG+=f'[{ndx}]scale=-2:720,pad=1280:720:(ow-iw)/2:(oh-ih)/2,setsar=1[{ndx}newPic];'
          #imgARG+=f'[{ndx}]scale=1080:-2,pad=width=max(iw\,ih*(16/9)):height=ow/(16/9):x=(ow-iw)/2:y=(oh-ih)/2,setsar=1[{ndx}newPic];'
          imgARG+=f'[{ndx}]scale=1024:720,setsar=1[{ndx}newPic];'   #,setsar=1 [scale=960:720:force_original_aspect_ratio=1
          offset+=vidDur-trnsDur
          xfd=xFilter[ndx]
          if not ndx:
            vidArg+=f'''[0newPic][1newPic]xfade=transition={xfd}:duration={trnsDur}:offset={offset}[1newVid]; '''
          elif ndx==fLen-1:
            extraARG=f'''-map "[{ndx}newVid]" -c:v libx264 -crf 17'''
          else:
            vidArg+=f"[{ndx}newVid][{ndx+1}newPic]xfade=transition={xfd}:duration={trnsDur}:offset={offset}[{ndx+1}newVid];"
        #if args.Album:
        #    dir=Album
        #    CMD=f"""ffmpeg {loopArg} -filter_complex "{imgARG} {vidArg}" {extraARG} -y {Dir}/output.mp4"""  #{argAVTB}
        #else:
        CMD=f"""ffmpeg {loopArg} -filter_complex "{imgARG} {vidArg}" {extraARG} -y """  #output.mp4{argAVTB}
        return CMD
        #print(cmd)
        #cmd = "ffprobe -v quiet -print_format json -show_streams"
        #ffprobe -v error -show_entries stream=width,height -of default=noprint_wrappers=1 -of csv=p=0:s=x  fecd/output.mp4
        #args = shlexSplit(cmd)
        #args.append(pathToInputVideo)
        # run the ffprobe process, decode stdout into utf-8 & convert to JSON
        #ffprobeOutput = sbprcssChkoutput(args).decode('utf-8')
        #print(ffprobeOutput)
        #ffprobeOutput = json.loads(ffprobeOutput)
        #curDir=f'{dir}/{curdir}'
        #absPath=abspath(curdir)
        #print('abspath=', absPath)
        #chdir(absPath)
        #CMD=CMD.split

        #output=Popen(CMD, stdout=PIPE, stderr=PIPE)  #, shell=True)
        #stdout, stderr = output.communicate()
        #stdOut=output.stdout.read()#.replace('\n', chr(10))
        #stdErr=output.stderr.read()
        #print(stdOut.decode('utf-8'), stdErr.decode('utf-8'))#, end='\n'
def vidCnct(args):
    from os import system
    from os.path import splitext
    FILEs=['新民start.mp4', '三分餘地.mp4', '白水老人.mp4', '知命立命.mp4', '論語1黃講師.mp4', '論語2楊講師.mp4', '論語3.mp4', '班務辦道.mp4', '知心之旅.mp4', '好吃餐點.mp4'] #'新民班開班.mp4', #'新民開班.mp4
    FILEs=args.files
    tsFILEs=[]
    for fname in FILEs:
        base, ext=splitext(fname)
        print(f'ffmpeg -i {fname} -c copy -bsf:v h264_mp4toannexb -f mpegts -y {base}.ts')
        tsFILEs.append(f'{base}.ts')
    cmdConcat='concat:'+'|'.join(tsFILEs)
    concatFFMPG=f'ffmpeg -i "{cmdConcat}" -c copy -y stage1.mp4'
    print(concatFFMPG)
    #ffmpeg -i "concat:三分餘地.ts|新民開班.ts|白水老人.ts|知心之旅.ts|論語2楊講師.ts|新民班開班.ts|班務辦道.ts|知命立命.ts|論語1黃講師.ts|論語3.ts" -c copy 新民班回顧全.mp4
    print('ffmpeg -i stage1.mp4 -c copy -an -y noAudio.mp4')
    print('ffmpeg -i noAudio.mp4 -f concat -i allMP3 -c:v copy -y -shortest stage2.mp4')
    print('ffmpeg -i stage2.mp4 -vf ass=~/新民班.ass -y finalASS.mp4')

def cnctVid(args, VIDs):
    cnctArg, cntDir, outVid, filterArg='', 0, 'ffmpeg ', ''
    finalNoaudio, bkgrndMusic, finalMerge='vidNoaudio.mp4', 'musicBckgrnd.mp3', 'finalMerge.mp4'
    #VIDs=args.files
    #ffmpeg -i jpn_op.m4a -i eng_nop.m4a -filter_complex "[0:a][1:a]concat=n=2:v=0:a=1" final.mp4
    for vid in VIDs:
        #cnctArg+=f"[{cntDir}:v]"
        cnctArg+=f"[v{cntDir}]"
        #[0:v]scale=1024:576:force_original_aspect_ratio=decrease,pad=1024:576:-1:-1,setsar=1[v0]; [1:v]scale=1024:576:force_original_aspect_ratio=decrease,pad=1024:576:-1:-1,setsar=1[v1]; [2:v]scale=1024:576:force_original_aspect_ratio=decrease,pad=1024:576:-1:-1,setsar=1[v2]; [v0][0:a][v1][1:a][v2][2:a]concat=n=3:v=1:a=1[v][a]"
        #scale=1024:576:force_original_aspect_ratio=decrease,pad=1024:576:-1:-1,setsar=1
        outVid+=f"-i {vid} "
        filterArg+=f'[{cntDir}:v]scale=1024:576:force_original_aspect_ratio=decrease,pad=1024:576:-1:-1,setsar=1[v{cntDir}];'
        cntDir+=1   # [{cntDir}:v]scale=1024:576:force_original_aspect_ratio=decrease,pad=1024:576:-1:-1,setsar=1[v{cntDir}];[v{cntDir}]
    stageI=f'{outVid} -filter_complex "{filterArg}{cnctArg}concat=n={cntDir}:v=1:a=0[v]" -map "[v]" -y {finalNoaudio}'
    print(stageI)    #unsafe=1:
#找出vid的大小及framerate
#ffprobe -v error -show_entries stream=width,height -of default=noprint_wrappers=1 fecd/output.mp4
#第二步要加入背景音樂 
    stageII=f'ffmpeg -i {finalNoaudio} -stream_loop -1 -i ~/{bkgrndMusic} -c:v copy -shortest -map 0:v -map 1:a  -y {finalMerge}'
    print(stageII)
    return stageI, stageII

def dir2Vid(args, dir):  #args, dir
    FILEs=glob(f'{dir}/*.{args.pic}')
    CMD=mkPicVid(args, FILEs)
    return CMD

def album2Vid(args):
    DIRs=listdir('.')
    VIDs=[]
    for ndx, Dir in enumerate(DIRs):
        if isdir(Dir):
            #print(dir)
            vid=f'{Dir}/output.mp4'
            #outVid+=f"-i {Dir}/output.mp4 "
            VIDs.append(vid)
            if args.Merge: print('Merge')
            else:
              CMD=dir2Vid(args, Dir)
              CMD+=f'{Dir}/output.mp4'
              fout=open(f'{Dir}/out.sh', 'w')
              fout.write(CMD)
              print(f'sh {Dir}/out.sh')
    VIDs.append('~/end.mp4')
    stageI, stageII=cnctVid(args, VIDs)
    fout=open('end.sh', 'w')
    fout.write(stageI)
    fout.write(stageII)
    print(f'sh end.sh')
    #CMD+=stageI+'\n'
    #CMD+=stageII+'\n'
    #fout.write(CMD)
#ffmpeg -i output1.mp4 -i output2.mp4 -i ~/Downloads/影像素材/開頭影片的背景特效.mp4  -filter_complex "[0:v][1:v][2:v]concat=n=3:v=1:a=0[v]" -map "[v]" -y mergeAll.mp4
#第二步要加入end.mp4 ffmpeg -i input1.mp4 -i input2.wmv -filter_complex "[0:0][0:1][1:0][1:1]concat=n=2:v=1:a=1[outv][outa]" -map "[outv]" -map "[outa]" output.mp4
#第二步要加入背景音樂 ffmpeg -i mergeAll.mp4 -stream_loop -1 -i musicBckgrnd.mp3 -c:v copy  -shortest -map 0:v -map 1:a  -y mergeFull6.mp4

if __name__=='__main__':
    parser = ArgumentParser(description='calculate stock to the total of SKY')
    #parser.add_argument('--Upload', '-U', action='store_true', default=False, help='uploadFiles')
    parser.add_argument('--Pic', '-P', action='store_true', default=False, help='execCmd')
    parser.add_argument('--Album', '-A', action='store_true', default=False, help='execCmd')
    parser.add_argument('--Merge', '-M', action='store_true', default=False, help='execCmd')
    parser.add_argument('--Vid', '-V', action='store_true', default=False, help='execCmd')
    parser.add_argument('--vext', '-v', type=str, default='mp4', help='nodes')
    parser.add_argument('--dur', '-d', type=int, default=5, help='duration')
    parser.add_argument('--pic', '-p', type=str, default='jpg', help='nodes')
    parser.add_argument('--xclsv', '-x', default=['output.mp4'], nargs='*', help='files') #type=str,
    parser.add_argument('--files', '-f', default=['output.mp4'], nargs='*', help='files') #type=str,
    parser.add_argument('--Anntt', '-N', action='store_true', default=False, help='execCmd')
    args = parser.parse_args()
    if args.Pic:
        FILEs=glob(f'*.{args.pic}')
        FILEs.sort()
        FILEs.reverse()
        #print('FILEs=', FILEs)
        cnctArg, filterArg='', ''
        finalNoaudio, bkgrndMusic, finalMerge='vidNoaudio.mp4', 'musicBckgrnd.mp3', 'finalMerge.mp4'
        CMD=mkPicVid(args, FILEs)
        #print(CMD)
        CMD+='output.mp4'
        outVid='out.sh'
        fout=open(outVid, 'w')
        #print(CMD)
        VIDs=['output.mp4', '~/道場end.mp4']
        stageI, stageII=cnctVid(args, VIDs)
        fout.write(f'{CMD}\n')
        print(f'sh {outVid}')
        fout=open('end.sh', 'w')
        fout.write(stageI+'\n')
        fout.write(stageII+'\n')
        print(f'sh end.sh')
        #print(f'{outVid} -i ~/道場end.mp4 -filter_complex "{filterArg} [{cntDir}:v]scale=1024:576:force_original_aspect_ratio=decrease,pad=1024:576:-1:-1,setsar=1[v{cntDir}];{cnctArg}[v{cntDir}]concat=n={cntDir+1}:v=1:a=0[v]" -map "[v]" -y {finalNoaudio}')    #unsafe=1:
#找出vid的大小及framerate
#ffprobe -v error -show_entries stream=width,height -of default=noprint_wrappers=1 fecd/output.mp4
#第二步要加入背景音樂 
#'''
#ffmpeg -i /home/josh/Downloads/NerdFonts/vidNoaudio.mp4 -i 發一崇德之歌.mp3 -i "yt1s.com - 大開普渡調 寄一暝三冬黃芷芳.mp3" -i "yt1s.com - 感恩道場.mp3" -i "yt1s.com - 善歌丨修辦百年丨白陽小徒兒.mp3" -i "yt1s.com - 綻放光芒VCDmpg.mp3"  -filter_complex "[0:v]scale=1024:576:force_original_aspect_ratio=decrease,pad=1024:576:-1:-1,setsar=1[v0];[1:a]scale=1024:576:force_original_aspect_ratio=decrease,pad=1024:576:-1:-1,setsar=1[a1];[2:a]scale=1024:576:force_original_aspect_ratio=decrease,pad=1024:576:-1:-1,setsar=1[a2];[3:a]scale=1024:576:force_original_aspect_ratio=decrease,pad=1024:576:-1:-1,setsar=1[a3];[4:a]scale=1024:576:force_original_aspect_ratio=decrease,pad=1024:576:-1:-1,setsar=1[a4];[5:a]scale=1024:576:force_original_aspect_ratio=decrease,pad=1024:576:-1:-1,setsar=1[a5];[v0][a1][a2][a3][a4][a5]concat=n=6:v=1:a=0[v]" -map "[v]" -y vidNoaudio.mp4
#'''
        #print(f'ffmpeg -i {finalNoaudio} -stream_loop -1 -i ~/{bkgrndMusic} -c:v copy -shortest -map 0:v -map 1:a  -y {finalMerge}')
    if args.Album: album2Vid(args)
    elif args.Vid:
      cnctVid(args, args.files)#vid2Vid(args)
    elif args.Merge: album2Vid(args)
    elif args.Anntt: annttPic(args)
