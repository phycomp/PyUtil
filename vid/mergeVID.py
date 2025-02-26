#!/usr/bin/env python
from stUtil import rndrCode
from sys import argv
from os.path import splitext
from os import system
VIDs, FILEs=[], argv[1:]
for vid in FILEs:
    base, ext=splitext(vid)
    VID=f'{base}.ts'
    VIDs.append(VID)
    cmd=f'ffmpeg -i "{vid}" -c copy -bsf:v h264_mp4toannexb -f mpegts -y "{VID}"' #-c copy -vcodec copy -acodec copy -bsf:v h264_mp4toannexb -c copy -bsf:v h264_mp4toannexb -f mpegts
    
    rndrCode(cmd)
    system(cmd)
#ffmpeg -i original.ts -i converted.mp4 -c:v copy -c:a copy -map 1:v -map 0:a:0 -map 0:a:1 output.mp4
#cnctMKV='ffmpeg -f concat -safe 0 -i allMKV -c copy output.mkv' #需要先準備檔列表
#print([f"{vid}" for vid in VIDs])
***********************  merge *************************
ffmpeg -i ~/神奇的天路.mkv -c copy -bsf:v h264_mp4toannexb -an -f mpegts -y tmp2.ts
ffmpeg -ss 00:00:00.33 -i tmp2.ts -f concat -i aa -c:v copy -y -shortest merge.mp4
需要準備mp3的文字檔


#allMergeTS='|'.join([vid for vid in VIDs]) #'"{}"' 
#mergeTS, allMerge='allMerge.ts', 'allMerge.mp4'
#cmd=f'cat {allMergeTS}> {mergeTS}'
#print(cmd)
#system(cmd)
#cmd=f'ffmpeg -y -f concat -safe 0 -i "concat:{allMergeTS}" {allMerge}'    #-acodec copy -ar 44100 -ab 96k -coder ac -vbsf h264_mp4toannexb 
#cmd=f'ffmpeg -copyts -i "concat:{allMergeTS}" -muxpreload 0 -muxdelay 0 -c copy {allMerge}'
#需要先準備檔案列表 mergeLIST.txt 才可以合併所有的檔案成為 最終的合併檔allMerge.ts
cmd='ffmpeg -safe 0 -f concat -i list.txt -c copy allMerge.ts'
rndrCode(cmd)
system(cmd)
