#!/usr/bin/env python
from os import system
from os.path import splitext
FILEs=['新民start.mp4', '三分餘地.mp4', '白水老人.mp4', '知命立命.mp4', '論語1黃講師.mp4', '論語2楊講師.mp4', '論語3.mp4', '班務辦道.mp4', '知心之旅.mp4', '好吃餐點.mp4']
#'新民班開班.mp4',
#'新民開班.mp4
tsFILEs=[]

for fname in FILEs:
    base, ext=splitext(fname)
    print(f'ffmpeg -i {fname} -c copy -bsf:v h264_mp4toannexb -f mpegts -y {base}.ts')
    tsFILEs.append(f'{base}.ts')
cmdConcat='concat:'+'|'.join(tsFILEs)
concatFFMPG=f'ffmpeg -i "{cmdConcat}" -c copy -y stage1.mp4'
print(concatFFMPG)
#system(concatFFMPG)
#ffmpeg -i "concat:三分餘地.ts|新民開班.ts|白水老人.ts|知心之旅.ts|論語2楊講師.ts|新民班開班.ts|班務辦道.ts|知命立命.ts|論語1黃講師.ts|論語3.ts" -c copy 新民班回顧全.mp4
print('ffmpeg -i stage1.mp4 -c copy -an -y noAudio.mp4')
print('ffmpeg -i noAudio.mp4 -f concat -i allMP3 -c:v copy -y -shortest stage2.mp4')
print('ffmpeg -i stage2.mp4 -vf ass=~/新民班.ass -y finalASS.mp4')
