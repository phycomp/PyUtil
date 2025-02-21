#!/usr/bin/env python3
from sys import argv
from os import system
from os.path import splitext

for file in argv[1:]:
    base, ext=splitext(file)
    base=base.replace(' ', '')
    #cmd='ffmpeg -i "%s" -codec:a libmp3lame -qscale:a 2 %s.mp3'%(file, base)
    cmd='ffmpeg -i "%s" -acodec libmp3lame %s.mp3'%(file, base)
    print(cmd)
    system(cmd)

ffmpeg -i [input_file] -vcodec copy -an [output_file]

ffmpeg -ss 00:00:10  -t 5 -i "video.mp4" -ss 0:00:01 -t 5 -i "music.m4a" -map 0:v:0 -map 1:a:0 -y out.mp4

