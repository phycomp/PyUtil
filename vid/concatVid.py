#!/usr/bin/env python
from moviepy.editor import VideoFileClip, concatenate_videoclips as cnctVid

FILEs='output.mp4 dbde814d-edec-43f4-8ca0-2e4d4d358518.mp4 2d2f6915-3112-4395-af73-f69e12dab2dd.mp4 d990ef9a-f039-4aab-96dc-a40f5bb597d1.mp4 dce8f416-2087-4c1a-b695-6998d6d90edb.mp4 95ff9812-36f7-4eae-bc43-5bf665e446bc.mp4 8a66c13b-792a-444d-85a3-f30150e91604.mp4 /home/josh/end.mp4'
FILEs=FILEs.split()
print(FILEs)
out=[]
for ndx, fname in enumerate(FILEs):
    clip = VideoFileClip(fname) #"myvideo.mp4"
    out.append(clip)
    #clip2 = VideoFileClip("myvideo2.mp4").subclip(50,60)
    #clip3 = VideoFileClip("myvideo3.mp4")
    #if not ndx:
    #    finalClip = clip    #cnctVid([finalClip, ])  #concatenate_videoclips
    #else: 
    #    finalClip = cnctVid([finalClip, clip])  #concatenate_videoclips
finalClip=cnctVid(out)
finalClip.write_videofile("finalOutput.mp4")
****************************  concat  *****************************
#!/bin/bash
# Example of concatenating 3 mp4s together with 1-second transitions between them.

./ffmpeg \
  -i media/0.mp4 \
  -i media/1.mp4 \
  -i media/2.mp4 \
  -filter_complex " \
    [0:v]split[v000][v010]; \
    [1:v]split[v100][v110]; \
    [2:v]split[v200][v210]; \
    [v000]trim=0:3[v001]; \
    [v010]trim=3:4[v011t]; \
    [v011t]setpts=PTS-STARTPTS[v011]; \
    [v100]trim=0:3[v101]; \
    [v110]trim=3:4[v111t]; \
    [v111t]setpts=PTS-STARTPTS[v111]; \
    [v200]trim=0:3[v201]; \
    [v210]trim=3:4[v211t]; \
    [v211t]setpts=PTS-STARTPTS[v211]; \
    [v011][v101]gltransition=duration=1:source=./crosswarp.glsl[vt0]; \
    [v111][v201]gltransition=duration=1[vt1]; \
    [v001][vt0][vt1][v211]concat=n=4[outv]" \
  -map "[outv]" \
  -c:v libx264 -profile:v baseline -preset slow -movflags faststart -pix_fmt yuv420p \
  -y out.mp4
