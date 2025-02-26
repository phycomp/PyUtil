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
