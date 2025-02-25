def pic2Vid():
  from moviepy.editor import ImageSequenceClip
  from PIL import Image

  thumbnail_dir = os.path.join(SAMPLE_OUTPUTS, "thumbnails")
  output_video = os.path.join(SAMPLE_OUTPUTS, 'final_pic_to_vid.mp4')

  this_dir = os.listdir(thumbnail_dir)
  filepaths = [os.path.join(thumbnail_dir, fname) for fname in this_dir if fname.endswith("jpg")]

  directory = {}
  for root, dirs, files in os.walk(thumbnail_dir):
      for fname in files:
          filepath = os.path.join(root, fname)
          try:
              key = float(fname.replace(".jpg", ""))
          except:
              key = None
          if key != None:
              directory[key] = filepath

  new_path = []
  for k in sorted(directory.keys()):
      filepath = directory[k]
      new_path.append(filepath)
    
  clip = ImageSequenceClip(new_path, fps=5)
  clip.write_videofile(output_video)


from moviepy.video.VideoClip import DataVideoClip

def data_to_frame(time):
    xn = x + np.sin(2 * np.pi * time / 10.0)
    yn = y + np.cos(2 * np.pi * time / 8.0)
    p[0].set_data(xn, yn)
    return mplfig_to_npimage(fig)

times = np.arange(2341973, 2342373)
clip = DataVideoClip(times, data_to_frame, fps=1) # one plot per second

#final animation is 15 fps, but still displays 1 plot per second
animation.write_videofile("test2.mp4", fps=15) 


import numpy as np
import matplotlib.pyplot as plt
from moviepy.video.io.bindings import mplfig_to_npimage
import moviepy.editor as mpy

fig = plt.figure()
ax = plt.axes()
x = np.random.randn(10, 1)
y = np.random.randn(10, 1)
p = plt.plot(x, y, 'ko')
time = np.arange(2341973, 2342373)
cache = {}

def animate(t):
    i = int(t)
    if i in cache:
        return cache[i]

    xn = x + np.sin(2 * np.pi * time[i] / 10.0)
    yn = y + np.cos(2 * np.pi * time[i] / 8.0)
    p[0].set_data(xn, yn)
    cache.clear()
    cache[i] = mplfig_to_npimage(fig)
    return cache[i]

duration = len(time)
fps = 15
animation = mpy.VideoClip(animate, duration=duration)
animation.write_videofile("test.mp4", fps=fps)

#**********************************************************
import os
from moviepy.editor import AudioFileClip, VideoFileClip
def mnplMovie():
  org_video_path = input("Enter the video path: ")
  audio_path = input("Enter the audio path: ")
  final_video_path = input("Enter the output folder path: ")
  final_video_name = input("Enter the final video name: ")
  start_dur = int(input("Enter the starting duration in seconds: "))
  end_dur = int(input("Enter the ending duration in seconds: "))

  final_video_path = os.path.join(final_video_path, final_video_name)

  video_clip = VideoFileClip(org_video_path)

  background_audio_clip = AudioFileClip(audio_path)
  bg_music = background_audio_clip.subclip(start_dur, end_dur)

  final_clip = video_clip.set_audio(bg_music)
  final_clip.write_videofile(final_video_path, codec='libx264', audio_codec="aac")

from stUtil import rndrCode

def mkEnd(oriClip):
  from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip, VideoClip
  from moviepy.video.tools.drawing import circle as mpyCircle

  from os.path import splitext
  #from moviepy.Clip import Clip
  #from moviepy.video.VideoClip import DataVideoClip
  #moviepy.video.VideoClip.VideoClip
  #clip = VideoClip() #make_frame=None, audio=False).subclip(25,31).add_mask()
  #clip=DataVideoClip(oriClip, 24)
  #stCode([  clip.__dict__ ])
  base, ext=splitext(oriClip)
  clip = VideoFileClip(oriClip, audio=False).subclip(26,31).add_mask()
  w,h = clip.size
  stCode(['w, h=', w, h])
  clip.mask.get_frame = lambda t: mpyCircle(screensize=(clip.w,clip.h), center=(clip.w/2,clip.h/4), radius=max(0,int(800-200*t)), col1=1, col2=0, blur=4) # The mask is a circle with vanishing radius r(t) = 800-200*t
  theEnd=TextClip("The End", font="Amiri-bold", color="white", fontsize=70).set_duration(clip.duration)
  final = CompositeVideoClip([theEnd.set_pos('center'),clip], size =clip.size)
  outFname=f"{base}End{ext}"
  stCode(['outFname', outFname])
  final.write_videofile(outFname)
  return final
  
from moviepy.editor import VideoFileClip, CompositeVideoClip
from moviepy.video.tools.segmenting import findObjects

# Load the image specifying the regions.
im = ImageClip("../../ultracompositing/motif.png")

# Loacate the regions, return a list of ImageClips
regions = findObjects(im)

allClip=[ "../../videos/romo_0004.mov", "../../videos/apis-0001.mov", "../../videos/romo_0001.mov", "../../videos/elma_s0003.mov", "../../videos/elma_s0002.mov", "../../videos/calo-0007.mov", "../../videos/grsm_0005.mov"]
# Load 7 clips from the US National Parks. Public Domain :D
clips = [VideoFileClip(n, audio=False).subclip(18,22) for n in allClip ]

# fit each clip into its region
comp_clips =  [c.resize(r.size).set_mask(r.mask).set_pos(r.screenpos) for c,r in zip(clips,regions)]

cc = CompositeVideoClip(comp_clips,im.size)
cc.resize(.6).write_videofile("../../composition.mp4")

# Note that this particular composition takes quite a long time of
# rendering (about 20s on my computer for just 4s of video).
