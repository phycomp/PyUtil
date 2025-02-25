clip = VideoClip(mkFrame, duration=4) # for custom animations (see below)
clip = VideoFileClip("my_video_file.mp4") # or .avi, .webm, .gif ...
clip = ImageSequenceClip(['image_file1.jpeg', ...], fps=24)
clip = ImageClip("my_picture.png") # or .jpeg, .tiff, ...
clip = TextClip("Hello !", font="Amiri-Bold", fontsize=70, color="black")
clip = ColorClip(size=(460,380), color=[R,G,B])

# AUDIO CLIPS
clip = AudioFileClip("my_audiofile.mp3") # or .ogg, .wav... or a video !
clip = AudioArrayClip(numpy_array, fps=44100) # from a numerical array
clip = AudioClip(mkFrame, duration=3) # uses a function mkFrame(t)

import gizeh
from moviepy.editor import VideoFileClip, VideoClip

def mkFrame(t):
    surface = gizeh.Surface(128,128) # width, height
    radius = W*(1+ (t*(2-t))**2 )/6 # the radius varies over time
    circle = gizeh.circle(radius, xy = (64,64), fill=(1,0,0))
    circle.draw(surface)
    return surface.get_npimage() # returns a 8-bit RGB array

clip = VideoClip(mkFrame, duration=2) # 2 seconds
clip.write_gif("circle.gif",fps=15)
