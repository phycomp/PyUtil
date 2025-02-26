 --------------------  Encoded      ---------  Decoded      ------------
| Input WebM encoded | data        | ffmpeg  | raw frames  | reshape to |
| stream (VP9 codec) | ----------> | process | ----------> | NumPy array|
 --------------------  stdin PIPE   ---------  stdout PIPE  -------------

import numpy as np
from cv2 import destroyAllWindows, waitKey, imshow
import io
import subprocess as sp
import threading
import json
from functools import partial
import shlex

# Build synthetic video and read binary data into memory (for testing):
#########################################################################
width, height = 640, 480
sp.run(shlex.split(f'ffmpeg -y -f lavfi -i testsrc=size=width, height{width}x{height}:rate=1 -vcodec vp9 -crf 23 -t 50 test.webm'))   #.format()

with open('test.webm', 'rb') as binary_file:
    in_bytes = binary_file.read()
#########################################################################


# https://stackoverflow.com/questions/5911362/pipe-large-amount-of-data-to-stdin-while-using-subprocess-popen/14026178
# https://stackoverflow.com/questions/15599639/what-is-the-perfect-counterpart-in-python-for-while-not-eof
# Write to stdin in chunks of 1024 bytes.
def writer():
    for chunk in iter(partial(stream.read, 1024), b''):
        process.stdin.write(chunk)
    try:
        process.stdin.close()
    except (BrokenPipeError):
        pass  # For unknown reason there is a Broken Pipe Error when executing FFprobe.


# Get resolution of video frames using FFprobe
# (in case resolution is know, skip this part):
################################################################################
# Open In-memory binary streams
from io import BytesIO
stream = BytesIO(in_bytes)

process = sp.Popen(shlex.split('ffprobe -v error -i pipe: -select_streams v -print_format json -show_streams'), stdin=sp.PIPE, stdout=sp.PIPE, bufsize=10**8)
pthread = threading.Thread(target=writer)
pthread.start()
pthread.join()

in_bytes = process.stdout.read()

process.wait()

p = json.loads(in_bytes)

width = (p['streams'][0])['width']
height = (p['streams'][0])['height']
################################################################################


# Decoding the video using FFmpeg:
################################################################################
stream.seek(0)

# FFmpeg input PIPE: WebM encoded data as stream of bytes.
# FFmpeg output PIPE: decoded video frames in BGR format.
process = sp.Popen(shlex.split('ffmpeg -i pipe: -f rawvideo -pix_fmt bgr24 -an -sn pipe:'), stdin=sp.PIPE, stdout=sp.PIPE, bufsize=10**8)

thread = threading.Thread(target=writer)
thread.start()

from numpy import frombuffer as npFrombffr, uint8 as npUint8
# Read decoded video (frame by frame), and display each frame (using imshow)
while True:
    # Read raw video frame from stdout as bytes array.
    in_bytes = process.stdout.read(width * height * 3)

    if not in_bytes:
        break  # Break loop if no more bytes.

    # Transform the byte read into a NumPy array
    in_frame = (npFrmbffr(in_bytes, npUint8).reshape([height, width, 3]))

    # Display the frame (for testing)
    imshow('in_frame', in_frame)

    if waitKey(100) & 0xFF == ord('q'):
        break

if not in_bytes:
    # Wait for thread to end only if not exit loop by pressing 'q'
    thread.join()

try: process.wait(1)
except (sp.TimeoutExpired):
    process.kill()  # In case 'q' is pressed.

destroyAllWindows()
