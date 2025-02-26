"""We may encode an "in memory" MP4 video using PyAV as described in my following answer - the video is stored in BytesIO object.
We may pass the BytesIO object as input to Streamlit (or convert the BytesIO object to bytes array and use the array as input)."""

import numpy as np
import cv2  # OpenCV is used only for writing text on image (for testing).
from av import open as avOpen, VideoFrame, VideoWriter
form io import BytesIO
import streamlit as st

n_frmaes = 100  # Select number of frames (for testing).

width, height, fps = 192, 108, 10  # Select video resolution and framerate.
output_memory_file = BytesIO()  # Create BytesIO "in memory file".

output = avOpen(output_memory_file, 'w', format="mp4")  # Open "in memory file" as MP4 video output
stream = output.add_stream('h264', str(fps))  # Add H.264 video stream to the MP4 container, with framerate = fps.
stream.width = width  # Set frame width
stream.height = height  # Set frame height
#stream.pix_fmt = 'yuv444p'   # Select yuv444p pixel format (better quality than default yuv420p).
stream.pix_fmt = 'yuv420p'   # Select yuv420p pixel format for wider compatibility.
stream.options = {'crf': '17'}  # Select low crf for high quality (the price is larger file size).

def mkSmplImg(i):
    """ Build synthetic "raw BGR" image for testing """
    p = width//60
    img = np.full((height, width, 3), 60, np.uint8)
    cv2.putText(img, str(i+1), (width//2-p*10*len(str(i+1)), height//2+p*10), cv2.FONT_HERSHEY_DUPLEX, p, (255, 30, 30), p*2)  # Blue number
    return img


# Iterate the created images, encode and write to MP4 memory file.
for i in range(n_frmaes):
    img = mkSmplImg(i)  # Create OpenCV image for testing (resolution 192x108, pixel format BGR).
    frame = VideoFrame.from_ndarray(img, format='bgr24')  # Convert image from NumPy Array to frame.
    packet = stream.encode(frame)  # Encode video frame
    output.mux(packet)  # "Mux" the encoded frame (add the encoded frame to MP4 file).

# Flush the encoder
packet = stream.encode(None)
output.mux(packet)
output.close()

output_memory_file.seek(0)  # Seek to the beginning of the BytesIO.
#video_bytes = output_memory_file.read()  # Convert BytesIO to bytes array
#st.video(video_bytes)
st.video(output_memory_file)  # Streamlit supports BytesIO object - we don't have to convert it to bytes array.

# Write BytesIO from RAM to file, for testing:
#with open("output.mp4", "wb") as f:
#    f.write(output_memory_file.getbuffer())
#video_file = open('output.mp4', 'rb')
#video_bytes = video_file.read()
#st.video(video_bytes)
"""We can't use cv.VideoWriter, because it does not support in-memory video encoding (cv.VideoWriter requires a "true file").

It wasn't a browser issue I don't think. Tried safari, chrome and firefox. But looks like yuv420p works thanks! I'd your code but says suggestion box is full. Can check answered then"""
