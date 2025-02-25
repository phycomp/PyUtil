import numpy as np
from cv2 import VideoWriter
size = 720*16//9, 720
duration = 2
fps = 2
out = VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (size[1], size[0]), False)
for i in range(aj.shape[2]):
    data = aj[:,:,i,:]
    out.write(data)
out.release()

python-3.x numpyopencv

Your video dimension is (13, 222, 356, 3)? I assume that the shape of the NumPy array that contains your video frames is (13, 222, 356, 3). According to the common convention for storing video frames in NumPy array: The size is not 720*16//9, 720, but 356, 222. The last argument of cv2.VideoWriter applies isColor=false, but your video has 3 color channels, so don't set it to false - keep the default value - true. i in range(aj.shape[2]) is wrong - use i in range(aj.shape[0]) for iterating your 13 frames.

data = aj[:,:,i,:] should be data = aj[i,:,:,:].
Here is a code sample (the sample builds synthetic frames for testing):

from numpy import zeros as npZeros
from cv2 import putText, FONT_HERSHEY_DUPLEX

# Create aj for testing, dimension (13, 222, 356, 3)
aj = npZeros((13, 222, 356, 3), np.uint8)
for ndx in range(aj.shape[0]):
    putText(aj[ndx, :, :], str(ndx+1), (aj.shape[2]//2-50*len(str(ndx+1)), aj.shape[1]//2+50), FONT_HERSHEY_DUPLEX, 5, (255, 30, 30), 10)  # Blue number

# size = 720 * 16 // 9, 720
# duration = 2

from cv2 import VideoWriter, VideoWriter_fourcc, cvtColor, COLOR_RGB2BGR
fps = 2
out = VideoWriter('output.mp4', VideoWriter_fourcc(*'mp4v'), fps, (aj.shape[2], aj.shape[1]))
for i in range(aj.shape[0]):
    data = aj[i, :, :, :]
    out.write(data)
out.release()

#Color channels ordering of OpenCV is BGR (blue is first). Many Python packages uses RGB convention (red is first). In case you see that red and blue are swapped, use RGB to BGR conversion:

data = cvtColor(data, COLOR_RGB2BGR)
