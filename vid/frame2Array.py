from numpy import empty as npEmpty
frames = npEmpty(shape=(160 * 120, 152, 360, 3), dtype=np.float32)
from os import listdir
from cv2 import VideoCapture

for file in listdir(directory):
    if file.endswith(".flv"):
        file_path = os.path.join(directory, file)
        nr_file = nr_file + 1
        print('File '+str(nr_file)+' of '+str(nb_files_in_dir)+' files: '+file_path)

        # Create a VideoCapture object and read from input file
        # If the input is the camera, pass 0 instead of the video file name
        cap = VideoCapture(file_path)

        # Check if camera opened successfully
        if cap.isOpened() == False:
            print("Error opening video stream or file")
        ret=True
        while ret:   #cap.isOpened() Read until video is completed
            ret, frame = cap.read() # Capture frame-by-frame
            if ret:
                # frames.append(frame.astype('float32')/255.)
                frames[nr_frame, :, :, :]=frame.astype('float32')/255.
                nr_frame += 1
                nb_frames_in_file += 1

'''
A regular image is represented as a 3D Tensor with the following shape: (height, width, channels). The channels value 3 if the image is RGB and 1 if it is grayscale.

A video is a collection of N frames, where each frame is an image. You'd want to represent this data as a 4D tensor: (frames, height, width, channels).

So for example if you have 1 minute of video with 30 fps, where each frame is RGB and has a resolution of 256x256, then your tensor would look like this: (1800, 256, 256, 3), where 1800 is the number of frames in the video: 30 (fps) * 60 (seconds).

To achieve this you can basically open each individual frame of the video, store them all in a list and concatenate them together along a new axis (i.e. the "frames" dimension).
Numpy Array to Video
In theory, the examples I've seen for using cv2.VideoWriter go something like
'''

# let `video` be an array with dimensionality (T, H, W, C)

from cv2 import VideoWriter, VideoWriter_fourcc, VideoCapture
from numpy import array as npArray, split as npSplit

video=VideoCapture(path)
num_frames, height, width, _ = video.shape

filename = "/path/where/video/will/be/saved.mp4"
codec_id = "mp4v" # ID for a video codec.
fourcc = VideoWriter_fourcc(*codec_id)
out = VideoWriter(filename, fourcc=fourcc, fps=20, frameSize=(width, height))

for frame in npSplit(video, num_frames, axis=0):
    out.write(frame)

vid = VideoCapture('path/to/video/file')

frames, check, ndx = [], True, 0

while check:
    check, arr = vid.read()
    if not ndx % 20:  # This line is if you want to subsample your video (i.e. keep one frame every 20)
        frames.append(arr)
    ndx += 1

frames = npArray(frames)  # convert list of frames to numpy array


'''
A regular image is represented as a 3D Tensor with the following shape: (height, width, channels). The channels value 3 if the image is RGB and 1 if it is grayscale.

A video is a collection of N frames, where each frame is an image. You'd want to represent this data as a 4D tensor: (frames, height, width, channels).

So for example if you have 1 minute of video with 30 fps, where each frame is RGB and has a resolution of 256x256, then your tensor would look like this: (1800, 256, 256, 3), where 1800 is the number of frames in the video: 30 (fps) * 60 (seconds).

To achieve this you can basically open each individual frame of the video, store them all in a list and concatenate them together along a new axis (i.e. the "frames" dimension).
Numpy Array to Video
In theory, the examples I've seen for using cv2.VideoWriter go something like
'''

# let `video` be an array with dimensionality (T, H, W, C)
num_frames, height, width, _ = video.shape

filename = "/path/where/video/will/be/saved.mp4"
codec_id = "mp4v" # ID for a video codec.
fourcc = cv2.VideoWriter_fourcc(*codec_id)
out = cv2.VideoWriter(filename, fourcc=fourcc, fps=20, frameSize=(width, height))

for frame in np.split(video, num_frames, axis=0):
    out.write(frame)

from cv2 import VideoCapture
vid = VideoCapture('path/to/video/file')

frames = []
check = True
i = 0

while check:
    check, arr = vid.read()
    if not i % 20:  # This line is if you want to subsample your video (i.e. keep one frame every 20)
        frames.append(arr)
    i += 1

frames = np.array(frames)  # convert list of frames to numpy array
