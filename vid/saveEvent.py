from pyimagesearch.keyclipwriter import KeyClipWriter
from stUtil import rndrCode
from imutils.video import VideoStream
from argparse import ArgumentParser
import datetime
import imutils
import time
from cv2 import GaussianBlur, cvtColor, inRange, erode, dilate, findContours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, contourArea, minEnclosingCircle, circle, VideoWriter_fourcc, imshow, waitKey, destroyAllWindows

# construct the argument parse and parse the arguments
ap = ArgumentParser()
ap.add_argument("-o", "--output", required=True, help="path to output directory")
ap.add_argument("-p", "--picamera", type=int, default=-1, help="whether or not the Raspberry Pi camera should be used")
ap.add_argument("-f", "--fps", type=int, default=20, help="FPS of output video")
ap.add_argument("-c", "--codec", type=str, default="MJPG", help="codec of output video")
ap.add_argument("-b", "--buffer-size", type=int, default=32, help="buffer size of video clip writer")
args = vars(ap.parse_args())
# args={"output":"output","picamera":-1,"fps":20,"codec":"MJPG","buffer_size":32}

# initialize the video stream and allow the camera sensor to
# warmup
rndrCode("[INFO] warming up camera...")
vs = VideoStream(usePiCamera=args["picamera"] > 0).start()
time.sleep(2.0)

# define the lower and upper boundaries of the "green" ball in
# the HSV color space
# greenLower = (29, 43, 46)
# greenUpper = (64, 255, 255)
orangeLower = (11, 43, 46)
orangeUpper = (25, 255, 255)

# initialize key clip writer and the consecutive number of
# frames that have *not* contained any action
kcw = KeyClipWriter(bufSize=args["buffer_size"])
consecFrames = 0

# keep looping
while True:
  # grab the current frame, resize it, and initialize a
  # boolean used to indicate if the consecutive frames
  # counter should be updated
  frame = vs.read()
  frame = imutils.resize(frame, width=600)
  updateConsecFrames = True

  # update the key frame clip buffer
  kcw.update(frame)

  # blur the frame and convert it to the HSV color space
  blurred = GaussianBlur(frame, (11, 11), 0)
  hsv = cvtColor(blurred, COLOR_BGR2HSV)
  # rndrCode(hsv.min(axis=(0,1)))
  # rndrCode(hsv.max(axis=(0,1)))

  # construct a mask for the color "green", then perform
  # a series of dilations and erosions to remove any small
  # blobs left in the mask
  mask = inRange(hsv, orangeLower, orangeUpper)
  mask = erode(mask, None, iterations=2)
  mask = dilate(mask, None, iterations=2)

  # find contours in the mask
  cnts = findContours(mask.copy(), RETR_EXTERNAL,
    CHAIN_APPROX_SIMPLE)
  cnts = imutils.grab_contours(cnts)

  # only proceed if at least one contour was found
  if len(cnts) > 0:
    # find the largest contour in the mask, then use it
    # to compute the minimum enclosing circle
    c = max(cnts, key=contourArea)
    ((x, y), radius) = minEnclosingCircle(c)
    updateConsecFrames = radius <= 10

    # only proceed if the redius meets a minimum size
    if radius > 10:
      # reset the number of consecutive frames with
      # *no* action to zero and draw the circle
      # surrounding the object
      consecFrames = 0
      circle(frame, (int(x), int(y)), int(radius),
        (0, 0, 255), 2)

      # if we are not already recording, start recording
      if not kcw.recording:
        timestamp = datetime.datetime.now()
        p = "{}/{}.avi".format(args["output"],
          timestamp.strftime("%Y%m%d-%H%M%S"))
        kcw.start(p, VideoWriter_fourcc(*args["codec"]),
          args["fps"])

  # otherwise, no action has taken place in this frame, so
  # increment the number of consecutive frames that contain
  # no action
  if updateConsecFrames:
    consecFrames += 1

  # if we are recording and reached a threshold on consecutive
  # number of frames with no action, stop recording the clip
  if kcw.recording and consecFrames == args["buffer_size"]:
    kcw.finish()

  # show the frame
  imshow("Frame", frame)
  key = waitKey(1) & 0xFF

  # if the `q` key was pressed, break from the loop
  if key == ord("q"):
    break

# if we are in the middle of recording a clip, wrap it up
if kcw.recording: kcw.finish()

# do a bit of cleanup
destroyAllWindows()
vs.stop()
