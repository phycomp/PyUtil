'''
Streamlit webcam stream processing using OpenCV, Python and Tensorflow
Streamlit Components
streamlit-webrtc
Hello! I have created a Python app that gets a webcam frame and then processes it with a ML classifier model, and detects Sign Language words, then sends the processed frame back. Since I’m new deploying and accessing user media devices, I used streamlit to make it easier. The problem is, that my predicts annotations are not reaching my localhost in streamlit, let me show you:
This is how it should work, the app is running with Flask on a Flask local deployment:
Working app using Flask on localhost
Working app using Flask on localhost
And this is how the app looks like, deployed on streamlit. I had to hardcode a string into cv2.putText() function in order to have something on screen.
Not working app using streamlit
Not working app using streamlit
So putText and the other cv2 functions are working, but I cant/dont know how to use logging in streamlit to see where the mistake is… I guess it has something to do with the ML frame processing part. Here it is the streamlit app.py code.
'''

from streamlit_webrtc import VideoProcessorBase, RTCConfiguration,WebRtcMode,webrtc_streamer
from utils import *
import cv2
import streamlit as st
import mediapipe as mp
import av
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)
def main():
    st.header("Live stream processing")
    sign_language_det = "Sign Language Live Detector"
    app_mode = st.sidebar.selectbox( "Choose the app mode",
        [
            sign_language_det
        ],
    )
    st.subheader(app_mode)
    if app_mode == sign_language_det:
        sign_language_detector()
def sign_language_detector():
    class OpenCVVideoProcessor(VideoProcessorBase):
        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")
            mp_holistic = mp.solutions.holistic # Holistic model
            mp_drawing = mp.solutions.drawing_utils # Drawing utilities
            # Actions that we try to detect
            actions = np.array(['hello', 'thanks', 'iloveyou'])
            # Load the model from Modelo folder:
            model = load_model('model.h5',actions)
            # 1. New detection variables
            sequence = []
            sentence = []
            threshold = 0.8
            # Set mediapipe model
            with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
                while True:
                    #img = frame.to_ndarray(format="bgr24")
                    flip_img = cv2.flip(img,1)
                    # Make detections
                    image, results = mediapipe_detection(flip_img, holistic)
                    # Draw landmarks
                    draw_styled_landmarks(image, results)
                    # 2. Prediction logic
                    keypoints = extract_keypoints(results)
                    sequence.append(keypoints)
                    sequence = sequence[-30:]
                    if len(sequence) == 30:
                        res = model.predict(np.expand_dims(sequence, axis=0))[0]
                        #print(actions[np.argmax(res)])
                    #3. Viz logic
                        if res[np.argmax(res)] > threshold:
                            if len(sentence) > 0:
                                if actions[np.argmax(res)] != sentence[-1]:
                                    sentence.append(actions[np.argmax(res)])
                            else:
                                sentence.append(actions[np.argmax(res)])
                        if len(sentence) > 5:
                            sentence = sentence[-5:]
                        # Viz probabilities
                        image = prob_viz(res, actions, image)
                    cv2.rectangle(image, (0,0), (640, 40), (234, 234, 77), 1)
                    cv2.putText(image, ' '.join(sentence), (3,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    return av.VideoFrame.from_ndarray(image,format="bgr24")
    webrtc_ctx = webrtc_streamer( key="opencv-filter", mode=WebRtcMode.SENDRECV, rtc_configuration=RTC_CONFIGURATION, video_processor_factory=OpenCVVideoProcessor, async_processing=True)
if __name__ == "__main__":
    main()
And here the working app.py with Flask:
from flask import Flask, render_template, Response
from imutils.video import VideoStream
from utils import *
import mediapipe as mp
import threading
import argparse
import time
import cv2
# initialize the output frame and a lock used to ensure thread-safe
# exchanges of the output frames (useful when multiple browsers/tabs
# are viewing the stream)
outputFrame = None
lock = threading.Lock()
app = Flask(__name__)
mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities
# initialize the video stream and allow the camera sensor to
# warmup
#vs = VideoStream(usePiCamera=1).start()
vs = VideoStream(src=0).start()
time.sleep(2.0)
@app.route('/')
def index():
    return render_template('index.html')
def classify():
        global vs, outputFrame, lock
        mp_holistic = mp.solutions.holistic # Holistic model
        mp_drawing = mp.solutions.drawing_utils # Drawing utilities
        # Actions that we try to detect
        actions = np.array(['hello', 'thanks', 'iloveyou'])
        # Load the model from Modelo folder:
        model = load_model('model.h5',actions)
        # 1. New detection variables
        sequence = []
        sentence = []
        threshold = 0.8
        # Set mediapipe model
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            while True:
                frame = vs.read()
                flip_img = cv2.flip(frame,1)
                # Make detections
                image, results = mediapipe_detection(flip_img, holistic)
                # Draw landmarks
                draw_styled_landmarks(image, results)
                # 2. Prediction logic
                keypoints = extract_keypoints(results)
                sequence.append(keypoints)
                sequence = sequence[-30:]
                if len(sequence) == 30:
                    res = model.predict(np.expand_dims(sequence, axis=0))[0]
                    #print(actions[np.argmax(res)])
                #3. Viz logic
                    if res[np.argmax(res)] > threshold:
                        if len(sentence) > 0:
                            if actions[np.argmax(res)] != sentence[-1]:
                                sentence.append(actions[np.argmax(res)])
                        else:
                            sentence.append(actions[np.argmax(res)])
                    if len(sentence) > 5:
                        sentence = sentence[-5:]
                    # Viz probabilities
                    image = prob_viz(res, actions, image)
                cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
                cv2.putText(image, ' '.join(sentence), (3,30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                #cv2.imshow('LIVE SIGN DETECTION', image)
                with lock:
                    outputFrame = image.copy()
def generate():
    # grab global references to the output frame and lock variables
    global outputFrame, lock
    # loop over frames from the output stream
    while True:
        # wait until the lock is acquired
        with lock:
            # check if the output frame is available, otherwise skip
            # the iteration of the loop
            if outputFrame is None:
                continue
            # encode the frame in JPEG format
            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)
            # ensure the frame was successfully encoded
            if not flag:
                continue
        # yield the output frame in the byte format
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
            bytearray(encodedImage) + b'\r\n')
@app.route("/video")
def video():
    # return the response generated along with the specific media
    # type (mime type)
    return Response(generate(),
        mimetype = "multipart/x-mixed-replace; boundary=frame")
if __name__ == '__main__':
    # construct the argument parser and parse command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--ip", type=str, required=True,
        help="ip address of the device")
    ap.add_argument("-o", "--port", type=int, required=True,
        help="ephemeral port number of the server (1024 to 65535)")
    args = vars(ap.parse_args())
    # start a thread that will perform motion detection
    t = threading.Thread(target=classify)
    t.daemon = True
    t.start()
    app.run(host=args["ip"],port=args["port"],debug=True,threaded=True,use_reloader=False)
    # release the video stream pointer
    vs.stop()
My custom utils module are just some functions to make the predictions and plot the detection landmarks on face and hands, as well as the ML model creation:
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import os
mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results
def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS) # Draw face connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) # Draw pose connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw right hand connections
def draw_styled_landmarks(image, results):
    # Draw face connections
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                             mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                             mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                             )
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             )
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             )
    # Draw right hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             )
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])
def prob_viz(res, actions, input_frame):
    colors = [(245,117,16), (117,245,16), (16,117,245)]
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
    return output_frame
def load_model(model_path,actions):
    model = Sequential()
    # Layers
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(actions.shape[0], activation='softmax'))
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    model.load_weights(model_path)
    return model
Sorry for the post length, but wanted to give full details of what I was trying to do…
