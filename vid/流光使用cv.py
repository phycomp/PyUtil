from cv2 import convertScaleAbs, GaussianBlur, detailEnhance, imread, cvtColor
import numpy as np
from PIL import Image
from streamlit import file_uploader as flUpldr, info as stInfo, sidebar, error as stError, write as stWrite, code as stCode, session_state, text_input, markdown as stMarkdown, slider

def brighten_image(image, amount):
    img_bright = convertScaleAbs(image, beta=amount)
    return img_bright

def blur_image(image, amount):
    img = cvtColor(image, 1)
    blur_img = GaussianBlur(img, (11, 11), amount)
    return blur_img

def enhance_details(img):
    hdr = detailEnhance(img, sigma_s=12, sigma_r=0.15)
    return hdr

from streamlit import title as stTitle, subheader as stSubheader, text as stTxt, sidebar, file_uploader as flUpldr, image as stImage
from numpy import array as npArray
from PIL.Image import open as imgOpen

MENUs=['imgMNPL', 'VIDs', 'vidProfile', 'mkEnd']  #, 'Annot', 'nerTagger', 'embedding', 'BILUO', 'viterbi', 'Metadata', '病歷文本', '影片profile', 
menu = sidebar.radio('Output', MENUs, index=0)
if menu==MENUs[0]:
    stTitle("OpenCV Demo App")
    stSubheader("This app allows you to play with Image filters!")
    stTxt("We use OpenCV and Streamlit for this demo")

    blur_rate = sidebar.slider("Blurring", min_value=0.5, max_value=3.5)
    brightness_amount = sidebar.slider("Brightness", min_value=-50, max_value=50, value=0)
    apply_enhancement_filter = sidebar.checkbox('Enhance Details')

    imgFname = flUpldr("Upload Your Image", type=['jpg', 'png', 'jpeg'])
    if imgFname:
        original_image = imgOpen(imgFname)
        original_image = npArray(original_image)

        processed_image = blur_image(original_image, blur_rate)
        processed_image = brighten_image(processed_image, brightness_amount)

        if apply_enhancement_filter:
            processed_image = enhance_details(processed_image)
        stTxt("Original Image vs Processed Image")
        stImage([original_image, processed_image])
    else: stCode(["None"])
elif menu==MENUs[1]:
    img = imread(filename='tony_stark.jpg')
# do some cool image processing stuff
    img = enhance_details(img)
    img = brighten_image(img, amount=25)
    img = blur_image(img, amount=0.2)
    imshow('Tony Stark', img)
    waitKey(0)
    destroyAllWindows()
