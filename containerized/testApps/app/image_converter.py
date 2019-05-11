import base64
import io
import cv2
from imageio import imread

def convert_base2image(b64_string):

    # reconstruct image as an numpy array
    img = imread(io.BytesIO(base64.b64decode(b64_string)))

    # finally convert RGB image to BGR for opencv
    # and save result
    cv2_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite("images/reconstructed.jpg", cv2_img)