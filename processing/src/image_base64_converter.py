import base64
import io
import cv2
from imageio import imread

def convert_base2image(b64_string):
    """Converting base64 string to cv2 image.

        Args:
            b64_string: Representing the base64 encoded image.
        
        Return:
            A cv2 image, that we decoded from the base64 string,
            stored in a numpy array.
    """
    # reconstruct image as an numpy array
    img = imread(io.BytesIO(base64.b64decode(b64_string)))

    # finally convert RGB image to BGR for opencv
    # and save result
    cv2_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return cv2_img