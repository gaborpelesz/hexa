import base64
import io
import cv2
from imageio import imread
import sys
from image_resize import image_resize

filename = sys.argv[1]

cv2_img = cv2.imread(filename, 1)
if cv2_img.shape[1] > 350:
    cv2_img = image_resize(cv2_img, 350)

cv2.imwrite(filename, cv2_img)
filename = sys.argv[1]
with open(filename, "rb") as fid:
    data = fid.read()

with open("./output.txt", "w") as out:
    b64_bytes = base64.b64encode(data)
    b64_string = b64_bytes.decode()
    out.write(b64_string)