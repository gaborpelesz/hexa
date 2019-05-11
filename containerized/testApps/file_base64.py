import base64
import io
import cv2
from imageio import imread
import sys

filename = sys.argv[1]
with open(filename, "rb") as fid:
    data = fid.read()

with open("./output.txt", "w") as out:
    b64_bytes = base64.b64encode(data)
    b64_string = b64_bytes.decode()
    out.write(b64_string)