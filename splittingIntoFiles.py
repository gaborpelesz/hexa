import cv2
import numpy as np
from hexagonSeperator import HexagonSeperator
from image_resize import image_resize

def main():
    img = cv2.imread('green.jpg', 1)

    # showing original pic
    cv2.namedWindow('original', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('original', 900, 600)
    cv2.imshow('original', img)
    if img.shape[1] > 1000:
        img = image_resize(img, 1000)

    seperator = HexagonSeperator(img)
    hexagonsRows = seperator.seperateHexagons()
    # Creating seperate images from the different hexagons
    hexagonImages = [] # this array will contain the individual hexagons sorted by rows and columns
    for row in hexagonsRows:
        for i, cnt in enumerate(row):
            mask = np.zeros(img.shape, np.uint8)
            pixelpoints = np.transpose(np.nonzero(mask))
            mask = cv2.bitwise_not(mask)
            for pp in pixelpoints:
                mask[pp[0], pp[1]] = img[pp[0], pp[1]]
            hexagonImages.append(mask)
    for i, hexagonImage in enumerate(hexagonImages):
        cv2.imwrite("{0}_croppedHexagon.jpg".format(i+1), hexagonImage)

    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('img', 900, 600)
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()