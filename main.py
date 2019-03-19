import cv2
import numpy as np
from image_resize import image_resize
from hexagonSeperator import HexagonSeperator
from processingHexagon import processingHexagon

def isBlack(bgr, thresholdValue):
    for color in bgr.tolist():
        if color > 170: return False
    b_g_difference = np.int32(bgr[0]) - np.int32(bgr[1])
    r_b_difference = np.int32(bgr[2]) - np.int32(bgr[0])
    r_g_difference = np.int32(bgr[2]) - np.int32(bgr[1])

    # filtering Purple
    if r_b_difference <= 10 and r_g_difference > 20:
        return False

    if r_g_difference < -10 or abs(r_g_difference) > thresholdValue:
        return False
    if r_b_difference < -10 or abs(r_b_difference) > thresholdValue:
        return False

    return False if abs(b_g_difference) > thresholdValue else True

# --------------- TESTING FUNCTION ---------------

def processingForTests(imgPath):
    img = cv2.imread(imgPath, 1)
    if img.shape[1] > 1000:
        img = image_resize(img, 1000)
    seperator = HexagonSeperator(img)
    hexagonRows = seperator.seperateHexagons()
    finalResultInBase3 = ""
    for row in hexagonRows:
        numsInRow = ["","",""]
        for i, cnt in enumerate(row):
            x,y,w,h = cv2.boundingRect(cnt)
            mask = np.zeros((h, w, 3), np.uint8)
            maskCnt = cnt.copy()
            for pixels in maskCnt:
                pixels = pixels[0]
                pixels[0] = pixels[0] - x
                pixels[1] = pixels[1] - y
            cv2.drawContours(mask, [maskCnt], 0, (255, 255, 255), -1)
            pixelpoints = np.transpose(np.nonzero(mask))
            mask = cv2.bitwise_not(mask)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            for i in range(x, x+w):
                for j in range(y, y+h):
                    maskPP = mask[j-y][i-x]
                    if maskPP == 0:
                        mask[j-y, i-x] = 0 if isBlack(img[j, i], 25) else 255
            numSequence = "err"
            numSequence = processingHexagon(mask, w*h)
            for j in range(3):  
                numsInRow[j] += numSequence[j]
        finalResultInBase3 += "".join(numsInRow)

    return finalResultInBase3

# --------------- END OF TESTING FUNCTION ---------------

def main():
    # img = cv2.imread('beta1.0/testCases/10_022220221121122.jpg', 1)
    # img = cv2.imread('test_morningCloudy/orange.jpg', 1)
    # img = cv2.imread('test_evening_warmLight/blue3.jpg', 1)
    # img = cv2.imread('1_croppedHexagon.jpg', 1)
    img = cv2.imread('new.jpg', 1)

    # if the width of the image is more than 1000 pixels we reduce its size
    print(img.shape)
    if img.shape[1] > 1000:
        img = image_resize(img, 1000)
    cv2.namedWindow('thresh3', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('thresh3', 900, 600)
    cv2.imshow('thresh3', img)
    # create the HexagonSeperator instance with the colored image and seperate the hexagons from it
    seperator = HexagonSeperator(img)
    # hexagonRows contains contours in the following order
    # { row1[hexa1, hexa2, ..., hexaM], row2[hexa2, ...], row2[hexa1, ...], ..., rowN[hexa1, ...]}
    hexagonRows = seperator.seperateHexagons()

    # we now have the sorted list of contours of the hexagons
    print("the image contains", len(hexagonRows), "rows of hexagons")
    
    # ----------------------------------------------------

    allMasks = [] # DEBUG

    finalResultInBase3 = ""
    for j, row in enumerate(hexagonRows):
        numsInRow = ["","",""]
        for i, cnt in enumerate(row):
            x,y,w,h = cv2.boundingRect(cnt)
            mask = np.zeros((h, w, 3), np.uint8)
            # mask = np.zeros(img.shape, np.uint8)
            print('imgshape:', img.shape)
            print('x,y,w,h:', x,y,w,h)
            maskCnt = cnt.copy()
            for pixels in maskCnt:
                pixels = pixels[0]
                pixels[0] = pixels[0] - x
                pixels[1] = pixels[1] - y
            # print(maskCnt)
            cv2.drawContours(mask, [maskCnt], 0, (255, 255, 255), -1)
            cv2.namedWindow('thresh2', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('thresh2', 900, 600)
            pixelpoints = np.transpose(np.nonzero(mask))
            mask = cv2.bitwise_not(mask)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            cv2.imshow('thresh2', mask)
            # print(cnt)
            for i in range(x, x+w):
                for j in range(y, y+h):
                    maskPP = mask[j-y][i-x]
                    if maskPP == 0:
                        # if j-y == 252 or i-x == 252:
                            # print(isBlack(img[252+x,262+y], 20))
                        mask[j-y, i-x] = 0 if isBlack(img[j, i], 25) else 255 # 266 254
                        # x=292 y=340
            numSequence = "err"
            # cv2.namedWindow('thresh', cv2.WINDOW_NORMAL)
            # cv2.resizeWindow('thresh', 900, 600)
            # cv2.imshow('thresh', mask)
            allMasks.append(mask)
            try:
                numSequence = processingHexagon(mask, w*h)
            except IndexError:
                print('Index error happened at {0}. rows {1}. element'.format(j+1,i+1))
            for j in range(3):  
                numsInRow[j] += numSequence[j]
        finalResultInBase3 += "".join(numsInRow)
    # ----------------------------------------------------
    maskResult = cv2.bitwise_not(allMasks[0]).copy()
    print(len(allMasks))
    # for mask in allMasks:
    #     maskResult = cv2.bitwise_or(maskResult, cv2.bitwise_not(mask))
    cv2.namedWindow('thresh', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('thresh', 900, 600)
    cv2.imshow('thresh', allMasks[2])
    # test = processingHexagon(allMasks[2], 285*321)

    print("result: {0} | from {1} hexagons".format(finalResultInBase3, len(finalResultInBase3)//3))

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()