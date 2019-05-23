import cv2
import numpy as np
import os
from image_resize import image_resize
from hexagonSeparator import HexagonSeparator
from processingHexagon import processingHexagon

sequence_number = 1
pictureLabel = "000"

def seperate(imgPath):
    global sequence_number
    img = cv2.imread(imgPath, 1)
    if img.shape[1] > 350:
        img = image_resize(img, 350)

    # ------ Getting white and black balance ------
    maximumBlackDifference = 30
    # 1000/25, 750/18,75 ~> (40,40) bal felso
    # 24*1000/25, 750/18,75 ~> (960, 40) jobb felso
    # 1000/25, 17,75*750/18,75 ~> (40, 710) bal also
    # 24*1000/25, 17,75*750/18,75  ~> (960, 710) jobb also
    y, x = img.shape[0], img.shape[1]
    upper_left_max_color = max(img[round(float(y)/18.75), round(float(x)/25.0)])
    upper_right_max_color = max(img[round(float(y)/18.75), round(24*float(x)/25.0)])
    bottom_left_max_color = max(img[round(17.75*float(y)/18.75), round(float(x)/25.0)])
    bottom_right_max_color = max(img[round(17.75*float(y)/18.75), round(24*float(x)/25.0)])
    minimumWhiteColor = (int(upper_left_max_color) + int(upper_right_max_color) + int(bottom_left_max_color) + int(bottom_right_max_color)) // 4
    minimumWhiteColor -= 15
    if minimumWhiteColor > 160:
        minimumWhiteColor = 170
    # ---------------------------------------------

    seperator = HexagonSeparator(img)
    hexagonRows = seperator.separateHexagons()
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
            for i in range(x, x+w):
                for j in range(y, y+h):
                    maskPP = mask[j-y][i-x]
                    if maskPP[0] == 0 and maskPP[1] == 0 and maskPP[2] == 0:
                        mask[j-y, i-x] = img[j, i]
            cv2.imwrite("individualHexagonsWithColors/{0}/IMG_{0}_{1}.jpg".format(pictureLabel, sequence_number), mask)
            sequence_number += 1

def main():
    
    testerDir = os.path.dirname(os.path.abspath(__file__))
    testCasesDir = testerDir+'/largeDataset/{0}'.format(pictureLabel)
    test_file_names = [x for x in os.listdir(testCasesDir) if x != '.DS_Store']
    # TEST
    # seperate(testCasesDir + '/' + test_file_names[0])
    # TEST END
    for i, test_file_name in enumerate(test_file_names):
        seperate(testCasesDir + '/' + test_file_name)
        print("{0}/{1}  - {2}".format(i, len(test_file_names), pictureLabel))

if __name__ == "__main__":
    main()