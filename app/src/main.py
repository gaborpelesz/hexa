import cv2
import numpy as np
import PySimpleGUI as sg
import datetime
from hexagonImage import HexagonImage
from image_resize import image_resize
from hexagonSeparator import HexagonSeparator
from processingHexagon import processingHexagon
from processingHexagon import processingHexagonFromMLEdgeModel

def isBlack(bgr, maximumDifference, minimumColorForWhite):
    for color in bgr.tolist():
        if color > minimumColorForWhite: return False
    b_g_difference = np.int32(bgr[0]) - np.int32(bgr[1])
    r_b_difference = np.int32(bgr[2]) - np.int32(bgr[0])
    r_g_difference = np.int32(bgr[2]) - np.int32(bgr[1])

    # filtering Purple
    if r_b_difference <= 10 and r_g_difference > 20:
        return False

    if r_g_difference < -15 or abs(r_g_difference) > maximumDifference:
        return False
    if r_b_difference < -15 or abs(r_b_difference) > maximumDifference:
        return False

    return False if abs(b_g_difference) > maximumDifference else True

# --------------- TESTING FUNCTION ---------------

def processingImage(img, imhelper=None):
    max_predicted_values = []

    # ------ Getting white and black balance ------
    maximumBlackDifference = 30
    # 1000/25, 750/18,75 ~> (40,40) bal felső
    # 24*1000/25, 750/18,75 ~> (960, 40) jobb felső
    # 1000/25, 17,75*750/18,75 ~> (40, 710) bal alsó
    # 24*1000/25, 17,75*750/18,75  ~> (960, 710) jobb alsó
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

    # ----- Detect and seperate the hexagons ------
    seperate_start_time = datetime.datetime.now() # TIME

    separator = HexagonSeparator(img)
    hexagonRows = separator.separateHexagons(imhelper)

    seperate_end_time = datetime.datetime.now() # TIME
    seperate_time_spent = (seperate_end_time - seperate_start_time) # TIME
    seperate_time_spent = "{0}.{1:03d}s".format(seperate_time_spent.seconds, seperate_time_spent.microseconds // 1000) # TIME
    # ---------------------------------------------

    # ------ Process the individual hexagons ------
    processing_start_time = datetime.datetime.now() # TIME
    finalResultInBase3 = ""

    for row in hexagonRows:
        numsInRow = ["","",""]
        for k, cnt in enumerate(row):
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
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
            numSequence, max_predicted = processingHexagonFromMLEdgeModel(mask, imhelper) # PROCESSING HEXAGON
            max_predicted_values.append(max_predicted)
            for j in range(3):
                numsInRow[j] += numSequence[j]
        finalResultInBase3 += "".join(numsInRow)
        

    processing_end_time = datetime.datetime.now() # TIME
    processing_time_spent = (processing_end_time - processing_start_time)
    processing_time_spent = "{0}.{1:03d}s".format(processing_time_spent.seconds, processing_time_spent.microseconds // 1000) # TIME
    # ---------------------------------------------

    print("seperation was: {}".format(seperate_time_spent))
    print("processing all individual hexagons was: {}".format(processing_time_spent))
    return (finalResultInBase3, max_predicted_values)

# --------------- END OF TESTING FUNCTION ---------------

def main():

    layout = [[sg.Text('Select an image:')],
          [sg.Input(), sg.FileBrowse()],
          [sg.OK('Process'), sg.Cancel('Exit')] ]

    window = sg.Window('Hexagon processing program').Layout(layout)

    while True:
        event, (path,) = window.Read()
        if event == 'Process':
            img = cv2.imread(path, 1)
            if img.shape[1] > 350:
                img = image_resize(img, 350)
            hexagonImage = HexagonImage(img.copy())

            # PROCESSING IMAGE
            time_total_start =  datetime.datetime.now()

            resultNum, _ = processingImage(img, hexagonImage)

            time_total_end =  datetime.datetime.now()
            time_total_millisec = (time_total_end - time_total_start).microseconds / 1000
            print("total processing time was: {}ms".format(time_total_millisec))
            # END OF PROCESSING

            hexagonImage.show()
            pupup_event = sg.Popup('Processing completed.', \
            'There is a popup window with the image and the visual representation.',\
            '{0} -> result in base 3'.format(resultNum), '{0} -> result in base 10'.format(int(resultNum, base=3)), "\nPress 'OK' to select another image in the main menu.")
            if pupup_event == 'OK':
                cv2.destroyAllWindows()
        if event == 'Exit':
            return

if __name__ == "__main__":
    main()
