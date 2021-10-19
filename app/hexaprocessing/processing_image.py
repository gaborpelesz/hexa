import cv2
import numpy as np
from hexaprocessing.hexagon_image import HexagonImage
from hexaprocessing.image_resize import image_resize
from hexaprocessing.hexagon_separator import HexagonSeparator
from hexaprocessing.processing_hexagon import processingHexagonFromMLEdgeModel

def processing_image(img, imhelper=None) -> str:
    """This function gets a downsized image (images should be downsized to
        350 pixel width from a close to 16:12 image, for maximum performance)
        of hexagons and returns the entire predicted value (in base3)
        as a string.
 
        It is starting the processing by getting sample pixels from the image 4
        different points. From that it tries to predict the white-black balance
        of the picture in words, how dark is the images background to ensure
        maximum potential when cutting the hexagons out. After that the function
        calls the HexagonSeparators separate function which separates the
        hexagons from eachother. It returns the contours from the
        hexagon images. Finally it cuts out the hexagons one by one and predicts
        the value of the hexagon which returns a string. We append this string
        to the variable storing the final value. At the end, when there is no
        other hexagon to be processed, we return the final value.
        (Note that it is considered to be a base3 number in string format)

        Args:
            img: The downsized image that we going to process.
            imhelper: An image helper class is used when we need
                the image to be displayed. When processing the image
                we pass the results to the image helper class so it can
                modify the base image according to that.

        Returns:
            A string (which only contains numbers and is in base3) of the final
            predicted value which is based on the processing.
            For example:

            "201012212012212"

            An example of the layout of the hexagons corresponding to the
            final value is:

            [[(2,0,2)(0,1,1)(1,2,2)], [(0,2,1),(1,2,2)]]

            Where we can see that there is 2 rows of hexagons
            (because 2 elements in the outer array),
            first row contains 3 hexagons, second row contains 2 hexagons and
            the first row's first hexagon's top triangle has the value of '2',
            mid triangle has '0', bottom triangle has '2'.

            ...
            TODO
    """

    # ---------------------------------------------

    separator = HexagonSeparator(img)
    hexagonRows = separator.separateHexagons(imhelper)

    # ---------------------------------------------

    # ------ Process the individual hexagons ------
    finalResultInBase3 = ""
    is_prediction_viable = True
    incorrect_hexagons = []
    formation_of_hexagons = []

    for row_index, row in enumerate(hexagonRows):
        numsInRow = ["","",""]
        formation_of_hexagons.append(len(row))
        for column_index, cnt in enumerate(row):
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
            try:
                numSequence, max_predicted_value = processingHexagonFromMLEdgeModel(
                                                    mask,
                                                    imhelper=imhelper)
                    
            except Exception as error:
                print("ERROR:", error)
                numSequence = "eee"
                is_prediction_viable = False
                incorrect_hexagons.append((row_index+1,column_index+1, str(error)))
                max_predicted_value = -1

            for j in range(3):
                numsInRow[j] += numSequence[j]
        finalResultInBase3 += "".join(numsInRow)
        
    # ---------------------------------------------

    return (finalResultInBase3, is_prediction_viable, formation_of_hexagons, incorrect_hexagons)