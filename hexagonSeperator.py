import cv2
import numpy as np

_DEBUG = False

class HexagonSeperator:
    def __init__(self, image):
        try:
            self.image = image.copy()
        except:
            print('An error occured.')

    def seperateHexagons(self):
        thresholded = self.threshold(self.image)
        contourReady = self.morphologicalOpen(thresholded)
        # TODO maybe morph close for safety? Not needed till now
        # img = self.morphologicalClose(img)

        # ---------- DEBUG ------------
        if _DEBUG:
            cv2.namedWindow('seperation', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('seperation', 900, 600)
            cv2.imshow('seperation', contourReady)
        # ---------- DEBUG END ------------

        contours = self.findContours(contourReady)
        contours = self.removeSmallShapes(contours)
        cv2.drawContours(contourReady, contours, -1, 100, 10)
        sortedRowsOfHexagonContours = self.sortHexagons(contours)
        return sortedRowsOfHexagonContours
    
    @staticmethod
    def threshold(img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # for debug
        # H: 0 - 180 (blue), S: 0 - 255 (green), V: 0 - 255 (red)

        # filtering the blacks and storing it in a binarized image
        lower_end = np.array([0,0,0])
        upper_end = np.array([180,255,140])
        blacks = cv2.inRange(img, lower_end, upper_end)

        # filtering the colors and storing it in a binarized image
        lower_end = np.array([0,60,0])
        upper_end = np.array([180,255,255])
        colors = cv2.inRange(img, lower_end, upper_end)

        # merge the image with the colors and the blacks to have the image with full hexagons
        bit_or = cv2.bitwise_or(blacks, colors)

        return bit_or
        # return cv2.inRange(img, np.array([180,255,255]), np.array([180,255,160]))

    @staticmethod
    def morphologicalOpen(img):
        kernel = np.ones((5,5), np.uint8)
        return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    
    @staticmethod
    def morphologicalClose(img):
        kernelClose = np.ones((5,5), np.uint8)
        return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernelClose)

    @staticmethod
    def findContours(img):
        contours, hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        return contours

    @staticmethod
    def removeSmallShapes(contours):
        areas = []
        for c in contours:
            areas.append(cv2.contourArea(c))
        mean = sum(areas) / len(areas)

        contours = [c for c in contours if cv2.contourArea(c) >= 3*mean/4]
        # and len(cv2.approxPolyDP(c, 0.05*cv2.arcLength(c, True), True)) in (6,7)]

        areas = []
        for c in contours:
            areas.append(cv2.contourArea(c))
        return contours

    def sortHexagons(self, hexagons):
        """
            Sorting algorithm:
                Searching the first instance from top (y-axis), that center of a hexagon
                will be the 'range calculator'.

                From the range calculator's y value we define a range -> like (y-100, y+100),
                and in that range we go for a left-to-right sorting.

                So we go through the boundingBoxes array and if a center point is in that range, then we
                put it to another list called row and we remove it from the original list.

                After we put every element found in the row to the array then we sort that array of boundingBoxes
                by the x-axis value.

                After we are done with the previous row we start again after all elements are removed from 
                the original list.
        """

        # this will be the output
        sortedHexagons = []

        while hexagons:
            # find the center of the bounding rectangles
            boundingBoxes = [cv2.boundingRect(hexagon) for hexagon in hexagons]
            centers = [(round(x + w/2), round(y + h/2)) for (x, y, w, h) in boundingBoxes]

            # the folowing 'list(zip(*centers))' creates a list of [(x1,x2...), (y1,y2...)] tuple
            # then we get the (y1,y2...) tuple and calculate the min() value of it and 
            #   that will be the center of our range (range calculator)
            # then we create a range from the range calculator
            rangeCalculator = min(list(zip(*centers))[1])
            rowRange = range(rangeCalculator - 100, rangeCalculator + 100)

            centersInRow = [center for center in centers if center[1] in rowRange]
            sortedCentersInRow = sorted(centersInRow, key = lambda x: x[0]) # sort the tuples by first value

            # now we have to add to the sortedHexagons list the contours by their sorted centers
            # returning 'hexagons' list because we pop the already sorted items from it
            hexagons, sortedRowOfHexagons = self.mappingCentersToHexagons(sortedCentersInRow, hexagons)
            sortedHexagons += [sortedRowOfHexagons] # adding it by creating a list from it because later we need the rows

        return sortedHexagons

    @staticmethod
    def mappingCentersToHexagons(centers, hexagons):
        sortedRowOfHexagons = []
        for center in centers:
            for i, hexagon in enumerate(hexagons):
                x,y,w,h = cv2.boundingRect(hexagon)
                if (round(x + w/2), round(y + h/2)) == center:
                    sortedRowOfHexagons.append(hexagon)
                    hexagons.pop(i)
        return [hexagons, sortedRowOfHexagons]