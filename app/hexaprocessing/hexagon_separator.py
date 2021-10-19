import cv2
import numpy as np
import sys

_DEBUG = False

class HexagonSeparator:
    """This class is responsible for separate Hexagons from an image input.

        The class has several methods that helps the previously mentioned,
        behavior. When an object is created from this class, it will initialize
        an image to this class's image attribute. Further function calls will
        work on this image and will return the processed results.
    """

    def __init__(self, image):
        """Gets the image parameter and sets the image attribute to its value.

            Args:
                image: This should be a numpy array containing a cv2 image,
                    which from we initialize our image attribute.
        """
        try:
            self.image = image.copy()
        except:
            print("Couldn't copy image in HexagonSeparator constructor...") # TODO raise

        self.hexagon_min_ratio = 0.5
        self.hexagon_max_ratio = 2
        self.hexagon_min_width = 30
        self.hexagon_min_height = 30

    def separateHexagons(self, imhelper=None):
        """Starts to process the class's image attribute and pass data,
        to the provided HexagonImage class.

        The easiest way to understand this function is to write a short
        list about what is does.
            1. It thresholds the image to a binary image.
            2. Does some morphological operations to reduce noise.
            3. Finds the contours on the image.
            4. Remove accidentally found smaller contours.
            5. Sort the contours according to the hexagons layout.

        Return:
            Sorted hexagons contours which are in
            format of cv2 contour nparrays.
        """
        hexagons = self.findHexagons(self.image)
        sortedRowsOfHexagonContours = self.sortHexagons(hexagons)

        # Draw to the image we want to visualize the steps on #
        if imhelper: imhelper.draw(sortedRowsOfHexagonContours[0]+sortedRowsOfHexagonContours[1])
        # --------------------------------------------------- #
        return sortedRowsOfHexagonContours

    def findHexagons(self, img):
        if len(img.shape) != 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
        thresholded = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 33, 7)
        thresholded = cv2.dilate(thresholded, None, iterations=1) # None -> 3x3 rect kernel

        contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = map(lambda cnt: cv2.convexHull(cnt), contours) # only storing convex hulls

        hexagons = filter(self.contourFilterForHexagons, contours)

        return list(hexagons)
        
    def contourFilterForHexagons(self, cnt):
        x,y,w,h = cv2.boundingRect(cnt)
        
        if w < self.hexagon_min_width:
            return False
        
        if h < self.hexagon_min_height:
            return False
        
        ratio = (float)(w) / h
        if ratio < self.hexagon_min_ratio or ratio > self.hexagon_max_ratio:
            return False

        epsilon = 0.035 * cv2.arcLength(cnt,True)
        approx = cv2.approxPolyDP(cnt,epsilon,True)
    
        # not an actual hexagon
        if len(approx) != 6:
            return False
        
        return True

    def sortHexagons(self, hexagons):
        """Sorting algorithm:
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
            rowRange = range(rangeCalculator - 40, rangeCalculator + 40)

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
