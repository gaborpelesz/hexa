import cv2
import numpy as np

# DEBUG variable
_DEBUG = True

def processingHexagon(hexagonImage, hexagonBoxArea):
    # hexagonImage = cv2.cvtColor(hexagonImage, cv2.COLOR_BGR2GRAY)
    hexagonImage = cv2.threshold(hexagonImage, 150, 255, cv2.THRESH_BINARY_INV)[1]

    kernel = np.ones((4,4), np.uint8) # maybe 4,4 kernel for safety reasons 

    # hexagonImage = cv2.morphologyEx(hexagonImage, cv2.MORPH_CLOSE, kernel)
    hexagonImage = cv2.erode(hexagonImage,kernel,iterations = 4)
    hexagonImage = cv2.dilate(hexagonImage,kernel,iterations = 4)

    # trying contour detection in 'hexagonImage'
    triangleCnts, hierarchy = cv2.findContours(hexagonImage,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    # cv2.RETR_EXTERNAL -> just find outer contours
    areas = []
    for c in triangleCnts:
        areas.append(cv2.contourArea(c))
    try:    
        mean = sum(areas) / len(areas)
    except ZeroDivisionError:
        print('ZERO DIVISON HAPPENED... no objects were found')
    triangleCnts = [c for c in triangleCnts if cv2.contourArea(c) >= 2*mean/3]

    # ---------- DEBUG ------------
    if _DEBUG:
        for i, cnt in enumerate(triangleCnts):
            x,y,w,h = cv2.boundingRect(cnt)
            cv2.putText(hexagonImage, str(i+1),(round(x+w/2), round(y+h/2)), cv2.FONT_HERSHEY_SIMPLEX, 4,150,5,cv2.LINE_AA)
        cv2.drawContours(hexagonImage, triangleCnts, -1, 100, 3)
        cv2.namedWindow('processing', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('processing', 900, 600)
        cv2.imshow('processing', hexagonImage)
        print("1 - 2",cv2.matchShapes(triangleCnts[0],triangleCnts[1],2,0.0))
        print("1 - 3", cv2.matchShapes(triangleCnts[0],triangleCnts[2],2,0.0))
        print("2 - 3", cv2.matchShapes(triangleCnts[1],triangleCnts[2],2,0.0))
    # ---------- DEBUG END ------------

    boundingBoxes = [cv2.boundingRect(triangleCnt) for triangleCnt in triangleCnts]
    centers = [(round(x + w/2), round(y + h/2)) for (x, y, w, h) in boundingBoxes]
    sortedCenters = sorted(centers, key = lambda y: y[1])

    sortedTriangles = []
    for center in sortedCenters:
        for i, triangleCnt in enumerate(triangleCnts):
            x,y,w,h = cv2.boundingRect(triangleCnt)
            if (round(x + w/2), round(y + h/2)) == center:
                sortedTriangles.append(triangleCnt)

    # ---------- DEBUG ------------
    if _DEBUG:
        print("---------------")
        print("{0}: {1}".format(hexagonBoxArea, cv2.contourArea(sortedTriangles[0])))
        print("1: {0}".format(hexagonBoxArea/cv2.contourArea(sortedTriangles[0])))
        approx = cv2.approxPolyDP(sortedTriangles[0], 0.05*cv2.arcLength(sortedTriangles[0], True), True)
        print("1.", len(approx))
        print("---------------")
        print("---------------")
        print("{0}: {1}".format(hexagonBoxArea, cv2.contourArea(sortedTriangles[1])))
        print("2: {0}".format(hexagonBoxArea/cv2.contourArea(sortedTriangles[1])))
        approx = cv2.approxPolyDP(sortedTriangles[1], 0.05*cv2.arcLength(sortedTriangles[1], True), True)
        print("2.", len(approx))
        print("---------------")
        print("---------------")
        print("{0}: {1}".format(hexagonBoxArea, cv2.contourArea(sortedTriangles[2])))
        print("3: {0}".format(hexagonBoxArea/cv2.contourArea(sortedTriangles[2])))
        approx = cv2.approxPolyDP(sortedTriangles[2], 0.05*cv2.arcLength(sortedTriangles[2], True), True)
        print("3.", len(approx))
        print("---------------")
        print("DEBUG !!!!!!!!!!!!!!!!!!!!")
        print("END DEBUG !!!!!!!!!!!!!!!!!!!!")
    # ---------- DEBUG END ------------

    testTo = sortedTriangles[1]
    approxTest = cv2.approxPolyDP(testTo, 0.05*cv2.arcLength(testTo, True), True)
    
    def vectorsDistance(point1, point2):
        x1, y1, x2, y2 = point1[0], point1[1], point2[0], point2[1]
        return float((abs(x1-x2)**2 + abs(y1-y2)**2))**0.5

    def sideLengths(approx):
        sides = []
        for i in range(len(approx)):
            point1 = approx[i][0]
            for j in range(i+1, len(approx)):
                point2 = approx[j][0]
                sides.append(vectorsDistance(point1, point2))
                print('point1 - point2 - distance | ' + str(point1) + ' - ' + str(point2) + ' - ' + str(vectorsDistance(point1, point2)))
        return sides



    numSequence = ""
    for sortedTriangle in sortedTriangles:
        approx = cv2.approxPolyDP(sortedTriangle, 0.04*cv2.arcLength(sortedTriangle, True), True)
        sides = sideLengths(approx)
        # If the contour has 3 sides then it is probably a triangle with 0 white object
        if len(approx) == 3:
            numSequence += "0"
        # If the contour hasn't got 3 sides
        # and the length ratio of the longest and shortest sides are less than 2.9
        # or (in case of the contour has some weird sides)
        # the ratio of the hexagons enclosing rectangle and the area of the contour is less than 11.5
        # then it is probably a triangle with 2 white object
        elif max(sides) / min(sides) < 2.85 or hexagonBoxArea/cv2.contourArea(sortedTriangle) > 11.5:
            numSequence += "2"
            # ---------- DEBUG ------------
            if _DEBUG: print(max(sides) / min(sides))
            # ---------- DEBUG END ------------
        # else it is probably a triangle with 1 white object
        else:
            numSequence += "1"
            # ---------- DEBUG ------------
            if _DEBUG: print(max(sides) / min(sides))
            # ---------- DEBUG END ------------
    # ---------- DEBUG ------------
    if _DEBUG: print("nums => ",numSequence[0],numSequence[1],numSequence[2])
    # ---------- DEBUG END ------------
    return numSequence