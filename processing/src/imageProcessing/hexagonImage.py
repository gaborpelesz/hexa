import cv2
import numpy as np

class HexagonImage:
    def __init__(self, img):
        self.image = img.copy()
        self.iter = 0
        self.contours = None

    def draw(self, contours):
        self.contours = contours.copy()
        self.image = cv2.drawContours(self.image, contours, -1, (0,0,255), 3)

    def drawNumbers(self, numbers):
        height = -25
        hexagon = self.contours[self.iter]
        (x,y),radius = cv2.minEnclosingCircle(hexagon)
        hexagon_center = (int(x), int(y))
        for num in numbers:
            num_coords = (hexagon_center[0], hexagon_center[1]+height)
            self.image = cv2.putText(self.image, num, num_coords, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv2.LINE_AA)
            height += 25

    def nextHexagon(self):
        self.iter += 1

    def show(self):
        cv2.imshow('Processed image', self.image)
