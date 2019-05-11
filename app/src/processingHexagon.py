import cv2
import numpy as np
from google.cloud import automl_v1beta1
from google.cloud.automl_v1beta1.proto import service_pb2
import tensorflow as tf

# DEBUG variable
_DEBUG = False

def processingHexagon(hexagonImage, hexagonBoxArea, imhelper=None):
    # hexagonImage = cv2.cvtColor(hexagonImage, cv2.COLOR_BGR2GRAY)
    hexagonImage = cv2.threshold(hexagonImage, 150, 255, cv2.THRESH_BINARY_INV)[1]

    kernel = np.ones((4,4), np.uint8) # maybe 4,4 kernel for safety reasons 

    # hexagonImage = cv2.morphologyEx(hexagonImage, cv2.MORPH_CLOSE, kernel)
    hexagonImage = cv2.erode(hexagonImage,kernel,iterations = 1)
    hexagonImage = cv2.dilate(hexagonImage,kernel,iterations = 1)

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
        approx = cv2.approxPolyDP(sortedTriangles[0], 0.04*cv2.arcLength(sortedTriangles[0], True), True)
        print("1.", len(approx))
        print("---------------")
        print("---------------")
        print("{0}: {1}".format(hexagonBoxArea, cv2.contourArea(sortedTriangles[1])))
        print("2: {0}".format(hexagonBoxArea/cv2.contourArea(sortedTriangles[1])))
        approx = cv2.approxPolyDP(sortedTriangles[1], 0.04*cv2.arcLength(sortedTriangles[1], True), True)
        print("2.", len(approx))
        print("---------------")
        print("---------------")
        print("{0}: {1}".format(hexagonBoxArea, cv2.contourArea(sortedTriangles[2])))
        print("3: {0}".format(hexagonBoxArea/cv2.contourArea(sortedTriangles[2])))
        approx = cv2.approxPolyDP(sortedTriangles[2], 0.04*cv2.arcLength(sortedTriangles[2], True), True)
        print("3.", len(approx))
        print("---------------")
        print("DEBUG !!!!!!!!!!!!!!!!!!!!")
        print("END DEBUG !!!!!!!!!!!!!!!!!!!!")
    # ---------- DEBUG END ------------

    testTo = sortedTriangles[1]
    approxTest = cv2.approxPolyDP(testTo, 0.04*cv2.arcLength(testTo, True), True)
    
    def vectorLength(point1, point2):
        x1, y1, x2, y2 = point1[0], point1[1], point2[0], point2[1]
        return float((abs(x1-x2)**2 + abs(y1-y2)**2))**0.5

    def parallelSidesRatio(approx):
        """
            This algorithm checks for the trapezoid's two parallel sides and then calculates the ratio of it
        """
        sides = []
        longest_side = ((0,0),(0,0))
        for i in range(len(approx)):
            point1 = approx[i][0]
            point2 = None
            if i == len(approx)-1: point2 = approx[0][0]
            else: point2 = approx[i+1][0]

            sides.append([point1, point2])
            if vectorLength(point1,point2) > vectorLength(longest_side[0], longest_side[1]):
                longest_side = (point1, point2)
        biggestCos = [0.0, "", ""]
        for j in range(0, len(sides)):
            p1, p2 = longest_side
            x1, x2 = sides[j][0], sides[j][1]
            if not (p1[0] == x1[0] and p1[1] == x1[1] and p2[0] == x2[0] and p2[1] == x2[1]):
                v1 = (p2[0] - p1[0], p2[1] - p1[1])
                v1_len = (float(v1[0]**2) + float(v1[1]**2))**0.5
                v2 = (x2[0] - x1[0], x2[1] - x1[1])
                v2_len = (float(v2[0]**2) + float(v2[1]**2))**0.5
                # vectorial multiplication
                vectMultp = v1[0] * v2[0] + v1[1] * v2[1]
                cosGamma = abs(float(vectMultp) / (v1_len*v2_len))
                if _DEBUG:
                    
                    # print('v1, v1len:', v1, v1_len)
                    # print('v2, v2len:', v2, v2_len)
                    # print('vectMultp:',vectMultp)
                    # print(float(vectMultp) ,'/', v1_len, '*', v2_len)
                    print('p1 p2 / x1 x2:',p1,p2,'/', x1, x2)
                    print('COSGAMMA =', cosGamma)
                if cosGamma > biggestCos[0]:
                    biggestCos[0] = cosGamma
                    biggestCos[2] = sides[j]
        if biggestCos[0] != 0.0:
            p1,p2 = longest_side
            x1,x2 = biggestCos[2][0],biggestCos[2][1]
            parallelSides = [vectorLength(p1,p2),vectorLength(x1,x2)]
            if _DEBUG: 
                print('p1 p2 / x1 x2:',p1,p2,'/', x1, x2)
                print('CHECK THIS: --------- ', max(parallelSides) / min(parallelSides))
            return max(parallelSides) / min(parallelSides)
        else:
            return -1    


        return -1

    numSequence = ["e","e","e"]
    for i, sortedTriangle in enumerate(sortedTriangles):
        approx = cv2.approxPolyDP(sortedTriangle, 0.04*cv2.arcLength(sortedTriangle, True), True)
        # If the contour has 3 sides then it is probably a triangle with 0 white object
        if len(approx) == 3:
            numSequence[i] = "0"
        else:
            sideRatio = parallelSidesRatio(approx)
            boxTriangleRatio = hexagonBoxArea/cv2.contourArea(sortedTriangle)
            if _DEBUG: 
                print('\n{0}. sideRatio: {1}'.format(i+1, sideRatio))
                print('{0}. boxTriangleRatio: {1}'.format(i+1, boxTriangleRatio))
                print('-----------------------------')
            if sideRatio == -1:
                raise Exception('NOT FOUND SIDES WITH LESS THAN 20 DEGREE ANGLES...')
            # If the contour hasn't got 3 sides
            # and the length ratio of the longest and shortest sides are less than 2.9
            # or (in case of the contour has some weird sides)
            # the ratio of the hexagons enclosing rectangle and the area of the contour is less than 11.5
            # then it is probably a triangle with 2 white object
            elif (sideRatio < 2.5 and boxTriangleRatio > 9.55): # or boxTriangleRatio > 14.0:
                # TODO vajon le lehet menni sideRatio > 2.0-ra?
                numSequence[i] = "2"
                # ---------- DEBUG ------------
                if _DEBUG: sideRatio
                # ---------- DEBUG END ------------
            # else it is probably a triangle with 1 white object
            else:
                numSequence[i] = "1"
                # ---------- DEBUG ------------
                if _DEBUG: sideRatio
                # ---------- DEBUG END ------------
    # ---------- DEBUG ------------
    if _DEBUG: print("nums => ",numSequence[0],numSequence[1],numSequence[2])
    # ---------- DEBUG END ------------
    if imhelper:
        imhelper.drawNumbers(numSequence)
        imhelper.nextHexagon()
    return numSequence

project_id = 'exahexa'
model_id = 'ICN6910564857651245076'

def processingHexagonFromMLCloud(hexagonImage, imhelper=None):
    # hexagon image is: [[255,255,255]]
    # print(hexagonImage)
    # encode image to byte string
    # numSequence = "000"
    encoded_str = cv2.imencode('.jpg', hexagonImage)[1].tostring()
    # get prediction from Google API
    prediction = get_prediction(encoded_str, project_id, model_id)
    # print(encoded_str)
    numSequence = prediction.payload[0].display_name
    
    if imhelper:
        imhelper.drawNumbers(numSequence)
        imhelper.nextHexagon()
    return numSequence

def get_prediction(content, project_id, model_id):
    prediction_client = automl_v1beta1.PredictionServiceClient()

    name = 'projects/{}/locations/us-central1/models/{}'.format(project_id, model_id)
    payload = {'image': {'image_bytes': content }}
    params = {}
    request = prediction_client.predict(name, payload, params)
    return request  # waits till request is returned

def processingHexagonFromMLEdgeModel(hexagonImage, imhelper=None):
    """ initialize tf lite model from file """
    # Load TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path="tf_model/tf_model.tflite")
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test model on random input data.
    input_shape = input_details[0]['shape']

    """ getting image ready to inference """
    resized = cv2.resize(hexagonImage, (224,224), interpolation = cv2.INTER_AREA)
    resized = [resized]

    interpreter.set_tensor(input_details[0]['index'], resized)

    """ run inference """
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    """ matching output to labels """

    # TODO raise if more than two indexes were max
    # TODO check max if it is greater than the MINIMUM we consider a valueable prediction
    max_prediction_value = max(output_data[0])
    all_indexes_of_max_precictions = [i for i, j in enumerate(output_data[0]) if j == max_prediction_value]

    # open labels file and create a list from the labels
    f = open("tf_model/tf_edge_labels.txt", "r")
    labels = f.read().split('\n')

    if _DEBUG and len(all_indexes_of_max_precictions) == 1:
        outputFormat = 'predicted label is: "{}"\nwith prediction score: {}'.format(labels[all_indexes_of_max_precictions[0]], max_prediction_value)
        print(outputFormat)

    numSequence = labels[all_indexes_of_max_precictions[0]]

    if imhelper:
        imhelper.drawNumbers(numSequence)
        imhelper.nextHexagon()

    print("-----------------------")
    print(output_data)
    print("-----------------------")

    return (numSequence, max_prediction_value)