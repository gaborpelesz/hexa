"""
    1. Search for testCases directory
    2. copy all the file names inside
    3. FOR every Image
        1. split at '_' to get the following [seqNum, Value.jpg]
        2. run main.py with arg seqNumValue.jpg
        3. test result against Value
    4. print -> all tests: n
                test succeeded: x/n
                test failed with value error: y/n
                test failed with Thrown error: k/n
    5. print array of all failed tests seqNum
        i.e -> [1, 5, 13, 14, 17, 20]
"""

import os
import cv2
from main import processingImage
import numpy as np
import PySimpleGUI as sg
import datetime
from hexagonImage import HexagonImage
from image_resize import image_resize
from hexagonSeparator import HexagonSeparator
from processingHexagon import processingHexagon
from processingHexagon import processingHexagonFromMLEdgeModel

# we assume that the testCases folder is in the same directory as the tester.py
# so we get the tester.py's path
testerDir = os.path.dirname(os.path.abspath(__file__))

if not 'testCases' in os.listdir(testerDir):
    raise Exception('testCases is not in this directory')
if not 'main.py' in os.listdir(testerDir):
    raise Exception('main.py is not in this directory')

# get testCases path and the file names that we are going to test
testCasesDir = testerDir+'/testCases'
test_file_names = [x for x in os.listdir(testCasesDir) if x != '.DS_Store']
# print(test_file_names)
tests_number = len(test_file_names)
tests_succeded = 0
tests_failed = 0
tests_that_failed = []
max_predicted_values = []

# LOADING BAR FOR USER EXPERIENCE
loading_bar = '[' + (50*' ') + ']'
print(30*'\n'+ 'PROCESSING TEST IMAGES:\n')
print("0/{0} Image processed".format(tests_number))
print("0% Completed")
print(loading_bar)
# LOADING BAR FOR USER EXPERIENCE

# we have file names in the following format seqNum_value.jpg
for i, test_file_name in enumerate(test_file_names):
    # we get seqNum by splitting at '_' and getting the first part
    seqNum = test_file_name.split('_')[0]
    # we get assertValue by splitting the second part at '.' and getting the first part
    assertValue = test_file_name.split('_')[1].split('.')[0]

    try:
        img = cv2.imread(testCasesDir + '/' + test_file_name, 1) # read in image
        if img.shape[1] > 350:
            img = image_resize(img, 350)
        result, predicted_values = processingImage(img)

        # ///////// DEBUG ////////////
        print(test_file_name)
        print("------------------")
        print(predicted_values)
        max_predicted_values.extend(predicted_values)
        # ///////// DEBUG ////////////

        if assertValue == result:
            tests_succeded += 1
        else:
            tests_failed += 1
            tests_that_failed.append([seqNum,'Assertion failed.\n'])
    except ZeroDivisionError as zero:
        tests_that_failed.append([seqNum,'Zero Division error accured, probably because the thresholding at the SEPERATION phase\
        \nfailed and no areas were detected...\nDebug at seperation phase NEEDED!\n'])
        tests_failed += 1
    except IndexError as ie:
        tests_that_failed.append([seqNum, 'Index out of bounds error accured, probably because the thresholding \
        \nat the PROCESSING phase failed and not all black triangles were detected...\nDebug at seperation phase NEEDED if the failed image is not tilted\n'])
        tests_failed += 1

    # progress with loading bar
    equal_signs = round(float(i+1)/float(tests_number)*50)
    spaces = 50 - equal_signs
    loading_bar = '[' + equal_signs*'=' + (spaces*' ') + ']'
    print(30*'\n'+ 'PROCESSING TEST IMAGES:\n')
    print("{0}/{1} Image processed".format(i+1,tests_number))
    print("{0:.00f}% Completed".format(float(i+1)/float(tests_number)*100))
    print(loading_bar)

print('------------------------')
print('Processing on test images completed:\n')
print('\tTotal number of tests:',tests_number)
print('\tTotal succeded:', tests_succeded)
print('\tTotal failed:', tests_failed)
print('\tTest success percent: {0:.2f}%'.format(float(tests_succeded)/float(tests_number)*100))

# ///////// DEBUG ////////////
print('------------------------')
print(max_predicted_values)
print('------------------------')
# ///////// DEBUG ////////////

if tests_that_failed:
    tests_that_failed = sorted(tests_that_failed, key=lambda x: int(x[0]))
    print('\n!!!!!!!!!!!!!')
    print('Failed tests are:\n')
    for test_that_failed in tests_that_failed:
        print('{0}. image failed.'.format(test_that_failed[0]))
        print('Cause:\n\t{0}'.format(test_that_failed[1]))
