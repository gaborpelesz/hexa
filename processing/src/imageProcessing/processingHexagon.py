import cv2
import numpy as np
import tensorflow as tf

def processingHexagonFromMLEdgeModel(hexagonImage, imhelper=None):
    # ----- initialize tf lite model from file ----- #
    # Load TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path="tf_model/tf_model.tflite")
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test model on random input data.
    input_shape = input_details[0]['shape']

    # ----- getting image ready to inference ----- #
    resized = cv2.resize(hexagonImage, (224,224), interpolation = cv2.INTER_AREA)
    resized = [resized]

    interpreter.set_tensor(input_details[0]['index'], resized)

    # ----- run inference ----- #
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # ----- matching output to labels ----- #

    # TODO raise if more than two indexes were max
    # TODO check max if it is greater than the MINIMUM we consider a valueable prediction
    max_prediction_value = max(output_data[0])
    all_indexes_of_max_precictions = [i for i, j in enumerate(output_data[0]) if j == max_prediction_value]

    # open labels file and create a list from the labels
    f = open("tf_model/tf_edge_labels.txt", "r")
    labels = f.read().split('\n')

    numSequence = labels[all_indexes_of_max_precictions[0]]

    if imhelper:
        imhelper.drawNumbers(numSequence)
        imhelper.nextHexagon()

    return (numSequence, max_prediction_value)