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

    # Get the shape of the input so we can resize the image according to that
    input_shape = input_details[0]['shape']

    # ----- getting image ready for inference ----- #
    resized = cv2.resize(hexagonImage, (224,224), interpolation = cv2.INTER_AREA)
    resized = [resized]

    # Set tensors according to the image we want to run inference on
    interpreter.set_tensor(input_details[0]['index'], resized)

    # ----- run inference ----- #
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # ----- matching output to labels ----- #
    max_prediction_value = max(output_data[0])
    all_indexes_of_max_precictions = [i for i, j in enumerate(output_data[0]) if j == max_prediction_value]

    # VALIDITY CHECKING FOR PREDICTED VALUE
    # To methods to try:
    #   1. 98.4375% of the predicted max values are over score 70
    #   2. if max_predicted is 30 score greater than the second max than it is almost certain
    # ---------------------------
    # METHOD 1. score is below 70
    if max_prediction_value < 70: 
        raise Exception("The prediction for a hexagon is not certain, because the predicted \
        value is below score 70. Please take another, better image.")
    # METHOD 2. two highest scores difference is less than 30
    sorted_predictions = sorted(output_data[0])
    if max_prediction_value - sorted_predictions[len(sorted_predictions)-2] < 30:
        raise Exception("The prediction for a hexagon is not certain, because the two highest \
        values difference is greater than 30 ({}-{} < 30). \
        Please take another, better image.".format(max_prediction_value, sorted_predictions[len(sorted_predictions)-2]))
    # ---------------------------

    # open labels file and create a list from the labels
    f = open("tf_model/tf_edge_labels.txt", "r")
    labels = f.read().split('\n')

    # We get the predicted sequence string (e.g: "201" or "002") from the labels array
    predicted_num_sequence = labels[all_indexes_of_max_precictions[0]]

    if imhelper:
        imhelper.drawNumbers(predicted_num_sequence)
        imhelper.nextHexagon()

    return (predicted_num_sequence, max_prediction_value)