import os
import cv2
import numpy as np
import tensorflow as tf
import sys

def processingHexagonFromMLEdgeModel(hexagonImage, imhelper=None):
    # ----- initialize tf lite model from file ----- #
    # Load TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path="app/tf_model/tf_model.tflite")
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Get the shape of the input so we can resize the image according to that
    input_shape = input_details[0]['shape']

    # ----- getting image ready for inference ----- #
    resized = cv2.resize(hexagonImage, (224,224), interpolation = cv2.INTER_LINEAR)
    resized = [resized]

    # Set tensors according to the image we want to run inference on
    interpreter.set_tensor(input_details[0]['index'], resized)

    # ----- run inference ----- #
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    probs = softmax_prob(output_data[0])
    max_prob_i = np.argmax(probs)
    if probs[max_prob_i] < 95:
        raise Exception(f"Uncertain prediction with probability: {probs[max_prob_i]}%")

    # open labels file and create a list from the labels
    f = open("app/tf_model/tf_edge_labels.txt", "r")
    labels = f.read().split('\n')

    # We get the predicted sequence string (e.g: "201" or "002") from the labels array
    predicted_num_sequence = labels[max_prob_i]

    if imhelper:
        imhelper.drawNumbers(predicted_num_sequence)
        imhelper.nextHexagon()

    return predicted_num_sequence, probs[max_prob_i]


def softmax_prob(v):
    exp = np.exp(v.astype(np.float64))
    sum_exp = np.sum(exp)
    res_softmax = exp / sum_exp
    return (res_softmax*100).astype(np.uint8)