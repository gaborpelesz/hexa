import numpy as np
import tensorflow as tf
import cv2

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="tf_model/tf_model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test model on random input data.
input_shape = input_details[0]['shape']

img = cv2.imread('individualHexagonsWithColors/000/IMG_000_2.jpg', 1)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
resized = cv2.resize(img, (224,224), interpolation = cv2.INTER_AREA)
resized = [resized]

interpreter.set_tensor(input_details[0]['index'], resized)

interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])

# -------------------------------
""" matching output to labels """

# TODO raise if more than two indexes were max
# TODO check max if it is greater than the MINIMUM we consider a valueable prediction
# TODO 70 score is good for separate
max_prediction_value = max(output_data[0])
all_indexes_of_max_precictions = [i for i, j in enumerate(output_data[0]) if j == max_prediction_value]

# open labels file and create a list from the labels
f = open("tf_model/tf_edge_labels.txt", "r")
labels = f.read().split('\n')

if len(all_indexes_of_max_precictions) == 1:
    outputFormat = 'predicted label is: "{}"\nwith prediction score: {}'.format(labels[all_indexes_of_max_precictions[0]], max_prediction_value)
    print(outputFormat)

print("-----------------------")
print(output_data)
print("-----------------------")
