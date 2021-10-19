import os
import time

import cv2
import numpy as np
from flask import Flask, request, jsonify
from flask_api import FlaskAPI, status
from flask_cors import CORS

from hexaprocessing.processing_image import processing_image
from hexaprocessing.image_resize import image_resize

app = FlaskAPI(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST" and request.files.get('image'):
        t0 = time.time()

        image_bytes = request.files["image"].read()
        cv2_img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), -1)

        if cv2_img.shape[1] > 350:
            cv2_img = image_resize(cv2_img, 350)

        ## DEBUG
        cv2.imwrite("app/temp/resized.jpg", cv2_img)

        final_predicted_value, is_viable, formation_of_hexagons, incorrect_hexagons_list = processing_image(cv2_img)

        t1 = time.time()

        json_response = {
            "predicted_value": final_predicted_value,
            "formation_of_hexagons": formation_of_hexagons,
            "prediction_status": is_viable,
            "runtime_ms": (t1-t0)*100
        }

        if not is_viable:
            json_response["incorrect_hexagons"] = [[x[:2], x[2]] for x in incorrect_hexagons_list]

        return jsonify(json_response), status.HTTP_200_OK

    # Request error handling
    elif request.method != "POST":
        return jsonify(
            Error="HTTP method for /predict must be POST."
        ), status.HTTP_400_BAD_REQUEST
    elif not request.files.get('image'):
        return jsonify(
            Error="Form field 'image' must be specified and must contain a valid image file."
        ), status.HTTP_400_BAD_REQUEST
    else:
        return jsonify(
            Error="Internal server error. Please contact gaborpelesz@gmail.com"
        ), status.HTTP_500_INTERNAL_SERVER_ERROR

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000)