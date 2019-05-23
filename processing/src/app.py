from flask import Flask, request, redirect
from flask_api import FlaskAPI
from image_base64_converter import convert_base2image
from imageProcessing.processingImage import processing_image
from imageProcessing.image_resize import image_resize
import socket

app = FlaskAPI(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST" and request.is_json:
        content = request.get_json()
        base64_image = content["image"]
        cv2_img = convert_base2image(base64_image)

        if cv2_img.shape[1] > 350:
            cv2_img = image_resize(cv2_img, 350)

        final_predicted_value = processing_image(cv2_img)
        #final_predicted_value = "0"
        return { "predicted_value": final_predicted_value }

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=3000)