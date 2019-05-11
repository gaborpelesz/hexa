from flask import Flask
from flask import request
from flask import redirect
from image_converter import convert_base2image
import socket

app = Flask(__name__)

@app.route("/ml", methods=["GET", "POST"])
def hello():
    if request.method == "POST":
        base64_image = request.form["image"]
        convert_base2image(base64_image)
        return redirect("/ml")
    else:
        html = "<h3>Hello {name}!</h3>" \
            "<b>Hostname:</b> {hostname}<br/>" \
            "<b>Visits:</b> {visits}"
        return html.format(name="Test", hostname=socket.gethostname(), visits="0")

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80)