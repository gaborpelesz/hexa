const fs = require('fs');
const app = require("express")();
const multer = require("multer");
const request = require("request");
const port = 3000;

const upload = multer({dest: "./"});
app.set("view engine", "pug");

app.get("/", (req, res) => {
    res.render("index", {isPredicted: false, predictedValue: null});
});

app.post("/",  upload.single("file"), (req, res) => {
    const tempPath = req.file.path;
    const base64img = fs.readFileSync(tempPath, "base64");

    request.post(
        "http://172.18.0.1:3000/predict",
        { json: { image: base64img } },
        function (error, response, body) {
            let predictedValue = "Some error has occured.";
            if (!error && response.statusCode == 200) {
                console.log("Body: " + body);
                predictedValue = body.predicted_value;
            } else {
                console.log("Error: " + error);
                if(response != undefined){
                    console.log("statusCode: " + response.statusCode);
                    console.log("message: " + response.body);
                }
            }
            fs.unlink(tempPath, () => {
                res.render("index", {isPredicted: true, predictedValue: predictedValue});
            });
        }
    );
});

app.listen(port, () => console.log(`App is listening on port ${port}!`));