# Hexa python computer vision backend

## Init/Install Project

```bash
cd hexa && make build && make run
```

## Makefile

```bash
  # Build the python service image and force remove previous instance
  make build

  # Push to docker hub
  make push

  # Run all the service containers with compose (docker-compose up) and force recreate container for python service
  make run

  # Run developement environment (make run with docker-compose.dev.yml)
  make dev

  # Get into the python service's container shell
  make exec

  # Turn down running docker-compose services
  make down
```

## Usage

Server will be listening on port 3000.

### Available endpoints:
- POST /predict

## POST /predict

### Request

The API is waiting for POST requests at **/predict**.

The request type must be **multipart/form-data** and it must have an **image** field containing an image file.

The original image, we want to inference, should have 350 width and around the 350w/262h ratio.

### Response

If the request was successful, the response will be a JSON string with a **predicted_value** property,
which contains a string of a number (in base 3), and a **predicted_status** which shows that the prediction is viable (**true**) or inappropriate (**false**).  
If **predicted_status** is **false** then the response JSON will contain an **incorrect_hexagons** list of uncertainly predicted hexagons. Every element will be in the following format: **[ [row, column], error_message ]** where row and column identifies the hexagon and the error_message shows why the prediction for this hexagon was uncertain. To be clear, **row** and **column** values are 1-based. Meaning that [1, 1] is the first rows first element. 

Response example:
```bash
{
  "predicted_value": "212122012222201",
  "formation_of_hexagons": [3, 2],
  "prediction_status": true
}
```

Response example when prediction failed (where the score 70 comes from PRED_MIN_SCORE_LIMIT which is currently hard coded):
```bash
{
  "predicted_value": "021e221e121e010112220022101010220",
  "formation_of_hexagons": [4, 3, 4]
  "prediction_status": false,
  "incorrect_hexagons": [
    [
      [1, 4],
      "The prediction for a hexagon is not certain, because the predicted score 55 is below score 70."
    ]
  ]
}
```

*JUST SOME INFO ON PREDICTION DETAILS*  
**prediction_details** is an array containing the following:  
**[ PRED_MIN_SCORE_LIMIT, PRED_MAX_DIFF_LIMIT ]**  where,  
**PRED_MIN_SCORE_LIMIT** (optional) is setting the minimum score limit that is allowed at making a prediction. Otherwise, if the highest scored label is below that limit, the prediction is considered weak.  
**PRED_MAX_DIFF_LIMIT** (optional) is setting the maximum difference from the highest score and the second highest score, thus avoid uncertainty.

There is a **Postman** example included, in the *hexa.postman_collection.json* file from the next section.

## Postman

Postman collection file: `postman_example/hexa.postman_collection.json`.

© gaborpelesz 2019
