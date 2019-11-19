# MC_Assignment_GestureIdentification

## Pre-Requesites
 - Flask
 - Python3
 - scikit-learn
 - pandas
 - numpy
 - joblib
 
- - - - 

## Installation
 * Use requirements.txt present inside src folder for installing the dependencies.
     * pip install -r requirements.txt
     
----

## Execution (Local Testing)
 * Run command `Python main.py` from src folder.
 * In postman, Pass URL as `http://127.0.0.1:80/test` and in body, pass the keyPoints.json.
 * Return type will be of form:
     * {"1": "predicted_label", "2": "predicted_label", "3": "predicted_label", "4": "predicted_label"}
