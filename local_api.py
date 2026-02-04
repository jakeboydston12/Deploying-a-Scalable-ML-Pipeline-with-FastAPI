import json

import requests

# TODO: send a GET using the URL http://127.0.0.1:8000
r_get = requests.get("http://127.0.0.1:8000")

# TODO: print the status code
print(f"GET Status Code: {r_get.status_code}")

# TODO: print the welcome message
print(f"GET Response Body: {r_get.json()}")



data = {
    "age": 37,
    "workclass": "Private",
    "fnlgt": 178356,
    "education": "HS-grad",
    "education-num": 10,
    "marital-status": "Married-civ-spouse",
    "occupation": "Prof-specialty",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States",
}

# TODO: send a POST using the data above
# We use the 'json' parameter to automatically set headers and encode the dict
r_post = requests.post("http://127.0.0.1:8000/data/", json=data)

# TODO: print the status code
print(f"POST Status Code: {r_post.status_code}")

# TODO: print the result
print(f"POST Prediction Result: {r_post.json()}")
