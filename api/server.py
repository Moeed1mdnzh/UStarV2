from __future__ import print_function
import os
import cv2
import json
import requests

addr = "http://localhost:5000"
test_url = addr + "/api/test"

# prepare headers for http request
content_type = "image/jpeg"
headers = {"content-type": content_type}

img = cv2.imread(os.sep.join(["raw", "teststar4.jpg"]))
# encode image as jpeg
_, img_encoded = cv2.imencode(".jpg", img)
# send http request with image and receive response
response = requests.post(test_url, data=img_encoded.tostring(), headers=headers)
# decode response
print(json.loads(response.text))

# expected output: {u'message': u'image received. size=124x124'}
