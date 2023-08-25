import os
import cv2
import sys
import jsonpickle
import numpy as np
cwd = os.getcwd()
sys.path.append(cwd.replace("api", ""))
from inference import Inference
from flask import Flask, request, Response

# Initialize the Flask application
app = Flask(__name__)
ustar_generator = Inference()
ustar_generator.initialize()


# route http posts to this method
@app.route("/api/test", methods=["POST"])
def inference():
    r = request
    # convert string of image data to uint8
    nparr = np.frombuffer(r.data, np.uint8)
    # decode image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # do some fancy processing here....
    image = ustar_generator.generate(img)
    cv2.imwrite(os.sep.join(["generated", "result.jpg"]), image)

    # build a response dict to send back to client
    response = {"message": "image received. size={}x{}".format(img.shape[1], img.shape[0])
                }
    # encode response using jsonpickle
    response_pickled = jsonpickle.encode(response)

    return Response(response=response_pickled, status=200, mimetype="application/json")


# start flask app
app.run(host="0.0.0.0", port=5000)
