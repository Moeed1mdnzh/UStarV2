import cv2
import numpy as np

def mirror(image):
    return [cv2.flip(image, i) for i in [0, 1]]

def lighting(image, rate):
    amount = np.ones(image.shape, np.uint8) * int(100*rate)
    bright = cv2.add(image, amount)
    dark = cv2.subtract(image, amount)
    return [bright, dark]

def rotate(image):
    res = []
    image_center = tuple(np.array(image.shape[:2][::-1]) / 2)
    for angle in np.arange(10, 351, 45):
        rotated = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        res.append(cv2.warpAffine(image, rotated, image.shape[:2][::-1], flags=cv2.INTER_LINEAR))
    return res

def zoom_out(image):
    H, W = image.shape[:2]
    res = []
    for rate in [0.75, 0.5, 0.25]:
        rW, rH = int(rate * W), int(rate * H)
        resized = cv2.resize(image, (rW, rH))
        blank = np.zeros((H, W, 3), np.uint8)
        blank[(H//2)-(rH//2): (H//2)+(rH//2),
              (W//2)-(rW//2): (W//2)+(rW//2)] = resized
        res.append(blank)
    return res

def transfer(image, limit):
    vol = limit * 100
    H, W = image.shape[:2]
    res = []
    pts = [np.float32([[1, 0, vol],[0, 1, vol]]),
            np.float32([[1, 0, -vol], [0, 1, vol]]), 
            np.float32([[1, 0, -vol], [0, 1, -vol]]), 
            np.float32([[1, 0, vol], [0, 1, -vol]]), 
            np.float32([[1, 0, 0], [0, 1, vol]]), 
            np.float32([[1, 0, vol], [0, 1, 0]]), 
            np.float32([[1, 0, -vol], [0, 1, 0]]), 
            np.float32([[1, 0, 0], [0, 1, -vol]])]
    for pt in pts:
        res.append(cv2.warpAffine(image, pt, (W, H)))
    return res
