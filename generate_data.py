import os
import cv2
import numpy as np

image = cv2.imread(os.sep.join(["data", "org.png"]))
clone = image.copy()
blurred = cv2.GaussianBlur(image, (5, 5), 0)
gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
thresh, bins = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
kernel = np.ones((5, 5), np.uint8)
bins = cv2.morphologyEx(bins, cv2.MORPH_CLOSE, kernel)

b = 0
index = 1
SZ = 17
for i in range(int(255 / SZ)):
    g = 0
    for j in range(int(255 / SZ)):
        r = 0
        for d in range(int(255 / SZ)):
            color = [b, g, r]
            clone[bins==255] = color
            clone[bins!=255] = [0, 0, 0]
            if clone.sum() > 45000000:
                bg = np.full(image.shape, color, dtype=np.uint8)
                bg[bins!=255] = [0, 0, 0]
                image[bins!=255] = [0, 0, 0]
                maximum = np.max(image[bins==255], axis=0)
                deficit = np.full(image.shape, maximum, dtype=np.float32) - image.astype("float32")
                res = clone.astype("float32")-deficit
                res[res < 0] = 0
                res[res > 255] = 255
                res = res.astype("uint8")
                cv2.imwrite(os.sep.join(["data", "images", f"img_{index}.jpg"]), clone)
                cv2.imwrite(os.sep.join(["data", "labels", f"img_{index}.jpg"]), res)
                print(f"[INFO]: Created image {index}/813")
                index += 1
            r += SZ
        g += SZ
    b += SZ