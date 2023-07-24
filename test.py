import os 
import cv2 
import progressbar
import numpy as np

image = cv2.imread(os.sep.join(["dataset", "data2.jpg"]))
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray[gray==26] = 0
thresh, bins = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
kernel = np.ones((1, 1), np.uint8)
bins = cv2.morphologyEx(bins, cv2.MORPH_CLOSE, kernel)
kernel = np.ones((5, 5), np.uint8)
dilated = cv2.dilate(bins, kernel)
eroded = cv2.erode(dilated, kernel)
cv2.imshow("", np.hstack([bins, eroded, dilated]))
cv2.waitKey(0)

#### Find the biggest contour   Create a circle around it   Draw a filled circle from the cirlce informations