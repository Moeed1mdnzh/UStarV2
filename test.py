import cv2 
import numpy as np

image = cv2.imread("star2.jpg")
image = cv2.resize(image, (192, 192))
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imwrite("data3.jpg", image)