import os 
import cv2 
import progressbar
import numpy as np
from datetime import datetime
from utilities.augmentor import Augmentor
from configs import SHIFT_LIMIT, widgets_2, DATA_NAME

os.system("mkdir " + os.sep.join(["dataset", "images"]))
os.system("mkdir " + os.sep.join(["dataset", "labels"]))

image = cv2.imread(os.sep.join(["dataset", DATA_NAME]))
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray[gray==26] = 0
thresh, bins = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
kernel = np.ones((1, 1), np.uint8)
bins = cv2.morphologyEx(bins, cv2.MORPH_CLOSE, kernel)
kernel = np.ones((5, 5), np.uint8)
dilated = cv2.dilate(bins, kernel)
eroded = cv2.erode(dilated, kernel)
index = 0 # For testing
augmentor = Augmentor(SHIFT_LIMIT) 
pbar = progressbar.ProgressBar(max_value=3*8*8*126, widgets=widgets_2)
for i in range(3):
    channels = [0, 1, 2]
    channels.remove(i)
    clone = gray.copy()
    clone = cv2.cvtColor(clone, cv2.COLOR_GRAY2BGR)
    clone[:, :, i] = gray
    for j in range(0, 186, 25):
        clone[:, :, channels[0]] = j
        for k in range(0, 186, 25):
            clone[:, :, channels[1]] = k
            for l in range(1, 72, 25):
                bg = np.ones(image.shape, np.uint8) * l 
                sample_1 = cv2.add(clone, bg)
                sample_1[bins==0] = 0
                bg_1 = np.zeros(image.shape, np.uint8)
                mean = (np.uint8(np.mean(sample_1[:, :, 0][eroded!=0])),
                        np.uint8(np.mean(sample_1[:, :, 1][eroded!=0])),
                        np.uint8(np.mean(sample_1[:, :, 2][eroded!=0])))
                bg_1[eroded==255] = mean
                raw_sample = augmentor.augment(bg_1)
                raw_label = augmentor.augment(sample_1)
                index_2 = index
                for cluster in raw_sample:
                    for img in cluster:
                        cv2.imwrite(os.sep.join(["dataset", "images", f"sample_{index}.jpg"]), img)
                        index += 1
                        pbar.update(index,ImageNumber=index)
                for cluster in raw_label:
                    for img in cluster:
                        cv2.imwrite(os.sep.join(["dataset", "labels", f"label_{index_2}.jpg"]), img)
                        index_2 += 1
            for m in range(1, 151, 25):               
                bg = np.ones(image.shape, np.uint8) * m
                sample_2 = cv2.subtract(clone, bg)
                sample_2[bins==0] = 0
                bg_2 = np.zeros(image.shape, np.uint8)
                mean = (np.uint8(np.mean(sample_2[:, :, 0][eroded!=0])),
                        np.uint8(np.mean(sample_2[:, :, 1][eroded!=0])),
                        np.uint8(np.mean(sample_2[:, :, 2][eroded!=0])))
                bg_2[eroded==255] = mean
                raw_sample = augmentor.augment(bg_2)
                raw_label = augmentor.augment(sample_2)
                index_2 = index
                for cluster in raw_sample:
                    for img in cluster:
                        cv2.imwrite(os.sep.join(["dataset", "images", f"sample_{index}.jpg"]), img)
                        index += 1
                        pbar.update(index,ImageNumber=index)
                for cluster in raw_label:
                    for img in cluster:
                        cv2.imwrite(os.sep.join(["dataset", "labels", f"label_{index_2}.jpg"]), img)
                        index_2 += 1
pbar.finish()
print(f"Total number of generated images: {index+index_2}")

    