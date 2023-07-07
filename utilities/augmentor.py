import cv2
import numpy as np

class Augmentor:
    def __init__(self, limit=0.4):
        self.limit = limit

    def rotate(self, image):
        res = []
        image_center = tuple(np.array(image.shape[:2][::-1]) / 2)
        for angle in np.arange(10, 211, 100):
            rotated = cv2.getRotationMatrix2D(image_center, angle, 1.0)
            res.append(cv2.warpAffine(image, rotated, image.shape[:2][::-1], flags=cv2.INTER_LINEAR))
        return res

    def zoom_out(self, image):
        H, W = image.shape[:2]
        res = []
        for rate in [0.75, 0.5]:
            rW, rH = int(rate * W), int(rate * H)
            resized = cv2.resize(image, (rW, rH))
            blank = np.zeros((H, W, 3), np.uint8)
            blank[(H//2)-(rH//2): (H//2)+(rH//2),
                (W//2)-(rW//2): (W//2)+(rW//2)] = resized
            res.append(blank)
        return res

    def transfer(self, image, limit):
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
    
    def augment(self, image):
        rotated = self.rotate(image)
        zoomed_out = self.zoom_out(image)
        transferred = self.transfer(image, self.limit)
        return [[image], rotated, zoomed_out, transferred]
    
    
    
# if __name__ == "__main__":
#     image = cv2.imread("test.jpg")
#     augmentor = Augmentor(image, 0.4)
#     raw = augmentor.augment()
#     images = []
#     for r in raw:
#         for img in r:
#             images.append(img)
#     for img in images:
#         cv2.imshow("", img)
#         cv2.waitKey(0)
#     print(len(images))