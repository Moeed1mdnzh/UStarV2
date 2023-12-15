import cv2
import numpy as np 
from utilities import tools, pages
from inference import Inference

class Integrate:
    def __init__(self):
        self.design = pages.Design()
        
    def run(self):
        cv2.namedWindow("UStar-V2", cv2.WINDOW_AUTOSIZE)
        cv2.createTrackbar("Pen Size: ", "UStar-V2", 5, 20, lambda x: None)
        cv2.resizeWindow("UStar-V2", self.design.size[0], self.design.size[1])
        self.design.drawing_window(self.design.bg)
        while True:
            self.design.bg = self.design.design()
            self.design.pen_size = self.design.sliders()
            self.design.tools(self.design.bg)
            self.design.buttons(self.design.bg)
            cv2.setMouseCallback("UStar-V2", self.design.check_events)
            self.design.show(self.design.bg)


def main():
    integrate = Integrate()
    integrate.run()    
    
if __name__ == "__main__":
    main()