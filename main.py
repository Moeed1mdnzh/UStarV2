import cv2
import numpy as np
from utilities import tools, pages
from inference import Inference


class Integrate:
    def __init__(self):
        self.design = pages.Design()

    def run(self, ):
        cv2.namedWindow("UStar-V2", cv2.WINDOW_AUTOSIZE)
        cv2.createTrackbar("Pen Size: ", "UStar-V2", 1, 20, lambda x: None)
        cv2.resizeWindow("UStar-V2", self.design.size[0], self.design.size[1])
        sample = np.zeros((384, 384, 3), dtype=np.uint8)
        self.design.drawing_window(self.design.bg)
        while True:
            self.design.bg = self.design.design()
            self.design.pen_size = self.design.sliders()
            if self.design.pen_color != (0, 0, 0):
                self.design.prev_pen_color = self.design.pen_color
            self.design.tools(self.design.bg)
            self.design.buttons(self.design.bg)
            cv2.setMouseCallback("UStar-V2", self.design.check_events)
            if self.design.erasing:
                self.design.bg[175+90+66: 175+90+66+90, 1174: 1264][self.design.bg[175 +
                                                                                   90+66: 175+90+66+90, 1174: 1264] != (200, 200, 200)] = 0
                self.design.erasing = False

            elif self.design.brushing:
                self.design.bg[175: 175+90, 1174: 1264][self.design.bg[175: 175 +
                                                                       90, 1174: 1264] != (200, 200, 200)] = 0
                self.design.brushing = False

            elif self.design.shuffle:
                self.design.bg[175+90+66+90+66: 175+90+66+90+66+90, 1174: 1264][self.design.bg[175 +
                                                                                               90+66+90+66: 175+90+66+90+66+90, 1174: 1264] != (200, 200, 200)] = 0
                self.design.shuffle = False

            elif self.design.generate:
                self.design.bg[-27+233: -27+233+100, 16:206] = cv2.bitwise_not(
                    self.design.bg[-27+233: -27+233+100, 16:206])
                sample = cv2.resize(
                    self.design.bg[380-192: 380+192, 640-192:640+192], (192, 192))
                self.design.generate = False
                break

            elif self.design.clear:
                self.design.bg[-27+233+100+25: -27+233+100+25+100, 16:206] = cv2.bitwise_not(
                    self.design.bg[-27+233+100+25: -27+233+100+25+100, 16:206])
                self.design.bg[380-192: 380+192, 640-192:640 +
                               192] = np.zeros((384, 384, 3), dtype=np.uint8)
                self.design.clear = False

            elif self.design.quit:
                self.design.bg[-27+233+100+25+100+25: -27+233+100+25+100+25+100, 16:206] = cv2.bitwise_not(
                    self.design.bg[-27+233+100+25+100+25: -27+233+100+25+100+25+100, 16:206])
                cv2.destroyAllWindows()
                quit()

            if self.design.super_resolute:
                cv2.rectangle(self.design.bg, (16, 650),
                              (56, 690), (34, 8, 2), -1)
                cv2.rectangle(self.design.bg, (16, 650),
                              (56, 690), (64, 38, 5), 3)

            elif not self.design.super_resolute:
                cv2.rectangle(self.design.bg, (16, 650),
                              (56, 690), (200, 200, 200), -1)
                cv2.rectangle(self.design.bg, (16, 650),
                              (56, 690), (64, 38, 5), 3)

            self.design.show(self.design.bg)
        cv2.destroyAllWindows()
        page_2 = pages.Finalize()
        ustar_generator = Inference()
        ustar_generator.initialize()
        image = ustar_generator.generate(sample, self.design.super_resolute)
        clone = cv2.resize(image.copy(), (384, 384))
        while True:
            bg = page_2.preview()
            page_2.text_placement(bg)
            event = page_2.check_events()
            if event == "quit":
                quit()
            elif event == "return":
                break
            elif event == "save":
                cv2.imwrite("result.png", image)
            page_2.show(bg, clone)
        integrate = Integrate()
        integrate.run()


def main():
    integrate = Integrate()
    integrate.run()


if __name__ == "__main__":
    main()
