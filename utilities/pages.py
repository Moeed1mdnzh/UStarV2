import cv2
import numpy as np


class Design:
    def __init__(self):
        self.size = (760, 1280)
        self.brush_image = cv2.resize(cv2.imread("assets/brush.png"), (90, 90))
        self.eraser_image = cv2.resize(
            cv2.imread("assets/eraser.png"), (90, 90))
        self.shuffle_image = cv2.resize(
            cv2.imread("assets/shuffle.png"), (90, 90))
        self.palette = cv2.resize(cv2.imread("assets/palette.png"), (70, 760))
        self.drawing = False
        self.pen_size = 5
        self.pen_color = (200, 100, 100)
        self.prev_pen_color = (200, 100, 100)
        self.bg = np.hstack(
            [np.full((self.size[0], self.size[1], 3), (200, 200, 200), np.uint8), self.palette])
        self.erasing = False
        self.brushing = False
        self.shuffle = False
        self.to_shuffle = False
        self.generate = False
        self.clear = False
        self.quit = False
        self.super_resolute = False

    def show(self, bg):
        halves = self.size[0]//2, self.size[1]//2
        cv2.rectangle(bg, (int(halves[1]-203), int(halves[0]-203)),
                      (int(halves[1]+203), int(halves[0]+203)), (64, 38, 5), 19)
        cv2.imshow("UStar-V2", self.bg)
        if cv2.waitKey(1) == ord("q") or cv2.waitKey(1) == ord("Q"):
            cv2.destroyAllWindows()
            quit()

    def sliders(self):
        cv2.setTrackbarMin("Pen Size: ", "UStar-V2", 1)
        cv2.setTrackbarMax("Pen Size: ", "UStar-V2", 150)
        return cv2.getTrackbarPos("Pen Size: ", "UStar-V2")

    def drawing_window(self, bg):
        halves = self.size[0]//2, self.size[1]//2
        cv2.rectangle(bg, (int(halves[1]-192), int(halves[0]-192)),
                      (int(halves[1]+192), int(halves[0]+192)), (0, 0, 0), -1)

    def tools(self, bg):
        bg[175: 175+90, 1174: 1264] = self.brush_image
        bg[175+90+66: 175+90+66+90, 1174: 1264] = self.eraser_image
        bg[175+90+66+90+66: 175+90+66+90+66+90, 1174: 1264] = self.shuffle_image
        # cv2.rectangle(bg, (1110, 650), (1260, 710), (34, 8, 2), -1)
        # cv2.rectangle(bg, (1110, 650), (1260, 710), (64, 38, 5), 3)

    def buttons(self, bg):
        cv2.rectangle(bg, (16, -27+233), (206, -27+233+100), (34, 8, 2), -1)
        cv2.rectangle(bg, (16, -27+233+100+25),
                      (206, -27+233+100+25+100), (34, 8, 2), -1)
        cv2.rectangle(bg, (16, -27+233+100+25+100+25),
                      (206, -27+233+100+25+100+25+100), (34, 8, 2), -1)
        cv2.rectangle(bg, (16, -27+233), (206, -27+233+100), (64, 38, 5), 3)
        cv2.rectangle(bg, (16, -27+233+100+25),
                      (206, -27+233+100+25+100), (64, 38, 5), 3)
        cv2.rectangle(bg, (16, -27+233+100+25+100+25),
                      (206, -27+233+100+25+100+25+100), (64, 38, 5), 3)
        cv2.putText(bg, "Generate", (35, 267),
                    cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255), 2)
        cv2.putText(bg, "Clear", (63, 393),
                    cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255), 2)
        cv2.putText(bg, "Quit", (72, 517),
                    cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255), 2)
        cv2.rectangle(bg, (16, 650), (56, 690), (64, 38, 5), 3)
        cv2.putText(bg, "Super-Resolute", (66, 680),
                    cv2.FONT_HERSHEY_TRIPLEX, 0.8, (64, 38, 5), 2)

    def check_resolute(self, x, y):
        if x >= 16 and x <= 56:
            if y >= 650 and y <= 690:
                self.super_resolute = not self.super_resolute

    def check_quit(self, x, y):
        if x >= 16 and x <= 206:
            if y >= -27+233+100+25+100+25 and y <= -27+233+100+25+100+25+100:
                print(True)
                self.quit = True

    def check_clear(self, x, y):
        if x >= 16 and x <= 206:
            if y >= -27+233+100+25 and y <= -27+233+100+25+100:
                self.clear = True

    def check_generate(self, x, y):
        if x >= 16 and x <= 206:
            if y >= -27+233 and y <= -27+233+100:
                self.generate = True

    def check_eraser(self, x, y):
        if x >= 1174 and x <= 1264:
            if y >= 175+90+66 and y <= 175+90+66+90:
                self.pen_color = (0, 0, 0)
                self.erasing = True
                self.to_shuffle = False

    def check_brush(self, x, y):
        if x >= 1174 and x <= 1264:
            if y >= 175 and y <= 175+90:
                self.pen_color = self.prev_pen_color
                self.brushing = True
                self.to_shuffle = False

    def check_drawing(self, x, y):
        if x >= 640-192 and x <= 640+192:
            if y >= 380-192 and y <= 380+192:
                self.drawing = True
                return True
        return False

    def check_shuffle(self, x, y):
        if x >= 1174 and x <= 1264:
            if y >= 175+90+66+90+66 and y <= 175+90+66+90+66+90:
                self.shuffle = True
                self.to_shuffle = True

    def check_color(self, x, y):
        if x >= 1280 and x <= 1280+70:
            if y >= 0 and y <= 760:
                self.pen_color = self.bg[y, x]
                self.pen_color = (int(self.pen_color[0]),
                                  int(self.pen_color[1]),
                                  int(self.pen_color[2]))

    def check_events(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.check_color(x, y)
            self.check_eraser(x, y)
            self.check_brush(x, y)
            self.check_shuffle(x, y)
            self.check_generate(x, y)
            self.check_clear(x, y)
            self.check_quit(x, y)
            self.check_resolute(x, y)
            if self.check_drawing(x, y):
                if self.to_shuffle:
                    for _ in range(5):
                        cv2.circle(self.bg, (int(x+np.random.randint(-self.pen_size, self.pen_size+1)),
                                             int(y+np.random.randint(-self.pen_size, self.pen_size+1))),
                                   1, self.pen_color, -1)
                else:
                    cv2.circle(self.bg, (x, y), self.pen_size,
                               self.pen_color, -1)

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing == True:
                if self.check_drawing(x, y):
                    if self.to_shuffle:
                        for _ in range(5):
                            cv2.circle(self.bg, (int(x+np.random.randint(-self.pen_size, self.pen_size+1)),
                                                 int(y+np.random.randint(-self.pen_size, self.pen_size+1))),
                                       1, self.pen_color, -1)
                    else:
                        cv2.circle(self.bg, (x, y), self.pen_size,
                                   self.pen_color, -1)

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False

    def design(self):
        return self.bg


class Finalize:
    def __init__(self):
        self.size = (760, 1280)
        self.bg = np.full(
            (self.size[0], self.size[1], 3), (200, 200, 200), np.uint8)

    def show(self, bg, sample):
        halves = self.size[0]//2, self.size[1]//2
        bg[halves[0]-192: halves[0]+192, halves[1]-192:halves[1]+192] = sample
        cv2.rectangle(bg, (int(halves[1]-203), int(halves[0]-203)),
                      (int(halves[1]+203), int(halves[0]+203)), (64, 38, 5), 21)
        cv2.imshow("UStar-V2", self.bg)
        if cv2.waitKey(1) == ord("q") or cv2.waitKey(1) == ord("Q"):
            cv2.destroyAllWindows()
            quit()

    def text_placement(self, bg):
        cv2.putText(bg, "Press", (60, 380-203+100),
                    cv2.FONT_HERSHEY_TRIPLEX, 1.7, (20, 20, 20), 6)
        cv2.putText(bg, "Press", (60, 380-203+100),
                    cv2.FONT_HERSHEY_TRIPLEX, 1.7, (64, 38, 5), 3)
        cv2.putText(bg, "- S to save", (60, 230+100),
                    cv2.FONT_HERSHEY_TRIPLEX, 1.1, (20, 20, 20), 7)
        cv2.putText(bg, "- S to save", (60, 230+100),
                    cv2.FONT_HERSHEY_TRIPLEX, 1.1, (15, 100, 15), 3)
        cv2.putText(bg, "- R to Return", (60, 300+100),
                    cv2.FONT_HERSHEY_TRIPLEX, 1.1, (20, 20, 20), 7)
        cv2.putText(bg, "- R to Return", (60, 300+100),
                    cv2.FONT_HERSHEY_TRIPLEX, 1.1, (15, 100, 100), 3)
        cv2.putText(bg, "- Q to Quit", (60, 370+100),
                    cv2.FONT_HERSHEY_TRIPLEX, 1.1, (20, 20, 20), 7)
        cv2.putText(bg, "- Q to Quit", (60, 370+100),
                    cv2.FONT_HERSHEY_TRIPLEX, 1.1, (15, 15, 100), 3)
        cv2.putText(bg, "Your Imaginary", (900, 380-203+100),
                    cv2.FONT_HERSHEY_TRIPLEX, 1.2, (20, 20, 20), 6)
        cv2.putText(bg, "Your Imaginary", (900, 380-203+100),
                    cv2.FONT_HERSHEY_TRIPLEX, 1.2, (64, 38, 5), 3)
        cv2.putText(bg, "Your Star", (970, 230+100),
                    cv2.FONT_HERSHEY_TRIPLEX, 1.2, (20, 20, 20), 8)
        cv2.putText(bg, "Your Star", (970, 230+100),
                    cv2.FONT_HERSHEY_TRIPLEX, 1.2, (100, 5, 64), 2)

    def check_events(self):
        order = None
        event = cv2.waitKey(1)
        if event == ord("q") or event == ord("Q"):
            order = "quit"
        elif event == ord("r") or event == ord("R"):
            order = "return"
        elif event == ord("s") or event == ord("S"):
            order = "save"
        return order

    def preview(self):
        return self.bg
