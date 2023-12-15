import cv2 
import numpy as np 

class Design:
    def __init__(self):
        self.size = (760, 1280)
        self.brush_image = cv2.resize(cv2.imread("assets/brush.png"), (90, 90))
        self.eraser_image = cv2.resize(cv2.imread("assets/eraser.png"), (90, 90))
        self.shuffle_image = cv2.resize(cv2.imread("assets/shuffle.png"), (90, 90))
        self.palette = cv2.resize(cv2.imread("assets/palette.png"), (70, 760))
        self.drawing = False 
        self.pen_size = 5
        self.pen_color = (255, 255, 255)
        self.bg = np.hstack([np.full((self.size[0], self.size[1], 3), (200, 200, 200), np.uint8), self.palette])
    
    def show(self, bg):
        halves = self.size[0]//2, self.size[1]//2
        cv2.rectangle(bg, (int(halves[1]-205), int(halves[0]-205)), 
                      (int(halves[1]+205), int(halves[0]+205)), (64, 38, 5), 23)
        cv2.imshow("UStar-V2", self.bg)
        if cv2.waitKey(1) == ord("q") or cv2.waitKey(1) == ord("Q"):
            cv2.destroyAllWindows()
            quit()
            
    def sliders(self):
        cv2.setTrackbarMin("Pen Size: ", "UStar-V2", 5)
        cv2.setTrackbarMax("Pen Size: ", "UStar-V2", 20)
        return cv2.getTrackbarPos("Pen Size: ", "UStar-V2")
    
    def drawing_window(self, bg):
        halves = self.size[0]//2, self.size[1]//2
        cv2.rectangle(bg, (int(halves[1]-192), int(halves[0]-192)), 
                      (int(halves[1]+192), int(halves[0]+192)), (0, 0, 0), -1)
        
    def tools(self, bg):
        bg[175: 175+90, 1174: 1264] = self.brush_image
        bg[175+90+66: 175+90+66+90, 1174: 1264] = self.eraser_image
        bg[175+90+66+90+66: 175+90+66+90+66+90, 1174: 1264] = self.shuffle_image
        cv2.rectangle(bg, (1110, 650), (1260, 710), (34, 8, 2), -1)
        cv2.rectangle(bg, (1110, 650), (1260, 710), (64, 38, 5), 3)
        
    def buttons(self, bg):
        cv2.rectangle(bg, (16, -27+233), (206, -27+233+100), (34, 8, 2), -1)
        cv2.rectangle(bg, (16, -27+233+100+25), (206, -27+233+100+25+100), (34, 8, 2), -1)
        cv2.rectangle(bg, (16, -27+233+100+25+100+25), (206, -27+233+100+25+100+25+100), (34, 8, 2), -1)
        cv2.rectangle(bg, (16, -27+233), (206, -27+233+100), (64, 38, 5), 3)
        cv2.rectangle(bg, (16, -27+233+100+25), (206, -27+233+100+25+100), (64, 38, 5), 3)
        cv2.rectangle(bg, (16, -27+233+100+25+100+25), (206, -27+233+100+25+100+25+100), (64, 38, 5), 3)
        cv2.putText(bg, "Generate", (35, 267), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255), 2)
        cv2.putText(bg, "Clear", (63, 393), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255), 2)
        cv2.putText(bg, "Quit", (72, 517), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255), 2)
        cv2.rectangle(bg, (16, 650), (56, 690), (64, 38, 5), 3)
        cv2.putText(bg, "Super-Resolute", (66, 680), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (64, 38, 5), 2)
        
    def check_drawing(self, x, y):
        if x >= 640-192 and x <= 640+192:
            if y >= 380-192 and y <= 380+192:
                self.drawing = True 
                return True 
        return False
    
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
            if self.check_drawing(x, y):
                cv2.circle(self.bg, (x, y), self.pen_size, self.pen_color, -1)

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing == True: 
                if self.check_drawing(x, y):
                    cv2.circle(self.bg, (x, y), self.pen_size, self.pen_color, -1)
                
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
    
    def design(self):
        return self.bg

class Finalize:
    pass