import cv2
import numpy as np
import matplotlib.pyplot as plt

class ParkingPlacesDetector():

    def __init__(self):
        pass

    def detect(self, img):

        gray = cv2.cvtColor(src=img, code=cv2.COLOR_RGB2GRAY)

        blur_gray = cv2.GaussianBlur(src=gray, ksize=(5, 5), sigmaX=0)
        edges = cv2.Canny(image=blur_gray, threshold1=50, threshold2=150, apertureSize=3)

        cnts = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        canvas = np.zeros(img.shape,dtype=np.uint8)
        canvas.fill(255)
        radius = 2
        color = (0,0,0)
        cv2.drawContours(canvas, cnts, -1, color , radius)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
        canvas = cv2.morphologyEx(canvas, cv2.MORPH_OPEN, kernel)

        return canvas, 0
