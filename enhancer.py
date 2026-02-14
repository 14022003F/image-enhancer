import cv2
import numpy as np

def reduce_noise(img):
    return cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

def sharpen(img):
    kernel = np.array([[0,-1,0],
                       [-1,5,-1],
                       [0,-1,0]])
    return cv2.filter2D(img, -1, kernel)

def enhance_contrast(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl,a,b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

def upscale(img, scale=2):
    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
