# -*- coding: utf-8 -*-
import cv2
import numpy as np

img1 = cv2.imread('MyPic.png', -1)
img2 = cv2.imread('MyPic.png', 0)
img3 = cv2.imread('MyPic.png', 1)

cv2.imshow('image1', img1)
cv2.imshow('image2', img2)
cv2.imshow('image3', img3)
cv2.waitKey(0) & 0xFF
cv2.destroyAllWindows()
