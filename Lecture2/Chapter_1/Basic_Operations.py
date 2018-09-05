import cv2
import numpy as np

img = cv2.imread('hammer.jpg')

print img.shape

print img.size

print img.dtype

hammer_head = img[360:450, 250:350]
img[360:450, 440:540] = hammer_head

while(1):

    cv2.imshow('frame', img)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows
