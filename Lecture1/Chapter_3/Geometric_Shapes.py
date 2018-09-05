import cv2
import numpy as np

# create a black image
img = np.zeros((512, 512, 3), np.uint8)

# draw a diagonal blue line with thickness of 5 px
cv2.line(img, (0,0), (511,511), (255, 0, 0), 5)

# draw a green rectangle
cv2.rectangle(img, (384, 0), (510, 128), (0, 255, 0), 3)

# draw a red circle
cv2.circle(img, (447, 63), 63, (0, 0, 255), -1)

# draw a blue ellipse
cv2.ellipse(img, (256, 256), (100, 50), 0, 0, 180, 255, -1)

# draw a polygon
points = np.array([[10, 5], [20, 30], [70, 20], [50, 10]], np.int32)
points = points.reshape((-1, 1, 2))
cv2. polylines(img, [points], True, (0, 255, 255))

# adding text
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img, 'OpenCV', (10, 500), font, 4, (255, 255, 255), 2, cv2.LINE_AA)

cv2.imshow('frame', img)
cv2.waitKey(5000)
cv2.destroyAllWindows
