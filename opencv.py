import cv2
import numpy as np

img = cv2.imread("eye.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

blur = cv2.GaussianBlur(gray, (7,7), 0)

circles = cv2.HoughCircles(
    blur,
    cv2.HOUGH_GRADIENT,
    1.2,
    30,
    param1=50,
    param2=20,
    minRadius=5,
    maxRadius=30
)

points = []

if circles is not None:
    circles = np.uint16(np.around(circles))

    for c in circles[0, :2]:
        x, y, r = c
        points.append((x, y))
        cv2.circle(img, (x, y), r, (0,255,0), 2)

if len(points) == 2:
    x1, y1 = points[0]
    x2, y2 = points[1]

    dist = np.sqrt((x2-x1)**2 + (y2-y1)**2)
    print("兩眼距離:", dist)

cv2.imshow("result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
