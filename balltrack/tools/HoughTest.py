import cv2
import matplotlib.pyplot as plt

img = cv2.imread('ball.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.medianBlur(gray,5)
para = {'param1': 10,
        'param2': 30,
        'minRadius': 7,
        'maxRadius': 20}
balls = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 100, **para)

for ball in balls[0]:
    x = int(ball[0])
    y = int(ball[1])
    r = int(ball[2])
    img = cv2.circle(img, (x, y), r, (0, 0, 255), -1)
    plt.imshow(img)

plt.show()
