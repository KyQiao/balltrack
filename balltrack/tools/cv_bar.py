import cv2
import numpy as np
import matplotlib.pyplot as plt


def nothing(x):
    pass


cv2.namedWindow('image')

# create trackbars for color change
cv2.createTrackbar('1st', 'image', 0, 100, nothing)
cv2.createTrackbar('2nd', 'image', 0, 20, nothing)
cv2.createTrackbar('3rd', 'image', 1, 10, nothing)
while(1):
    k = cv2.waitKey(0) & 0xFF
    if k == 27:
        break

    # get current positions of four trackbars
    r = cv2.getTrackbarPos('1st', 'image')
    g = cv2.getTrackbarPos('2nd', 'image')
    b = cv2.getTrackbarPos('3rd', 'image')
    img = cv2.imread('008.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, img = cv2.threshold(img, 150, 255, cv2.THRESH_TOZERO)
    kernel1 = np.ones((4, 4), np.uint8)
    kernel2 = np.ones((5, 5), np.uint8)
    # img = cv2.erode(img, kernel1, iterations=b)
    # img = cv2.dilate(img, kernel2, iterations=b)
    for i in range(b):
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, (7,7))
    gray = cv2.GaussianBlur(img, (3, 3), 1, 1)
    # gray = cv2.Canny(gray, r, g)
    try:
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, 1, 5, param1=r, param2=g, minRadius=3, maxRadius=15)
        for circle in circles[0]:
            x = int(circle[0])
            y = int(circle[1])
            r = int(circle[2]) - 3
            if r <= 7:
                img = cv2.circle(img, (x, y), r, (0, 0, 255), -1)
            else:
                img = cv2.circle(img, (x, y), r, (0, 255, 0), -1)
        pass
    except Exception as e:
        pass
    else:
        pass
    finally:
        pass

    cv2.imshow('image', img)


cv2.destroyAllWindows()
