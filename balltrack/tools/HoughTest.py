from ..balltrack import balltrack
import cv2


class HoughTest(balltrack):
    def getPara(self):
        if "skiptime" not in self.setting:
            self.save({"skiptime": 0})

        cap = self.timeskip()

        tracker = self.trackparaInit()

        self.trackInit(cap, tracker)

        def nothing(x):
            pass
        cv2.namedWindow('frame')
        cv2.createTrackbar('threshold1', 'frame', 2, 100, nothing)
        cv2.createTrackbar('threshold2', 'frame', 1, 100, nothing)
        cv2.createTrackbar('maxRadius', 'frame', 2, 100, nothing)
        cv2.createTrackbar('minRadius', 'frame', 1, 100, nothing)

        while(cap.isOpened()):
            # Capture frame-by-frame
            ret, frame = cap.read()
            if frame is None:
                break
            frame = self.resize(frame)
            (H, W) = frame.shape[:2]

            param1 = cv2.getTrackbarPos('threshold1', 'frame')
            param2 = cv2.getTrackbarPos('threshold2', 'frame')
            minRadius = cv2.getTrackbarPos('minRadius', 'frame')
            maxRadius = cv2.getTrackbarPos('maxRadius', 'frame')

            para = {
                "param1": param1,
                "param2": param2,
                "minRadius": minRadius,
                "maxRadius": maxRadius
            }
            self.fpsInit()
            (success, box) = tracker.update(frame)
            if success:
                (x, y, w, h) = [int(v) for v in box]
                cv2.rectangle(frame, (x, y), (x + w, y + h),
                              (0, 255, 0), 2)
                self.fpsUpdate()
                self.drwaInfo(H, frame)
                imCrop = frame[y: y+h, x:x+w]
                gray = cv2.cvtColor(imCrop, cv2.COLOR_BGR2GRAY)
                gray = cv2.medianBlur(gray, 5)
                balls = cv2.HoughCircles(
                    gray, cv2.HOUGH_GRADIENT, 1, 100, **para)
                if balls is None:
                    continue
                else:
                    for ball in balls[0]:
                        x_ball = int(ball[0])
                        y_ball = int(ball[1])
                        r = int(ball[2])
                        frame = cv2.circle(
                            frame, (x+x_ball, y+y_ball),
                            2, (0, 0, 255), -1)

            cv2.imshow('frame', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.save({"para": para})
                self.saveSettings()
                break
        self.save({"para": para})
        self.saveSettings()
        cap.release()
        cv2.destroyAllWindows()
