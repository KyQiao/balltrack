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

        if "HoughPara" in self.setting:
            HoughPara = self.setting["HoughPara"]
            cv2.createTrackbar('minDist', 'frame',
                               HoughPara["param1"], 1000, nothing)
            cv2.createTrackbar('threshold1', 'frame',
                               HoughPara["param1"], 100, nothing)
            cv2.createTrackbar('threshold2', 'frame',
                               HoughPara["param2"], 100, nothing)
            cv2.createTrackbar('maxRadius', 'frame',
                               HoughPara["maxRadius"], 100, nothing)
            cv2.createTrackbar('minRadius', 'frame',
                               HoughPara["minRadius"], 100, nothing)

        else:
            cv2.createTrackbar('minDist', 'frame', 50, 1000, nothing)
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
            frame = self.preprocess(frame)
            minDist = cv2.getTrackbarPos('minDist', 'frame')
            param1 = cv2.getTrackbarPos('threshold1', 'frame')
            param2 = cv2.getTrackbarPos('threshold2', 'frame')
            minRadius = cv2.getTrackbarPos('minRadius', 'frame')
            maxRadius = cv2.getTrackbarPos('maxRadius', 'frame')

            HoughPara = {
                "minDist":minDist,
                "param1": param1,
                "param2": param2,
                "minRadius": minRadius,
                "maxRadius": maxRadius
            }

            feature = self.methodInit()
            self.fpsInit()

            (success, box) = tracker.update(frame)
            if success:
                (x, y, w, h) = [int(v) for v in box]
                cv2.rectangle(frame, (x, y), (x + w, y + h),
                              (0, 255, 0), 2)
                self.fpsUpdate()
                self.drwaInfo(H, frame)
                imCrop = frame[y: y+h, x:x+w]
                feature(frame, imCrop, HoughPara, x, y, saveData=False)

            cv2.imshow('frame', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.save({"HoughPara": HoughPara})
                self.saveSettings()
                break

        self.save({"HoughPara": HoughPara})
        self.saveSettings()
        cap.release()
        cv2.destroyAllWindows()
