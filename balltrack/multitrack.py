from .balltrack import balltrack
import cv2
import imutils
import numpy as np


class multitrack(balltrack):
    """docstring for multitrack"""

    def __init__(self, file, setting={}):
        balltrack.__init__(self, file, setting=setting)
        assert "objNumber" in self.setting, print("set objNumber in dict")
        self.trace = np.zeros(
            [self.setting["totalFrame"]-1, 4,
             self.setting["objNumber"]], dtype=np.int16)
        self.t = 0
        # leave the root color range as the global setting

    def label(self):
        self.save({"class": "multitrack"})

    def drwaInfo(self, H, frame):
        info = [
            ("FPS", "{:.2f}".format(self.fps.fps())),
        ]

        # loop over the info tuples and draw them on our frame
        for (i, (k, v)) in enumerate(info):
            text = "{}: {}".format(k, v)
            cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    def trackparaInit(self):
        if "trackers" in self.setting:
            _tmp_dict = self.setting["trackers"]
            self.trackerList = [_tmp_dict[str(i)]["tracker"]
                                for i in range(self.setting["objNumber"])]
        else:
            # using csrt by default
            self.trackerList = ["csrt" for i in range(
                self.setting["objNumber"])]
            _tmp_dict = {str(i): {"tracker": "csrt"}
                         for i in range(self.setting["objNumber"])}
            self.save({"trackers": _tmp_dict})

    def trackInit(self, cap, tracker):
        OPENCV_OBJECT_TRACKERS = {
            "csrt": cv2.TrackerCSRT_create,
            "kcf": cv2.TrackerKCF_create,
            "boosting": cv2.TrackerBoosting_create,
            "mil": cv2.TrackerMIL_create,
            "tld": cv2.TrackerTLD_create,
            "medianflow": cv2.TrackerMedianFlow_create,
            "mosse": cv2.TrackerMOSSE_create}
        self.trackparaInit()
        _tmp_dict = self.setting["trackers"]
        ret, frame = cap.read()
        frame = imutils.resize(
            frame, width=self.setting["size"][0]//self.setting["resize"])
        cv2.imshow('frame', frame)
        for i in range(self.setting["objNumber"]):
            trackerName = self.trackerList[i]
            if "initBB" not in _tmp_dict[str(i)]:
                initBB = cv2.selectROI("frame", frame, fromCenter=False,
                                       showCrosshair=True)
                _tmp_dict[str(i)].update({"initBB": initBB})
            else:
                initBB = tuple(_tmp_dict[str(i)]["initBB"])
            tracker.add(OPENCV_OBJECT_TRACKERS[trackerName](), frame, initBB)
        self.save({"trackers": _tmp_dict})
        self.saveSettings()

    def HoughTransform(self, frame, imCrop, HoughPara, x, y, saveData=True):
        gray = cv2.cvtColor(imCrop, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        balls = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, 1,  **HoughPara)
        if balls is None:
            pass
        else:
            for ball in balls[0]:
                x_ball = int(ball[0])
                y_ball = int(ball[1])
                r = int(ball[2])
                frame = cv2.circle(
                    frame, (x+x_ball, y+y_ball),
                    2, (0, 0, 255), -1)
                if saveData:
                    self.trace[self.t, :, self.i] = [
                        x+x_ball, y+y_ball, r, self.t]

    def process(self):
        # skiptime
        cap = self.timeskip()

        # creata tracker

        tracker = cv2.MultiTracker_create()

        # start track
        self.trackInit(cap, tracker)

        # parameter for houghcircle
        if "HoughPara" in self.setting:
            HoughPara = self.setting["HoughPara"]
        else:
            HoughPara = {'minDist': 100,
                         'param1': 10,
                         'param2': 20,
                         'minRadius': 7,
                         'maxRadius': 20}
            self.save({"HoughPara": HoughPara})

        feature = self.methodInit()
        self.fpsInit()

        while(cap.isOpened()):
            # Capture frame-by-frame
            ret, frame = cap.read()
            if not ret:
                break

            frame = self.resize(frame)
            (H, W) = frame.shape[:2]

            frame = self.preprocess(frame)

            (success, boxes) = tracker.update(frame)
            if success:
                self.fpsUpdate()
                for index, box in enumerate(boxes):
                    (x, y, w, h) = [int(v) for v in box]

                    imCrop = frame[y: y+h, x:x+w]
                    imCrop = self.output(imCrop, None)
                    cv2.rectangle(frame, (x, y), (x + w, y + h),
                                  (0, 255, 0), 2)
                    self.i = index
                    feature(frame, imCrop, HoughPara, x, y)
                    self.drwaInfo(H, frame)

            cv2.imshow('frame', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

            self.t += 1
        self.label()
        self.saveSettings()
        cap.release()
        cv2.destroyAllWindows()
