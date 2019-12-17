'''
trakcer options
{"csrt","kcf","boosting","mil","tld","medianflow","mosse"}
'''
from .videoProcessor import videoProcessor
import cv2
import imutils
import numpy as np


class balltrack(videoProcessor):
    """basic class for normal balltrack"""

    def __init__(self, file, setting={}):
        videoProcessor.__init__(self, file, setting=setting)
        assert "totalFrame" in self.setting,print("emmm????")
        start_frame_number = self.setting["skiptime"]*self.setting["fps"]
        self.trace = np.zeros([self.setting["totalFrame"]-start_frame_number, 4], dtype=np.int16)
        self.t = 0
        # You can change the range after initialize
        if "lower_range" in self.setting:
            self.lower_range = np.array(
                self.setting["lower_range"], dtype=np.uint8)
            self.upper_range = np.array(
                self.setting["upper_range"], dtype=np.uint8)
        else:
            self.lower_range = np.array([20, 100, 100], dtype=np.uint8)
            self.upper_range = np.array([30, 255, 255], dtype=np.uint8)
            self.setting.update(
                {"lower_range": [20, 100, 100], "upper_range": [30, 255, 255]})
        self.saveSettings()

    def preprocess(self, frame):
        # An api left for preprocess the image
        return frame

    def output(self,imCrop):
        # An api left for output the image
        return imCrop

    def trackparaInit(self):
        OPENCV_OBJECT_TRACKERS = {
            "csrt": cv2.TrackerCSRT_create,
            "kcf": cv2.TrackerKCF_create,
            "boosting": cv2.TrackerBoosting_create,
            "mil": cv2.TrackerMIL_create,
            "tld": cv2.TrackerTLD_create,
            "medianflow": cv2.TrackerMedianFlow_create,
            "mosse": cv2.TrackerMOSSE_create}
        if "tracker" in self.setting:
            tracker = OPENCV_OBJECT_TRACKERS[self.setting["tracker"]]()
        else:
            # using csrt by default
            self.save({"tracker": "csrt"})
            tracker = cv2.TrackerCSRT_create()
        return tracker

    def trackInit(self, cap, tracker):
        ret, frame = cap.read()
        frame = imutils.resize(
            frame, width=self.setting["size"][0]//self.setting["resize"])
        cv2.imshow('frame', frame)
        if "initBB" not in self.setting:
            initBB = cv2.selectROI("frame", frame, fromCenter=False,
                                   showCrosshair=True)
            self.save({"initBB": initBB})
        else:
            initBB = tuple(self.setting["initBB"])
        tracker.init(frame, initBB)

    def label(self):
        self.save({"class": "balltrack"})

    def drwaInfo(self, H, frame):
        info = [
            ("Tracker", self.setting["tracker"]),
            ("FPS", "{:.2f}".format(self.fps.fps())),
        ]

        # loop over the info tuples and draw them on our frame
        for (i, (k, v)) in enumerate(info):
            text = "{}: {}".format(k, v)
            cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

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
                    self.trace[self.t, :] = [x+x_ball, y+y_ball, r, self.t]

    def HoughTransform_Mask(self, frame, imCrop, HoughPara, x, y, saveData=True):
        hsv = cv2.cvtColor(imCrop, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_range, self.upper_range)

        balls = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, 1, **HoughPara)
        if balls is None:
            pass
        else:
            for ball in balls[0]:
                x_ball = int(ball[0])
                y_ball = int(ball[1])
                r = int(ball[2])
                frame = cv2.circle(
                    frame, (x+x_ball, y+y_ball), 2, (0, 0, 255), -1)
                if saveData:
                    self.trace[self.t, :] = [x+x_ball, y+y_ball, r, self.t]

    def methodInit(self):
        METHODS = {
            "HT": self.HoughTransform,
            "HTMask": self.HoughTransform_Mask}
        if "feature" in self.setting:
            feature = METHODS[self.setting["feature"]]
        else:
            # using csrt by default
            self.save({"feature": "HT"})
            feature = self.HoughTransform
        return feature

    def process(self):
        # skiptime
        cap = self.timeskip()

        # creata tracker
        tracker = self.trackparaInit()

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

            (success, box) = tracker.update(frame)
            if success:
                (x, y, w, h) = [int(v) for v in box]

                imCrop = frame[y: y+h, x:x+w]
                imCrop = self.output(imCrop)

                cv2.rectangle(frame, (x, y), (x + w, y + h),
                              (0, 255, 0), 2)
                self.fpsUpdate()
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
