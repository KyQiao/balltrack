'''
trakcer options
{"csrt","kcf","boosting","mil","tld","medianflow","mosse"}
'''
from .videoProcessor import videoProcessor
import cv2
import imutils


class balltrack(videoProcessor):
    """basic class for normal balltrack"""

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
        self.setting
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

    def balltracklabel(self):
        self.save({"class": "balltrack"})

    def drwaInfo(self, H,frame):
        info = [
            ("Tracker", self.setting["tracker"]),
            ("FPS", "{:.2f}".format(self.fps.fps())),
        ]

        # loop over the info tuples and draw them on our frame
        for (i, (k, v)) in enumerate(info):
            text = "{}: {}".format(k, v)
            cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    def process(self):
        # skiptime
        cap = self.timeskip()

        # creata tracker
        tracker = self.trackparaInit()

        # start track
        self.trackInit(cap, tracker)

        # parameter for houghcircle
        if "para" in self.setting:
            para = self.setting["para"]
        else:
            para = {'param1': 10,
                    'param2': 20,
                    'minRadius': 7,
                    'maxRadius': 20}
            self.save({"para": para})

        self.fpsInit()
        trace = []
        t = 0
        while(cap.isOpened()):
            # Capture frame-by-frame
            ret, frame = cap.read()
            if frame is None:
                break

            frame = self.resize(frame)
            (H, W) = frame.shape[:2]

            (success, box) = tracker.update(frame)
            if success:
                (x, y, w, h) = [int(v) for v in box]
                cv2.rectangle(frame, (x, y), (x + w, y + h),
                              (0, 255, 0), 2)

                self.fpsUpdate()
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
                        trace.append([x+x_ball, y+y_ball, r, t])
                self.drwaInfo(H,frame)

            cv2.imshow('frame', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

            t += 1
        self.balltracklabel()
        self.saveSettings()
        cap.release()
        cv2.destroyAllWindows()
