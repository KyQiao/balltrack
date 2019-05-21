from videoProcess import videoProcessor
import cv2
import imutils


class balltrack(videoProcessor):
    """docstring for balltrack"""

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
            self.setting.update({"tracker": "csrt"})
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
            self.setting.update({"initBB": initBB})
        else:
            initBB = tuple(self.setting["initBB"])
        tracker.init(frame, initBB)

    def balltracklabel(self):
        self.setting.update({"class": "balltrack"})

    def Houghtest():
        pass

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
            self.setting.update({"para": para})

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

            cv2.imshow('frame', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

            t += 1
        self.balltracklabel()
        self.saveSettings()


if __name__ == '__main__':
    '''
    trakcer options
    {"csrt","kcf","boosting","mil","tld","medianflow","mosse",}
    '''

    setting = {"skiptime": 5,
               "tracker": "csrt",
               "feature": "Hough",
               "fps": 25,
               "size": (1920, 1080),
               "resize": 4
               }
    # test = videoProcessor(r'C:\Users\kyqiao\Desktop\ball\MVI_0517.MP4')

    test = balltrack(
        r'C:\Users\kyqiao\Desktop\ball\MVI_0517.MP4', setting=setting)

    # print(test.outputSetting())
    test.process()

# ref https://www.pyimagesearch.com/2018/07/30/opencv-object-tracking/
# yellow object http://aishack.in/tutorials/tracking-colored-objects-opencv/
