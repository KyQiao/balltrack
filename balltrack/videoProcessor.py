import cv2
import imutils
from imutils.video import FPS
import json
from pathlib import Path
import os


class videoProcessor(object):
    """basic class for tracking"""

    def __init__(self, file, setting={}):
        super(videoProcessor, self).__init__()
        assert isinstance(file, str), print("filename should be string")

        # make sure there is folder save_conf
        self.checkFolder('save_conf')

        # initialise
        self.file = Path(file)
        assert self.file.exists(), print("file not find")
        self.filename = self.file.parts[-1].split('.')[0]

        # filename with full path absolute
        self.conf_file = os.path.join(
            os.getcwd(), "save_conf", self.filename+".json")
        self.conf = Path(self.conf_file).is_file()

        if self.conf:
            self.readSettings()

        self.updateSetting(setting)
        self.saveSettings()

    def updateSetting(self, setting):
        # have nothing, use default
        if (not bool(setting)) and not self.conf:
            self.setting = {"file": str(self.file.resolve()),
                            "skiptime": 0,
                            "tracker": "csrt",
                            "feature": "HT",
                            "fps": 25,
                            "size": (1024, 860),
                            }
        # has input but no record
        elif bool(setting) and (not self.conf):
            self.setting = {"file": str(self.file.resolve())}
            self.setting.update(setting)
        # has record, update setting
        elif bool(setting) and self.conf:
            self.setting.update(setting)

    def checkFolder(self, name):
        folderPath = os.path.join(os.getcwd(), name)
        if not Path(folderPath).is_dir():
            os.mkdir(folderPath)

    def readSettings(self):
        with open(self.conf_file, 'r') as f:
            load_setting = json.load(f)
            self.setting = load_setting

    def saveSettings(self):
        with open(self.conf_file, 'w') as f:
            f.write(json.dumps(self.setting, indent=2))

    def outputSetting(self):
        return json.dumps(self.setting, indent=2)

    def fpsInit(self):
        self.fps = FPS().start()

    def fpsUpdate(self):
        self.fps.update()
        self.fps.stop()

    def timeskip(self):
        start_frame_number = self.setting["skiptime"]*self.setting["fps"]
        cap = cv2.VideoCapture(self.setting["file"])
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_number)
        return cap

    def resize(self, frame):
        # return the height and width of resized frame
        frame = imutils.resize(frame,
                               width=self.setting["size"][0]//self.setting["resize"])
        return frame

    def save(self, d):
        self.setting.update(d)

    def process(self):
        para = self.setting
        if "class" in para:
            print("Explicitly naming the class is recommended")
            # add new class py file to balltrack
            import balltrack
            f = getattr(balltrack, para["class"])
            _ = f(self.setting["file"])
            _.process()
            self.__dict__ = _.__dict__.copy()
        else:
            cap = self.timeskip()

            while(cap.isOpened()):
                # Capture frame-by-frame
                ret, frame = cap.read()
                if not ret:
                    break

                frame = self.resize(frame)
                (H, W) = frame.shape[:2]

                cv2.imshow('frame', frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
            cap.release()
            cv2.destroyAllWindows()
