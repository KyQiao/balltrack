import cv2
import numpy as np
import imutils
import json
from pathlib import Path
import os


class videoProcessor(object):
    """docstring for videoProcessor"""

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
            print("conf load success")

        self.updateSetting(setting)
        self.saveSettings()

    def updateSetting(self, setting):
        # have nothing, use default
        if (not bool(setting)) and not self.conf:
            self.setting = {"file": str(self.file.resolve()),
                            "skiptime": 0,
                            "method": "CSRT",
                            "feature": "Hough",
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
            Path.mkdir(folderPath)

    def readSettings(self):
        with open(self.conf_file, 'r') as f:
            load_setting = json.load(f)
            self.setting = load_setting

    def saveSettings(self):
        with open(self.conf_file, 'w') as f:
            f.write(json.dumps(self.setting, indent=2))

    def outputSetting(self):
        return json.dumps(self.setting, indent=2)

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

    def process(self):
        para = self.setting
        if "class" in para:
            # add new class py file to balltrack
            import balltrack
            f = getattr(balltrack, para["class"])
            f.process(self)

        else:
            cap = self.timeskip()

            while(cap.isOpened()):
                # Capture frame-by-frame
                ret, frame = cap.read()
                if frame is None:
                    break

                frame = self.resize(frame)
                (H, W) = frame.shape[:2]

                cv2.imshow('frame', frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break


if __name__ == '__main__':
    setting = {"skiptime": 0,
               "method": "CSRT",
               "feature": "Hough",
               "fps": 25,
               "size": (1680, 1277),
               "resize": 4
               }
    test = videoProcessor(r'../test/h264_bubble5_crf28.avi', setting=setting)
    test.process()
