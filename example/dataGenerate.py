import cv2
from balltrack import balltrack
from balltrack.tools import HoughTest
import numpy as np
import matplotlib.pyplot as plt

# def f(frame):
#     frame = frame + 50
#     return frame

setting = {
    "skiptime": 10,
    "resize": 2
}


def Hough(file, setting={}):
    test = HoughTest(file, setting=setting)
    # test.preprocess=f
    test.getPara()


def output(self,imCrop, out=None):
    cv2.imwrite("./data/655/655M{0:0>5d}.png".format(self.t), imCrop)
    return imCrop


def dataGenerate(file):
    test = balltrack(file)
    # test.preprocess = f
    test.output = output
    test.process()
    data = test.trace
    np.save(file.split('.')[0]+'.npy', data)
    data = np.delete(data, np.where(data[:, 3] == 0)[0], 0)
    plt.axis('equal')
    plt.plot(data[:, 0], data[:, 1], '-*', ms=5)
    plt.show()


if __name__ == '__main__':
    # 654 655 656
    # Hough("MVI_0654.MP4")
    # dataGenerate("MVI_0654.MP4")

    dataGenerate("MVI_0655.MP4")
    # Hough("MVI_0655.MP4")
    # Hough("MVI_0649.MP4",setting=setting)
    # Hough("MVI_0649.MP4")
    # dataGenerate("MVI_0649.MP4")
