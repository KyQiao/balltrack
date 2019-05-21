import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt


class trace(object):
    """docstring for trace"""

    def __init__(self, filename):
        super(trace, self).__init__()
        self.data = np.load(filename)
        print('get ', len(self.data[:, 1]), ' total frames')
        if np.unique(self.data[:, 2])[0] == 0:
            self.missing_list = np.where(self.data[:, 2] == 0)[0]
            print(len(self.missing_list), ' particle missing')

            self.data[self.missing_list] = (
                (self.data[self.missing_list-1]+self.data[self.missing_list+1])/2).astype(np.uint16)
            # self.data = np.delete(self.data, self.missing_list, 0)

        self.frames = len(self.data[:, 1])
        self.time = self.data[-1, 3]-self.data[0, 3]
        print('get ', self.frames, ' efficient frames')
        print('final time is ', self.time)

    def velocity(self, x1, x2):
        v = la.norm(np.abs(x1[0:2].astype(np.float16) -
                           x2[0:2].astype(np.float16))/abs(x2[3]-x1[3]))
        return v

    def velocity_test(self, interval=1):
        v = []
        for i in range(self.frames-interval):
            v.append(self.velocity(self.data[i, :], self.data[i+interval, :]))
        # plt.plot(self.data[0:self.frames-interval, 3], v, '-*', ms=10)
        plt.hist(v, bins=20)
        plt.show()

    def trace(self):
        plt.axis('equal')
        plt.plot(self.data[:, 0], self.data[:, 1], '-*', ms=5)
        plt.show()

    def modify(self, interval, vlim):
        for i in range(self.frames-interval-1):
            if self.velocity(self.data[i, :], self.data[i+interval, :]) > vlim:
                self.data[i+interval, 0:2] = ((self.data[i, 0:2] +
                                               self.data[i+interval+1, 0:2])/2).astype(np.uint16)

    def delete_modify(self, interval, vlim):
        missing_list2 = []
        for i in range(self.frames-interval-1):
            if self.velocity(self.data[i, :], self.data[i+interval, :]) > vlim:
                missing_list2.append(i+1)
        self.data = np.delete(self.data, missing_list2, 0)
        self.frames = len(self.data[:, 1])


if __name__ == "__main__":
    print("c = trace('trace_without_water.npy')\n\
c.velocity_test(interval=1)\n\
c.modify(1,4)\n\
c.modify(2,4)\n\
c.delete_modify(1,4)\n\
c.delete_modify(2,4)\n\
c.trace()\n")

