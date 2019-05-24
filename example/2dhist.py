import numpy as np
from dataAnalysis import trace
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt


def preprocess():
    water = trace('trace_with_water.npy')
    water.modify(1, 4)
    water.modify(2, 4)
    no_water = trace('trace_without_water.npy')
    no_water.modify(1, 4)
    no_water.modify(2, 4)

    water_data = water.data
    no_water_data = no_water.data

    np.save('Pwater', water_data)
    np.save('Pno_water', no_water_data)


def main():
    water = np.load('Pwater.npy')
    nowater = np.load('Pno_water.npy')
    fig, axes = plt.subplots(nrows=1, ncols=2)

    for ax, data in zip(axes, [water,nowater]):
        ax.axis('equal')
        ax.hist2d(data[:, 0], data[:, 1],
                  bins=100,)

    fig.tight_layout()

    plt.show()



if __name__ == '__main__':
    main()