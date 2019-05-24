import colloidpy as cp
import numpy as np
from dataAnalysis import trace
import matplotlib.pyplot as plt

water = trace('trace_with_water.npy')
water.modify(1, 4)
water.modify(2, 4)
no_water = trace('trace_without_water.npy')
no_water.modify(1, 4)
no_water.modify(2, 4)

print(cp.__version__)

water_data = water.data
no_water_data = no_water.data


def track(data):
    N = len(data[:, 3])
    dis = np.where((data[1:N, 3] - data[0:N-1, 3]) != 1)
    dis_con = np.zeros(N-1)
    dis_con[dis] = 1
    label = np.hstack((0, np.cumsum(dis_con, dtype=int)))
    for i in range(len(dis)-1):
        data[dis(i)+1:dis(i+1)+1, 3] = np.arange(0, dis(i+1)-dis(i))
    return np.hstack((0, np.cumsum(dis_con, dtype=int)))


pID = track(water_data)
pID2 = track(no_water_data)
print(pID.shape)
print(water_data.shape)


zero = np.zeros_like(water_data[:, 3])
zero2 = np.zeros_like(no_water_data[:, 3])
water_msd = cp.ColloidData(
    water_data[:, 0], water_data[:, 1], water_data[:, 3], pID, zero)
no_water_msd = cp.ColloidData(
    no_water_data[:, 0], no_water_data[:, 1], no_water_data[:, 3], pID2, zero2)

dts, msd, counts = water_msd.msd()
dts2, msd2, counts2 = no_water_msd.msd()


plt.figure(figsize=(14, 7))
plt.title('with water')
plt.subplot(121)
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$T(s)$', fontsize=25)
plt.ylabel(r'${\Delta r}^2$', fontsize=25)
plt.plot(dts/25, msd[:, 2], '-o', label='$dx^2$')
plt.plot(dts/25, msd[ :, 3], '-o', label='$dy^2$')
plt.plot(dts/25, msd[ :, 4], '-o', label='$dr^2$')
plt.plot(dts/25,4**np.exp(1)*dts/25)
plt.legend(prop={'size': 20})
plt.subplot(122)
plt.xscale('log')
plt.plot(dts/25,counts)
plt.ylabel(r'$counts$', fontsize=25)
plt.savefig('water.png', dpi=200)
plt.show()

plt.figure(figsize=(14, 7))
plt.title('no water')
plt.subplot(121)
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$T(s)$', fontsize=25)
plt.ylabel(r'${\Delta r}^2$', fontsize=25)

plt.plot(dts2/25, msd2[:, 2], '-o', label='$dx^2$')
plt.plot(dts2/25, msd2[ :, 3], '-o', label='$dy^2$')
plt.plot(dts2/25, msd2[ :, 4], '-o', label='$dr^2$')
plt.plot(dts2/25,4*np.exp(1.8)*dts2/25)
plt.legend(prop={'size': 20})

plt.subplot(122)
plt.xscale('log')
plt.plot(dts2/25,counts2)
plt.ylabel(r'$counts$', fontsize=25)
plt.savefig('no_water.png', dpi=200)
plt.show()
