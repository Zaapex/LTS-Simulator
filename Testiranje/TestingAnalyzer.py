import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
from scipy.interpolate import splprep, splev
from scipy.interpolate import UnivariateSpline
from mpl_toolkits import mplot3d

lapone = (24380, 30000)
laptwo = (41995, 47000)
start_point = 41995
end_point = 48000

data = pd.read_csv("Logatec 26.10.2021/logatec_vsi.csv", nrows=end_point)


time = data["Time"][start_point:end_point]
x = (data["IMU_Longitude"][start_point:end_point] - data["IMU_Longitude"][start_point])*10**7
y = (data["IMU_Latitude"][start_point:end_point] - data["IMU_Latitude"][start_point])*10**7
velocity = data["Log_MotSpd1"][start_point:end_point]*(2*np.pi*0.228/60/5.45)
power = data["Log_Power"][start_point:end_point]  # Watts

average_velocity = np.average(velocity)
print("Average velocity: " + str(average_velocity) + " m/s")
timespent = 621/average_velocity
print("Time: " + str(timespent) + " s")
energy = 0
for i in range(start_point, end_point-1):
    delta_time = (data.loc[i+1, "Time"] - data.loc[i, "Time"])/1000
    energyUsed = delta_time * data.loc[i, "Log_Power"]
    energy += energyUsed

totalTime = (data.loc[end_point - 1, "Time"] - data.loc[start_point, "Time"])/1000

print(energy)

time = time.to_numpy()
x = x.to_numpy()
y = y.to_numpy()

sply = UnivariateSpline(time, y, s=0.0, k=2)
splx = UnivariateSpline(time, x, s=0.0, k=2)
x_g = np.linspace(time[0], time[-1], 150)

data2 = pd.read_csv("/home/alex/Desktop/LTS-Simulator/Simulations/csv/Logatec_newnormal")
x2 = data2["x"]
y2 = data2["y"]
velocity2 = data2["vx_exit"]

"""plt.plot(time, y, label='Aproksimacijkse tocke')
plt.plot(x_g, sply(x_g), label='Zlepek y')"""



"""plt.plot(x, y, label="Točke")
plt.plot(splx(x_g), sply(x_g), "o-", label="Zlepek")
plt.show()"""

"""plt.scatter(x, y, linewidths=0.001,
           edgecolor='white',
           s=10,
           c=velocity, label="velocity")

cbar = plt.colorbar()"""


ax = plt.axes(projection='3d')
ax.plot3D(x2, y2, velocity2, label="Simulator")
ax.plot3D(x, y, velocity, label="Test data")
plt.legend()
plt.show()


df = pd.DataFrame({"x":splx(x_g),
                   "y":sply(x_g)})

#df.to_csv("Logatec5.csv")