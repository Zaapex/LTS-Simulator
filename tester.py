from scipy import optimize
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import Functions

df_data = pd.read_csv("Svarog data", index_col=0)

# basic parameters
mass = float(df_data["Value"][df_data.loc[df_data['Parameter'] == "Mass"].index[0]])
CG = float(df_data["Value"][df_data.loc[df_data['Parameter'] == "CG in y"].index[0]])
height_CG = float(df_data["Value"][df_data.loc[df_data['Parameter'] == "CG in z"].index[0]])
track_width = float(df_data["Value"][df_data.loc[df_data['Parameter'] == "Track width"].index[0]])
wheelbase = float(df_data["Value"][df_data.loc[df_data['Parameter'] == "Wheelbase"].index[0]])
W = float(df_data["Value"][df_data.loc[df_data['Parameter'] == "W"].index[0]])

# suspension parameters
coef_friction = float(df_data["Value"][df_data.loc[df_data['Parameter'] == "Coefficient of Friction"].index[0]])

# aerodynamics parameters
air_density = float(df_data["Value"][df_data.loc[df_data['Parameter'] == "Air density"].index[0]])
frontal_area = float(df_data["Value"][df_data.loc[df_data['Parameter'] == "Frontal area"].index[0]])
coef_of_df = float(df_data["Value"][df_data.loc[df_data['Parameter'] == "Coefficient of DF"].index[0]])
coef_of_drag = float(df_data["Value"][df_data.loc[df_data['Parameter'] == "Coefficient of Drag"].index[0]])
alpha_Cl = air_density * frontal_area * coef_of_df/2
alpha_Cd = air_density * frontal_area * coef_of_drag/2
CP = float(df_data["Value"][df_data.loc[df_data['Parameter'] == "CoP in y"].index[0]])
CoPy = CP / 100 * wheelbase
CoPz = float(df_data["Value"][df_data.loc[df_data['Parameter'] == "CoP in z"].index[0]])

# drivetrain parameters
KF = 0  # which axis is driven, front equals zero
KR = 1
g = 9.81
a = wheelbase*CG/100
b = wheelbase*(100-CG)/100
mass_rear = mass * a / wheelbase
mass_front = mass * b / wheelbase
w = float(df_data["Value"][df_data.loc[df_data['Parameter'] == "Tire radius"].index[0]])*2
max_rpm = float(df_data["Value"][df_data.loc[df_data['Parameter'] == "Max RPM"].index[0]])
tire_radius = float(df_data["Value"][df_data.loc[df_data['Parameter'] == "Tire radius"].index[0]])
gear_ratio = float(df_data["Value"][df_data.loc[df_data['Parameter'] == "Gear ratio"].index[0]])
v_max_teo = max_rpm*np.pi*2*tire_radius/(60*gear_ratio)
max_torque = float(df_data["Value"][df_data.loc[df_data['Parameter'] == "Max torque"].index[0]])
max_rear_wheel_torque = max_torque*gear_ratio/tire_radius*2

# braking system parameters
brake_bias = float(df_data["Value"][df_data.loc[df_data['Parameter'] == "Brake bias"].index[0]])

"""df1 = pd.read_csv("Simulations/csv/FSG260kg_newmaxvel", index_col=0, low_memory=False)
df2 = pd.read_csv("Simulations/csv/FSG260kg", index_col=0, low_memory=False)

plt.plot(df1["pot"], df1["vx_max"], label="New")
plt.plot(df2["pot"], df2["vx_max"], label="Old")
plt.legend()"""

r = 15.25 + track_width/2 + 0.1

Functions.add_new_points("/home/alex/Desktop/LTS-Simulator/Tracks/Logatec2_lesspoints.csv")
Functions.radius_of_corner("/home/alex/Desktop/LTS-Simulator/Tracks/Logatec2_lesspoints.csv")