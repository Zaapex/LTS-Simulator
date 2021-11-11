from Functions import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

def lap_time_simulation(track, formula_data, simulations_name):
    name_of_track = "Tracks/" + track + ".csv"
    df = pd.read_csv(name_of_track, index_col=0, low_memory=False)
    df_data = pd.read_csv(formula_data, index_col=0)

    initial_conditions = {"vx_initial": 0, "t_initial": 0, "a_initial": 0}

    df.at[0, "vx_entry"] = initial_conditions["vx_initial"]
    df.at[0, "time"] = initial_conditions["t_initial"]
    df.at[0, "acceleration"] = initial_conditions["a_initial"]

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
    v_max_teo = max_rpm*math.pi*2*tire_radius/(60*gear_ratio)
    max_torque = float(df_data["Value"][df_data.loc[df_data['Parameter'] == "Max torque"].index[0]])
    max_rear_wheel_torque = max_torque*gear_ratio/tire_radius*2

    # braking system parameters
    brake_bias = float(df_data["Value"][df_data.loc[df_data['Parameter'] == "Brake bias"].index[0]])

    for x in range(len(df.index)):
        """"Here we calculate max velocity in given corner"""

        # track width added to the radius -> we need the center of the car
        radius = df.loc[x, 'R'] + track_width/2 + 0.1 # get radius at this segment, between two points

        if radius < 4:
            radius = 4

        if radius < 75:

            max_vel_rear_inner = max_vel_rear_inner_v2(track_width, a, mass, height_CG, alpha_Cl, wheelbase, CoPy, alpha_Cd, CoPz, coef_friction,
                          v_max_teo, tire_radius, r=radius, mass_rear=mass_rear)

            max_vel_front_inner = max_vel_front_inner_v2(track_width, b, mass, height_CG, alpha_Cl, wheelbase, CoPy, alpha_Cd, CoPz,
                          coef_friction, v_max_teo, tire_radius, r=radius, mass_front=mass_front)

            df.at[x, "vel_fi"] = max_vel_front_inner
            df.at[x, "vel_ri"] = max_vel_rear_inner
            df.at[x, "vx_max"] = min((max_vel_rear_inner + max_vel_front_inner)*W, v_max_teo)
        else:
            df.at[x, "vel_fi"] = v_max_teo
            df.at[x, "vel_ri"] = v_max_teo
            df.at[x, "vx_max"] = v_max_teo

    df.fillna(method="ffill", inplace=True)

    for x in range(len(df.index)):

        vx_entry = df.loc[x, 'vx_entry']  # entry velocity in this section
        acc_x = df.loc[x, "acceleration"]
        radius = df.loc[x, "R"] + track_width/2
        vx_max = df.loc[x, "vx_max"]
        s = df.loc[x, "s"]

        if radius < 4:
            radius = 4

        # normal force on each tire
        f_nor_r_o = normal_force_rear_outer(a, b, m=mass, g=g, h=height_CG, w=w, alfa_cl=alpha_Cl, l=wheelbase, CoPy=CoPy,
                                             alfa_cd=alpha_Cd, CoPz=CoPz, r=radius, d=track_width, v=vx_entry, acc=acc_x)
        f_nor_r_i = normal_force_rear_inner(a, b, m=mass, g=g, h=height_CG, w=w, alfa_cl=alpha_Cl, l=wheelbase, CoPy=CoPy,
                                             alfa_cd=alpha_Cd, CoPz=CoPz, r=radius, d=track_width, v=vx_entry, acc=acc_x)
        f_nor_f_o = normal_force_front_outer(a, b, m=mass, g=g, h=height_CG, w=w, alfa_cl=alpha_Cl, l=wheelbase, CoPy=CoPy,
                                             alfa_cd=alpha_Cd, CoPz=CoPz, r=radius, d=track_width, v=vx_entry, acc=acc_x)
        f_nor_f_i = normal_force_front_inner(a, b, m=mass, g=g, h=height_CG, w=w, alfa_cl=alpha_Cl, l=wheelbase, CoPy=CoPy,
                                             alfa_cd=alpha_Cd, CoPz=CoPz, r=radius, d=track_width, v=vx_entry, acc=acc_x)

        # friction force for each tire
        f_fri_r_o = f_nor_r_o * coef_friction
        f_fri_r_i = f_nor_r_i * coef_friction
        f_fri_f_o = f_nor_f_o * coef_friction
        f_fri_f_i = f_nor_f_i * coef_friction

        # some total forces on car
        F_centripental = (mass * vx_entry ** 2 / radius)
        F_centripental_front = (mass_front*vx_entry**2/radius)
        F_centripental_rear = (mass_rear * vx_entry ** 2 / radius)
        F_drag = alpha_Cd * vx_entry ** 2
        F_nor_total = f_nor_f_i + f_nor_f_o + f_nor_r_i + f_nor_r_o


        if (f_fri_r_i**2 - (F_centripental_rear/2)**2 < 0) or (f_fri_f_i**2 - (F_centripental_front/2)**2 < 0):
            F_acc_r_i = 0
            F_acc_r_o = 0

        else:
            F_acc_r_o = np.sqrt(f_fri_r_o ** 2 - (F_centripental_rear/2) ** 2)
            F_acc_r_i = np.sqrt(f_fri_r_i ** 2 - (F_centripental_rear/2) ** 2)
            #F_acc_f_o = np.sqrt(f_fri_f_o ** 2 - (F_centripental_front/2) ** 2)
            #F_acc_f_i = np.sqrt(f_fri_f_i ** 2 - (F_centripental_front/2) ** 2)

        if vx_entry < vx_max:
            F_acceleration = min(F_acc_r_o, F_acc_r_i) * 2 - F_drag
            F_limit = max_rear_wheel_torque - F_drag
            acc = min(F_acceleration, F_limit) / mass
            vx_exit = np.sqrt(vx_entry ** 2 + 2 * s * acc)

            if vx_exit < vx_max:
                df.at[x, "vx_exit"] = vx_exit
                df.at[x + 1, "vx_entry"] = vx_exit
                df.at[x + 1, "acceleration"] = acc

            else:
                df.at[x, "vx_exit"] = vx_max
                df.at[x + 1, "vx_entry"] = vx_max
                df.at[x + 1, "acceleration"] = acc

        else:
            df.at[x, "vx_exit"] = vx_max
            df.at[x + 1, "vx_entry"] = vx_max
            df.at[x + 1, "acceleration"] = 0


    df.fillna(method="ffill", inplace=True)

    for x in range(len(df.index) - 2, 0, -1):
        vx_entry = df.loc[x, 'vx_entry']
        vx_exit = df.loc[x, "vx_exit"]
        radius = df.loc[x, "R"] + track_width/2
        acc_x = df.loc[x+1, "acceleration"]
        vx_max = df.loc[x, "vx_max"]

        s = df.loc[x, "s"]

        if radius < 4:
            radius = 4

        F_centripental_front = (mass_front * vx_exit ** 2 / radius)
        F_centripental_rear = (mass_rear * vx_exit ** 2 / radius)
        F_drag = alpha_Cd * vx_exit ** 2

        # normal force on each tire
        f_nor_r_o = normal_force_rear_outer(a, b, m=mass, g=g, h=height_CG, w=w, alfa_cl=alpha_Cl, l=wheelbase, CoPy=CoPy,
                                            alfa_cd=alpha_Cd, CoPz=CoPz, r=radius, d=track_width, v=vx_exit, acc=acc_x)
        f_nor_r_i = normal_force_rear_inner(a, b, m=mass, g=g, h=height_CG, w=w, alfa_cl=alpha_Cl, l=wheelbase, CoPy=CoPy,
                                            alfa_cd=alpha_Cd, CoPz=CoPz, r=radius, d=track_width, v=vx_exit, acc=acc_x)
        f_nor_f_o = normal_force_front_outer(a, b, m=mass, g=g, h=height_CG, w=w, alfa_cl=alpha_Cl, l=wheelbase, CoPy=CoPy,
                                             alfa_cd=alpha_Cd, CoPz=CoPz, r=radius, d=track_width, v=vx_exit, acc=acc_x)
        f_nor_f_i = normal_force_front_inner(a, b, m=mass, g=g, h=height_CG, w=w, alfa_cl=alpha_Cl, l=wheelbase, CoPy=CoPy,
                                             alfa_cd=alpha_Cd, CoPz=CoPz, r=radius, d=track_width, v=vx_exit, acc=acc_x)

        # friction force for each tire
        f_fri_r_o = f_nor_r_o * coef_friction
        f_fri_r_i = f_nor_r_i * coef_friction
        f_fri_f_o = f_nor_f_o * coef_friction
        f_fri_f_i = f_nor_f_i * coef_friction

        # average of inner and outer tire friction
        F_front_friction = min(f_fri_f_o, f_fri_f_i) #(f_fri_f_o + f_fri_f_i)/2
        F_rear_friction = min(f_fri_r_o, f_fri_r_i) #(f_fri_r_o + f_fri_r_i)/2

        if vx_entry >= vx_exit:
            if ((F_rear_friction ** 2 - (F_centripental_rear / 2) ** 2) < 0) or ((F_front_friction ** 2 - (F_centripental_front / 2) ** 2) < 0):
                F_braking = 0

            # added brake bias
            else:
                F_brake_f = np.sqrt((F_front_friction ** 2 - (F_centripental_front / 2) ** 2))
                F_brake_r = np.sqrt((F_rear_friction ** 2 - (F_centripental_rear/ 2) ** 2))
                if F_brake_f*brake_bias/100 < F_brake_r*(100-brake_bias)/100:
                    F_braking = F_brake_f*4

                else:
                    F_braking = F_brake_r*4


            F_deceleretion = F_drag + F_braking
            max_deceleration = - F_deceleretion / mass

            new_v_entry = np.sqrt(vx_exit ** 2 - 2 * max_deceleration * s)
            df.at[x, "vx_entry"] = new_v_entry
            df.at[x - 1, "vx_exit"] = new_v_entry
            df.at[x, "acceleration"] = max_deceleration


    plt.plot(list(range(len(df.index))), df["acceleration"][:], "r.", label="Pospešek")
    plt.plot(list(range(len(df.index))), df["vx_max"][:], "g", label="V max")
    plt.plot(list(range(len(df.index))), df["vx_entry"][:], "b", label="V vstopna")
    plt.title("Vstopna in maksimalna hitrost ter pojemek")
    plt.xlabel("Številka odseka")
    plt.ylim(-40, 40)
    plt.ylabel("Hitrost [m/s]" "\n"
               "Pojemek [m/s²]")

    plt.legend()
    plt.show()
    df.fillna(method="ffill", inplace=True)

    for x in range(len(df.index)):
        vx_entry = df.loc[x, 'vx_entry']
        vx_exit = df.loc[x, "vx_exit"]
        s = df.loc[x, "s"]
        acc = df.loc[x, "acceleration"]
        radij = df.loc[x, "R"]

        if acc > 0:
            energy = acc*mass*s
            df.at[x, "energy_spent"] = energy
        else:
            df.at[x, "energy_spent"] = 0

        time = abs(s / ((vx_entry + vx_exit) / 2))

        df.at[x, "lateral_acc"] = ((vx_entry + vx_exit)/2)**2/radij
        df.at[x, "time"] = time
        df.at[x, "time_total"] = df["time"].sum()

    total_energy_used_jouls = df["energy_spent"].sum()
    max_lateral_acc = max(df["lateral_acc"])
    max_longitudinal_acc = max((df["acceleration"]))
    max_veloctiy = max((df["vx_exit"]))
    max_deacc = min((df["acceleration"]))
    avg_velocity = np.average((df["vx_exit"]))


    df.at[0, "pot"] = df.loc[0, "s"]
    for x in range(len(df.index) - 1):
        pot = df.loc[x, "pot"] + df.loc[x + 1, "s"]
        df.at[x + 1, "pot"] = pot


    skidpadtime = skidpad_time(track_width, a, b, mass, height_CG, w, alpha_Cl, wheelbase, CoPy, alpha_Cd, CoPz,
                               coef_friction, KF, KR, v_max_teo)

    # print(skidpadtime)

    return df["time"].sum(), total_energy_used_jouls, max_lateral_acc, max_longitudinal_acc, max_veloctiy, \
           max_deacc, avg_velocity, df

