import os

def writeInfo(author, name_of_sim, notes, df, lap_time):
    directory = name_of_sim
    parent_dir = "Simulations"
    path = os.path.join(parent_dir, directory)

    if os.path.isdir(path):
        pass
    else:
        os.mkdir(path)

    f = open(str(path) + "/" + name_of_sim + ".txt", "w+")

    f.write("Author: " + str(author) + "\n")
    f.write("Name of Simulation: " + str(name_of_sim) + "\n")
    f.write("Notes: " + str(notes) + "\n")
    f.write("--------------------------------------------------------" + "\n")
    f.write("DATA: " + "\n")

    for line in range(len(df.index)):
        f.write(df["Parameter"][line] + ": " + df["Value"][line] + "\n")

    f.write("--------------------------------------------------------" + "\n")
    f.write("RESULTS: " + "\n")
    f.write("Total time [s]: " + str(lap_time[0]) + "\n")
    f.write("Total energy used [J]: " + str(lap_time[1]) + "\n")
    f.write("Total energy used [kWh]: " + str(lap_time[1] / (3600 * 1000)) + "\n")
    f.write("Max lateral acceleration [m/s²]: " + str(lap_time[2]) + "\n")
    f.write("Max longitudinal acceleration [m/s²]: " + str(lap_time[3]) + "\n")
    f.write("Max acceleration [m/s²]: " + str(lap_time[3]) + "\n")
    f.write("Max deceleration [m/s²]: " + str(lap_time[5]) + "\n")
    f.write("Max velocity [m/s]: " + str(lap_time[4]) + "\n")
    f.write("Average velocity [m/s]: " + str(lap_time[6]) + "\n")

    f.close()

    lap_time[7].to_csv(str(path) + "/" + name_of_sim + ".csv")