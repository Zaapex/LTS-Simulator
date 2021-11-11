import Code
import pandas as pd
import writeScript

author = input("Author (Default: User): ")
name_of_sim = input("Name of Simulation (Default: Sim): ")
notes = input("Notes (Default: None): ")
select_track = input("Track name (Default: Acceleration): ")
selected_settings = input("Data name (Default: Svarog data): ")

if not author:
    author = "User"
else:
    pass

if not name_of_sim:
    name_of_sim = "Sim"
else:
    pass

if not notes:
    notes = "None"
else:
    pass

if not select_track:
    select_track = "Acceleration"
else:
    pass

if not selected_settings:
    selected_settings = "Svarog data"
else:
    pass

# Read data
df = pd.read_csv(selected_settings)

# Start simulation
lap_time = Code.lap_time_simulation(select_track, selected_settings, name_of_sim)

# Write results
writeScript.writeInfo(author, name_of_sim, notes, df, lap_time)
