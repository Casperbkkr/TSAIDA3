import numpy as np
import pandas as pd

file_names = ["Home_energy",
              "machine_temperature_system_failure",
              "ambient_temperature_system_failure",
              "cpu_utilization_asg_misconfiguration",
              "ec2_request_latency_system_failure",
              "nyc_taxi",
              "rogue_agent_key_hold",
              "rogue_agent_key_updown"]

def read_power_data(start, end):
    Data = pd.read_csv("../Data/Home_energy.csv")
    print("----- Data_output loaded -----\n")

    Data["timestamp"] = pd.to_datetime(Data["timestamp"], unit='ms').dt.tz_localize('utc').dt.tz_convert('Europe/Amsterdam')

    Data['Year'] = Data['timestamp'].dt.year
    Data['Month'] = Data['timestamp'].dt.month
    Data['Day'] = Data['timestamp'].dt.day
    Data['Time'] = Data['timestamp'].dt.time
    Data['Watts'] = Data['value']
    Data = Data.drop(labels=['value'], axis='columns')

    column_names = Data.columns.tolist()[1:]
    for i in range(len(start)):
        Data = Data.loc[Data[column_names[i]] >= start[i]]
        Data = Data.loc[Data[column_names[i]] <= end[i]]

    Data_plot = Data[["Watts", "timestamp"]]
    Data_np = Data_plot["Watts"].to_numpy()

    Data_np = Data_np.astype(np.float32)
    Data_np = Data_np[:, np.newaxis]
    print("----- Data_output converted to numpy -----\n")

    return Data_np, Data_plot