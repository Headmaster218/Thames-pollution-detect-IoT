import matplotlib
import numpy as np
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import os
from datetime import datetime, timedelta

# Generate a time series of 24 hours for the current day (hours only)
time_steps = [f"{hour}:00" for hour in range(24)]


def get_monitoring_data(date=None):
    np.random.seed(42)
    if date:
        try:
            now = datetime.strptime(date, "%Y-%m-%d")
        except ValueError:
            raise ValueError(f"Invalid date format: {date}. Expected format is 'YYYY-MM-DD'.")
    else:
        now = datetime.now()
        now = now.replace(minute=0, second=0, microsecond=0)  # Round to the nearest hour

    time_steps = [(now - timedelta(hours=i)).strftime("%H:%M") for i in range(24)][::-1]

    return [
        {
            "point_id": i + 1,
            "location": loc,
            "data": {
                t: {
                    "pH": np.random.uniform(6.8, 7.5),
                    "Turbidity": np.random.uniform(3.0, 5.0),
                    "DO2": np.random.uniform(5.5, 7.0),
                    "Conductivity": np.random.uniform(240, 280),
                    "Ecoli": np.random.uniform(1, 10),
                }
                for t in time_steps
            }
        }
        for i, loc in enumerate([(51.55, -0.025), (51.53, -0.018), (51.51, -0.012)])
    ]

def generate_plots(data_list, filename, date=None):
    import matplotlib.pyplot as plt

    if not data_list:
        print("Error: Empty data_list received!")
        return

    first_point_data = data_list[0]["data"]  # Use data from the first monitoring point
    time_steps = list(first_point_data.keys())  # Get time points
    print("Expected time steps:", time_steps)  # Debug information

    fig, ax = plt.subplots(figsize=(8, 4))

    for param in ["pH", "Turbidity", "DO2", "Conductivity", "Ecoli"]:
        try:
            ax.plot(
                time_steps, 
                [first_point_data[t][param] for t in time_steps], 
                marker="o", linestyle="-", label=param
            )
        except KeyError as e:
            print(f"KeyError: {e} - Check if time format matches!")  
            continue  

    ax.set_xlabel("Time")
    ax.set_ylabel("Measurement Value")
    ax.set_title("Water Quality Monitoring Data Changes")
    ax.legend()
    plt.xticks(rotation=45)
    if date:
        plt.savefig(f"static/{filename}_{date}.png")
    else:
        plt.savefig(f"static/{filename}.png")



if __name__ == "__main__":
    monitoring_data = get_monitoring_data()
    generate_plots(monitoring_data, "monitoring")