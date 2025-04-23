import matplotlib
import numpy as np
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import os
from datetime import datetime, timedelta
import json
import sqlite3
import pandas as pd
import time
import paho.mqtt.client as mqtt

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
                    "coli": np.random.uniform(1, 10),
                }
                for t in time_steps
            }
        }
        for i, loc in enumerate([(51.55, -0.025)])
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

    for param in ["pH", "Turbidity", "DO2", "Conductivity", "coli"]:
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

# MQTT broker address, port, topic
broker_address = "192.168.137.1"  # Replace with the correct IP address
broker_port = 1883
send_topic = "AQ/send"
request_topic = "AQ/request"
response_topic = "AQ/response"

realtime_data = []
history_data = []

# Directory to save received data
data_save_dir = "data/received_data"
os.makedirs(data_save_dir, exist_ok=True)

def save_realtime_data(payload):
    """Save the new data to a file."""
    try:
        # Ensure payload is in the correct format
        if isinstance(payload, list) and len(payload) == 7:
            data = {
                "timestamp": payload[0],  # Keep timestamp as a string
                "DOxy": float(payload[1]),
                "TDSs": float(payload[2]),
                "Tur": float(payload[3]),
                "pH": float(payload[4]),
                "Temp": float(payload[5]),
                "coli": float(payload[6])
            }
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = os.path.join(data_save_dir, f"data_{timestamp}.json")
            with open(filename, "w") as f:
                json.dump(data, f, indent=4)
            print(f"Data saved to {filename}")
        else:
            print("Error: Payload format is incorrect. Data not saved.")
    except Exception as e:
        print(f"Error saving data: {e}")

def load_realtime_data():
    """Load all saved data from files."""
    data = []
    try:
        for file in sorted(os.listdir(data_save_dir)):
            if file.endswith(".json"):
                filepath = os.path.join(data_save_dir, file)
                with open(filepath, "r") as f:
                    data.append(json.load(f))
    except Exception as e:
        print(f"Error loading data: {e}")
    return data

def get_latest_saved_data():
    """Retrieve the latest saved data from the received_data directory."""
    try:
        files = [f for f in os.listdir(data_save_dir) if f.endswith(".json")]
        if not files:
            return None
        latest_file = max(files, key=lambda f: os.path.getmtime(os.path.join(data_save_dir, f)))
        with open(os.path.join(data_save_dir, latest_file), "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error retrieving latest saved data: {e}")
        return None

# Callback functions
def on_message(client, userdata, msg):
    try:
        payload_str = msg.payload.decode('utf-8')
        payload = json.loads(payload_str)
        
        if msg.topic == send_topic:
            # Handle real-time data
            realtime_data.append(payload)
            # print("\n=== New Realtime Data ===")
            # print(f"Message: {payload}")

            # Save data to file
            save_realtime_data(payload)
            
        elif msg.topic == response_topic:
            global history_data
            history_data = payload
    except Exception as e:
        print(f"Error processing message: {e}")

def on_connect(client, userdata, flags, reason_code, properties):
    if reason_code == 0:
        print(f"Successfully connected to MQTT Broker: {broker_address}:{broker_port}")
        client.subscribe([(send_topic, 0), (response_topic, 0), (request_topic, 0)])
        print(f"Subscribed to topics: {send_topic} and {response_topic} and {request_topic}")
    else:
        print(f"Connection failed, error code: {reason_code}")

def get_user_input():
    while True:
        try:
            start_time = input("Enter start time (YYYY-MM-DD HH:MM:SS): ")
            end_time = input("Enter end time (YYYY-MM-DD HH:MM:SS): ")

            # Validate time format
            datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
            datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")

            return start_time, end_time
        except ValueError:
            print("Invalid time format. Please use YYYY-MM-DD HH:MM:SS.")

def request_history(start_time, end_time):
    payload = json.dumps({
        "start_time": start_time,
        "end_time": end_time
    })
    client.publish(request_topic, payload)
    print(f"Sent request for data from {start_time} to {end_time}")

def send_date_to_broker(selected_date):
    """Send the selected date to the MQTT broker."""
    try:
        payload = json.dumps({
            "start_time": selected_date,
            "end_time": selected_date
        })
        client.publish(request_topic, payload)
        print(f"Sent date {selected_date} to broker on topic {request_topic}")
    except Exception as e:
        print(f"Error sending date to broker: {e}")


# MQTT client setup
client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.on_connect = on_connect
client.on_message = on_message

client.connect(broker_address, broker_port)
client.loop_start()

def check_for_new_messages():
    """Check if there are new messages on the response topic."""
    time.sleep(0.5)
    global history_data
    if history_data:
        print("\n=== New Message on Response Topic ===")
        print(history_data)
        print("\n=== New Message on Response Topic ===")

        return history_data
    else:
        print("No new messages on the response topic.")
        return None


# Example usage
if __name__ == "__main__":
    monitoring_data = get_monitoring_data()
    generate_plots(monitoring_data, "monitoring")
    
    # Start listening for real-time data
    try:
        print("\nListening for real-time data... Press Ctrl+C to stop")
        while True:
            pass  # Keep the main loop running to process messages
    except KeyboardInterrupt:
        print("\nUser interrupted, disconnecting...")
        client.loop_stop()
        client.disconnect()
        print("Disconnected from MQTT broker")