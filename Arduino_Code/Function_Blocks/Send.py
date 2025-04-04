import csv
import serial
import sqlite3
import json
import serial.tools.list_ports
from datetime import datetime
import paho.mqtt.client as mqtt

def init_db():
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sensor_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            DO FLOAT,
            TDS FLOAT,
            Tur FLOAT,
            pH FLOAT,
            Temp FLOAT
                )
    ''')
    conn.commit()
    conn.close()

def save_to_db(data):

    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO sensor_data (DO, TDS, Tur, pH, Temp) VALUES (?, ?, ?, ?, ?)", 
                   (data[0], data[1], data[2], data[3], data[4]))
    conn.commit()
    conn.close()

def query_history(start_time, end_time):
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute("SELECT timestamp, DO, TDS, Tur, pH, Temp FROM sensor_data WHERE timestamp BETWEEN ? AND ?", (start_time, end_time))
    data = cursor.fetchall()
    conn.close()
    return data

def on_message(client, userdata, msg):
    if msg.topic == request_topic:
        payload = json.loads(msg.payload.decode())
        start_time = payload.get("start_time")
        end_time = payload.get("end_time")
        print(f"Received request for data from {start_time} to {end_time}")

        history_data = query_history(start_time, end_time)
        response = json.dumps(history_data)

        client.publish(response_topic, response)
        print(f"Sent history data to {response_topic}")


# Initialize the serial port
ser = serial.Serial("COM3", 9600)

# Configuration of the database
db_file = "C:\\Users\\XCH\\Desktop\\Design_Sensor_System\\code\\Function_Blocks\\sensor_data.db"
init_db() 

# MQTT broker address and port
broker_address = "127.0.0.1"  
broker_port = 1883
send_topic = "AQ/send"  
request_topic = "AQ/request"  
response_topic = "AQ/response" 

client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.on_message = on_message
client.connect(broker_address, broker_port)
client.subscribe(request_topic) 
client.loop_start()

# Check if the serial port is opened
if ser.isOpen():
    print("Serial port is opened.")

    # save in csv file
    csv_filename = "sensor_data.csv"
    with open(csv_filename, "a", newline="") as file:
        writer = csv.writer(file)

        if file.tell() == 0:
            #writer.writerow(["Date", "Time", "DO", "TDS", "Tur", "pH", "Temp"])
            writer.writerow(["Timestamp", "DO", "TDS", "Tur", "pH", "Temp"])

    while True:
        try:
            com_input = ser.readline().decode("utf-8").strip()

            if com_input:
                # split by ,
                data_list = com_input.split(",") 

                if len(data_list) == 5:  # Ensure that the data is complete
                    now = datetime.now()
                    date = now.strftime("%Y-%m-%d")  
                    time = now.strftime("%H:%M:%S")  
                    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
                    
                    #data_output = [date, time] + data_list
                    data_output = [timestamp] + data_list
                    message = f"Data: {data_output}" # MQTT message
                    print(message)

                    # Save data to database & publish to MQTT broker
                    save_to_db(data_list)
                    client.publish(send_topic, message) 

                    with open(csv_filename, "a", newline="") as file:
                        writer = csv.writer(file)
                        writer.writerow(data_output)
        except Exception as e:
            print("Error:", e)
            break
else:
    print("Failed to open serial port.")

ser.close()
client.loop_stop()
print("Serial port is closed.")
