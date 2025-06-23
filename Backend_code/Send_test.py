import csv
import serial
import sqlite3
import json
import random
from datetime import datetime, timedelta
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
            Temp FLOAT,
            coli FLOAT
                )
    ''')
    conn.commit()
    conn.close()

def save_to_db(data):

    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute('''
            INSERT INTO sensor_data (timestamp, DO, TDS, Tur, pH, Temp, coli)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (data[0], data[1], data[2], data[3], data[4], data[5], data[6]))
    conn.commit()
    conn.close()

def query_history(start_time, end_time):
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    #cursor.execute("SELECT timestamp, DO, TDS, Tur, pH, Temp FROM sensor_data WHERE timestamp BETWEEN ? AND ?", (start_time, end_time))
    cursor.execute("""
        SELECT timestamp, DO, TDS, Tur, pH, Temp, coli
        FROM sensor_data 
        WHERE DATE(timestamp) BETWEEN ? AND ?
        ORDER BY timestamp
    """, (start_time, end_time))
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

def generate_historical_data():
    """生成前7天的随机数据，间隔10秒"""
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    
    end_time = datetime.now()
    start_time = end_time - timedelta(days=7)
    
    current_time = start_time
    while current_time <= end_time:
        # 生成随机传感器数据（范围可根据实际调整）
        data = [
            current_time.strftime("%Y-%m-%d %H:%M:%S"),
            round(random.uniform(1.0, 10.0), 2),  # DO
            round(random.uniform(100.0, 2000.0), 2),  # TDS
            round(random.uniform(100.0, 3000.0), 2),  # Tur
            round(random.uniform(6.0, 8.5), 2),  # pH
            round(random.uniform(15.0, 35.0), 2),   # Temp
            round(random.uniform(100.0, 200.0), 2)   # coli
        ]
        
        # 存入数据库
        cursor.execute('''
            INSERT INTO sensor_data (timestamp, DO, TDS, Tur, pH, Temp, coli)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', data)
        
        # 前进10秒
        current_time += timedelta(seconds=10)
    
    conn.commit()
    conn.close()
    print(f"Generated historical data from {start_time} to {end_time}")

# Initialize the serial port
ser = serial.Serial("COM3", 9600)

# Configuration of the database
db_file = "C:\\Users\\XCH\\Desktop\\Design_Sensor_System\\code\\Function_Blocks\\sensor_data.db"
init_db() 
generate_historical_data()

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
                    coli = str(round(random.uniform(100.0, 200.0), 2)) # Random coli value需要后期被修
                    data_output = [timestamp] + data_list + [coli]
                    message = f"Data: {data_output}" # MQTT message
                    print(message)

                    # Save data to database & publish to MQTT broker
                    save_to_db(data_output)
                
                    client.publish(send_topic, json.dumps(data_output)) 

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
