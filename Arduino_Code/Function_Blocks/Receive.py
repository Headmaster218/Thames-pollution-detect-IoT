import json
import sqlite3
import pandas as pd
from datetime import datetime
import paho.mqtt.client as mqtt

# MQTT broker address, port, topic
broker_address = "127.0.0.1"  # ip address 换成xch的ip地址
broker_port = 1883
send_topic = "AQ/data"  
request_topic = "AQ/request"  
response_topic = "AQ/response"  

received_data = []
history_data = []


# Callback functions
def on_message(client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode())
            
            if msg.topic == send_topic:
                received_data.append(payload)
                print(f"Received real-time data: {payload}")
                
            elif msg.topic == response_topic:
                if "data" in payload:
                    history_data = payload["data"]
                    print(f"Received history data:")
                    df = pd.DataFrame(history_data, 
                                     columns=["Timestamp", "DO", "TDS", "Tur", "pH", "Temp"])
                    print(df)
                
        except Exception as e:
            print(f"Error processing message: {e}")

def on_connect(client, userdata, flags, reason_code, properties):
    if reason_code == 0:
        print(f"Successfully connect to MQTT Broker: {broker_address}:{broker_port}")
        client.subscribe(send_topic)  
        client.subscribe(response_topic)  
        print(f"Subscribed topics: {send_topic} and {response_topic}")
    else:
        print(f"Connection failed, error code: {reason_code}")

def get_user_input():
    while True:
        try:
            start_time = input("Enter start time (YYYY-MM-DD HH:MM:SS): ")
            end_time = input("Enter end time (YYYY-MM-DD HH:MM:SS): ")

            # 验证时间格式
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


# Connect to the MQTT broker
client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.on_connect = on_connect
client.on_message = on_message  

client.subscribe(send_topic)  
client.subscribe(response_topic)  
client.loop_start()

#start_time, end_time = get_user_input()
start_time = "2025-03-24 16:10:32"
end_time = "2025-03-24 16:11:23"

#request_history(start_time, end_time)

while True:
    try:
        client.connect(broker_address, broker_port)
        request_history(start_time, end_time)

    except KeyboardInterrupt:
        print("\n用户中断，正在断开连接...")
        client.loop_start()
        client.disconnect()
        print("已断开MQTT连接")
        break

