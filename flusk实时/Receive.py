import json
import sqlite3
import pandas as pd
from datetime import datetime
import paho.mqtt.client as mqtt

# MQTT broker address, port, topic
broker_address = "192.168.0.197"  # ip address 换成xch的ip地址
broker_port = 1883
send_topic = "AQ/send"  
request_topic = "AQ/request"  
response_topic = "AQ/response"  

realtime_data = []
history_data = []


# Callback functions
def on_message(client, userdata, msg):
        try:
            #print(f"Received message on {msg.topic}: {msg.payload}")
            payload_str = msg.payload.decode('utf-8')
            payload = json.loads(payload_str)
            
            if msg.topic == send_topic:
                # 处理实时数据
                realtime_data.append(payload)
                print("\n=== New Realtime Data ===")
                print(f"Message: {payload}")

                
            elif msg.topic == response_topic:
                if "data" in payload:
                    history_data = payload["data"]
                    print("\n=== History Data ===")
                    df = pd.DataFrame(payload["data"], 
                                     columns=["Timestamp", "DO", "TDS", "Tur", "pH", "Temp"])
                    print(df)
                
        except Exception as e:
            print(f"Error processing message: {e}")

def on_connect(client, userdata, flags, reason_code, properties):
    if reason_code == 0:
        print(f"Successfully connect to MQTT Broker: {broker_address}:{broker_port}")
        client.subscribe([(send_topic, 0), (response_topic, 0)])
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

client.connect(broker_address, broker_port) 
client.loop_start()

#start_time, end_time = get_user_input()
start_time = "2025-04-14 10:33:34"
end_time = "2025-04-14 10:34:02"
request_history(start_time, end_time)


try:
    print("\nListening for realtime data... Press Ctrl+C to stop")
    while True:
        pass  # 主循环保持运行，让回调函数处理消息

except KeyboardInterrupt:
    print("\nUser interrupted, disconnecting...")
    client.loop_stop()
    client.disconnect()
    print("Disconnected from MQTT broker")

