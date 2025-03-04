#include <SoftwareSerial.h>

#define LORA_RXD 0  // 连接到 LoRa 模块的 TXD
#define LORA_TXD 1  // 连接到 LoRa 模块的 RXD

// 创建软串口对象
SoftwareSerial loraSerial(LORA_RXD, LORA_TXD);

void setup() {
  // 初始化硬件串口（用于与电脑通信）
  Serial.begin(9600);
  while (!Serial);

  // 初始化软串口（用于与 LoRa 模块通信）
  loraSerial.begin(9600);
  Serial.println("LoRa Sender Initialized!");
}

void loop() {
  Serial.println("Sending packet: Hello LoRa");

  loraSerial.println("Hello LoRa");
  delay(5000);
}