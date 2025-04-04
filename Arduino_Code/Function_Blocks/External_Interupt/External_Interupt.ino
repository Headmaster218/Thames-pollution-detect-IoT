#include <avr/wdt.h>  
#include <avr/sleep.h>
#include <avr/power.h>

const byte swPin = 2;   // 按钮（外部中断引脚）
const byte ledPin = 25; // LED 引脚
byte times = 0;         // 记录执行次数
volatile byte state = 0;// 记录是否从睡眠中唤醒

// 中断服务函数
void wakeISR() {
    state = 1; // 唤醒后标记状态
}

void enterSleep() {
    Serial.println("Go to sleep...");
    Serial.flush(); // 确保所有数据发送完成

    sleep_enable();                            // 允许进入睡眠模式
    set_sleep_mode(SLEEP_MODE_PWR_DOWN);       // 设定为掉电模式
    sleep_cpu();                               // 进入睡眠模式

    // 唤醒后执行
    sleep_disable(); // 禁止睡眠，防止立即进入睡眠
}

void setup() {
    Serial.begin(9600);
    pinMode(ledPin, OUTPUT);
    pinMode(swPin, INPUT_PULLUP); // 使用内部上拉电阻

    attachInterrupt(digitalPinToInterrupt(swPin), wakeISR, CHANGE); // 设置外部中断

    Serial.println("Running...");
}

void loop() {
    if (state == 1) {
        Serial.println("Was sleeping...");
    }

    state = 0;
    digitalWrite(ledPin, !digitalRead(ledPin)); // 切换 LED 状态
    delay(500);
    times++;
    Serial.println(times);

    if (times > 5) {
        times = 0;
        enterSleep(); // 进入睡眠模式
    }
}
