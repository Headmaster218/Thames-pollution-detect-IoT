#include <avr/sleep.h>
#include <avr/interrupt.h>

// RTC 计数器值（10 秒 = 10 × 32.768kHz / 32）
#define RTC_PERIOD 1250   // 320 计数 ≈ 10 秒

volatile bool rtc_wakeup = false; // 用于标志 RTC 是否唤醒

void setup() {
  Serial.begin(115200);
  while (!Serial);  // 等待串口初始化

  setupRTC();   // 配置 RTC
}

void loop() {
  enterSleep();

  if (rtc_wakeup) {
    Serial.println("RTC 唤醒系统");
    rtc_wakeup = false; // 复位标志
  }
  
  Serial.println("被 RTC 唤醒！");
  Serial.flush();
  delay(2000); // 模拟任务执行
}

// 配置 RTC 定时器
void setupRTC() {
  // 1. 使能 RTC，并选择 32.768kHz 内部振荡器作为时钟源
  while (RTC.STATUS > 0); 
  RTC.CLKSEL = RTC_CLKSEL_INT32K_gc; // 选择内部 32kHz 振荡器
  // 2. 设置 RTC 计数器周期（每 10 秒触发一次中断）
  RTC.PER = RTC_PERIOD;
  // 3. 使能 RTC 溢出中断
  RTC.INTCTRL = RTC_OVF_bm;
  // 4. 使能 RTC 并启动计数
  RTC.CTRLA = RTC_RTCEN_bm | RTC_PRESCALER_DIV256_gc | RTC_RUNSTDBY_bm; // 预分频 256（125Hz 计数）
  // 5. 开启全局中断
  sei();
}

// RTC 中断处理（当 RTC 计数溢出时触发）
ISR(RTC_CNT_vect) {
  RTC.INTFLAGS = RTC_OVF_bm; // 清除中断标志
  rtc_wakeup = true;
}

// 进入低功耗睡眠模式
void enterSleep() {
  Serial.println("进入睡眠...");
  Serial.flush();
  set_sleep_mode(SLEEP_MODE_STANDBY); // 进入 Standby 模式（RTC 仍然运行）
  sleep_enable();
  sleep_cpu();  // 进入睡眠模式，等待 RTC 唤醒

  sleep_disable();
}