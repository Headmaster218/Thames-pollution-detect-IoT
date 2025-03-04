#include <avr/sleep.h>
#include <avr/interrupt.h>
#include <SoftwareSerial.h>

#define TDS_PIN A0
#define TUR_PIN A1
#define PH_PIN A2
#define DO_PIN A3
#define LORA_RXD 0  // Connect the TXD to the LoRa module
#define LORA_TXD 1  // Connect the RXD to the LoRa module

#define VREF 5.0     
#define ADS_RES 1024.0
#define SCOUNT  30           // sum of sample point, array length
#define OFFSET 0.00            //deviation compensate
#define SAMPLE_INTERVAL 100U
#define PRINT_INTERVAL 2000U

//single-point calibration
#define CAL1_V (1600) //unit mv
#define CAL1_T (25)   //unit ℃

// RTC The counter value ---- sleep time
#define RTC_PERIOD 1250

// store the analog value in the array, read from ADC
struct SensorData {
  int Array[SCOUNT];
  int bufferTemp[SCOUNT];
  float voltage;
  float value;
};
SensorData tdsData, turbidityData, pHData, doData;

volatile bool rtc_wakeup = false; // Used to signal whether the RTC is awake
volatile bool rtc_ifsleep = false; // Determine whether the device needs to sleep
int analogBufferIndex = 0,copyIndex = 0;
// !!! remove it when add temp sensor; this variable is also used in DO calculation
int temperature = 25;

// Create soft serial ports (to avoid communication serial port conflict)
SoftwareSerial loraSerial(LORA_RXD, LORA_TXD);

const int DO_Table[41] = {
  14460, 14220, 13820, 13440, 13090, 12740, 12420, 12110, 11810, 11530,
  11260, 11010, 10770, 10530, 10300, 10080, 9860, 9660, 9460, 9270,
  9080, 8900, 8730, 8570, 8410, 8250, 8110, 7960, 7820, 7690,
  7560, 7430, 7300, 7180, 7070, 6950, 6840, 6730, 6630, 6530, 6410
};

void readSensorData(SensorData *Do, SensorData *pH, SensorData *tur, SensorData *tds){
  Do->voltage = getMedianNum(Do->bufferTemp, SCOUNT) * (float)VREF / ADS_RES;
  int V_saturation = CAL1_V + 35 * temperature - CAL1_T * 35;
  Do->value = Do->voltage * DO_Table[temperature] / (V_saturation * 1000);

  tds->voltage = getMedianNum(tds->bufferTemp, SCOUNT) * (float)VREF / ADS_RES;
  float compensationCoefficient = 1.0 + 0.02 * (temperature - 25.0);
  float compensationVolatge = tds->voltage / compensationCoefficient;
  float volSquare_tds = compensationVolatge * compensationVolatge;
  float volCube_tds = volSquare_tds * compensationVolatge;
  tds->value = (133.42f * volCube_tds - 255.86f * volSquare_tds + 857.39f * compensationVolatge) * 0.5f; 

  tur->voltage = getMedianNum(tur->bufferTemp, SCOUNT) * (float)VREF / ADS_RES;
  float volSquare_tur = tur->voltage * tur->voltage;
  tur->value = -1120.4 * volSquare_tur + 5742.3 * tur->voltage - 4352.9;

  pH->voltage = getMedianNum(pH->bufferTemp, SCOUNT) * (float)VREF / ADS_RES;
  pH->value = 3.5 * pH->voltage + OFFSET;

  /*Serial.print("DO Value: ");
  Serial.println(Do->value, 2); 
  Serial.print("TDS Value: ");
  Serial.print(tds->value, 0);
  Serial.println(" ppm");
  Serial.print("Turbidity: ");
  Serial.print(tur->value, 0);
  Serial.println(" NTU");
  Serial.print("pH Value: ");
  Serial.println(pH->value, 2);*/

  loraSerial.print(Do->value); loraSerial.print(",");
  loraSerial.print(tds->value); loraSerial.print(",");
  loraSerial.print(tur->value); loraSerial.print(",");
  loraSerial.println(pH->value); 
}

void sampleData(){
  //every 40ms, read the analog value from the ADC
  static unsigned long analogSampleTime = millis();
  // every 800ms, process the analog data
  static unsigned long printTime = millis();

  if (millis() - analogSampleTime > SAMPLE_INTERVAL) {
    analogSampleTime = millis();
    tdsData.Array[analogBufferIndex] = analogRead(TDS_PIN);
    turbidityData.Array[analogBufferIndex] = analogRead(TUR_PIN);
    pHData.Array[analogBufferIndex] = analogRead(PH_PIN);
    doData.Array[analogBufferIndex] = analogRead(DO_PIN);
    analogBufferIndex++;
    if (analogBufferIndex == SCOUNT) 
      analogBufferIndex = 0;
  }

  if (millis() - printTime > PRINT_INTERVAL) {
    printTime = millis();
    for (int i = 0; i < SCOUNT; i++) {
      tdsData.bufferTemp[i] = tdsData.Array[i];
      turbidityData.bufferTemp[i] = turbidityData.Array[i];
      pHData.bufferTemp[i] = pHData.Array[i];
      doData.bufferTemp[i] = doData.Array[i];
    }
    readSensorData(&doData, &pHData, &turbidityData, &tdsData);
    rtc_ifsleep = true;
  }
}

// medium filter, remove outliers
int getMedianNum(int bArray[], int iFilterLen) {
  int bTab[iFilterLen];
  memcpy(bTab, bArray, iFilterLen * sizeof(int));

  int i, j, bTemp;
  for (j = 0; j < iFilterLen - 1; j++) {
    for (i = 0; i < iFilterLen - j - 1; i++) {
      if (bTab[i] > bTab[i + 1]) {
        bTemp = bTab[i];
        bTab[i] = bTab[i + 1];
        bTab[i + 1] = bTemp;
      }
    }
  }

  if ((iFilterLen & 1) > 0)
    return bTemp = bTab[(iFilterLen - 1) / 2];
  else
    return bTemp = (bTab[iFilterLen / 2] + bTab[iFilterLen / 2 - 1]) / 2;
}

// Setup RTC timer
void setupRTC() {
  // 1. 使能 RTC，并选择 32kHz 内部振荡器作为时钟源
  while (RTC.STATUS > 0); 
  RTC.CLKSEL = RTC_CLKSEL_INT32K_gc; // 选择内部 32kHz 振荡器
  // 2. 设置 RTC 计数器周期
  RTC.PER = RTC_PERIOD;
  // 3. 使能 RTC 溢出中断
  RTC.INTCTRL = RTC_OVF_bm;
  // 4. 使能 RTC 并启动计数
  RTC.CTRLA = RTC_RTCEN_bm | RTC_PRESCALER_DIV256_gc | RTC_RUNSTDBY_bm; // 预分频 256（125Hz 计数）
  // 5. 开启全局中断
  sei();
}

void enterSleep() {
  Serial.println("Fall asleep...");
  Serial.flush();
  set_sleep_mode(SLEEP_MODE_STANDBY); // Enter Standby mode (RTC still running)
  sleep_enable();
  sleep_cpu();  // Enter the sleep mode and wait for the RTC to wake up

  sleep_disable();
}

ISR(RTC_CNT_vect) {
  RTC.INTFLAGS = RTC_OVF_bm; // Clear interrupt flags
  rtc_wakeup = true;
}

void setup(){
    Serial.begin(115200);
    while (!Serial);

    pinMode(TDS_PIN,INPUT);
    pinMode(TUR_PIN,INPUT);
    pinMode(PH_PIN,INPUT);
    pinMode(DO_PIN, INPUT);
    Serial.println("Pins Initialized!");

    loraSerial.begin(9600);
    Serial.println("LoRa Sender Initialized!");
    setupRTC();
    Serial.println("RTC Initialized!");
}

void loop(){
  if (rtc_ifsleep) {
    enterSleep();

    if (rtc_wakeup) {
      Serial.println("RTC wakes up the system");
      rtc_wakeup = false; // Reset flag
    }
    Serial.println("Wake up by RTC!");
    Serial.flush();
  }

  rtc_ifsleep = false;
  
  sampleData();
  delay(1000);
}
