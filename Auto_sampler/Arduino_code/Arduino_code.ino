#include <Servo.h>
#include <avr/interrupt.h> 

#define In1 9
#define In2 8
#define In3 7
#define In4 6

#define RTC_PERIOD 1250 //normally 1250=10s

Servo myServo;  // 创建Servo对象


void setup() {
  myServo.attach(2);  
  
  pinMode(In1, OUTPUT);
  pinMode(In2, OUTPUT);
  pinMode(In3, OUTPUT);
  pinMode(In4, OUTPUT);

}

void loop() {
  
  static int angle = 0;  
  static bool done = false;

  if (done) {
    return;
  }

  myServo.write(angle);
  delay(1000);
  
  digitalWrite(In3, HIGH);
  digitalWrite(In4, LOW);
  delay(2000);
  digitalWrite(In3, LOW);
  digitalWrite(In4, LOW);

  digitalWrite(In1, LOW);
  digitalWrite(In2, HIGH);
  delay(1000);
  digitalWrite(In1, LOW);
  digitalWrite(In2, LOW);
  delay(2000);


  angle += 15;
  if (angle > 180) {
    done = true; 
  }
}
