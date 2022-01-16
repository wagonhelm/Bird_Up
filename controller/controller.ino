#include <Servo.h>
#define VERT_PIN A0
#define HORZ_PIN A1
#define SEL_PIN  2
#define light A2 // define input pin
Servo x_servo;  // base servo
Servo y_servo;  // second servo on top
Servo door;

void setup() {
  x_servo.attach(9);  // attaches the servo on pin 9 to the servo object
  y_servo.attach(10);
  door.attach(11);
  pinMode(VERT_PIN, INPUT);
  pinMode(HORZ_PIN, INPUT);
  pinMode(SEL_PIN, INPUT_PULLUP);
  Serial.begin(19200);
}


void loop() {
  int vert = map(analogRead(VERT_PIN), 0, 1023, 0, 180);
  int horz = map(analogRead(HORZ_PIN), 0, 1023, 0, 180);
  int Lvalue = analogRead(light);// read the light
  int mVolt = map(Lvalue,0, 1023, 0, 100);// map analogue reading to 5000mV
  x_servo.write(horz);              // tell servo to go to position in variable 'pos'
  y_servo.write(vert);
  Serial.print(Lvalue);
  Serial.println();
    if (Lvalue <= 35) {
    door.write(150);
  } else {
    door.write(180);
  }
  delay(100);
}
