#include <Servo.h>

// =====================================================
// L298N - DC Motor
// =====================================================

// Motor A
const int ENA = 3;
const int IN1 = 5;
const int IN2 = 6;

// Motor B
const int ENB = 11;
const int IN3 = 7;
const int IN4 = 8;

// DC Motor PWM Speed
const int MOTOR_SPEED = 150;

// Motor safety timeout
const unsigned long MOTOR_TIMEOUT = 300;

// 마지막 모터 명령을 받은 시간
unsigned long lastMotorCommandTime = 0;


// =====================================================
// Servo
// =====================================================

Servo panServo;
Servo tiltServo;

const int PAN_PIN  = 9;
const int TILT_PIN = 10;


// =====================================================
// Pan Servo
// 0° ~ 180°
// =====================================================

const int PAN_CENTER = 90;
const int PAN_MIN = 0;
const int PAN_MAX = 180;

int panAngle = PAN_CENTER;


// =====================================================
// Tilt Servo
// 60° ~ 120°
// =====================================================

const int TILT_CENTER = 90;
const int TILT_MIN = 60;
const int TILT_MAX = 120;

int tiltAngle = TILT_CENTER;


// =====================================================
// Servo movement step
// =====================================================

const int SERVO_STEP = 1;


// =====================================================
// Setup
// =====================================================

void setup() {

  // DC Motor
  pinMode(ENA, OUTPUT);
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);

  pinMode(ENB, OUTPUT);
  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);

  stopMotor();


  // Servo
  panServo.attach(PAN_PIN);
  tiltServo.attach(TILT_PIN);

  // Pan 중앙
  panServo.write(panAngle);

  // Tilt 중앙
  tiltServo.write(tiltAngle);


  // UART
  Serial.begin(115200);


  // 시작 시 안전 상태
  lastMotorCommandTime = millis();
}


// =====================================================
// Main Loop
// =====================================================

void loop() {

  // ---------------------------------------------------
  // Serial Command
  // ---------------------------------------------------

  if (Serial.available() > 0) {

    char command = Serial.read();

    switch (command) {

      // -----------------------------------------------
      // DC Motor
      // -----------------------------------------------

      case 'W':
      case 'w':
        forward();
        lastMotorCommandTime = millis();
        break;


      case 'S':
      case 's':
        backward();
        lastMotorCommandTime = millis();
        break;


      case 'A':
      case 'a':
        left();
        lastMotorCommandTime = millis();
        break;


      case 'D':
      case 'd':
        right();
        lastMotorCommandTime = millis();
        break;


      case 'X':
      case 'x':
        stopMotor();
        lastMotorCommandTime = millis();
        break;


      // -----------------------------------------------
      // Pan
      //
      // 실제 카메라 장착 방향 때문에
      // J / L 방향 반전
      // -----------------------------------------------

      case 'J':
      case 'j':
        panRight();
        break;


      case 'L':
      case 'l':
        panLeft();
        break;


      // -----------------------------------------------
      // Tilt
      //
      // 실제 카메라 장착 방향에 맞춰
      // I / K 방향 반전
      // -----------------------------------------------

      case 'I':
      case 'i':
        tiltUp();
        break;


      case 'K':
      case 'k':
        tiltDown();
        break;


      // -----------------------------------------------
      // Pan / Tilt Center
      // -----------------------------------------------

      case 'C':
      case 'c':
        centerServo();
        break;
    }
  }


  // ---------------------------------------------------
  // Motor Safety Timeout
  // ---------------------------------------------------

  if (millis() - lastMotorCommandTime > MOTOR_TIMEOUT) {
    stopMotor();
  }
}


// =====================================================
// DC Motor
// =====================================================

// Forward
void forward() {

  digitalWrite(IN1, HIGH);
  digitalWrite(IN2, LOW);

  digitalWrite(IN3, HIGH);
  digitalWrite(IN4, LOW);

  analogWrite(ENA, MOTOR_SPEED);
  analogWrite(ENB, MOTOR_SPEED);
}


// =====================================================
// Backward
// =====================================================

void backward() {

  digitalWrite(IN1, LOW);
  digitalWrite(IN2, HIGH);

  digitalWrite(IN3, LOW);
  digitalWrite(IN4, HIGH);

  analogWrite(ENA, MOTOR_SPEED);
  analogWrite(ENB, MOTOR_SPEED);
}


// =====================================================
// Left
// =====================================================

void left() {

  digitalWrite(IN1, LOW);
  digitalWrite(IN2, HIGH);

  digitalWrite(IN3, HIGH);
  digitalWrite(IN4, LOW);

  analogWrite(ENA, MOTOR_SPEED);
  analogWrite(ENB, MOTOR_SPEED);
}


// =====================================================
// Right
// =====================================================

void right() {

  digitalWrite(IN1, HIGH);
  digitalWrite(IN2, LOW);

  digitalWrite(IN3, LOW);
  digitalWrite(IN4, HIGH);

  analogWrite(ENA, MOTOR_SPEED);
  analogWrite(ENB, MOTOR_SPEED);
}


// =====================================================
// Stop
// =====================================================

void stopMotor() {

  analogWrite(ENA, 0);
  analogWrite(ENB, 0);

  digitalWrite(IN1, LOW);
  digitalWrite(IN2, LOW);

  digitalWrite(IN3, LOW);
  digitalWrite(IN4, LOW);
}


// =====================================================
// Pan Servo
// =====================================================

void panLeft() {

  if (panAngle > PAN_MIN) {

    panAngle -= SERVO_STEP;

    if (panAngle < PAN_MIN) {
      panAngle = PAN_MIN;
    }

    panServo.write(panAngle);
  }
}


// =====================================================
// Pan Right
// =====================================================

void panRight() {

  if (panAngle < PAN_MAX) {

    panAngle += SERVO_STEP;

    if (panAngle > PAN_MAX) {
      panAngle = PAN_MAX;
    }

    panServo.write(panAngle);
  }
}


// =====================================================
// Tilt Servo
// =====================================================
// 실제 동작 기준으로 방향을 반전함.
//
// I → tiltUp() → 각도 감소 → 실제 위로
// K → tiltDown() → 각도 증가 → 실제 아래로
// =====================================================

void tiltUp() {

  if (tiltAngle > TILT_MIN) {

    tiltAngle -= SERVO_STEP;

    if (tiltAngle < TILT_MIN) {
      tiltAngle = TILT_MIN;
    }

    tiltServo.write(tiltAngle);
  }
}


// =====================================================
// Tilt Down
// =====================================================

void tiltDown() {

  if (tiltAngle < TILT_MAX) {

    tiltAngle += SERVO_STEP;

    if (tiltAngle > TILT_MAX) {
      tiltAngle = TILT_MAX;
    }

    tiltServo.write(tiltAngle);
  }
}


// =====================================================
// Center Pan / Tilt
// =====================================================

void centerServo() {

  // Pan 중앙 = 90°
  panAngle = PAN_CENTER;

  // Tilt 중앙 = 90°
  tiltAngle = TILT_CENTER;

  panServo.write(panAngle);
  tiltServo.write(tiltAngle);
}
