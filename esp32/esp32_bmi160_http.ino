/*
 =====================================================
   ESP32 + BMI160 — Thu tín hiệu rung → HTTP POST
 =====================================================
 Thư viện cần cài (Arduino IDE / PlatformIO):
   - BMI160-Arduino  (by DFRobot)
   - ArduinoJson     (by Benoit Blanchon) >= v6
   - WiFi            (built-in ESP32)
   - HTTPClient      (built-in ESP32)

 Sơ đồ nối dây BMI160 (I2C):
   BMI160 VCC  → ESP32 3.3V
   BMI160 GND  → ESP32 GND
   BMI160 SDA  → ESP32 GPIO 21
   BMI160 SCL  → ESP32 GPIO 22
   BMI160 SDO  → GND  (I2C address = 0x68)
 =====================================================
*/

#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include <BMI160Gen.h>

// ============================================================
// CẤU HÌNH — chỉnh sửa phần này
// ============================================================
const char* WIFI_SSID     = "YOUR_WIFI_SSID";
const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";
const char* SERVER_IP     = "192.168.1.100";  // IP máy tính chạy predict_server.py
const int   SERVER_PORT   = 5000;
const int   SEGMENT_LEN   = 512;   // số samples mỗi lần gửi
                                    // (BMI160 max 1600Hz → 512 samples = ~0.32s)
const int   SAMPLE_RATE   = 1600;  // Hz — BMI160 ODR
const int   SAMPLE_DELAY_US = 1000000 / SAMPLE_RATE;  // ~625 µs

// I2C address BMI160
const int BMI160_ADDR = 0x68;  // SDO=GND → 0x68, SDO=VCC → 0x69

// ============================================================
// BIẾN TOÀN CỤC
// ============================================================
float samples[SEGMENT_LEN];
int   sampleCount = 0;
unsigned long lastSampleTime = 0;

String serverURL;
bool   wifiConnected = false;

// ============================================================
// SETUP
// ============================================================
void setup() {
    Serial.begin(115200);
    delay(1000);

    Serial.println("\n========================================");
    Serial.println("  ESP32 + BMI160 Motor Fault Detection");
    Serial.println("========================================");

    // --- Khởi động BMI160 ---
    Serial.print("Khởi động BMI160... ");
    BMI160.begin(BMI160GenClass::I2C_MODE, BMI160_ADDR);

    // Cài đặt accelerometer
    BMI160.setAccelerometerRate(1600);    // 1600 Hz ODR
    BMI160.setAccelerometerRange(2);      // ±2g range (nhạy nhất)
    BMI160.autoCalibrateXAccelOffset(0);
    BMI160.autoCalibrateYAccelOffset(0);
    BMI160.autoCalibrateZAccelOffset(1); // z = 1g (hướng xuống)
    Serial.println("OK");

    // --- Kết nối WiFi ---
    Serial.printf("Kết nối WiFi: %s\n", WIFI_SSID);
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
    int retry = 0;
    while (WiFi.status() != WL_CONNECTED && retry < 20) {
        delay(500);
        Serial.print(".");
        retry++;
    }

    if (WiFi.status() == WL_CONNECTED) {
        wifiConnected = true;
        Serial.printf("\n✓ WiFi OK — IP: %s\n", WiFi.localIP().toString().c_str());
    } else {
        Serial.println("\n✗ WiFi thất bại — chạy offline");
    }

    serverURL = "http://" + String(SERVER_IP) + ":" +
                String(SERVER_PORT) + "/predict";

    Serial.printf("Server URL: %s\n", serverURL.c_str());
    Serial.printf("Segment: %d samples @ %d Hz\n", SEGMENT_LEN, SAMPLE_RATE);
    Serial.println("========================================\n");
}

// ============================================================
// LOOP — Thu mẫu liên tục
// ============================================================
void loop() {
    unsigned long now = micros();

    // Lấy mẫu đúng tần số
    if (now - lastSampleTime >= SAMPLE_DELAY_US) {
        lastSampleTime = now;

        // Đọc gia tốc trục Z (hướng rung chính của motor)
        int16_t ax, ay, az;
        BMI160.readAccelerometer(ax, ay, az);

        // Chuyển raw → g (±2g range: 1g = 16384 LSB)
        float gz = (float)az / 16384.0f;

        samples[sampleCount++] = gz;

        // Đủ SEGMENT_LEN samples → gửi lên server
        if (sampleCount >= SEGMENT_LEN) {
            sampleCount = 0;
            sendToServer();
        }
    }
}

// ============================================================
// GỬI DATA LÊN SERVER
// ============================================================
void sendToServer() {
    if (!wifiConnected || WiFi.status() != WL_CONNECTED) {
        Serial.println("[WARN] WiFi mất kết nối");
        WiFi.reconnect();
        return;
    }

    // --- Tạo JSON ---
    // Dùng DynamicJsonDocument để chứa mảng lớn
    DynamicJsonDocument doc(SEGMENT_LEN * 12 + 256);
    doc["fs"] = SAMPLE_RATE;

    JsonArray arr = doc.createNestedArray("samples");
    for (int i = 0; i < SEGMENT_LEN; i++) {
        arr.add(round(samples[i] * 10000) / 10000.0f);  // 4 chữ số thập phân
    }

    String jsonStr;
    serializeJson(doc, jsonStr);

    // --- HTTP POST ---
    HTTPClient http;
    http.begin(serverURL);
    http.addHeader("Content-Type", "application/json");
    http.setTimeout(5000);  // 5 giây timeout

    unsigned long t0 = millis();
    int httpCode = http.POST(jsonStr);
    unsigned long elapsed = millis() - t0;

    if (httpCode == 200) {
        String response = http.getString();

        // Parse response
        DynamicJsonDocument res(512);
        deserializeJson(res, response);

        String fault      = res["fault"].as<String>();
        float  confidence = res["confidence"].as<float>();
        int    label      = res["label"].as<int>();

        // In kết quả ra Serial
        Serial.printf("[%lu ms] Fault: %-20s Confidence: %.1f%%  (HTTP: %dms)\n",
                      millis(), fault.c_str(), confidence, (int)elapsed);

        // Cảnh báo nếu không phải Normal
        if (label != 0) {
            Serial.printf("  ⚠️  CẢNH BÁO: Phát hiện %s!\n", fault.c_str());
            // TODO: Bật LED cảnh báo, buzzer, v.v.
            // digitalWrite(LED_PIN, HIGH);
        }
    } else {
        Serial.printf("[ERROR] HTTP %d — %s\n", httpCode, http.errorToString(httpCode).c_str());
    }

    http.end();
}
