// #include <Brain.h>

// // Set up the brain parser, pass it the hardware serial object you want to listen on.
// Brain brain(Serial);

// void setup() {
//   // Start the hardware serial.
//   Serial.begin(9600);
// }

// void loop() {
//   // Expect packets about once per second.
//   // The .readCSV() function returns a string (well, char *)
//   // listing the most recent brain data, in the following format:
//   // "signal strength, attention, meditation, delta, theta, low alpha,
//   //  high alpha, low beta, high beta, low gamma, high gamma"
//   if (brain.update()) {
//     Serial.println(brain.readCSV());
//   }
// }

#include <Brain.h>
#include <WiFi.h>
#include <WebServer.h>

// Access point network credentials
const char* ap_ssid = "BrainSensor";      // Name of the WiFi network to broadcast
const char* ap_password = "12345678";  // Password for the WiFi network (min 8 characters)

// IP Address for the AP
IPAddress local_ip(192, 168, 4, 1);
IPAddress gateway(192, 168, 4, 1);
IPAddress subnet(255, 255, 255, 0);

// Set up the brain parser on Serial2
Brain brain(Serial);

// Web server on port 80
WebServer server(80);

// Variables to store the latest brain data
String latestBrainData = "No data yet";

void setup() {
  // Start debug serial
  Serial.begin(115200);
  
//   // Start the serial for the brain sensor
  Serial.begin(9600);
  
  // Configure ESP32 as Access Point
  WiFi.mode(WIFI_AP);
  WiFi.softAPConfig(local_ip, gateway, subnet);
  WiFi.softAP(ap_ssid, ap_password);
  
  Serial.println("Access Point Started");
  Serial.print("IP address: ");
  Serial.println(local_ip.toString());
  Serial.print("Network name: ");
  Serial.println(ap_ssid);
  
  // Define server routes
  server.on("/", handleRoot);
  server.on("/data", handleData);
  server.onNotFound(handleNotFound);
  
  // Start the server
  server.begin();
  Serial.println("HTTP server started");
}

void loop() {
  // Handle client requests
  server.handleClient();
  
  // Check for brain data updates
  if (brain.update()) {
    // Store the latest brain data
    // Serial.println(brain.readCSV());

    latestBrainData = brain.readCSV();
    
    // Print to serial monitor for debugging
    Serial.println(latestBrainData);
  }
  
  // Small delay to prevent watchdog timer issues
  delay(10);
}

// Handle root path
void handleRoot() {
  String html = "<html><head>";
  html += "<title>Brain Sensor Monitor</title>";
  html += "<meta http-equiv='refresh' content='1'>"; // Auto-refresh every 1 second
  html += "<style>";
  html += "body { font-family: Arial, sans-serif; margin: 20px; }";
  html += "h1 { color: #333366; }";
  html += "table { border-collapse: collapse; width: 100%; }";
  html += "th, td { text-align: left; padding: 8px; border: 1px solid #ddd; }";
  html += "th { background-color: #f2f2f2; }";
  html += "button { padding: 10px 20px; font-size: 16px; background-color: #333366; color: white; border: none; border-radius: 5px; cursor: pointer; }";
  html += "button:hover { background-color: #555588; }";
  html += "</style></head><body>";
  html += "<h1>Brain Sensor Monitor</h1>";
  
  // Data display section
  if (latestBrainData != "No data yet") {
    html += "<table><tr>";
    html += "<th>Signal</th><th>Attention</th><th>Meditation</th><th>Delta</th><th>Theta</th>";
    html += "<th>Low Alpha</th><th>High Alpha</th><th>Low Beta</th><th>High Beta</th>";
    html += "<th>Low Gamma</th><th>High Gamma</th>";
    html += "</tr><tr>";
    
    int commaIndex = 0;
    int nextCommaIndex = 0;
    
    for (int i = 0; i < 11; i++) {
      nextCommaIndex = latestBrainData.indexOf(',', commaIndex);
      
      if (nextCommaIndex != -1) {
        html += "<td>" + latestBrainData.substring(commaIndex, nextCommaIndex) + "</td>";
        commaIndex = nextCommaIndex + 1;
      } else {
        // Last value
        html += "<td>" + latestBrainData.substring(commaIndex) + "</td>";
      }
    }
    
    html += "</tr></table>";
  } else {
    html += "<p>Waiting for brain data...</p>";
  }

  html += "<h2>Data Logger</h2>";
  html += "<button id='logBtn'>Hold to Log Data</button>";
  html += "<p id='status'></p>";

  html += "<script>";
  html += "let isLogging = false;";
  html += "let loggedData = 'Signal,Attention,Meditation,Delta,Theta,LowAlpha,HighAlpha,LowBeta,HighBeta,LowGamma,HighGamma\\n';";
  html += "let fetchInterval;";
  html += "const logBtn = document.getElementById('logBtn');";
  html += "const statusText = document.getElementById('status');";
  
  html += "logBtn.addEventListener('mousedown', startLogging);";
  html += "logBtn.addEventListener('mouseup', stopLogging);";
  html += "logBtn.addEventListener('mouseleave', stopLogging);";

  html += "function startLogging() {";
  html += "  isLogging = true;";
  html += "  statusText.innerText = 'Logging...';";
  
  html += "  fetchInterval = setInterval(async () => {";
  html += "    const response = await fetch('/data');";
  html += "    const text = await response.text();";
  html += "    loggedData += text + '\\n';";
  html += "  }, 200);"; // Fetch every 200ms
  
  html += "}";
  
  html += "function stopLogging() {";
  html += "  if (!isLogging) return;";
  html += "  clearInterval(fetchInterval);";
  html += "  isLogging = false;";
  html += "  statusText.innerText = 'Download ready.';";
  
  html += "  // Trigger download";
  html += "  const blob = new Blob([loggedData], { type: 'text/csv' });";
  html += "  const url = URL.createObjectURL(blob);";
  
  html += "  const a = document.createElement('a');";
  html += "  a.href = url;";
  html += "  a.download = 'brain_data.csv';";
  html += "  a.click();";
  
  html += "  URL.revokeObjectURL(url);";
  html += "}";
  html += "</script>";

  html += "<p><a href='/data'>View raw data</a></p>";
  html += "</body></html>";
  
  server.send(200, "text/html", html);
}


// Handle data path - returns raw CSV
void handleData() {
  server.send(200, "text/plain", latestBrainData);
}

// Handle not found
void handleNotFound() {
  server.send(404, "text/plain", "404: Not found");
}
