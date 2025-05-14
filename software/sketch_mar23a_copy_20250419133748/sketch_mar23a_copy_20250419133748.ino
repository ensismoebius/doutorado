const int analogPin = A0;    // Analog pin to read the value from
const int windowSize = 10;  // Number of values for the moving average

int values[windowSize];     // Circular buffer to store the values
int index = 0;              // Current index in the buffer
long sum = 0;               // Sum of values in the buffer
bool bufferFilled = false;  // Flag to check if buffer is full

void setup() {
  Serial.begin(9600);
}

void loop() {
  int analogValue = analogRead(analogPin);

  // Subtract the value that is going to be overwritten
  sum -= values[index];

  // Store the new value in the buffer
  values[index] = analogValue;

  // Add the new value to the sum
  sum += analogValue;

  // Move index and wrap around if needed
  index = (index + 1) % windowSize;

  // Calculate average only if buffer is filled at least once
  int average;
  if (bufferFilled) {
    average = sum / windowSize;
  } else {
    average = sum / (index == 0 ? 1 : index);  // prevent division by zero
    if (index == windowSize - 1) bufferFilled = true;
  }

  // Print average value overwriting previous line
  Serial.print("\r          ");
  Serial.print("\r");
  Serial.print("Avg: ");
  Serial.print(average);

  delay(50);
}
