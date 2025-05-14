import serial
import time

# Set up the serial connection to /dev/ttyACM0
ser = serial.Serial('/dev/ttyACM0', 115200)

# Set the window size in seconds (3 seconds)
window_size = 3
start_time = time.time()
sample_count = 0
data_received = 0

while True:
    # Read the buffer size (128 bytes)
    data = ser.read(128)
    
    # Increment the sample count by the number of bytes received
    sample_count += len(data)
    
    # Keep track of the total number of bytes received
    data_received += len(data)
    
    # Check if the 3-second window has passed
    if time.time() - start_time >= window_size:
        # Calculate and print the average sample rate
        avg_sample_rate = sample_count / window_size
        print(f"Average Sample Rate: {avg_sample_rate:.2f} Hz")
        
        # Reset counters for the next 3-second window
        start_time = time.time()
        sample_count = 0
