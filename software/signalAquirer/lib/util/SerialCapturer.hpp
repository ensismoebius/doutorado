// SerialCapturer.hpp
#ifndef SERIAL_CAPTURER_H
#define SERIAL_CAPTURER_H

#include "ICapturer.hpp"
#include <fcntl.h>   // File control definitions (open, fcntl)
#include <termios.h> // POSIX terminal control for configuring serial ports
#include <unistd.h>  // Standard symbolic constants and types (close, read)
#include <cstring>   // C string handling
#include <vector>    // std::vector for storing samples
#include <iostream>  // std::cout, std::cerr

// SerialCapturer class handles serial communication
class SerialCapturer : public ICapturer
{
private:
    std::string devicePath;        // Serial device path (e.g., "/dev/ttyUSB0")
    int baudRate;                  // Baud rate (e.g., B9600)
    int fd;                        // File descriptor for serial port
    bool capturing;                // Capturing status flag
    mutable std::string lastError; // Stores last error message

public:
    // Constructor initializes serial device path and baud rate
    SerialCapturer(const std::string &device, int baudrate);

    // Starts serial communication
    bool start() override;

    // Stops serial communication
    void stop() override;

    // Checks if capturing is active
    bool isCapturing() const override;

    // Returns the last error message
    const std::string &last_error() const override;

    // Reads available data from the serial port and returns it as a vector of floats
    const std::vector<float> getAvailableSamples() override;

    // Destructor ensures the serial port is closed when the object is destroyed
    ~SerialCapturer();
};

#endif // SERIAL_CAPTURER_H

// Constructor implementation
SerialCapturer::SerialCapturer(const std::string &device, int baudrate)
    : devicePath(device), baudRate(baudrate), fd(-1), capturing(false) {}

// Opens the serial port and configures it
bool SerialCapturer::start()
{
    fd = open(devicePath.c_str(), O_RDWR | O_NOCTTY);
    if (fd == -1)
    {
        lastError = "Failed to open serial port";
        return false;
    }

    struct termios options;
    tcgetattr(fd, &options); // Get current serial port settings

    // Set baud rate
    cfsetispeed(&options, baudRate);
    cfsetospeed(&options, baudRate);

    // Configure serial port options
    options.c_cflag |= (CLOCAL | CREAD); // Enable receiver and ignore modem control lines
    options.c_cflag &= ~PARENB;          // Disable parity
    options.c_cflag &= ~CSTOPB;          // Use one stop bit
    options.c_cflag &= ~CSIZE;           // Clear size bits
    options.c_cflag |= CS8;              // Set 8-bit character size

    tcsetattr(fd, TCSANOW, &options); // Apply settings immediately

    capturing = true;
    return true;
}

// Closes the serial port and stops capturing
void SerialCapturer::stop()
{
    if (fd != -1)
    {
        close(fd);
        fd = -1;
    }
    capturing = false;
}

// Returns whether data capturing is active
bool SerialCapturer::isCapturing() const
{
    return capturing;
}

// Returns the last error message
const std::string &SerialCapturer::last_error() const
{
    return lastError;
}

// Reads data from the serial port and converts it to a float vector
const std::vector<float> SerialCapturer::getAvailableSamples()
{
    static std::vector<float> samples;

    if (!capturing || fd == -1)
    {
        return samples;
    }

    char buffer[256];
    int bytesRead = read(fd, buffer, sizeof(buffer) - 1);

    if (bytesRead > 0)
    {
        buffer[bytesRead] = '\0'; // Null-terminate received data
        try
        {
            samples.push_back(std::stof(buffer)); // Convert received data to float
        }
        catch (...)
        {
            lastError = "Invalid data received";
        }
    }

    return samples;
}

// Destructor ensures the serial port is closed
SerialCapturer::~SerialCapturer()
{
    stop();
}
