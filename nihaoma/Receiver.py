import serial

# Open the second virtual serial port
receiver = serial.Serial('/tmp/ttyS2', baudrate=9600, timeout=1)

while True:
    if receiver.in_waiting > 0:  # Check if data is available
        message = receiver.readline().decode().strip()  # Read and decode the message
        print(f"Received: {message}")