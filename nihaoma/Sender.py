import serial
import time

# Open the first virtual serial port
sender = serial.Serial('/tmp/ttyS1', baudrate=9600, timeout=1)

while True:
    message = "Hello from Sender!\n"
    sender.write(message.encode())  # Send message
    print(f"Sent: {message.strip()}")
    time.sleep(1)  # Wait 1 second before sending the next message