import cv2
import serial
import time

# Initialize UART communication (Replace with your serial port)
# For testing, you can use virtual ports created with `socat`.
# Example: port='/tmp/ttyS1' for sender
ser = serial.Serial(port='/tmp/ttyS1', baudrate=9600, timeout=1)

# Load pre-trained Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize video capture (0 for default camera)
cap = cv2.VideoCapture(0)

print("Starting facial recognition simulation...")

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert to grayscale (required for Haar cascades)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        # Mimic sending a success message to the UART (e.g., "Face recognized")
        ser.write(b"Face recognized\n")
        print("Face recognized - Message sent via UART")
        time.sleep(1)  # Add a delay to prevent spamming

    # Display the resulting frame
    cv2.imshow('Facial Recognition Simulation', frame)

    # Break the loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()