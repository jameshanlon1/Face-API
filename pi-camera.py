from flask import Flask, render_template, Response
import cv2
import threading
import time

app = Flask(__name__)

# Open the default camera (usually your laptop's webcam)
camera = cv2.VideoCapture(0)

# This will store the latest frame from the camera
current_frame = None

# Continuously capture frames from the camera in a background thread
def capture_frames():
    global current_frame
    while True:
        success, frame = camera.read()
        if success:
            current_frame = frame
        time.sleep(0.03)  # Slight delay to reduce CPU usage

# Start the frame capture thread
threading.Thread(target=capture_frames, daemon=True).start()

# This function turns camera frames into a stream of JPEGs
def generate_frames():
    global current_frame
    while True:
        if current_frame is not None:
            ret, buffer = cv2.imencode('.jpg', current_frame)
            if ret:
                # This format lets the browser display a live stream
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.05)

# Route to get the live video feed
@app.route('/camera_feed')
def camera_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Simple page to show the camera stream
@app.route('/camera')
def camera_page():
    return render_template('camera.html')

if __name__ == '__main__':
    # Run the app on all network interfaces (good for testing on phone too)
    app.run(host='0.0.0.0', port=5000)
