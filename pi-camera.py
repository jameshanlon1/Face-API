from flask import Flask, render_template, Response
import cv2
import threading
import time

app = Flask(__name__)
camera = cv2.VideoCapture(0)
current_frame = None

def capture_frames():
    global current_frame
    while True:
        success, frame = camera.read()
        if success:
            current_frame = frame
        time.sleep(0.03)

threading.Thread(target=capture_frames, daemon=True).start()

def generate_frames():
    global current_frame
    while True:
        if current_frame is not None:
            ret, buffer = cv2.imencode('.jpg', current_frame)
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.05)

@app.route('/camera_feed')
def camera_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/camera')
def camera_page():
    return render_template('camera.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
