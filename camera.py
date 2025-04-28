#!/usr/bin/env python3

import cv2
import time
from picamera2 import Picamera2

# Setup camera
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"size": (1280, 720)}))
picam2.start()
time.sleep(1)  # Warm-up

print("Press 'q' to exit preview")

try:
    while True:
        frame = picam2.capture_array()
        
        # Show the frame in a window
        cv2.imshow("Live Camera Preview", frame)
        
        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    pass

finally:
    # Cleanup
    picam2.close()
    cv2.destroyAllWindows()
    print("Camera preview closed.")
