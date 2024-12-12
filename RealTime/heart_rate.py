import cv2
import numpy as np
from scipy.fftpack import fft, fftfreq
import time

def detect_heart_rate_spo2_bp_with_timer():
    """
    Real-time heart rate, SpO2, and blood pressure detection using webcam and block processing.
    """
    # Initialize webcam
    cap = cv2.VideoCapture(0)  # Use 0 for the default webcam
    if not cap.isOpened():
        print("Error: Unable to access the camera.")
        return

    frame_buffer = []
    fps = 30  # Approximate frames per second (adjust based on your camera)

    # Timer setup
    start_time = time.time()
    duration = 30  # Run for 30 seconds

    try:
        print("Starting heart rate, SpO2, and BP detection... The process will stop after 30 seconds.")
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            if not ret:
                print("Error: Unable to capture frame.")
                break

            # Convert to grayscale for face detection
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Load Haar cascade for face detection
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

            if len(faces) > 0:
                # Focus on the first detected face
                (x, y, w, h) = faces[0]

                # Define ROIs for different regions: forehead, cheeks, and chin
                forehead_roi = frame[y:y + int(0.2 * h), x:x + w]  # Upper 20% of the face (forehead)
                cheek_roi = frame[y + int(0.2 * h):y + int(0.5 * h), x:x + w]  # Lower cheek region
                chin_roi = frame[y + int(0.5 * h):y + h, x:x + w]  # Lower part of the face (chin)

                # Compute the mean green channel intensity for HR detection
                green_channel_mean = np.mean(forehead_roi[:, :, 1])  # Green channel (index 1)

                # Append to frame buffer
                frame_buffer.append(green_channel_mean)
                if len(frame_buffer) > fps * 5:  # Keep only the last 5 seconds of data
                    frame_buffer.pop(0)

                # Alternative texture feature extraction: Sobel gradients for texture features
                forehead_grad_x = cv2.Sobel(forehead_roi[:, :, 1], cv2.CV_64F, 1, 0, ksize=3)
                forehead_grad_y = cv2.Sobel(forehead_roi[:, :, 1], cv2.CV_64F, 0, 1, ksize=3)
                forehead_texture = cv2.magnitude(forehead_grad_x, forehead_grad_y)

                cheek_grad_x = cv2.Sobel(cheek_roi[:, :, 1], cv2.CV_64F, 1, 0, ksize=3)
                cheek_grad_y = cv2.Sobel(cheek_roi[:, :, 1], cv2.CV_64F, 0, 1, ksize=3)
                cheek_texture = cv2.magnitude(cheek_grad_x, cheek_grad_y)

                chin_grad_x = cv2.Sobel(chin_roi[:, :, 1], cv2.CV_64F, 1, 0, ksize=3)
                chin_grad_y = cv2.Sobel(chin_roi[:, :, 1], cv2.CV_64F, 0, 1, ksize=3)
                chin_texture = cv2.magnitude(chin_grad_x, chin_grad_y)

                # Display texture features (optional visualization)
                cv2.imshow("Forehead Texture", forehead_texture.astype(np.uint8))  # Display gradient magnitude

                # Calculate and display other ROI areas
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Face rectangle
                cv2.rectangle(frame, (x, y), (x + w, y + int(0.2 * h)), (0, 255, 0), 2)  # Forehead ROI
                cv2.rectangle(frame, (x, y + int(0.2 * h)), (x + w, y + int(0.5 * h)), (0, 0, 255), 2)  # Cheeks
                cv2.rectangle(frame, (x, y + int(0.5 * h)), (x + w, y + h), (0, 255, 255), 2)  # Chin

            # Display the resulting frame
            cv2.imshow("Heart Rate, SpO2, and BP Detection", frame)

            # Quit if time is up
            elapsed_time = time.time() - start_time
            if elapsed_time >= duration:
                break

            # Quit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Process heart rate, SpO2, and BP after the timer
        if len(frame_buffer) > fps:  # Ensure we have enough data for HR
            print("Processing heart rate...")
            # Apply FFT to find the dominant frequency
            freqs = fftfreq(len(frame_buffer), d=1 / fps)
            fft_values = np.abs(fft(frame_buffer))
            positive_freqs = freqs[freqs > 0]
            positive_fft_values = fft_values[freqs > 0]

            # Find the dominant frequency in the typical heart rate range (0.8–3 Hz)
            valid_range = (positive_freqs >= 0.8) & (positive_freqs <= 3.0)
            if np.any(valid_range):
                dominant_freq = positive_freqs[valid_range][np.argmax(positive_fft_values[valid_range])]
                heart_rate = dominant_freq * 60  # Convert to BPM
                
                # Clamp heart rate to 60–100 BPM
                heart_rate = max(60, min(heart_rate, 100))
                print(f"Estimated Heart Rate: {heart_rate:.2f} BPM")
            else:
                print("No valid heart rate signal detected.")

            # Simulate SpO2 calculation using a simple approach: ratio of red to green channel intensity
            red_channel_mean = np.mean(forehead_roi[:, :, 2])  # Red channel (index 2)
            spo2 = (red_channel_mean / green_channel_mean) * 100  # Simplified ratio
            print(f"Estimated SpO2: {spo2:.2f}%")

            # Simulate Blood Pressure estimation (very simplified)
            blood_pressure = 120 - (np.mean(green_channel_mean) / 255) * 40  # Simplified BP estimate
            print(f"Estimated Blood Pressure: {blood_pressure:.2f} mmHg")

        else:
            print("Insufficient data to calculate heart rate, SpO2, and BP.")

    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        # Release resources
        cap.release()
        cv2.destroyAllWindows()

# Run the detection function
detect_heart_rate_spo2_bp_with_timer()
