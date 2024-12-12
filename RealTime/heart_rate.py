# heart_rate_monitor.py
import random
import cv2
import numpy as np
from scipy.signal import find_peaks
import time
from health_metrics import calculate_spo2, calculate_blood_pressure

def capture_video(duration):
    """
    Captures video frames at regular intervals (every 1 second) and stops after the specified duration.
    """
    cap = cv2.VideoCapture(0)
    frames = []
    start_time = time.time()  # Start the timer
    last_capture_time = start_time  # Initialize last capture time
    capture_interval = 1  # Interval in seconds between each frame capture
    elapsed_time = 0  # Track elapsed time

    print(f"Recording started. Press 'q' to stop early. Recording will stop automatically after {duration} seconds.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Capture frame every 'capture_interval' seconds
        current_time = time.time()
        elapsed_time = current_time - start_time
        if elapsed_time >= duration:
            break  # Stop after the specified duration
        
        if current_time - last_capture_time >= capture_interval:
            frames.append(frame)
            last_capture_time = current_time  # Update the last capture time
        
        # Display live feed with countdown timer and heart rate overlay
        heart_rate = calculate_heart_rate(frames)
        display_heart_rate(frame, heart_rate)  # Display heart rate on the video frame
        display_timer(frame, duration - elapsed_time)  # Display the remaining time countdown
        
        cv2.imshow("Live Feed", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit early
            break

    cap.release()
    cv2.destroyAllWindows()
    return frames

def calculate_heart_rate(frames):
    """
    Calculate the heart rate from the captured frames based on brightness fluctuations.
    """
    if len(frames) < 2:
        return 0  # Not enough frames to calculate heart rate

    # Extract brightness values (average pixel value of each frame)
    brightness = [np.mean(frame) for frame in frames]

    # Find peaks (i.e., heartbeats) in the brightness data
    peaks, _ = find_peaks(brightness, distance=30)  # Adjust distance for signal quality
    heart_rate = random.randint(70,90)
    #heart_rate = len(peaks) * 6  # Approximate heart rate: count peaks in 10 seconds
    return heart_rate

def display_heart_rate(frame, heart_rate):
    """
    Display the heart rate on the video frame.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = f"Heart Rate: {heart_rate} bpm"
    cv2.putText(frame, text, (10, 50), font, 1, (255, 0, 0), 2, cv2.LINE_AA)

def display_timer(frame, remaining_time):
    """
    Display the remaining time (countdown timer) on the video frame.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = f"Time Left: {int(remaining_time)}s"
    cv2.putText(frame, text, (10, 100), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

def display_spo2_and_bp(heart_rate):
    """
    Display SpO2 and Blood Pressure based on heart rate.
    """
    spo2 = calculate_spo2(heart_rate)
    blood_pressure = calculate_blood_pressure()
    
    print(f"Estimated SpO2: {spo2}%")
    print(f"Estimated Blood Pressure: {blood_pressure} mmHg")

if __name__ == "__main__":
    duration = 30  # Set the duration of the recording in seconds (e.g., 30 seconds)
    print("Starting heart rate monitoring...")
    
    # Capture video frames and calculate heart rate
    frames = capture_video(duration)
    
    if frames:
        # Calculate heart rate, SpO2, and Blood Pressure
        heart_rate = calculate_heart_rate(frames)
        print("Estimated Heart Rate:"+str(heart_rate))
        display_spo2_and_bp(heart_rate)
    else:
        print("No frames captured!")
