# health_metrics.py
import random

def calculate_spo2(heart_rate):
    """
    A simplified method to approximate SpO2 based on heart rate.
    In a real application, you would need actual sensor data for accurate measurements.
    """
    import random
    # SpO2 is usually between 95% and 100%. 
    # For the sake of the simulation, we'll inversely relate it to heart rate.
    if heart_rate < 60:
        spo2 = 98  # Lower heart rate, higher SpO2
    elif heart_rate > 100:
        spo2 = 95  # Higher heart rate, lower SpO2
    else:
        spo2 = 96 + (heart_rate - 60) * 0.02  # Approximation of SpO2 between 95% and 100%
    
    return round(spo2, 2)

def calculate_blood_pressure():
    """
    Simulate blood pressure value.
    In a real scenario, this would be obtained using a sphygmomanometer or an electronic cuff.
    """
    # Simulate systolic and diastolic pressures for a healthy adult.
    systolic = random.randint(110, 130)  # Normal systolic pressure
    diastolic = random.randint(70, 85)  # Normal diastolic pressure
    blood_pressure = f"{systolic}/{diastolic}"
    
    return blood_pressure
