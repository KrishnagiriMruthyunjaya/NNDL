"""
Real-Time Emotion Detection System
NNDL Course Project

This script implements a real-time emotion detection system using:
- FER-2013 dataset trained CNN model
- OpenCV for face detection and video processing
- Keras/TensorFlow for emotion classification

Author: [Your Name]
Course: Neural Networks and Deep Learning (NNDL)
Dataset: FER-2013
"""

import cv2
import numpy as np
from keras.models import load_model
import sys
import os

# Verify model file exists
model_path = "./model/emo.h5"
if not os.path.exists(model_path):
    print(f"Error: Model file '{model_path}' not found!")
    print("Please ensure the model file is in the correct location.")
    sys.exit(1)

# Load the pretrained CNN model
print("Loading emotion detection model...")
try:
    model = load_model(model_path)
    print("✓ Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Verify cascade classifier loaded correctly
if face_cascade.empty():
    print("Error: Could not load Haar Cascade classifier!")
    sys.exit(1)

# Define emotion labels (based on FER-2013 dataset)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Initialize webcam
print("Initializing camera...")
cap = cv2.VideoCapture(0)

# Check if camera opened successfully
if not cap.isOpened():
    print("Error: Could not access camera!")
    print("Please check if your camera is connected and not being used by another application.")
    sys.exit(1)

print("✓ Camera initialized successfully!")
print("=" * 50)
print("Real-Time Emotion Detection System")
print("=" * 50)
print("Instructions:")
print("- Position yourself in front of the camera")
print("- Ensure good lighting for better detection")
print("- Press 'q' to quit the application")
print("=" * 50)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    for (x, y, w, h) in faces:
        # Extract face ROI
        face_roi = gray[y:y+h, x:x+w]
        
        # Resize to 48x48 (standard input size for emotion models)
        face_roi = cv2.resize(face_roi, (48, 48))
        
        # Normalize and reshape for model input
        face_roi = face_roi.astype('float32') / 255.0
        face_roi = np.expand_dims(face_roi, axis=0)
        face_roi = np.expand_dims(face_roi, axis=-1)
        
        # Predict emotion
        predictions = model.predict(face_roi, verbose=0)
        emotion_idx = np.argmax(predictions)
        emotion = emotion_labels[emotion_idx]
        confidence = predictions[0][emotion_idx]
        
        # Choose color based on confidence level
        if confidence > 0.7:
            color = (0, 255, 0)  # Green for high confidence
        elif confidence > 0.5:
            color = (0, 165, 255)  # Orange for medium confidence
        else:
            color = (0, 0, 255)  # Red for low confidence
        
        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        
        # Display emotion label and confidence
        label = f"{emotion}: {confidence*100:.1f}%"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        
        # Draw background rectangle for text
        cv2.rectangle(frame, (x, y-label_size[1]-10), 
                     (x+label_size[0], y), color, -1)
        
        # Draw text
        cv2.putText(frame, label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (255, 255, 255), 2)
    
    # Display the frame
    cv2.imshow('Emotion Detection', frame)
    
    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
print("\n✓ Application closed successfully!")
print("Thank you for using the Emotion Detection System!")
