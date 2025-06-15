import cv2
import numpy as np
import os
from datetime import datetime
import time

class ProctoringSystem:
    def __init__(self):
        # Initialize face detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.face_locations = []
        self.last_face_detection_time = 0
        self.detection_interval = 1  # seconds
        self.anomalies = []
        
    def start_monitoring(self, student_id):
        """Initialize the proctoring system for a student"""
        self.student_id = student_id
        self.anomalies = []
        self.start_time = datetime.now()
        
        # Initialize webcam
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise Exception("Could not open webcam")
            
        # Load student's reference image if available
        self._load_student_reference()
        
    def stop_monitoring(self):
        """Stop the proctoring system"""
        if hasattr(self, 'cap'):
            self.cap.release()
        return self.anomalies
        
    def _load_student_reference(self):
        """Load student's reference image for face comparison"""
        self.reference_image = None
        reference_path = f"static/uploads/students/{self.student_id}.jpg"
        if os.path.exists(reference_path):
            self.reference_image = cv2.imread(reference_path)
            if self.reference_image is not None:
                self.reference_image = cv2.cvtColor(self.reference_image, cv2.COLOR_BGR2GRAY)
            
    def process_frame(self):
        """Process a single frame from the webcam"""
        current_time = time.time()
        if current_time - self.last_face_detection_time < self.detection_interval:
            return None
            
        self.last_face_detection_time = current_time
        
        # Capture frame
        ret, frame = self.cap.read()
        if not ret:
            return None
            
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        self.face_locations = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        # Check for anomalies
        self._check_anomalies()
        
        # Draw rectangles around faces
        for (x, y, w, h) in self.face_locations:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        return frame
        
    def _check_anomalies(self):
        """Check for proctoring anomalies"""
        current_time = datetime.now()
        
        # No face detected
        if len(self.face_locations) == 0:
            self.anomalies.append({
                'timestamp': current_time,
                'type': 'face_not_detected',
                'confidence': 1.0
            })
            return
            
        # Multiple faces detected
        if len(self.face_locations) > 1:
            self.anomalies.append({
                'timestamp': current_time,
                'type': 'multiple_faces',
                'confidence': 1.0,
                'num_faces': len(self.face_locations)
            })
            return
            
        # Face verification if reference image is available
        if self.reference_image is not None:
            # Get the first detected face
            x, y, w, h = self.face_locations[0]
            
            # Extract face region
            face_region = self.cap.read()[1][y:y+h, x:x+w]
            face_region = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            
            # Resize to match reference image size
            face_region = cv2.resize(face_region, (self.reference_image.shape[1], self.reference_image.shape[0]))
            
            # Calculate similarity using template matching
            result = cv2.matchTemplate(face_region, self.reference_image, cv2.TM_CCOEFF_NORMED)
            similarity = result[0][0]
            
            if similarity < 0.5:  # Threshold for face matching
                self.anomalies.append({
                    'timestamp': current_time,
                    'type': 'face_mismatch',
                    'confidence': 1.0 - similarity
                })
                
    def get_anomalies(self):
        """Get all detected anomalies"""
        return self.anomalies 