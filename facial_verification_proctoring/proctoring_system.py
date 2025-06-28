import cv2
import numpy as np
import os
from datetime import datetime
import time
import mediapipe as mp
import math
import base64
from io import BytesIO
import threading

class ProctoringSystem:
    def __init__(self):
        # Initialize face detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.face_locations = []
        self.last_face_detection_time = 0
        self.detection_interval = 1  # seconds
        self.anomalies = []
        # MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
        # Eye indices
        self.LEFT_EYE_IDX = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144]
        # 3D model points for head pose
        self.model_points = np.array([
            (0.0, 0.0, 0.0),           # Nose tip
            (0.0, -330.0, -65.0),      # Chin
            (-225.0, 170.0, -135.0),   # Left eye corner
            (225.0, 170.0, -135.0),    # Right eye corner
            (-150.0, -150.0, -125.0),  # Left mouth corner
            (150.0, -150.0, -125.0)    # Right mouth corner
        ], dtype="double")
        self.latest_status = {
            'num_faces': 0,
            'head_pose': 'Unknown',
            'left_eye_dir': 'Unknown',
            'right_eye_dir': 'Unknown',
            'left_eye_img': '',
            'right_eye_img': ''
        }
        self.latest_frame = None
        self.monitoring = False
        self.thread = None
        
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
        
        self.monitoring = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        
    def _monitor_loop(self):
        while self.monitoring:
            self.process_frame()
            time.sleep(1)
        
    def stop_monitoring(self):
        """Stop the proctoring system"""
        self.monitoring = False
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
        """Process a single frame from the webcam, including advanced head/eye monitoring"""
        current_time = time.time()
        if current_time - self.last_face_detection_time < self.detection_interval:
            return None
        self.last_face_detection_time = current_time
        ret, frame = self.cap.read()
        if not ret:
            return None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.face_locations = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        face_count = len(self.face_locations)
        # Anomaly: No face
        if face_count == 0:
            self.anomalies.append({'timestamp': datetime.now(), 'type': 'face_not_detected'})
        elif face_count > 1:
            self.anomalies.append({'timestamp': datetime.now(), 'type': 'multiple_faces', 'num_faces': face_count})
        # Head pose & eye direction (if face detected)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        if results.multi_face_landmarks:
            for landmarks in results.multi_face_landmarks:
                h, w = frame.shape[:2]
                image_points = np.array([
                    (landmarks.landmark[1].x * w, landmarks.landmark[1].y * h),   # Nose
                    (landmarks.landmark[152].x * w, landmarks.landmark[152].y * h),  # Chin
                    (landmarks.landmark[263].x * w, landmarks.landmark[263].y * h),  # Right eye
                    (landmarks.landmark[33].x * w, landmarks.landmark[33].y * h),    # Left eye
                    (landmarks.landmark[287].x * w, landmarks.landmark[287].y * h),  # Right mouth
                    (landmarks.landmark[57].x * w, landmarks.landmark[57].y * h)     # Left mouth
                ], dtype="double")
                focal_length = w
                center = (w / 2, h / 2)
                camera_matrix = np.array([
                    [focal_length, 0, center[0]],
                    [0, focal_length, center[1]],
                    [0, 0, 1]
                ], dtype="double")
                dist_coeffs = np.zeros((4, 1))
                success, rot_vec, _ = cv2.solvePnP(self.model_points, image_points, camera_matrix, dist_coeffs)
                # Head direction
                rmat, _ = cv2.Rodrigues(rot_vec)
                sy = math.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
                singular = sy < 1e-6
                if not singular:
                    pitch = math.degrees(math.atan2(rmat[2, 1], rmat[2, 2]))
                    yaw = math.degrees(math.atan2(-rmat[2, 0], sy))
                else:
                    pitch = math.degrees(math.atan2(-rmat[1, 2], rmat[1, 1]))
                    yaw = math.degrees(math.atan2(-rmat[2, 0], sy))
                # Log head pose anomalies
                if yaw < -15:
                    self.anomalies.append({'timestamp': datetime.now(), 'type': 'head_left'})
                elif yaw > 15:
                    self.anomalies.append({'timestamp': datetime.now(), 'type': 'head_right'})
                if pitch < -15:
                    self.anomalies.append({'timestamp': datetime.now(), 'type': 'head_up'})
                elif pitch > 15:
                    self.anomalies.append({'timestamp': datetime.now(), 'type': 'head_down'})
                # Eye direction
                def get_eye_image(frame, landmarks, indices):
                    h, w = frame.shape[:2]
                    eye_points = [(int(landmarks.landmark[i].x * w), int(landmarks.landmark[i].y * h)) for i in indices]
                    x_coords, y_coords = zip(*eye_points)
                    x_min, x_max = max(min(x_coords) - 5, 0), min(max(x_coords) + 5, w)
                    y_min, y_max = max(min(y_coords) - 5, 0), min(max(y_coords) + 5, h)
                    return frame[y_min:y_max, x_min:x_max]
                def detect_eye_direction(eye_img):
                    if eye_img.size == 0 or eye_img.shape[0] < 10 or eye_img.shape[1] < 10:
                        return "N/A"
                    eye_img = cv2.resize(eye_img, (100, 60), interpolation=cv2.INTER_CUBIC)
                    gray = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)
                    gray = cv2.equalizeHist(gray)
                    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                                   cv2.THRESH_BINARY_INV, 11, 3)
                    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        c = max(contours, key=cv2.contourArea)
                        M = cv2.moments(c)
                        if M['m00'] != 0:
                            cx = int(M['m10'] / M['m00'])
                            w = eye_img.shape[1]
                            if cx < w // 3:
                                return "Left"
                            elif cx > 2 * w // 3:
                                return "Right"
                            else:
                                return "Center"
                    return "Undetected"
                left_eye_img = get_eye_image(frame, landmarks, self.LEFT_EYE_IDX)
                right_eye_img = get_eye_image(frame, landmarks, self.RIGHT_EYE_IDX)
                left_eye_dir = detect_eye_direction(left_eye_img)
                right_eye_dir = detect_eye_direction(right_eye_img)
                # Log eye direction anomalies
                if left_eye_dir != "Center":
                    self.anomalies.append({'timestamp': datetime.now(), 'type': f'left_eye_{left_eye_dir.lower()}'})
                if right_eye_dir != "Center":
                    self.anomalies.append({'timestamp': datetime.now(), 'type': f'right_eye_{right_eye_dir.lower()}'})
                # Head direction
                head_pose = 'Center'
                if yaw < -15:
                    head_pose = 'Left'
                elif yaw > 15:
                    head_pose = 'Right'
                if pitch < -15:
                    head_pose = 'Up'
                elif pitch > 15:
                    head_pose = 'Down'
                self.latest_status['head_pose'] = head_pose
                # Encode eye previews as base64
                def img_to_base64(img):
                    if img is None or img.size == 0:
                        return ''
                    _, buf = cv2.imencode('.jpg', img)
                    return base64.b64encode(buf).decode('utf-8')
                self.latest_status['left_eye_img'] = img_to_base64(left_eye_img)
                self.latest_status['right_eye_img'] = img_to_base64(right_eye_img)
                self.latest_status['left_eye_dir'] = left_eye_dir
                self.latest_status['right_eye_dir'] = right_eye_dir
        self.latest_frame = frame
        return frame
        
    def get_anomalies(self):
        """Get all detected anomalies"""
        return self.anomalies 

    def get_live_status(self):
        return self.latest_status 