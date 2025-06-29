import cv2
import mediapipe as mp
import numpy as np
import math
import base64

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

LEFT_EYE_IDX = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144]
model_points = np.array([
    (0.0, 0.0, 0.0),           # Nose tip
    (0.0, -330.0, -65.0),      # Chin
    (-225.0, 170.0, -135.0),   # Left eye corner
    (225.0, 170.0, -135.0),    # Right eye corner
    (-150.0, -150.0, -125.0),  # Left mouth corner
    (150.0, -150.0, -125.0)    # Right mouth corner
], dtype="double")

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

def get_head_direction(rotation_vector):
    rmat, _ = cv2.Rodrigues(rotation_vector)
    sy = math.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        pitch = math.degrees(math.atan2(rmat[2, 1], rmat[2, 2]))
        yaw = math.degrees(math.atan2(-rmat[2, 0], sy))
    else:
        pitch = math.degrees(math.atan2(-rmat[1, 2], rmat[1, 1]))
        yaw = math.degrees(math.atan2(-rmat[2, 0], sy))
    horizontal = "Center"
    if yaw < -15:
        horizontal = "Left"
    elif yaw > 15:
        horizontal = "Right"
    vertical = "Center"
    if pitch < -15:
        vertical = "Up"
    elif pitch > 15:
        vertical = "Down"
    return horizontal, vertical

def img_to_base64(img):
    if img is None or img.size == 0:
        return ''
    _, buf = cv2.imencode('.jpg', img)
    return base64.b64encode(buf).decode('utf-8')

def analyze_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    face_count = len(faces)
    status = {
        'num_faces': face_count,
        'head_pose': 'Unknown',
        'left_eye_dir': 'Unknown',
        'right_eye_dir': 'Unknown',
        'left_eye_img': '',
        'right_eye_img': ''
    }
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
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
            success, rot_vec, _ = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)
            face_horizontal, face_vertical = get_head_direction(rot_vec)
            status['head_pose'] = f'{face_horizontal}, {face_vertical}'
            left_eye_img = get_eye_image(frame, landmarks, LEFT_EYE_IDX)
            right_eye_img = get_eye_image(frame, landmarks, RIGHT_EYE_IDX)
            left_eye_dir = detect_eye_direction(left_eye_img)
            right_eye_dir = detect_eye_direction(right_eye_img)
            status['left_eye_dir'] = left_eye_dir
            status['right_eye_dir'] = right_eye_dir
            status['left_eye_img'] = img_to_base64(left_eye_img)
            status['right_eye_img'] = img_to_base64(right_eye_img)
    return status 