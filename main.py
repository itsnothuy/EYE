import cv2
import numpy as np
import dlib
from math import hypot
import time
import json

# Initialize video capture, dlib's face detector, and the shape predictor
cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Font for text on the video frame
font = cv2.FONT_HERSHEY_PLAIN

# Prompt for user input on activity type
# activity_type = input("Enter the type of activity (e.g., reading, coding, browsing): ")

activity_type = "coding"

# Session Data
session_data = {
    'start_time': time.strftime("%Y-%m-%d %H:%M:%S"),
    'end_time': None,
    'blink_count': 0,
    'breaks': [],
    'activity_type': activity_type,  # Manually set for now
    'gaze_transitions': []  # Track changes in gaze direction
}

# Initialize variables for tracking
gaze_start_time = None
gaze_duration = 0
currently_looking_center = False
break_start_time = None
current_gaze = "CENTER"

# Initialize variables to track total gaze time for each direction
total_time = 0
last_update_time = time.time()  # Tracks the last update time for any gaze direction


# Define the calibration thresholds for gaze direction
calibrated_left_threshold = 0.8
calibrated_right_threshold = 2.0

def midpoint(p1, p2):
    return int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)

def get_blinking_ratio(eye_points, facial_landmarks):
    left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
    center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
    center_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))

    hor_line_length = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
    ver_line_length = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))

    return hor_line_length / ver_line_length

def get_gaze_ratio(eye_points, facial_landmarks, gray):
    eye_region = np.array([(facial_landmarks.part(point).x, facial_landmarks.part(point).y) for point in eye_points], np.int32)
    
    min_x = np.min(eye_region[:, 0])
    max_x = np.max(eye_region[:, 0])
    min_y = np.min(eye_region[:, 1])
    max_y = np.max(eye_region[:, 1])

    eye = gray[min_y:max_y, min_x:max_x]
    _, threshold_eye = cv2.threshold(eye, 70, 255, cv2.THRESH_BINARY)
    height, width = threshold_eye.shape
    left_side_white = cv2.countNonZero(threshold_eye[0: height, 0: int(width / 2)])
    right_side_white = cv2.countNonZero(threshold_eye[0: height, int(width / 2): width])

    if right_side_white == 0:
        return 1
    return left_side_white / right_side_white

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    if not faces:
        if not break_start_time:  # Break starts now
            break_start_time = time.time()
    else:
        if break_start_time:  # Break ended
            break_end_time = time.time()
            session_data['breaks'].append({'start': break_start_time, 'end': break_end_time})
            break_start_time = None

        for face in faces:
            landmarks = predictor(gray, face)

            # Calculate the blinking and gaze ratios
            left_eye_ratio = get_blinking_ratio([36, 37, 38, 39, 40, 41], landmarks)
            right_eye_ratio = get_blinking_ratio([42, 43, 44, 45, 46, 47], landmarks)
            blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2
            gaze_ratio = get_gaze_ratio([36, 37, 38, 39, 40, 41], landmarks, gray) + get_gaze_ratio([42, 43, 44, 45, 46, 47], landmarks, gray)

            # Determine if blinking
            if blinking_ratio > 5.7:
                session_data['blink_count'] += 1
                cv2.putText(frame, "BLINKING", (50, 150), font, 7, (255, 0, 0))

            # Gaze direction logic
            if gaze_ratio < calibrated_left_threshold:
                new_gaze = "LEFT"
            elif gaze_ratio > calibrated_right_threshold:
                new_gaze = "RIGHT"
            else:
                new_gaze = "CENTER"

            if new_gaze != current_gaze:
                session_data['gaze_transitions'].append({'time': time.time(), 'from': current_gaze, 'to': new_gaze})
                current_gaze = new_gaze

            # Show current gaze direction
            cv2.putText(frame, f"LOOKING {current_gaze}", (50, 100), font, 2, (0, 0, 255), 3)
            cv2.putText(frame, f"Gaze Ratio: {gaze_ratio:.2f}", (50, 340), font, 2, (255, 255, 255), 3)
            
        now = time.time()
        time_diff = now - last_update_time
        last_update_time = now

        if current_gaze == "CENTER" or  current_gaze == "RIGHT" or  current_gaze == "LEFT":
            total_time += time_diff

    cv2.putText(frame, f"Time: {int(total_time)}s", (10, 50), font, 1, (255, 255, 255), 2)
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)
    if key == 27:  # ESC key to exit
        session_data['end_time'] = time.strftime("%Y-%m-%d %H:%M:%S")
        break

# Save session data to a file upon exit
with open('session_data.json', 'w') as f:
    json.dump(session_data, f, indent=4)

cap.release()
cv2.destroyAllWindows()
