import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.PlotModule import LivePlot
import math
from collections import deque
import time
import numpy as np
import csv
import datetime
import os

# --- INPUT CONFIGURATION ---
# Set USE_WEBCAM to True for live feed, False for video file
USE_WEBCAM = True
INPUT_VIDEO_PATH = "path/to/your/video.mp4" # Only used if USE_WEBCAM = False

# --- CONFIGURATION ---
RIGHT_EYE_POINTS = [33, 133, 160, 144, 158, 153]
LEFT_EYE_POINTS = [362, 263, 385, 380, 387, 373]

# Landmarks for Pitch Detection
FACE_TOP = 10
FACE_BOTTOM = 152
# Inner Eyebrow Points (Closest points between eyebrows)
FACE_LEFT = 55   # Right Inner Brow
FACE_RIGHT = 285 # Left Inner Brow

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
SMOOTHING_WINDOW = 5
PERCLOS_WINDOW_FRAMES = 300 
ROTATION_THRESHOLD = 0.8 

# Debugging: Set to True to see all 468 available face points
SHOW_ALL_LANDMARKS = False 

# --- DATA COLLECTION CONFIG ---
ENABLE_DATA_COLLECTION = True  # Master switch
RECORD_CSV = True              # Save numerical stats
RECORD_VIDEO = True            # Save visual feed (with overlays)
DATA_FOLDER_NAME = "./data"    # Works with nested paths

# --- SETUP OUTPUTS ---
csv_file = None
writer = None
video_writer = None

if ENABLE_DATA_COLLECTION:
    current_time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create directory if missing
    if not os.path.exists(DATA_FOLDER_NAME):
        os.makedirs(DATA_FOLDER_NAME)
    
    # 1. Setup CSV
    if RECORD_CSV:
        csv_filename = os.path.join(DATA_FOLDER_NAME, f"data_{current_time_str}.csv")
        csv_file = open(csv_filename, mode='w', newline='')
        writer = csv.writer(csv_file)
        writer.writerow(["Timestamp", "Selected_EAR", "Smooth_EAR", "Threshold", "Pitch_Ratio", "Pitch_Threshold", "Yaw_Ratio", "Active_Eye", "PERCLOS_Score", "State"])
        print(f"[DATA] Logging CSV to: {csv_filename}")

    # 2. Setup Video
    if RECORD_VIDEO:
        video_filename = os.path.join(DATA_FOLDER_NAME, f"video_{current_time_str}.mp4")
        # 'mp4v' is a widely supported codec. Use 'XVID' for .avi if this fails.
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        # Note: Must match the image size exactly (FRAME_WIDTH, FRAME_HEIGHT)
        video_writer = cv2.VideoWriter(video_filename, fourcc, 30.0, (FRAME_WIDTH, FRAME_HEIGHT))
        print(f"[DATA] Recording Video to: {video_filename}")

# --- INPUT SOURCE SELECTION ---
if USE_WEBCAM:
    print("[INPUT] Starting Webcam...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
else:
    print(f"[INPUT] Opening Video File: {INPUT_VIDEO_PATH}")
    cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
    if not cap.isOpened():
        print(f"Cannot open video file: {INPUT_VIDEO_PATH}")
        exit()

detector = FaceMeshDetector(maxFaces=1)
cv2.namedWindow("Drowsiness Detector", cv2.WINDOW_NORMAL)

# Variables
ear_history = deque(maxlen=SMOOTHING_WINDOW)
perclos_history = deque(maxlen=PERCLOS_WINDOW_FRAMES)
plotY = None 

# --- CALIBRATION STATE ---
calib_state = "IDLE"
calib_data = []
calib_pitch_data = [] 
calib_counter = 0
calib_max_frames = 90 

# Calibration Results
avg_open_ear = 0
avg_closed_ear = 0
avg_pitch = 0        
blink_threshold = 22 
pitch_threshold = 0.8

def calculate_ear(face, indices, img_draw=None):
    p1 = face[indices[0]] 
    p4 = face[indices[1]] 
    p2 = face[indices[2]] 
    p6 = face[indices[3]] 
    p3 = face[indices[4]] 
    p5 = face[indices[5]] 

    if img_draw is not None:
        for p in [p1, p2, p3, p4, p5, p6]:
            cv2.circle(img_draw, p, 2, (0, 255, 0), cv2.FILLED)
        cv2.line(img_draw, p2, p6, (255, 0, 0), 1)
        cv2.line(img_draw, p3, p5, (255, 0, 0), 1)
        cv2.line(img_draw, p1, p4, (0, 0, 255), 1)

    v1 = math.dist(p2, p6)
    v2 = math.dist(p3, p5)
    h = math.dist(p1, p4)
    v_avg = (v1 + v2) / 2.0

    if h == 0: return 0, 0, 0
    ear = ((v1 + v2) / (2.0 * h)) * 100
    return ear, h, v_avg

while True:
    success, img = cap.read()
    
    # Handle end of video or read error
    if not success:
        print("End of input source or read error.")
        break

    # Resize to standard frame size for consistent processing
    img = cv2.resize(img, (FRAME_WIDTH, FRAME_HEIGHT))
    
    img, faces = detector.findFaceMesh(img, draw=False)

    if faces:
        face = faces[0]
        
        if SHOW_ALL_LANDMARKS:
            for point in face:
                cv2.circle(img, point, 1, (50, 50, 50), cv2.FILLED)

        # Draw Face Geometry Landmarks
        pt_top = face[FACE_TOP]
        pt_bottom = face[FACE_BOTTOM]
        pt_left = face[FACE_LEFT]
        pt_right = face[FACE_RIGHT]
        
        for pt in [pt_top, pt_bottom, pt_left, pt_right]:
             cv2.circle(img, pt, 3, (0, 255, 255), cv2.FILLED)
        cv2.line(img, pt_top, pt_bottom, (0, 255, 255), 1) 
        cv2.line(img, pt_left, pt_right, (0, 255, 255), 1) 

        # Calculate Stats
        ear_right, w_right, v_right = calculate_ear(face, RIGHT_EYE_POINTS, img)
        ear_left, w_left, v_left = calculate_ear(face, LEFT_EYE_POINTS, img)
        
        # --- SMART EYE SELECTION ---
        current_ear = 0
        active_eye_label = "NONE"
        if w_right == 0 and w_left == 0:
            current_ear = 0
        elif w_right > w_left:
            current_ear = ear_right
            active_eye_label = "RIGHT"
        else:
            current_ear = ear_left
            active_eye_label = "LEFT"

        # --- PITCH CHECK ---
        valid_frame = True
        ratio = 0 
        
        if w_right > 0 and w_left > 0:
            ratio = min(w_right, w_left) / max(w_right, w_left)

        face_h = math.dist(face[FACE_TOP], face[FACE_BOTTOM])
        face_w = math.dist(face[FACE_LEFT], face[FACE_RIGHT])
        current_pitch_ratio = 0

        if face_w == 0:
            valid_frame = False
        else:
            current_pitch_ratio = face_h / face_w
            
            cutoff = pitch_threshold
            if calib_state != "CALIB_OPEN":
                if current_pitch_ratio < cutoff:
                    valid_frame = False
                    cvzone.putTextRect(img, f'BAD PITCH: {current_pitch_ratio:.2f} < {cutoff:.2f}', (30, 430), scale=1, thickness=1, colorR=(0, 0, 255))
        

        # ============================================================
        # STATE MACHINE
        # ============================================================

        if calib_state == "IDLE":
            cvzone.putTextRect(img, "STEP 1: OPEN EYES CALIBRATION", (50, 50), scale=2, thickness=2)
            cvzone.putTextRect(img, "Look at camera naturally.", (50, 100), scale=1.5, thickness=2)
            cvzone.putTextRect(img, "Press 'c' to start", (50, 150), scale=1.5, thickness=2)
            
            if cv2.waitKey(1) & 0xFF == ord('c'):
                calib_state = "CALIB_OPEN"
                calib_data = []
                calib_pitch_data = [] 
                calib_counter = 0

        elif calib_state == "CALIB_OPEN":
            if valid_frame:
                calib_data.append(current_ear)
                calib_pitch_data.append(current_pitch_ratio) 
                calib_counter += 1
            
            cv2.rectangle(img, (50, 200), (50 + calib_counter * 4, 230), (0, 255, 0), cv2.FILLED)
            cvzone.putTextRect(img, "RECORDING OPEN EYES...", (50, 150), scale=1.5)

            if calib_counter >= calib_max_frames:
                avg_open_ear = np.mean(calib_data)
                avg_pitch = np.mean(calib_pitch_data)
                pitch_threshold = avg_pitch * 0.85 
                calib_state = "WAIT_FOR_CLOSE"
                calib_counter = 0

        elif calib_state == "WAIT_FOR_CLOSE":
            cvzone.putTextRect(img, "STEP 2: CLOSED EYES CALIBRATION", (50, 50), scale=2, thickness=2)
            cvzone.putTextRect(img, f"Open EAR: {int(avg_open_ear)}", (50, 100), scale=1.5)
            cvzone.putTextRect(img, f"Pitch: {avg_pitch:.2f}", (300, 100), scale=1.5) 
            cvzone.putTextRect(img, "Press 'c', wait 2s, then CLOSE eyes", (50, 150), scale=1.5)
            
            if cv2.waitKey(1) & 0xFF == ord('c'):
                cvzone.putTextRect(img, "PREPARE TO CLOSE...", (100, 240), scale=3, colorR=(0,0,255))
                cv2.imshow("Drowsiness Detector", img)
                cv2.waitKey(2000) 
                calib_state = "CALIB_CLOSED"
                calib_data = []
                calib_counter = 0

        elif calib_state == "CALIB_CLOSED":
            if valid_frame:
                calib_data.append(current_ear)
                calib_counter += 1
            
            cv2.rectangle(img, (50, 200), (50 + calib_counter * 4, 230), (0, 0, 255), cv2.FILLED)
            cvzone.putTextRect(img, "RECORDING CLOSED EYES...", (50, 150), scale=1.5)

            if calib_counter >= calib_max_frames:
                avg_closed_ear = np.mean(calib_data)
                eye_range = avg_open_ear - avg_closed_ear
                blink_threshold = avg_closed_ear + (0.3 * eye_range)
                plotY = LivePlot(FRAME_WIDTH, FRAME_HEIGHT, [blink_threshold - 15, blink_threshold + 15], invert=True)
                calib_state = "WAIT_FOR_OPEN"

        elif calib_state == "WAIT_FOR_OPEN":
             cvzone.putTextRect(img, "CALIBRATION COMPLETE!", (50, 50), scale=2, thickness=2)
             cvzone.putTextRect(img, "Open your eyes to begin...", (50, 100), scale=1.5, thickness=2)
             if current_ear > blink_threshold:
                 calib_state = "RUNNING"

        elif calib_state == "RUNNING":
            if valid_frame:
                ear_history.append(current_ear)
                smooth_ear = sum(ear_history) / len(ear_history)
                
                is_closed = smooth_ear < blink_threshold
                perclos_history.append(1 if is_closed else 0)
                
                imgPlot = plotY.update(smooth_ear)
                
                state_text = "N/A"
                perclos_score = 0.0
                if len(perclos_history) > 0:
                    perclos_score = (sum(perclos_history) / len(perclos_history)) * 100
                    if perclos_score > 30: state_text = "SLEEPY"
                    elif perclos_score > 15: state_text = "DROWSY"
                    else: state_text = "ALERT"

                # --- EXPORT DATA ---
                if ENABLE_DATA_COLLECTION and RECORD_CSV:
                    timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
                    writer.writerow([timestamp, f"{current_ear:.2f}", f"{smooth_ear:.2f}", f"{blink_threshold:.2f}", 
                                     f"{current_pitch_ratio:.2f}", f"{pitch_threshold:.2f}", f"{ratio:.2f}", 
                                     active_eye_label, f"{perclos_score:.1f}", state_text])
                
                # --- VISUALS ---
                cvzone.putTextRect(img, f'EAR: {int(smooth_ear)}', (30, 50), scale=2, colorR=(0, 0, 0))
                cvzone.putTextRect(img, f'Thresh: {blink_threshold:.1f}', (30, 100), scale=1, colorR=(50, 50, 50))
                cvzone.putTextRect(img, f'Pitch Cur/Avg: {current_pitch_ratio:.2f} / {avg_pitch:.2f}', (30, 140), scale=1, colorR=(100, 100, 100))
                cvzone.putTextRect(img, f'Yaw Ratio: {ratio:.2f}', (30, 180), scale=1, colorR=(100, 100, 100))
                cvzone.putTextRect(img, f'Active Eye: {active_eye_label}', (30, 220), scale=1, colorR=(100, 100, 100))

                state_color = (0, 255, 0)
                if state_text == "SLEEPY": state_color = (0, 0, 255)
                elif state_text == "DROWSY": state_color = (0, 165, 255)
                
                cv2.rectangle(img, (450, 30), (620, 130), (50, 50, 50), cv2.FILLED)
                cv2.putText(img, f"PERCLOS: {int(perclos_score)}%", (460, 60), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2)
                cv2.putText(img, state_text, (460, 100), cv2.FONT_HERSHEY_PLAIN, 2, state_color, 3)

            else:
                imgPlot = plotY.update(blink_threshold + 10) 

            imgStack = cvzone.stackImages([img, imgPlot], 2, 1.0)
            cv2.imshow("Drowsiness Detector", imgStack)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # Write Video (Running state)
            if ENABLE_DATA_COLLECTION and RECORD_VIDEO and video_writer is not None:
                video_writer.write(img)
                
            continue 

    else:
        # Fallback if no face
        pass

    # Write Video (Non-Running or No Face)
    if ENABLE_DATA_COLLECTION and RECORD_VIDEO and video_writer is not None:
        video_writer.write(img)

    if calib_state != "RUNNING":
        cv2.imshow("Drowsiness Detector", img)
    elif calib_state == "RUNNING" and not faces:
         cv2.imshow("Drowsiness Detector", img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
if csv_file: csv_file.close()
if video_writer: video_writer.release()
cap.release()
cv2.destroyAllWindows()