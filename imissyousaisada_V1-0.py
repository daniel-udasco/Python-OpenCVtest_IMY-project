import cv2
import mediapipe as mp
import math
import time

# ---------------- MediaPipe Setup ----------------
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils

# Increased confidence to reduce flickering
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)
face_mesh = mp_face.FaceMesh(max_num_faces=1)

cap = cv2.VideoCapture(0)

# ---------------- Finger Logic (IMPROVED) ----------------
# Standard tip IDs: Index(8), Middle(12), Ring(16), Pinky(20)
finger_tips = [8, 12, 16, 20]

def count_fingers_robust(lm, label):
    """
    Counts fingers based on hand label (Left/Right) to fix Thumb detection.
    """
    count = 0
    
    # --- Thumb Logic ---
    # Thumb motion is horizontal (X-axis), fingers are vertical (Y-axis)
    # Determine handedness logic because camera is flipped (Mirror mode)
    thumb_tip = 4
    thumb_ip = 3 # Joint below tip
    
    # In a flipped image:
    # "Right" label usually appears on the right side of the screen
    if label == "Right": 
        # Right hand thumb: Open if Tip is to the LEFT of the joint (smaller X)
        if lm[thumb_tip].x < lm[thumb_ip].x:
            count += 1
    else: 
        # Left hand thumb: Open if Tip is to the RIGHT of the joint (larger X)
        if lm[thumb_tip].x > lm[thumb_ip].x:
            count += 1

    # --- 4 Fingers Logic ---
    # Check if Tip is ABOVE the PIP joint (Y increases downwards in OpenCV)
    for tip in finger_tips:
        if lm[tip].y < lm[tip - 2].y:
            count += 1
            
    return count

# ---------------- Emotion Logic (UNCHANGED) ----------------
def detect_emotion(face):
    left = face[61]
    right = face[291]
    top = face[13]
    bottom = face[14]

    width = math.hypot(right.x - left.x, right.y - left.y)
    height = math.hypot(bottom.y - top.y, bottom.y - top.y)
    ratio = height / width

    if ratio > 0.35:
        return "Surprised"
    elif ratio > 0.20:
        return "Happy"
    elif ratio < 0.15:
        return "Sad"
    else:
        return "Neutral"

# ---------------- State Variables ----------------
typed_text = ""
current_letter = ""

# Timer variables for "Hold to Type"
selection_start_time = 0
hold_duration = 3 # Seconds to hold before typing
prev_index = -1

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame for mirror effect
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    hand_result = hands.process(rgb)
    face_result = face_mesh.process(rgb)

    joke_text = ""

    # -------- Emotion Processing (UNCHANGED) --------
    if face_result.multi_face_landmarks:
        face = face_result.multi_face_landmarks[0].landmark
        emotion = detect_emotion(face)

        if emotion == "Sad":
            joke_text = "FAK U"
        elif emotion == "Surprised":
            joke_text = "NIGGA"
        elif emotion == "Happy":
            joke_text = "SBAPN"

    # -------- Hand Input & Alphabet Logic --------
    left_count = 0
    right_count = 0
    hands_detected = 0

    if hand_result.multi_hand_landmarks:
        for hand_lms, handedness in zip(hand_result.multi_hand_landmarks, hand_result.multi_handedness):
            label = handedness.classification[0].label
            hands_detected += 1
            
            # Use improved counter
            fingers = count_fingers_robust(hand_lms.landmark, label)

            if label == "Left":
                left_count = fingers
            else:
                right_count = fingers

            mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)

    # -------- Letter Calculation --------
    # Logic: Left Hand = Groups of 5, Right Hand = Individual 1-5
    # Total Index = (Left * 5) + Right
    
    # Only calculate if we actually see hands, otherwise reset
    if hands_detected > 0:
        index = (left_count * 5) + right_count
        
        # 0-25 are letters, 26 can be Space, 27 can be Backspace
        target_char = ""
        if 0 <= index < 26:
            target_char = chr(ord('A') + index)
        elif index == 26:
            target_char = "SPACE"
        elif index == 27:
            target_char = "BACK"
        else:
            target_char = "?"

        # -------- Hold-to-Type Logic --------
        if index == prev_index and target_char != "":
            # Calculate how long we've held this gesture
            time_held = time.time() - selection_start_time
            progress = min(time_held / hold_duration, 1.0)
            
            # Draw Progress Bar
            bar_width = 200
            cv2.rectangle(frame, (w//2 - 100, 150), (w//2 + 100, 170), (50, 50, 50), -1)
            cv2.rectangle(frame, (w//2 - 100, 150), (w//2 - 100 + int(bar_width * progress), 170), (0, 255, 0), -1)
            
            if time_held >= hold_duration:
                if target_char == "SPACE":
                    typed_text += " "
                elif target_char == "BACK":
                    typed_text = typed_text[:-1]
                elif len(target_char) == 1:
                    typed_text += target_char
                
                # Reset timer so it doesn't spam type
                selection_start_time = time.time()
                # Optional: Visual flash or sound could go here
                cv2.rectangle(frame, (0, 0), (w, h), (255, 255, 255), 10) 
                
        else:
            # Gesture changed, reset timer
            prev_index = index
            selection_start_time = time.time()
            
        current_letter = target_char
    else:
        # No hands detected
        current_letter = ""
        selection_start_time = time.time()

    # -------- UI Rendering --------
    # Top Bar
    cv2.rectangle(frame, (20, 20), (w - 20, 80), (0, 0, 0), -1)
    cv2.putText(frame, f"Emotion: {joke_text}", (30, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Status Info
    cv2.putText(frame, f"L:{left_count} R:{right_count} -> {current_letter}",
                (30, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Instructions
    cv2.putText(frame, "Hold steady to type", (w - 250, 120), 
                cv2.FONT_HERSHEY_PLAIN, 1, (200, 200, 200), 1)

    # Bottom Typing Area
    cv2.rectangle(frame, (20, h - 100), (w - 20, h - 20), (0, 0, 0), -1)
    cv2.putText(frame, f"Typed: {typed_text}",
                (30, h - 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 200, 255), 3)

    cv2.imshow("Final Satirical CV Project", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()