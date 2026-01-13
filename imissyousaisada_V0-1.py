import cv2
import mediapipe as mp
import math
import time

# ---------- MediaPipe ----------
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

face_mesh = mp_face.FaceMesh(
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

# ---------- Finger Logic ----------
finger_tips = [8, 12, 16, 20]
thumb_tip = 4

def count_fingers(landmarks):
    count = 0
    if landmarks[thumb_tip].x < landmarks[thumb_tip - 1].x:
        count += 1
    for tip in finger_tips:
        if landmarks[tip].y < landmarks[tip - 2].y:
            count += 1
    return count

# ---------- Emotion Logic ----------
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

# ---------- Letter Groups ----------
letter_groups = {
    1: ["A", "B", "C"],
    2: ["D", "E", "F"],
    3: ["G", "H", "I"],
    4: ["J", "K", "L"],
    5: ["M", "N", "O"]
}

current_group = None
group_index = 0
typed_text = ""

last_action_time = time.time()

CONFIRM_GESTURE = 0   # closed fist

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    hand_result = hands.process(rgb)
    face_result = face_mesh.process(rgb)

    joke_text = ""

    # ----- Face Emotion -----
    if face_result.multi_face_landmarks:
        face = face_result.multi_face_landmarks[0].landmark
        emotion = detect_emotion(face)

        if emotion == "Sad":
            joke_text = "1"
        elif emotion == "Surprised":
            joke_text = "2"
        elif emotion == "Happy":
            joke_text = "3"

    # ----- Hand Input -----
    if hand_result.multi_hand_landmarks:
        hand = hand_result.multi_hand_landmarks[0]
        fingers = count_fingers(hand.landmark)

        now = time.time()

        # Select letter group
        if fingers in letter_groups and now - last_action_time > 0.8:
            if current_group != fingers:
                current_group = fingers
                group_index = 0
            last_action_time = now

        # Confirm gesture (closed fist)
        if fingers == CONFIRM_GESTURE and current_group is not None:
            if now - last_action_time > 0.8:
                letter = letter_groups[current_group][group_index]
                typed_text += letter
                current_group = None
                group_index = 0
                last_action_time = now

        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

    # ----- UI -----
    cv2.rectangle(frame, (20, 20), (w - 20, 80), (0, 0, 0), -1)
    cv2.putText(frame, joke_text, (30, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    if current_group:
        preview = letter_groups[current_group][group_index]
        cv2.putText(frame, f"Selected: {preview}",
                    (30, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    cv2.rectangle(frame, (20, h - 100), (w - 20, h - 20), (0, 0, 0), -1)
    cv2.putText(frame, f"Typed: {typed_text}", (30, h - 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 200, 255), 3)

    cv2.imshow("Satirical CV Project v2", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
