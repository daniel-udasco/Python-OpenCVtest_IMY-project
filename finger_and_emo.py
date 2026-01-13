import cv2
import mediapipe as mp
import math

# ---------- MediaPipe Setup ----------
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

# ---------- Webcam ----------
cap = cv2.VideoCapture(0)

# ---------- Finger Logic ----------
finger_tips = [8, 12, 16, 20]
thumb_tip = 4

def count_fingers(landmarks):
    count = 0

    # Thumb
    if landmarks[thumb_tip].x < landmarks[thumb_tip - 1].x:
        count += 1

    # Other fingers
    for tip in finger_tips:
        if landmarks[tip].y < landmarks[tip - 2].y:
            count += 1

    return count

# ---------- Emotion Logic ----------
def emotion_from_face(face):
    left = face[61]
    right = face[291]
    top = face[13]
    bottom = face[14]

    mouth_width = math.hypot(right.x - left.x, right.y - left.y)
    mouth_height = math.hypot(bottom.y - top.y, bottom.y - top.y)

    ratio = mouth_height / mouth_width

    if ratio > 0.35:
        return "Surprised"
    elif ratio > 0.20:
        return "Happy"
    elif ratio < 0.15:
        return "Sad"
    else:
        return "Neutral"

# ---------- UI Helper ----------
def draw_panel(img, text, x, y, w, h):
    cv2.rectangle(img, (x, y), (x + w, y + h), (30, 30, 30), -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), (200, 200, 200), 2)
    cv2.putText(
        img, text,
        (x + 15, y + int(h / 1.6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2
    )

# ---------- Main Loop ----------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    hand_result = hands.process(rgb)
    face_result = face_mesh.process(rgb)

    finger_text = "Fingers: 0"
    face_text = "Face: Not Detected"
    emotion_text = "Emotion: ---"

    # ----- Hand -----
    if hand_result.multi_hand_landmarks:
        hand = hand_result.multi_hand_landmarks[0]
        finger_count = count_fingers(hand.landmark)
        finger_text = f"Fingers: {finger_count}"

        mp_draw.draw_landmarks(
            frame, hand, mp_hands.HAND_CONNECTIONS
        )

    # ----- Face + Emotion -----
    if face_result.multi_face_landmarks:
        face = face_result.multi_face_landmarks[0].landmark
        face_text = "Face: Detected"
        emotion_text = f"Emotion: {emotion_from_face(face)}"

    # ----- Clean UI Panels -----
    draw_panel(frame, finger_text, 20, 20, 220, 60)
    draw_panel(frame, face_text, w - 260, 20, 240, 60)
    draw_panel(frame, emotion_text, int(w / 2 - 180), h - 100, 360, 70)

    cv2.imshow("Hand, Face & Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
