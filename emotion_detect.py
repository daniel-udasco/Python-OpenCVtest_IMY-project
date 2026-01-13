import cv2
import mediapipe as mp
import math

# Initialize MediaPipe Face Mesh
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

def distance(p1, p2):
    return math.hypot(p2.x - p1.x, p2.y - p1.y)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    emotion = "No Face"

    if result.multi_face_landmarks:
        face = result.multi_face_landmarks[0].landmark

        # Mouth landmarks
        left_mouth = face[61]
        right_mouth = face[291]
        top_mouth = face[13]
        bottom_mouth = face[14]

        mouth_width = distance(left_mouth, right_mouth)
        mouth_height = distance(top_mouth, bottom_mouth)

        ratio = mouth_height / mouth_width

        # Simple rules
        if ratio > 0.35:
            emotion = "Surprised"
        elif ratio > 0.20:
            emotion = "Happy"
        elif ratio < 0.15:
            emotion = "Sad"
        else:
            emotion = "Neutral"

    cv2.putText(
        frame,
        f"Emotion: {emotion}",
        (30, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        (0, 255, 0),
        3
    )

    cv2.imshow("Emotion Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
