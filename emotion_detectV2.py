import cv2
import mediapipe as mp
import math
import random
import numpy as np

# ---------- MediaPipe Face ----------
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

# ---------- Particle Settings ----------
NUM_POINTS = 80
MAX_DIST = 80
POINT_SPEED = 0.2

particles = []

def init_particles(cx, cy):
    global particles
    particles = []
    for _ in range(NUM_POINTS):
        particles.append({
            "x": cx + random.randint(-100, 100),
            "y": cy + random.randint(-100, 100),
            "vx": random.uniform(-POINT_SPEED, POINT_SPEED),
            "vy": random.uniform(-POINT_SPEED, POINT_SPEED)
        })

def distance(p1, p2):
    return math.hypot(p2["x"] - p1["x"], p2["y"] - p1["y"])

# ---------- Emotion Logic ----------
def mouth_ratio(face):
    left = face[61]
    right = face[291]
    top = face[13]
    bottom = face[14]

    width = math.hypot(right.x - left.x, right.y - left.y)
    height = math.hypot(bottom.y - top.y, bottom.y - top.y)
    return height / width

initialized = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    overlay = frame.copy()
    emotion = "No Face"

    if result.multi_face_landmarks:
        face = result.multi_face_landmarks[0].landmark

        # Face center (nose)
        nose = face[1]
        cx, cy = int(nose.x * w), int(nose.y * h)

        if not initialized:
            init_particles(cx, cy)
            initialized = True

        # Move particles
        for p in particles:
            p["x"] += p["vx"]
            p["y"] += p["vy"]

            # Pull particles toward face
            p["x"] += (cx - p["x"]) * 0.01
            p["y"] += (cy - p["y"]) * 0.01

        # Draw connections
        for i in range(len(particles)):
            for j in range(i + 1, len(particles)):
                d = distance(particles[i], particles[j])
                if d < MAX_DIST:
                    alpha = 1 - (d / MAX_DIST)
                    cv2.line(
                        overlay,
                        (int(particles[i]["x"]), int(particles[i]["y"])),
                        (int(particles[j]["x"]), int(particles[j]["y"])),
                        (255, 255, 255),
                        1
                    )

        # Draw particles
        for p in particles:
            cv2.circle(
                overlay,
                (int(p["x"]), int(p["y"])),
                2,
                (255, 255, 255),
                -1
            )

        # Emotion detection
        ratio = mouth_ratio(face)
        if ratio > 0.35:
            emotion = "Surprised 😮"
        elif ratio > 0.20:
            emotion = "Happy 😀"
        elif ratio < 0.15:
            emotion = "Sad ☹️"
        else:
            emotion = "Neutral 😐"

    # Blend overlay for low opacity
    frame = cv2.addWeighted(overlay, 0.25, frame, 0.75, 0)

    cv2.putText(
        frame,
        f"Emotion: {emotion}",
        (30, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.3,
        (0, 255, 0),
        3
    )

    cv2.imshow("Emotion + Node Effect", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
