import cv2
import mediapipe as mp

# Initialize MediaPipe Hand module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

# Open webcam
cap = cv2.VideoCapture(0)

# Finger tip landmark indexes (MediaPipe specific)
finger_tips = [8, 12, 16, 20]
thumb_tip = 4

while True:
    success, frame = cap.read()
    if not success:
        break

    # Flip for mirror effect
    frame = cv2.flip(frame, 1)

    # Convert BGR to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    finger_count = 0

    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]
        landmarks = hand_landmarks.landmark

        # Thumb (horizontal movement)
        if landmarks[thumb_tip].x < landmarks[thumb_tip - 1].x:
            finger_count += 1

        # Other fingers (vertical movement)
        for tip in finger_tips:
            if landmarks[tip].y < landmarks[tip - 2].y:
                finger_count += 1

        # Draw hand landmarks
        mp_draw.draw_landmarks(
            frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
        )

    # Display finger count
    cv2.putText(
        frame,
        f'Fingers: {finger_count}',
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        2,
        (0, 255, 0),
        3
    )

    cv2.imshow("Finger Counter", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
