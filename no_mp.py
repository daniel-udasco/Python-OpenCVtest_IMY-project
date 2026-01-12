import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    roi = frame[100:400, 100:400]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (35, 35), 0)
    _, thresh = cv2.threshold(blur, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    if contours:
        cnt = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(cnt, returnPoints=False)

        if hull is not None and len(hull) > 3:
            defects = cv2.convexityDefects(cnt, hull)
            if defects is not None:
                finger_count = sum(
                    1 for i in range(defects.shape[0])
                ) + 1
            else:
                finger_count = 0
        else:
            finger_count = 0

        cv2.putText(frame, f"Fingers: {finger_count}",
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0), 2)

    cv2.rectangle(frame, (100, 100), (400, 400), (255, 0, 0), 2)
    cv2.imshow("Finger Counter (No MediaPipe)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
