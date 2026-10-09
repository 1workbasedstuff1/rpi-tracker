import cv2

# 0 is usually the first/default camera; try 1, 2, ... for others
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    raise RuntimeError("Could not open camera")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    cv2.imshow("Live Feed", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
