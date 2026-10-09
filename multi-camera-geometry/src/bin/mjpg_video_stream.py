import cv2
import time

cap = cv2.VideoCapture(4, cv2.CAP_V4L2)

cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 90)

print("FPS reported:", cap.get(cv2.CAP_PROP_FPS))

# Manual exposure from inside OpenCV (alternative to v4l2-ctl).
# On V4L2, 1 = manual, 3 = auto. Exposure units are 100 µs.
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
cap.set(cv2.CAP_PROP_EXPOSURE, 50)

n, t0 = 0, time.time()
while True:
    ret, frame = cap.read()
    if not ret:
        break
    n += 1
    if n % 90 == 0:
        print(f"Measured: {n / (time.time() - t0):.1f} fps")
        n, t0 = 0, time.time()
    cv2.imshow("Live Feed", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
