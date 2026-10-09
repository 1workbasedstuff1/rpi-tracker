"""
Live undistorted camera stream using camera_intrinsics.npz.

Keys: u = toggle undistortion on/off, q = quit
"""
import time

import cv2
import numpy as np

CAM_INDEX = 4
WIDTH, HEIGHT, FPS = 1280, 720, 90  # MUST match the calibration resolution
CALIB_FILE = "camera_intrinsics.npz"

# alpha = 0: crop to only valid pixels (no black borders, loses some field of view)
# alpha = 1: keep every original pixel (black borders appear, full field of view)
ALPHA = 0.0

data = np.load(CALIB_FILE)
K, dist = data["K"], data["dist"]
calib_w, calib_h = (int(v) for v in data["image_size"])
assert (calib_w, calib_h) == (WIDTH, HEIGHT), (
    f"Calibrated at {calib_w}x{calib_h} but streaming at {WIDTH}x{HEIGHT}"
)

# New camera matrix for the undistorted image. Use new_K (not K) for anything
# downstream (pose estimation, triangulation, etc.) on the undistorted frames.
new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist, (WIDTH, HEIGHT), ALPHA, (WIDTH, HEIGHT))

# Precompute the pixel remapping once; remap() per frame is much faster than undistort().
map1, map2 = cv2.initUndistortRectifyMap(K, dist, None, new_K, (WIDTH, HEIGHT), cv2.CV_16SC2)
print("new_K:\n", new_K)

cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
cap.set(cv2.CAP_PROP_FPS, FPS)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
cap.set(cv2.CAP_PROP_EXPOSURE, 50)

undistort_on = True
n, t0 = 0, time.time()
while True:
    ret, frame = cap.read()
    if not ret:
        break

    if undistort_on:
        frame = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)

    n += 1
    if n % 90 == 0:
        print(f"{n / (time.time() - t0):.1f} fps")
        n, t0 = 0, time.time()

    cv2.putText(frame, f"undistort: {undistort_on}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Live Feed", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord("u"):
        undistort_on = not undistort_on

cap.release()
cv2.destroyAllWindows()
