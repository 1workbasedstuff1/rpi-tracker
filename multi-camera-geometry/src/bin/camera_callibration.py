"""
Intrinsic camera calibration with a checkerboard.

Step 1 - capture images:   python calibrate_camera.py capture
Step 2 - run calibration:  python calibrate_camera.py calibrate

IMPORTANT: capture at the SAME resolution / settings you use at runtime
(1280x720 MJPG here). K is only valid for the resolution it was calibrated at.
"""
import sys
import glob
import json
import os

import cv2
import numpy as np

# ---------------- CONFIG ----------------
CAM_INDEX = 4
WIDTH, HEIGHT, FPS = 1280, 720, 90

# Number of INNER corners (not squares!). A board with 10x7 squares has 9x6 inner corners.
BOARD_COLS, BOARD_ROWS = 9, 6
SQUARE_SIZE_M = 0.025  # measured side length of one square, in metres

IMG_DIR = "calib_images"
OUT_NPZ = "camera_intrinsics.npz"
OUT_JSON = "camera_intrinsics.json"
# ----------------------------------------


def open_camera():
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)
    # Lock focus: changing focus changes the focal length, which ruins calibration.
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    # Manual exposure (same as your script) to keep corners sharp / low blur.
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
    cap.set(cv2.CAP_PROP_EXPOSURE, 50)
    return cap


def capture():
    os.makedirs(IMG_DIR, exist_ok=True)
    cap = open_camera()
    pattern = (BOARD_COLS, BOARD_ROWS)
    count = len(glob.glob(os.path.join(IMG_DIR, "*.png")))
    print("SPACE = save frame (only when board is detected), q = quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(
            gray, pattern,
            cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK,
        )

        view = frame.copy()
        if found:
            cv2.drawChessboardCorners(view, pattern, corners, found)
        cv2.putText(view, f"saved: {count}  detected: {found}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if found else (0, 0, 255), 2)
        cv2.imshow("Calibration capture", view)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" ") and found:
            path = os.path.join(IMG_DIR, f"img_{count:03d}.png")
            cv2.imwrite(path, frame)  # save the clean frame, not the annotated one
            count += 1
            print("saved", path)

    cap.release()
    cv2.destroyAllWindows()


def calibrate():
    pattern = (BOARD_COLS, BOARD_ROWS)

    # 3D points of the board corners in the board's own frame (Z = 0 plane)
    objp = np.zeros((BOARD_ROWS * BOARD_COLS, 3), np.float32)
    objp[:, :2] = np.mgrid[0:BOARD_COLS, 0:BOARD_ROWS].T.reshape(-1, 2)
    objp *= SQUARE_SIZE_M

    obj_points, img_points = [], []
    image_size = None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    files = sorted(glob.glob(os.path.join(IMG_DIR, "*.png")))
    if not files:
        sys.exit(f"No images found in {IMG_DIR}/. Run 'capture' first.")

    for f in files:
        img = cv2.imread(f)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if image_size is None:
            image_size = gray.shape[::-1]  # (width, height)
        elif gray.shape[::-1] != image_size:
            print("skip (different size):", f)
            continue

        found, corners = cv2.findChessboardCorners(gray, pattern)
        if not found:
            print("no board found:", f)
            continue
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        obj_points.append(objp)
        img_points.append(corners)

    print(f"Using {len(obj_points)} / {len(files)} images")
    if len(obj_points) < 10:
        print("WARNING: fewer than 10 good views; aim for 20-30.")

    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, image_size, None, None
    )

    # Per-image reprojection error
    errs = []
    for i in range(len(obj_points)):
        proj, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], K, dist)
        proj = proj.reshape(-1, 2)          # (N, 2)
        pts = img_points[i].reshape(-1, 2)  # (N, 2)
        errs.append(np.sqrt(np.mean(np.sum((pts - proj) ** 2, axis=1))))

    print(f"\nRMS reprojection error (OpenCV): {rms:.4f} px   (good: < ~0.5)")
    print(f"Mean per-image error:            {np.mean(errs):.4f} px")
    print(f"Worst image: index {int(np.argmax(errs))} ({max(errs):.3f} px) -> consider deleting it")

    print("\nCamera matrix K:\n", K)
    print("\nDistortion coeffs (k1, k2, p1, p2, k3):\n", dist.ravel())
    print(f"\nfx={K[0,0]:.2f}  fy={K[1,1]:.2f}  cx={K[0,2]:.2f}  cy={K[1,2]:.2f}")

    np.savez(OUT_NPZ, K=K, dist=dist, image_size=np.array(image_size), rms=rms)
    with open(OUT_JSON, "w") as fh:
        json.dump({
            "image_size": list(image_size),
            "K": K.tolist(),
            "dist": dist.ravel().tolist(),
            "rms": rms,
        }, fh, indent=2)
    print(f"\nSaved to {OUT_NPZ} and {OUT_JSON}")

    # Quick visual check: undistort the first image
    img = cv2.imread(files[0])
    und = cv2.undistort(img, K, dist)
    cv2.imshow("original | undistorted", np.hstack([img, und]))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("capture", "calibrate"):
        sys.exit("usage: python calibrate_camera.py [capture|calibrate]")
    capture() if sys.argv[1] == "capture" else calibrate()
