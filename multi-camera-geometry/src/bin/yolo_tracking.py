"""
Track people with YOLO + DeepSORT and print each bounding-box centre.

Install:
    pip install ultralytics deep-sort-realtime

Usage:
    python track_people.py --model yolov8n.pt
    python track_people.py --model yolov8n.engine --no-show      # headless, TensorRT
    python track_people.py --model yolov8n.pt --no-undistort

Keys (when the window is shown): u = toggle undistortion, q = quit
"""
import argparse
import threading
import time

import cv2
import numpy as np
import torch
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO

CAM_INDEX = 4
WIDTH, HEIGHT, FPS = 1280, 720, 90  # MUST match the calibration resolution
CALIB_FILE = "camera_intrinsics.npz"
ALPHA = 0.0  # 0 = crop to valid pixels, 1 = keep all pixels (black borders)
PERSON_CLASS = 0  # COCO class id for "person"


class LatestFrameReader:
    """Reads the camera in a background thread and keeps only the newest frame.

    YOLO + DeepSORT will usually run slower than the 90 fps camera. Without this,
    the driver buffer fills with old frames and the tracker runs on stale video.
    """

    def __init__(self, cap):
        self.cap = cap
        self.lock = threading.Lock()
        self.frame = None
        self.frame_id = 0
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                self.running = False
                break
            with self.lock:
                self.frame = frame
                self.frame_id += 1

    def read(self, last_id):
        """Return (frame, id) if a newer frame than last_id exists, else (None, last_id)."""
        with self.lock:
            if self.frame is None or self.frame_id == last_id:
                return None, last_id
            return self.frame, self.frame_id

    def stop(self):
        self.running = False
        self.thread.join(timeout=1)


def open_camera():
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
    cap.set(cv2.CAP_PROP_EXPOSURE, 50)
    return cap


def load_undistort_maps():
    data = np.load(CALIB_FILE)
    K, dist = data["K"], data["dist"]
    calib_w, calib_h = (int(v) for v in data["image_size"])
    assert (calib_w, calib_h) == (WIDTH, HEIGHT), (
        f"Calibrated at {calib_w}x{calib_h} but streaming at {WIDTH}x{HEIGHT}"
    )
    new_K, _ = cv2.getOptimalNewCameraMatrix(K, dist, (WIDTH, HEIGHT), ALPHA, (WIDTH, HEIGHT))
    map1, map2 = cv2.initUndistortRectifyMap(K, dist, None, new_K, (WIDTH, HEIGHT), cv2.CV_16SC2)
    return map1, map2, new_K


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to YOLO weights (.pt / .engine / .onnx)")
    ap.add_argument("--conf", type=float, default=0.4, help="Detection confidence threshold")
    ap.add_argument("--imgsz", type=int, default=640, help="YOLO inference size")
    ap.add_argument("--max-age", type=int, default=30, help="Frames to keep a lost track alive")
    ap.add_argument("--n-init", type=int, default=3, help="Detections needed to confirm a track")
    ap.add_argument("--no-undistort", action="store_true", help="Skip lens undistortion")
    ap.add_argument("--no-show", action="store_true", help="Headless: don't open a window")
    args = ap.parse_args()

    use_gpu = torch.cuda.is_available()
    device = 0 if use_gpu else "cpu"

    model = YOLO(args.model)

    # DeepSORT: appearance embedder (MobileNet) + Kalman filter + Hungarian matching.
    tracker = DeepSort(
        max_age=args.max_age,
        n_init=args.n_init,
        max_cosine_distance=0.3,
        embedder="mobilenet",
        embedder_gpu=use_gpu,
        half=use_gpu,
        bgr=True,  # we feed OpenCV BGR frames
    )

    undistort_on = not args.no_undistort
    map1 = map2 = None
    if undistort_on:
        map1, map2, new_K = load_undistort_maps()
        print("Centres are in undistorted pixel coords; use new_K downstream:\n", new_K)

    cap = open_camera()
    reader = LatestFrameReader(cap)

    last_id = 0
    n, t0 = 0, time.time()
    try:
        while reader.running:
            frame, last_id = reader.read(last_id)
            if frame is None:
                time.sleep(0.001)
                continue

            if undistort_on:
                frame = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)

            # --- YOLO: people only ---
            predict_kwargs = dict(
                classes=[PERSON_CLASS],
                conf=args.conf,
                imgsz=args.imgsz,
                device=device,
                verbose=False,
            )
            if use_gpu:
                predict_kwargs["quantize"] = "fp16"   # replaces half=True

            result = model.predict(frame, **predict_kwargs)[0]

            detections = []
            for (x1, y1, x2, y2), conf in zip(
                result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()
            ):
                # deep-sort-realtime wants ([left, top, w, h], confidence, class)
                detections.append(([x1, y1, x2 - x1, y2 - y1], float(conf), "person"))

            # --- DeepSORT ---
            tracks = tracker.update_tracks(detections, frame=frame)

            for track in tracks:
                # Skip unconfirmed tracks and tracks coasting on prediction only
                if not track.is_confirmed() or track.time_since_update > 0:
                    continue
                l, t, r, b = track.to_ltrb()
                cx, cy = (l + r) / 2.0, (t + b) / 2.0
                print(f"id={track.track_id} centre=({cx:.1f}, {cy:.1f})")

                if not args.no_show:
                    cv2.rectangle(frame, (int(l), int(t)), (int(r), int(b)), (0, 255, 0), 2)
                    cv2.circle(frame, (int(cx), int(cy)), 4, (0, 0, 255), -1)
                    cv2.putText(frame, f"ID {track.track_id}", (int(l), max(int(t) - 8, 12)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            n += 1
            if n % 60 == 0:
                print(f"[{n / (time.time() - t0):.1f} fps]")
                n, t0 = 0, time.time()

            if not args.no_show:
                cv2.putText(frame, f"undistort: {undistort_on}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("People Tracking", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("u") and map1 is not None:
                    undistort_on = not undistort_on
    except KeyboardInterrupt:
        pass
    finally:
        reader.stop()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
