"""
Live face detection on Raspberry Pi 5 + Hailo using retinaface_mobilenet_v1.hef

Setup (Pi):
    sudo apt install hailo-all
    python3 -m venv --system-site-packages venv && source venv/bin/activate
    pip install opencv-python numpy

Run:
    python hailo_retinaface.py --hef retinaface_mobilenet_v1.hef
    python hailo_retinaface.py --hef retinaface_mobilenet_v1.hef --no-show   # headless

Keys: q = quit
"""
import argparse
import threading
import time

import cv2
import numpy as np
from hailo_platform import (HEF, ConfigureParams, FormatType, HailoStreamInterface,
                            InferVStreams, InputVStreamParams, OutputVStreamParams, VDevice)

CAM_INDEX = 4
WIDTH, HEIGHT, FPS = 1280, 720, 30

# Standard Pytorch_Retinaface settings (what the Model Zoo network was trained with)
MIN_SIZES = {8: [16, 32], 16: [64, 128], 32: [256, 512]}
VARIANCES = (0.1, 0.2)
KIND_BY_CHANNELS = {8: "loc", 4: "conf", 20: "lm"}  # 2 anchors x (4 box / 2 cls / 10 landmark)


class LatestFrameReader:
    def __init__(self, cap):
        self.cap, self.lock = cap, threading.Lock()
        self.frame, self.frame_id, self.running = None, 0, True
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self.running:
            ok, f = self.cap.read()
            if not ok:
                self.running = False
                break
            with self.lock:
                self.frame, self.frame_id = f, self.frame_id + 1

    def read(self, last_id):
        with self.lock:
            if self.frame is None or self.frame_id == last_id:
                return None, last_id
            return self.frame, self.frame_id

    def stop(self):
        self.running = False


def open_camera():
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def letterbox(img, in_w, in_h):
    h, w = img.shape[:2]
    s = min(in_w / w, in_h / h)
    nw, nh = int(round(w * s)), int(round(h * s))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((in_h, in_w, 3), np.uint8)
    px, py = (in_w - nw) // 2, (in_h - nh) // 2
    canvas[py:py + nh, px:px + nw] = resized
    return canvas, s, px, py


def priors_for(stride, H, W, in_w, in_h):
    """Anchor centres/sizes (normalised), ordered cell-major then anchor, matching the output layout."""
    ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    cx = (xs.reshape(-1, 1) + 0.5) * stride / in_w
    cy = (ys.reshape(-1, 1) + 0.5) * stride / in_h
    sizes = np.array(MIN_SIZES[stride], np.float32)
    sw, sh = sizes / in_w, sizes / in_h
    n = len(sizes)
    cx = np.repeat(cx, n, axis=1).reshape(-1)
    cy = np.repeat(cy, n, axis=1).reshape(-1)
    pw = np.tile(sw, H * W)
    ph = np.tile(sh, H * W)
    return cx, cy, pw, ph


def softmax_fg(conf):
    """conf: (N,2). Return foreground prob; skip softmax if the HEF already did it."""
    if conf.min() >= 0 and conf.max() <= 1.0 and np.allclose(conf.sum(1), 1.0, atol=1e-2):
        return conf[:, 1]
    e = np.exp(conf - conf.max(1, keepdims=True))
    return (e / e.sum(1, keepdims=True))[:, 1]


def decode(outs, in_w, in_h, conf_thr, nms_thr):
    groups = {}
    for a in outs.values():
        a = np.asarray(a)
        if a.ndim == 4:
            a = a[0]
        H, W, C = a.shape
        stride = int(round(in_h / H))
        groups.setdefault(stride, {})[KIND_BY_CHANNELS[C]] = a

    B, S, L = [], [], []
    for stride, g in groups.items():
        H, W, _ = g["loc"].shape
        cx, cy, pw, ph = priors_for(stride, H, W, in_w, in_h)
        loc = g["loc"].reshape(-1, 4)
        score = softmax_fg(g["conf"].reshape(-1, 2))
        lm = g["lm"].reshape(-1, 10)
        keep = score > conf_thr
        if not keep.any():
            continue
        loc, score, lm = loc[keep], score[keep], lm[keep]
        cx, cy, pw, ph = cx[keep], cy[keep], pw[keep], ph[keep]

        bcx = cx + loc[:, 0] * VARIANCES[0] * pw
        bcy = cy + loc[:, 1] * VARIANCES[0] * ph
        bw = pw * np.exp(loc[:, 2] * VARIANCES[1])
        bh = ph * np.exp(loc[:, 3] * VARIANCES[1])
        boxes = np.stack([(bcx - bw / 2) * in_w, (bcy - bh / 2) * in_h,
                          (bcx + bw / 2) * in_w, (bcy + bh / 2) * in_h], 1)
        pts = lm.reshape(-1, 5, 2)
        pts[..., 0] = (cx[:, None] + pts[..., 0] * VARIANCES[0] * pw[:, None]) * in_w
        pts[..., 1] = (cy[:, None] + pts[..., 1] * VARIANCES[0] * ph[:, None]) * in_h
        B.append(boxes); S.append(score); L.append(pts)

    if not B:
        return np.zeros((0, 4)), np.zeros(0), np.zeros((0, 5, 2))
    B, S, L = np.concatenate(B), np.concatenate(S), np.concatenate(L)
    xywh = np.stack([B[:, 0], B[:, 1], B[:, 2] - B[:, 0], B[:, 3] - B[:, 1]], 1)
    idx = cv2.dnn.NMSBoxes(xywh.tolist(), S.tolist(), conf_thr, nms_thr)
    idx = np.array(idx).reshape(-1)
    return B[idx], S[idx], L[idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hef", required=True)
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--nms", type=float, default=0.4)
    ap.add_argument("--rgb", action="store_true", help="Feed RGB instead of BGR (try if detections are poor)")
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()

    hef = HEF(args.hef)
    in_info = hef.get_input_vstream_infos()[0]
    in_h, in_w = in_info.shape[0], in_info.shape[1]
    print(f"Model input: {in_w}x{in_h}")
    for o in hef.get_output_vstream_infos():
        print("output:", o.name, o.shape)

    cap = open_camera()
    reader = LatestFrameReader(cap)

    with VDevice() as target:
        cfg = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
        ng = target.configure(hef, cfg)[0]
        ng_params = ng.create_params()
        try:  # newer HailoRT
            in_p = InputVStreamParams.make(ng, format_type=FormatType.UINT8)
            out_p = OutputVStreamParams.make(ng, format_type=FormatType.FLOAT32)
        except AttributeError:  # older HailoRT
            in_p = InputVStreamParams.make_from_network_group(ng, quantized=False, format_type=FormatType.UINT8)
            out_p = OutputVStreamParams.make_from_network_group(ng, quantized=False, format_type=FormatType.FLOAT32)

        last_id, n, t0 = 0, 0, time.time()
        try:
            with InferVStreams(ng, in_p, out_p) as pipe, ng.activate(ng_params):
                while reader.running:
                    frame, last_id = reader.read(last_id)
                    if frame is None:
                        time.sleep(0.001)
                        continue

                    inp, scale, px, py = letterbox(frame, in_w, in_h)
                    if args.rgb:
                        inp = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)
                    outs = pipe.infer({in_info.name: inp[None]})
                    boxes, scores, lms = decode(outs, in_w, in_h, args.conf, args.nms)

                    # back to original frame coordinates
                    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - px) / scale
                    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - py) / scale
                    lms[..., 0] = (lms[..., 0] - px) / scale
                    lms[..., 1] = (lms[..., 1] - py) / scale

                    for (x1, y1, x2, y2), s, pts in zip(boxes, scores, lms):
                        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                        print(f"face centre=({cx:.1f}, {cy:.1f}) score={s:.2f}")
                        if not args.no_show:
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                            for px_, py_ in pts:
                                cv2.circle(frame, (int(px_), int(py_)), 2, (0, 0, 255), -1)

                    n += 1
                    if n % 60 == 0:
                        print(f"[{n / (time.time() - t0):.1f} fps]")
                        n, t0 = 0, time.time()

                    if not args.no_show:
                        cv2.imshow("Hailo RetinaFace", frame)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break
        except KeyboardInterrupt:
            pass
        finally:
            reader.stop()
            cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
