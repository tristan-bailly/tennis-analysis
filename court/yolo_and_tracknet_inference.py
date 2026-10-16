import torch
import cv2
import numpy as np
from tqdm import tqdm
import argparse
from itertools import groupby
from scipy.spatial import distance

from ultralytics import YOLO

from tracknet_model import BallTrackerNet
from court_reference import CourtReference
from player_reference import PlayerTracker
from stroke_analysis import StrokeDetector
from stats_panel import make_stats_panel
import os
import subprocess
import imageio_ffmpeg



def postprocess(mask):
    """
    mask: numpy array with shape (1, H*W)
    Returns (x, y) or (None, None)
    """
    mask = mask.reshape(360, 640)
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None, None
    return int(np.mean(xs)), int(np.mean(ys))


def infer_tracknet(frames, model, device, batch_size=16):
    """Run TrackNet on all frames to get ball positions (batched for high performance)"""
    height = 360
    width = 640

    dists = [-1, -1]
    ball_track = [(None, None), (None, None)]

    print("Running TrackNet ball detection...")
    total = len(frames)
    for start_idx in tqdm(range(2, total, batch_size)):
        end_idx = min(start_idx + batch_size, total)
        batch_tensors = []
        for num in range(start_idx, end_idx):
            img = cv2.resize(frames[num], (width, height))
            img_prev = cv2.resize(frames[num - 1], (width, height))
            img_preprev = cv2.resize(frames[num - 2], (width, height))
            imgs = np.concatenate([img, img_prev, img_preprev], axis=2).astype(np.float32) / 255.0
            imgs = np.transpose(imgs, (2, 0, 1))
            batch_tensors.append(imgs)

        inp = torch.from_numpy(np.array(batch_tensors)).float().to(device)
        with torch.no_grad():
            out = model(inp, testing=True)
            preds = out.argmax(dim=1).cpu().numpy()

        for mask in preds:
            x_pred, y_pred = postprocess(mask)
            ball_track.append((x_pred, y_pred))
            if (ball_track[-1][0] is not None and ball_track[-2][0] is not None and
                ball_track[-1][1] is not None and ball_track[-2][1] is not None):
                dist = distance.euclidean(ball_track[-1], ball_track[-2])
            else:
                dist = -1
            dists.append(dist)

    return ball_track, dists


def remove_outliers(ball_track, dists, max_dist=100):
    """Remove outlier ball detections"""
    outliers = np.where(np.array(dists) > max_dist)[0]

    for i in outliers:
        if i + 1 < len(dists) and (dists[i + 1] > max_dist or dists[i + 1] == -1):
            ball_track[i] = (None, None)
        elif i > 0 and dists[i - 1] == -1:
            ball_track[i - 1] = (None, None)

    return ball_track


def split_track(ball_track, max_gap=4, max_dist_gap=80, min_track=5):
    """Split ball track into segments"""
    list_det = [0 if p[0] else 1 for p in ball_track]
    groups = [(k, sum(1 for _ in g)) for k, g in groupby(list_det)]

    cursor = 0
    min_value = 0
    result = []

    for i, (k, l) in enumerate(groups):
        if (k == 1) and (0 < i < len(groups) - 1):
            dist = distance.euclidean(ball_track[cursor - 1], ball_track[cursor + l])
            if (l >= max_gap) or (dist / l > max_dist_gap):
                if cursor - min_value > min_track:
                    result.append([min_value, cursor])
                min_value = cursor + l - 1
        cursor += l

    if len(list_det) - min_value > min_track:
        result.append([min_value, len(list_det)])

    return result


def interpolation(coords):
    """Interpolate missing ball positions"""
    def nan_helper(arr):
        return np.isnan(arr), lambda z: z.nonzero()[0]

    x = np.array([c[0] if c[0] is not None else np.nan for c in coords])
    y = np.array([c[1] if c[1] is not None else np.nan for c in coords])

    nans_x, idx = nan_helper(x)
    x[nans_x] = np.interp(idx(nans_x), idx(~nans_x), x[~nans_x])

    nans_y, idy = nan_helper(y)
    y[nans_y] = np.interp(idy(nans_y), idy(~nans_y), y[~nans_y])

    return list(zip(x, y))


REMAPPING = {
    0: 0, 6: 1, 7: 2, 8: 3, 9: 4, 10: 5, 11: 6, 12: 7, 13: 8,
    1: 9, 2: 10, 3: 11, 4: 12, 5: 13
}

BALL_CLASS_ID = 14
PLAYER_CLASS_ID = 15


def get_homography(boxes, frame_shape, court_ref):
    """Compute homography from court keypoints"""
    img_h, img_w = frame_shape[:2]

    ref_pts = []
    img_pts = []

    for b in boxes:
        if b.cls.numel() == 0:
            continue

        cls_id = int(b.cls[0])
        if cls_id not in REMAPPING:
            continue

        mapped_id = REMAPPING[cls_id]

        x1, y1, x2, y2 = b.xyxy[0]
        cx = float((x1 + x2) / 2)
        cy = float((y1 + y2) / 2)

        img_pts.append([cx, cy])
        ref_pts.append(list(court_ref.court_conf[mapped_id]))

    if len(ref_pts) < 5:
        return None

    ref_pts = np.array(ref_pts, dtype=np.float32).reshape(-1, 1, 2)
    img_pts = np.array(img_pts, dtype=np.float32).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(ref_pts, img_pts, cv2.RANSAC, 5.0)
    return H


def _put_label(frame, text, cx, cy, color=(60, 150, 80)):
    """Draw a small centered label at (cx, cy)."""
    font = cv2.FONT_HERSHEY_DUPLEX
    scale = 0.38
    thick = 1
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    cv2.putText(frame, text,
                (cx - tw // 2, cy + th // 2),
                font, scale, color, thick, cv2.LINE_AA)


# COCO keypoint connectivity for human pose skeleton
SKELETON_EDGES = [
    (5, 6),             # shoulders
    (5, 7), (7, 9),     # left arm
    (6, 8), (8, 10),    # right arm
    (5, 11), (6, 12),   # torso sides
    (11, 12),           # hips
    (11, 13), (13, 15), # left leg
    (12, 14), (14, 16), # right leg
]


def draw_skeleton(frame, keypoints, player_boxes, min_conf=0.25):
    """Draw clean motion skeleton on the players matching reference aesthetic."""
    if keypoints is None or len(keypoints) == 0:
        return

    for person in keypoints:
        kpts = person.data[0].cpu().numpy()  # (17, 3)

        # Ensure person matches one of the tracked player regions
        valid_pts = [kpts[i][:2] for i in range(17) if kpts[i][2] > 0.2]
        if len(valid_pts) < 4:
            continue
        cx, cy = np.mean(valid_pts, axis=0)

        matched = False
        for box in player_boxes:
            if box is None:
                continue
            x1, y1, x2, y2 = box
            if (x1 - 50 <= cx <= x2 + 50) and (y1 - 50 <= cy <= y2 + 50):
                matched = True
                break
        if not matched:
            continue

        # Neck to nose line
        if kpts[0][2] >= min_conf and kpts[5][2] >= min_conf and kpts[6][2] >= min_conf:
            neck_x = int((kpts[5][0] + kpts[6][0]) / 2)
            neck_y = int((kpts[5][1] + kpts[6][1]) / 2)
            nose_x = int(kpts[0][0])
            nose_y = int(kpts[0][1])
            cv2.line(frame, (neck_x, neck_y), (nose_x, nose_y), (245, 245, 245), 2, cv2.LINE_AA)

        # Bones
        for p1, p2 in SKELETON_EDGES:
            x1, y1, c1 = kpts[p1]
            x2, y2, c2 = kpts[p2]
            if c1 >= min_conf and c2 >= min_conf:
                cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (245, 245, 245), 2, cv2.LINE_AA)

        # Joint circles
        for i in (0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16):
            x, y, conf = kpts[i]
            if conf >= min_conf:
                cv2.circle(frame, (int(x), int(y)), 3, (190, 245, 190), -1, cv2.LINE_AA)
                cv2.circle(frame, (int(x), int(y)), 4, (40, 40, 40), 1, cv2.LINE_AA)


def add_audio_to_video(temp_video_path, source_video_path, final_output_path):
    """Mux original audio from source_video_path into the final output video."""
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg_exe, "-y",
        "-i", temp_video_path,
        "-i", source_video_path,
        "-c:v", "copy",
        "-c:a", "aac",
        "-map", "0:v:0",
        "-map", "1:a:0?",
        "-shortest",
        final_output_path
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0 and os.path.exists(final_output_path):
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)
            print(f"Audio muxed successfully -> {final_output_path}")
            return
    except Exception as e:
        print(f"Audio muxing notice: {e}")

    # Fallback if audio muxing failed
    if os.path.exists(temp_video_path) and not os.path.exists(final_output_path):
        os.rename(temp_video_path, final_output_path)


def draw_projected_court(frame, H, court_ref, alpha=0.6):

    """Draw court lines with transparency"""
    if H is None:
        return

    lines = court_ref.get_court_lines()
    overlay = frame.copy()

    pts = []
    for p1, p2 in lines:
        pts.append(p1)
        pts.append(p2)

    pts = np.array([pts], dtype=np.float32)

    try:
        warped = cv2.perspectiveTransform(pts, H)[0]
    except:
        return

    for i in range(0, len(warped), 2):
        x1, y1 = warped[i]
        x2, y2 = warped[i + 1]

        if (np.isnan(x1) or np.isnan(y1) or np.isnan(x2) or np.isnan(y2) or
            abs(x1) > 10000 or abs(y1) > 10000):
            continue

        cv2.line(overlay, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)  # red


    # Blend with alpha
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


class ExponentialSmoother:
    """Exponential moving average smoother"""
    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.value = None

    def update(self, new_value):
        if new_value is None:
            return self.value

        if self.value is None:
            self.value = new_value
        else:
            self.value = self.alpha * new_value + (1 - self.alpha) * self.value

        return self.value


def read_video(path_video):
    """Load all frames from video"""
    cap = cv2.VideoCapture(path_video)
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    frames = []
    print("Loading video frames...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    print(f"Loaded {len(frames)} frames at {fps} FPS")
    return frames, fps


def run_inference(video_path, tracknet_model_path, yolo_model_path, output_path, 
                  enable_interpolation=True, ball_smoothing=0.5, trace_length=7):
    """
    Main inference pipeline:
    1. Load video frames
    2. Run TrackNet for ball detection
    3. Run YOLO for court + players
    4. Overlay everything and save
    """
    
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon GPU (MPS) ✓")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA GPU ✓")
    else:
        device = torch.device("cpu")
        print("Using CPU (no GPU available)")
    
    # Load TrackNet
    print("Loading TrackNet model...")
    tracknet_model = BallTrackerNet().to(device)
    state = torch.load(tracknet_model_path, map_location="cpu")
    tracknet_model.load_state_dict(state, strict=True)
    tracknet_model.eval()

    # Load YOLO
    print("Loading YOLO model...")
    yolo_model = YOLO(yolo_model_path)

    # Load Pose model for motion skeleton
    print("Loading YOLO Pose model...")
    pose_model = YOLO("yolov8n-pose.pt").to(device)

    # Court reference
    court_ref = CourtReference()

    # Player tracker
    player_tracker = PlayerTracker(smoothing=0.7, max_missing=15)

    frames, fps = read_video(video_path)
    
    ball_track, dists = infer_tracknet(frames, tracknet_model, device)
    ball_track = remove_outliers(ball_track, dists)

    # Optional interpolation
    if enable_interpolation:
        print("Interpolating ball trajectory...")
        segments = split_track(ball_track)
        for start, end in segments:
            sub = interpolation(ball_track[start:end])
            ball_track[start:end] = sub

    # Ball position smoother (for display coordinates)
    ball_smoother_x = ExponentialSmoother(alpha=ball_smoothing)
    ball_smoother_y = ExponentialSmoother(alpha=ball_smoothing)

    # Process Video with YOLO + Overlay 
    print("Processing video with YOLO and overlaying...")
    
    height, width = frames[0].shape[:2]
    PANEL_H = 320
    temp_output_path = output_path + ".temp.mp4"
    out = cv2.VideoWriter(
        temp_output_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height + PANEL_H)
    )

    last_H = None
    stroke_detector = StrokeDetector(min_gap_frames=18)

    for num in tqdm(range(len(frames))):
        frame = frames[num].copy()

        results = yolo_model(frame, conf=0.25, verbose=False)[0]
        boxes = results.boxes

        H = get_homography(boxes, frame.shape, court_ref)
        if H is not None:
            last_H = H

        if last_H is not None:
            draw_projected_court(frame, last_H, court_ref, alpha=0.5)

        player_boxes = []
        for b in boxes:
            if b.cls.numel() == 0:
                continue
            cls_id = int(b.cls[0])
            if cls_id != PLAYER_CLASS_ID:
                continue
            x1, y1, x2, y2 = b.xyxy[0]
            player_boxes.append([float(x1), float(y1), float(x2), float(y2)])

        player_tracker.update(player_boxes)
        boxes_by_id = player_tracker.get_boxes()

        # Motion skeleton detection and rendering on players
        pose_res = pose_model(frame, conf=0.2, verbose=False)[0]
        draw_skeleton(frame, pose_res.keypoints,
                      [b for b in boxes_by_id.values() if b is not None])

        # Clean player indicator badges above player heads
        for pid, box in boxes_by_id.items():
            if box is None:
                continue

            x1, y1, x2, y2 = box
            mid_x = int((x1 + x2) / 2)
            top_y = int(y1)
            label = "Player Top" if pid == 0 else "Player Bottom"
            color = (0, 180, 0) if pid == 0 else (180, 0, 180)

            # Elegant pill label
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.45, 1)
            cv2.rectangle(frame,
                          (mid_x - tw // 2 - 5, max(0, top_y - th - 8)),
                          (mid_x + tw // 2 + 5, top_y - 2),
                          color, -1)
            cv2.putText(frame, label,
                        (mid_x - tw // 2, top_y - 4),
                        cv2.FONT_HERSHEY_DUPLEX, 0.45,
                        (255, 255, 255), 1, cv2.LINE_AA)

        # Fine, smooth ball trail matching reference
        if num < len(ball_track):
            trail_pts = []
            for i in range(trace_length):
                idx = num - i
                if 0 <= idx < len(ball_track):
                    pt = ball_track[idx]
                    if pt[0] is not None and pt[1] is not None:
                        x_d = int(pt[0] * width / 640)
                        y_d = int(pt[1] * height / 360)
                        trail_pts.append((x_d, y_d))

            # Draw fine connected line segments with color gradient (yellow -> red)
            for j in range(len(trail_pts) - 1):
                p_curr = trail_pts[j]
                p_prev = trail_pts[j + 1]
                frac = 1.0 - (j / max(1, len(trail_pts) - 1))
                g = int(220 * frac)
                color = (0, g, 255)  # BGR gradient
                cv2.line(frame, p_prev, p_curr, color, 2, cv2.LINE_AA)

            # Draw current ball position: fine circular badge
            if trail_pts:
                curr_x, curr_y = trail_pts[0]
                curr_x = ball_smoother_x.update(curr_x)
                curr_y = ball_smoother_y.update(curr_y)
                if curr_x is not None and curr_y is not None:
                    cx, cy = int(curr_x), int(curr_y)
                    cv2.circle(frame, (cx, cy), 5, (255, 255, 255), -1, cv2.LINE_AA)
                    cv2.circle(frame, (cx, cy), 6, (0, 160, 255), 2, cv2.LINE_AA)

        # ── Stroke detection ──────────────────────────────────────────
        boxes_by_id_for_stroke = player_tracker.get_boxes()
        # Scale player boxes from original frame space → TrackNet 640×360 space
        scale_x_tn = 640.0 / width
        scale_y_tn = 360.0 / height
        player_boxes_scaled = {}
        for pid, b in boxes_by_id_for_stroke.items():
            if b is not None:
                x1, y1, x2, y2 = b
                player_boxes_scaled[pid] = [
                    x1 * scale_x_tn, y1 * scale_y_tn,
                    x2 * scale_x_tn, y2 * scale_y_tn,
                ]
            else:
                player_boxes_scaled[pid] = None

        stroke_detected = stroke_detector.update(
            ball_track=ball_track,
            frame_idx=num,
            player_boxes_scaled=player_boxes_scaled,
            H=last_H,
            fps=fps,
            frame_size=(width, height),
        )


        # ── Ball stroke number labels ─────────────────────────────────
        for i, stroke in enumerate(stroke_detector.strokes):
            sx, sy = stroke["ball_px"]
            # scale from tracknet 640x360 space to display space
            sx_d = int(sx * width / 640)
            sy_d = int(sy * height / 360)
            label = str(i + 1)
            # Clean circular marker matching reference aesthetic
            cv2.circle(frame, (sx_d, sy_d), 13, (255, 255, 255), -1)
            cv2.circle(frame, (sx_d, sy_d), 13, (60, 150, 80), 2)
            _put_label(frame, label, sx_d, sy_d)

        # ── Stats panel ───────────────────────────────────────────────
        panel = make_stats_panel(
            stroke_detector.strokes,
            width,
            PANEL_H,
            current_frame=num + 1,
            total_frames=len(frames)
        )
        combined = np.vstack([frame, panel])
        out.write(combined)

    out.release()
    print("Muxing audio into final output...")
    add_audio_to_video(temp_output_path, video_path, output_path)
    print(f"Output saved to: {output_path}")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Tennis Analysis Inference")
    parser.add_argument("--video_path", type=str, required=True, help="Input video path")
    parser.add_argument("--tracknet_model", type=str, required=True, help="TrackNet model path")
    parser.add_argument("--yolo_model", type=str, required=True, help="YOLO model path")
    parser.add_argument("--output_path", type=str, default="output.mp4", help="Output video path")
    parser.add_argument("--no_interpolation", action="store_true", help="Disable ball interpolation")
    parser.add_argument("--ball_smoothing", type=float, default=0.5, help="Ball smoothing factor (0-1)")
    parser.add_argument("--trace_length", type=int, default=7, help="Ball trail length")

    args = parser.parse_args()

    run_inference(
        video_path=args.video_path,
        tracknet_model_path=args.tracknet_model,
        yolo_model_path=args.yolo_model,
        output_path=args.output_path,
        enable_interpolation=not args.no_interpolation,
        ball_smoothing=args.ball_smoothing,
        trace_length=args.trace_length
    )

# example usage:
# python court/final_inference.py --video_path dataset/videos/5.mp4 --tracknet_model weights/tracknet.pt 
# --yolo_model weights/best.pt --output_path output5-new.mp4 --ball_smoothing 0.5 --trace_length 7