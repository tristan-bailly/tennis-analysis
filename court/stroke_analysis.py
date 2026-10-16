"""
stroke_analysis.py
------------------
Stroke detection, forehand/backhand classification, and ball speed estimation.

Coordinate conventions:
* ball_track positions   : TrackNet space (640 × 360)
* player_boxes_scaled    : TrackNet space (640 × 360)
* H (homography)         : maps court-reference coords (0-1) → full-res image pixels
* frame_size (fw, fh)    : original video resolution
"""

import numpy as np
from scipy.spatial import distance

# Real-world court dimensions (metres)
COURT_WIDTH_M  = 10.97
COURT_HEIGHT_M = 23.77


def _linear_speed(positions_m, times_s):
    """
    Estimate speed (m/s) via OLS linear regression:
    v = sqrt(vx^2 + vy^2)
    where vx = Σ(t - t̄)(x - x̄) / Σ(t - t̄)²
    """
    if len(positions_m) < 3:
        return None

    t  = np.array(times_s, dtype=np.float64)
    cx = np.array([p[0] for p in positions_m], dtype=np.float64)
    cy = np.array([p[1] for p in positions_m], dtype=np.float64)

    t_c   = t - t.mean()
    denom = np.dot(t_c, t_c)
    if denom < 1e-9:
        return None

    vx = np.dot(t_c, cx) / denom
    vy = np.dot(t_c, cy) / denom
    speed_ms = float(np.sqrt(vx**2 + vy**2))
    return speed_ms


def _dist_to_box(bx, by, box):
    """Euclidean distance from point (bx, by) to bounding box [x1, y1, x2, y2]."""
    if box is None:
        return float("inf")
    x1, y1, x2, y2 = box
    dx = max(0.0, x1 - bx, bx - x2)
    dy = max(0.0, y1 - by, by - y2)
    return float(np.sqrt(dx * dx + dy * dy))


class StrokeDetector:
    """
    Detects tennis strokes by identifying trajectory reversals (extrema in Y)
    that occur in close proximity to a tracked player.
    Speed is estimated using the outgoing trajectory mapped to real-world metres.
    """

    def __init__(self, min_gap_frames: int = 18):
        self.min_gap_frames = min_gap_frames
        self._last_stroke_frame = -999
        self._last_stroke_pid = None
        self.strokes: list[dict] = []

    def update(self,
               ball_track: list,
               frame_idx: int,
               player_boxes_scaled: dict,
               H,
               fps: float,
               frame_size: tuple = (1280, 720)) -> bool:
        """
        Check if a stroke occurred at (or just before) frame_idx.
        Returns True if a new stroke was detected.
        """
        # We need a small lookahead to confirm a direction reversal
        # Evaluate candidate stroke at candidate_f = frame_idx - 3
        candidate_f = frame_idx - 3
        if candidate_f < 4 or candidate_f >= len(ball_track) - 3:
            return False

        if candidate_f - self._last_stroke_frame < self.min_gap_frames:
            return False

        bx, by = ball_track[candidate_f]
        if bx is None or by is None:
            return False

        # Check local trajectory around candidate_f
        # f-3, f-1, f, f+1, f+3
        pts_prev = [ball_track[candidate_f - k][1] for k in (3, 2, 1) if ball_track[candidate_f - k][1] is not None]
        pts_next = [ball_track[candidate_f + k][1] for k in (1, 2, 3) if ball_track[candidate_f + k][1] is not None]
        if len(pts_prev) < 2 or len(pts_next) < 2:
            return False

        y_prev = np.mean(pts_prev)
        y_curr = by
        y_next = np.mean(pts_next)

        box_top = player_boxes_scaled.get(0)
        box_bot = player_boxes_scaled.get(1)

        d_top = _dist_to_box(bx, by, box_top)
        d_bot = _dist_to_box(bx, by, box_bot)

        # Stroke by Player Top (pid=0, facing camera, top baseline, y small):
        # Ball approaches Player Top (y decreases), reaches local minimum, then goes towards bottom (y increases)
        # Condition: near Player Top box (dist <= 35px), local minimum in y, clear outgoing displacement
        is_top_stroke = (
            d_top <= 38.0 and
            (y_curr <= y_prev + 1.0) and
            (y_next - y_curr >= 2.5) and
            (by < 180.0)  # Must be in top half
        )

        # Stroke by Player Bottom (pid=1, facing net, bottom baseline, y large):
        # Ball approaches Player Bottom (y increases), reaches local maximum, then goes towards top (y decreases)
        # Condition: near Player Bottom box (dist <= 50px), local maximum in y, clear outgoing displacement
        is_bot_stroke = (
            d_bot <= 55.0 and
            (y_curr >= y_prev - 1.0) and
            (y_curr - y_next >= 2.5) and
            (by > 150.0)  # Must be in bottom half
        )

        if not (is_top_stroke or is_bot_stroke):
            return False

        pid = 0 if is_top_stroke else 1

        # Check player alternation: if same player, require a longer gap (e.g. >= 32 frames)
        if pid == self._last_stroke_pid and (candidate_f - self._last_stroke_frame < 32):
            return False

        # ── Classify side (forehand vs backhand) ──────────────────────
        box = box_top if pid == 0 else box_bot
        side = self._classify_side(bx, box, pid)

        # ── Estimate speed on outgoing trajectory (frames candidate_f+1 to +7) ──
        speed_kmh = self._estimate_speed_outgoing(
            ball_track, candidate_f, H, fps, frame_size
        )

        # ── Normalised court position ─────────────────────────────────
        court_pos = self._to_court_pos(bx, by, H, frame_size)

        self.strokes.append({
            "frame":     candidate_f,
            "player_id": pid,
            "side":      side,
            "speed_kmh": speed_kmh,
            "ball_px":   (bx, by),
            "court_pos": court_pos,
        })
        self._last_stroke_frame = candidate_f
        self._last_stroke_pid   = pid

        player_name = "Player Top" if pid == 0 else "Player Bottom"
        spd_str = f"{speed_kmh:.1f} km/h" if speed_kmh is not None else "N/A"
        print(f"[stroke {len(self.strokes)}] frame={candidate_f:03d} "
              f"player={player_name} side={side:8s} speed={spd_str}")
        return True

    def _classify_side(self, bx, box, pid):
        if box is None:
            return "forehand"
        px = (box[0] + box[2]) / 2.0
        # Player Bottom (pid=1) faces away from camera:
        # Ball to screen-right of player -> player's right hand -> Forehand
        if pid == 1:
            return "forehand" if bx >= px else "backhand"
        else:
            # Player Top (pid=0) faces toward camera:
            # Ball to screen-right of player -> player's left hand -> Backhand
            return "forehand" if bx <= px else "backhand"

    def _estimate_speed_outgoing(self, ball_track, impact_f, H, fps, frame_size):
        """
        Estimate exit speed of the shot by fitting linear regression across the
        outgoing trajectory (frames impact_f+1 .. impact_f+7).
        """
        if H is None:
            return None

        fw, fh = frame_size
        sx = fw / 640.0
        sy = fh / 360.0

        try:
            H_inv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            return None

        court_pts = []
        times_s   = []

        max_lookahead = min(impact_f + 8, len(ball_track))
        for fi in range(impact_f + 1, max_lookahead):
            pt = ball_track[fi]
            if pt[0] is None or pt[1] is None:
                continue
            x_full = pt[0] * sx
            y_full = pt[1] * sy
            try:
                w = H_inv @ np.array([x_full, y_full, 1.0])
                if abs(w[2]) < 1e-9:
                    continue
                w /= w[2]
                cx_m = float(w[0]) * COURT_WIDTH_M
                cy_m = float(w[1]) * COURT_HEIGHT_M

                # Generous court boundary check
                if not (-5.0 <= cx_m <= COURT_WIDTH_M + 5.0):
                    continue
                if not (-5.0 <= cy_m <= COURT_HEIGHT_M + 5.0):
                    continue

                court_pts.append((cx_m, cy_m))
                times_s.append(fi / fps)
            except Exception:
                continue

        speed_ms = _linear_speed(court_pts, times_s)
        if speed_ms is None:
            return None

        speed_kmh = speed_ms * 3.6
        if 25.0 <= speed_kmh <= 240.0:
            return speed_kmh
        return None

    def _to_court_pos(self, bx, by, H, frame_size):
        if H is None or bx is None or by is None:
            return None
        fw, fh = frame_size
        try:
            H_inv = np.linalg.inv(H)
            w = H_inv @ np.array([bx * fw / 640.0, by * fh / 360.0, 1.0])
            w /= w[2]
            return (float(np.clip(w[0], 0.0, 1.0)),
                    float(np.clip(w[1], 0.0, 1.0)))
        except Exception:
            return None
