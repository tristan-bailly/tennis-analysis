"""
stats_panel.py
--------------
Renders the statistics overlay panel below the video frame.
Matches the clean, modern aesthetic of the reference:
  ┌──────────────┬───────────────────┬─────────────────┐
  │  strokes     │ forehand/backhand │  shot speed     │
  ├──────────────┴───────────────────┴─────────────────┤
  │  speed history (bar chart)       │  shot placement │
  └──────────────────────────────────┴─────────────────┘
"""

import cv2
import numpy as np


# Palette (BGR)
_BG        = (255, 255, 255)   # Clean crisp white
_DIVIDER   = (215, 215, 215)   # Subtle light-grey divider
_ACCENT    = (180, 50,  20)    # Clean deep blue (matches reference)
_TEXT_DK   = (30,  30,  30)    # Bold dark grey / black
_TEXT_LT   = (110, 110, 110)   # Neutral secondary text
_BAR_FH    = (80,  190, 90)    # Forehand green
_BAR_BH    = (50,  140, 240)   # Backhand vibrant orange
_BAR_PEAK  = (35,  75,  235)   # Peak shot highlight
_COURT_BG  = (250, 250, 248)   # Court floor off-white
_COURT_LN  = (130, 130, 130)   # Court line grey
_DOT_BG    = (255, 255, 255)
_DOT_BORDER= (60,  150, 80)    # Green ring for shot dots


def _put(img, text, pos, scale, color, thick=1, bold=False):
    font = cv2.FONT_HERSHEY_DUPLEX if bold else cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, text, pos, font, scale, color, thick, cv2.LINE_AA)


def make_stats_panel(strokes: list[dict],
                     frame_w: int,
                     panel_h: int = 320,
                     current_frame: int = 0,
                     total_frames: int = 0) -> np.ndarray:
    """
    Build a (panel_h x frame_w x 3) uint8 array containing the metrics UI.
    """
    panel = np.full((panel_h, frame_w, 3), _BG, dtype=np.uint8)

    n_strokes  = len(strokes)
    n_fh       = sum(1 for s in strokes if s.get("side") == "forehand")
    n_bh       = n_strokes - n_fh
    speeds     = [s["speed_kmh"] for s in strokes if s.get("speed_kmh") is not None]
    speed_now  = speeds[-1] if speeds else None
    speed_peak = max(speeds) if speeds else None
    fh_speeds  = [s["speed_kmh"] for s in strokes
                  if s.get("side") == "forehand" and s.get("speed_kmh") is not None]
    bh_speeds  = [s["speed_kmh"] for s in strokes
                  if s.get("side") == "backhand" and s.get("speed_kmh") is not None]
    fh_avg     = float(np.mean(fh_speeds)) if fh_speeds else None
    bh_avg     = float(np.mean(bh_speeds)) if bh_speeds else None

    # ── Divider grid ──────────────────────────────────────────────────
    top_h  = panel_h // 2
    col_w  = frame_w // 3

    # Outer border & dividers
    cv2.line(panel, (0, 0), (frame_w, 0), _DIVIDER, 1)
    cv2.line(panel, (0, top_h), (frame_w, top_h), _DIVIDER, 1)
    cv2.line(panel, (col_w, 0), (col_w, top_h), _DIVIDER, 1)
    cv2.line(panel, (2 * col_w, 0), (2 * col_w, top_h), _DIVIDER, 1)
    cv2.line(panel, (frame_w // 2, top_h), (frame_w // 2, panel_h), _DIVIDER, 1)

    pad = 24

    # ── TOP 1: strokes ────────────────────────────────────────────────
    _put(panel, "strokes", (pad, pad + 16), 0.55, _ACCENT, 1, bold=True)
    _put(panel, str(n_strokes), (pad, pad + 72), 2.0, _TEXT_DK, 3, bold=True)

    # ── TOP 2: forehand / backhand ────────────────────────────────────
    mx = col_w + pad
    _put(panel, "forehand / backhand", (mx, pad + 16), 0.55, _ACCENT, 1, bold=True)
    fh_bh_text = f"{n_fh}  /  {n_bh}"
    _put(panel, fh_bh_text, (mx, pad + 72), 2.0, _TEXT_DK, 3, bold=True)

    # ── TOP 3: shot speed ─────────────────────────────────────────────
    rx = 2 * col_w + pad
    _put(panel, "shot speed", (rx, pad + 16), 0.55, _ACCENT, 1, bold=True)
    if speed_now is not None:
        spd_val = str(int(round(speed_now)))
        _put(panel, spd_val, (rx, pad + 72), 2.0, _TEXT_DK, 3, bold=True)
        # Offset km/h based on digit length
        val_w = 40 * len(spd_val)
        _put(panel, "km/h", (rx + val_w + 10, pad + 72), 0.7, _TEXT_LT, 1)

        sub_y = pad + 96
        if speed_peak is not None:
            _put(panel, f"peak: {int(round(speed_peak))} km/h", (rx, sub_y), 0.42, _TEXT_LT, 1)
            sub_y += 18
        if fh_avg is not None:
            _put(panel, f"forehand avg: {int(round(fh_avg))} km/h", (rx, sub_y), 0.42, _TEXT_LT, 1)
            sub_y += 18
        if bh_avg is not None:
            _put(panel, f"backhand avg: {int(round(bh_avg))} km/h", (rx, sub_y), 0.42, _TEXT_LT, 1)
    else:
        _put(panel, "--", (rx, pad + 72), 2.0, _TEXT_DK, 3, bold=True)

    # ── BOTTOM LEFT: speed history ────────────────────────────────────
    _draw_speed_history(
        panel,
        speeds=speeds,
        speed_peak=speed_peak,
        x0=pad,
        y0=top_h + pad // 2,
        w=frame_w // 2 - 2 * pad,
        h=panel_h - top_h - pad,
        strokes=strokes,
    )

    # ── BOTTOM RIGHT: shot placement ──────────────────────────────────
    court_pts = [(s["court_pos"], s) for s in strokes if s.get("court_pos") is not None]
    _draw_shot_placement(
        panel,
        court_pts=court_pts,
        x0=frame_w // 2 + pad,
        y0=top_h + pad // 2,
        w=frame_w // 2 - 2 * pad,
        h=panel_h - top_h - pad,
        current_frame=current_frame,
        total_frames=total_frames,
    )

    return panel


def _draw_speed_history(panel, speeds, speed_peak, x0, y0, w, h, strokes):
    """Draw a clean bar chart with axes and stroke labels."""
    _put(panel, "speed history", (x0, y0 + 16), 0.52, _ACCENT, 1, bold=True)
    if speed_peak is not None:
        _put(panel, f"max  {int(round(speed_peak))} km/h", (x0 + w - 120, y0 + 16), 0.42, _TEXT_LT, 1)

    chart_x0 = x0 + 44
    chart_y0 = y0 + 28
    chart_w  = w - 48
    chart_h  = h - 52

    # Y-axis labels & gridlines
    max_spd = max(100.0, float(speed_peak) if speed_peak else 100.0)
    for tick_val in (0, 50, 100):
        ty = chart_y0 + chart_h - int((tick_val / max_spd) * chart_h)
        _put(panel, str(tick_val), (x0 + 6, ty + 4), 0.35, _TEXT_LT, 1)
        cv2.line(panel, (chart_x0, ty), (chart_x0 + chart_w, ty), (230, 230, 230), 1)

    # Base line
    cv2.line(panel, (chart_x0, chart_y0 + chart_h),
             (chart_x0 + chart_w, chart_y0 + chart_h), (180, 180, 180), 1)

    # Vertical axis title
    _put(panel, "km/h", (x0 + 4, chart_y0 - 4), 0.34, _TEXT_LT, 1)

    if not speeds:
        return

    n = len(speeds)
    slot_w = min(42, max(12, chart_w // max(n, 1)))
    bar_w  = max(8, int(slot_w * 0.72))

    for i, s in enumerate(speeds):
        bar_h_px = int(chart_h * (s / max_spd))
        bx = chart_x0 + 10 + i * slot_w
        by = chart_y0 + chart_h - bar_h_px

        stroke = strokes[i] if i < len(strokes) else {}
        is_peak = (speed_peak is not None and abs(s - speed_peak) < 0.5)

        if is_peak:
            color = _BAR_PEAK
        elif stroke.get("side") == "forehand":
            color = _BAR_FH
        else:
            color = _BAR_BH

        # Rounded top effect: draw filled rectangle
        cv2.rectangle(panel, (bx, by), (bx + bar_w, chart_y0 + chart_h), color, -1)

        # Stroke index number below bar
        _put(panel, str(i + 1), (bx + bar_w // 2 - 4, chart_y0 + chart_h + 15), 0.36, _TEXT_LT, 1)

    # Stroke # axis label
    total_bar_w = n * slot_w
    _put(panel, "stroke #", (chart_x0 + total_bar_w // 2 - 20, chart_y0 + chart_h + 28), 0.36, _TEXT_LT, 1)


def _draw_shot_placement(panel, court_pts, x0, y0, w, h, current_frame=0, total_frames=0):
    """
    Draw a clean, properly-proportioned top-down tennis court view with impact points.
    Court dimensions: 23.77m length x 10.97m width -> 2.167:1 ratio.
    """
    _put(panel, "shot placement", (x0, y0 + 16), 0.52, _ACCENT, 1, bold=True)

    avail_w = w - 20
    avail_h = h - 38

    # Render a horizontal court: length along X (23.77m), width along Y (10.97m)
    court_ratio = 23.77 / 10.97  # ~2.167
    court_h = int(min(avail_h, avail_w / court_ratio))
    court_w = int(court_h * court_ratio)

    cx0 = x0 + (avail_w - court_w) // 2
    cy0 = y0 + 26 + (avail_h - court_h) // 2

    # Court background & border
    cv2.rectangle(panel, (cx0, cy0), (cx0 + court_w, cy0 + court_h), _COURT_BG, -1)
    cv2.rectangle(panel, (cx0, cy0), (cx0 + court_w, cy0 + court_h), _COURT_LN, 2)

    # Court lines (horizontal court layout)
    # Court length is along X (0 = Top baseline, 1 = Bottom baseline)
    # Court width is along Y (0 = Left doubles line, 1 = Right doubles line)
    singles_margin_y = (1.0 - (8.23 / 10.97)) / 2.0
    sy_top = cy0 + int(singles_margin_y * court_h)
    sy_bot = cy0 + int((1.0 - singles_margin_y) * court_h)

    # Singles sidelines (horizontal lines)
    cv2.line(panel, (cx0, sy_top), (cx0 + court_w, sy_top), _COURT_LN, 1)
    cv2.line(panel, (cx0, sy_bot), (cx0 + court_w, sy_bot), _COURT_LN, 1)

    # Net in middle (vertical line)
    net_x = cx0 + court_w // 2
    cv2.line(panel, (net_x, cy0), (net_x, cy0 + court_h), (80, 80, 80), 2)

    # Service lines: 5.485m from each baseline
    srv_dist_x = int((5.485 / 23.77) * court_w)
    srv_x1 = cx0 + srv_dist_x
    srv_x2 = cx0 + court_w - srv_dist_x
    cv2.line(panel, (srv_x1, sy_top), (srv_x1, sy_bot), _COURT_LN, 1)
    cv2.line(panel, (srv_x2, sy_top), (srv_x2, sy_bot), _COURT_LN, 1)

    # Center service line
    mid_y = cy0 + court_h // 2
    cv2.line(panel, (srv_x1, mid_y), (srv_x2, mid_y), _COURT_LN, 1)

    # Draw shot dots with green ring and stroke number
    for i, (court_pos, stroke) in enumerate(court_pts):
        nx, ny = court_pos
        # Clamp coordinates to [0, 1]
        nx = max(0.02, min(0.98, nx))
        ny = max(0.02, min(0.98, ny))

        # nx is width (Y-axis), ny is length (X-axis)
        px = cx0 + int(ny * court_w)
        py = cy0 + int(nx * court_h)

        # Crisp circular marker matching reference
        cv2.circle(panel, (px, py), 9, _DOT_BG, -1)
        cv2.circle(panel, (px, py), 9, _DOT_BORDER, 2)
        num_str = str(i + 1)
        shift_x = 4 if len(num_str) == 1 else 7
        _put(panel, num_str, (px - shift_x, py + 4), 0.35, _DOT_BORDER, 1, bold=True)

    # Frame counter at bottom right corner (like reference image)
    if total_frames > 0:
        f_str = f"frame  {current_frame}  /  {total_frames}"
        _put(panel, f_str, (x0 + w - 140, y0 + h - 2), 0.34, (160, 160, 160), 1)
