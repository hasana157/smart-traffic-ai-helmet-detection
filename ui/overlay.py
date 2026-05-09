import cv2
import numpy as np


# ── colour palette (BGR format for OpenCV) ───────────────────────────────────
#   Legend:
#   🟢 Green  — Helmet detected
#   🔴 Red    — No helmet (violation)
#   🟡 Yellow — Number plate
#   🟠 Orange — Motorcycle/Bike
#   🔵 Blue   — Rider/Person

COLOR_GREEN  = ( 50, 205,  50)   # green  — with helmet
COLOR_RED    = ( 56,  56, 220)   # red    — no helmet
COLOR_YELLOW = (  0, 215, 255)   # yellow — number plate
COLOR_ORANGE = ( 30, 140, 255)   # orange — bike/motorcycle
COLOR_BLUE   = (235, 140,  50)   # blue   — rider/person

FONT       = cv2.FONT_HERSHEY_DUPLEX
FONT_SMALL = cv2.FONT_HERSHEY_SIMPLEX


def _put_label(frame, text, x1, y1, color, font_scale=0.55, thickness=1):
    """Draw a filled pill label above a bounding box."""
    (tw, th), baseline = cv2.getTextSize(text, FONT, font_scale, thickness)
    pad_x, pad_y = 8, 5
    box_x1 = x1
    box_y1 = max(y1 - th - pad_y * 2 - baseline, 0)
    box_x2 = x1 + tw + pad_x * 2
    box_y2 = y1

    cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), color, -1)
    cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (0, 0, 0), 1)
    cv2.putText(
        frame, text,
        (box_x1 + pad_x, box_y2 - baseline - 2),
        FONT, font_scale, (255, 255, 255), thickness, cv2.LINE_AA
    )


def _put_text_below(frame, text, x1, y2, color, font_scale=0.5, thickness=1):
    """Draw a small label below a bounding box."""
    (tw, th), baseline = cv2.getTextSize(text, FONT_SMALL, font_scale, thickness)
    pad_x, pad_y = 6, 4
    bx1 = x1
    by1 = y2 + 2
    bx2 = x1 + tw + pad_x * 2
    by2 = y2 + th + pad_y * 2 + baseline + 2

    h, w = frame.shape[:2]
    if by2 > h:
        by1 = y2 - th - pad_y * 2 - baseline - 4
        by2 = y2 - 2

    cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, -1)
    cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 0, 0), 1)
    cv2.putText(
        frame, text,
        (bx1 + pad_x, by1 + th + pad_y),
        FONT_SMALL, font_scale, (255, 255, 255), thickness, cv2.LINE_AA
    )


def _glow_rect(frame, x1, y1, x2, y2, color, thickness=2):
    """Draw a bounding box with a soft glow effect."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), color, thickness + 4)
    cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)


def draw_overlay(frame, detections):
    """
    Draws detection boxes and labels on frame.

    Box colours match the legend:
    - 'rider'        → BLUE   box, "RIDER X%"       (from yolov8n COCO)
    - 'bike'         → ORANGE box, "BIKE X%"         (from yolov8n COCO)
    - 'helmet_check' → GREEN  box, "WITH HELMET X%"  (from best.pt)
                     → RED    box, "NO HELMET X%"    (from best.pt — violation)
    - 'number_plate' → YELLOW box, plate text        (from best.pt + OCR)
    """

    if not detections:
        return frame

    h, w = frame.shape[:2]

    plate_boxes = [d for d in detections if d.get("type") == "number_plate"]

    def _rider_has_nearby_plate(rider_det):
        rx1, ry1, rx2, ry2 = (int(rider_det[k]) for k in ("x1", "y1", "x2", "y2"))
        for p in plate_boxes:
            px1, py1, px2, py2 = (int(p[k]) for k in ("x1", "y1", "x2", "y2"))
            v_overlap = not (py2 < ry1 or py1 > ry2 + int(0.5 * (ry2 - ry1)))
            h_overlap = not (px2 < rx1 - 30 or px1 > rx2 + 30)
            if v_overlap and h_overlap:
                return True
        return False

    for det in detections:
        x1, y1, x2, y2 = (int(det[k]) for k in ("x1", "y1", "x2", "y2"))
        dtype     = det.get("type", "")
        show_conf = det.get("show_conf", True)
        conf      = det.get("confidence", 0)

        # ── RIDER → BLUE box ──────────────────────────────────────────────
        if dtype == "rider":
            label = "RIDER"
            if show_conf:
                label += f"  {conf * 100:.0f}%"
            _glow_rect(frame, x1, y1, x2, y2, COLOR_BLUE, thickness=2)
            _put_label(frame, label, x1, y1, COLOR_BLUE)

            # NO PLATE corner ticks
            if not _rider_has_nearby_plate(det):
                tick = 18
                for (cx, cy, sx, sy) in [
                    (x1, y1, 1, 1), (x2, y1, -1, 1),
                    (x1, y2, 1, -1), (x2, y2, -1, -1)
                ]:
                    cv2.line(frame, (cx, cy), (cx + sx * tick, cy), COLOR_YELLOW, 2)
                    cv2.line(frame, (cx, cy), (cx, cy + sy * tick), COLOR_YELLOW, 2)
                _put_text_below(frame, "NO PLATE", x2 - 90, y2, COLOR_YELLOW, font_scale=0.42)

        # ── BIKE → ORANGE box ─────────────────────────────────────────────
        elif dtype == "bike":
            label = "BIKE"
            if show_conf:
                label += f"  {conf * 100:.0f}%"
            _glow_rect(frame, x1, y1, x2, y2, COLOR_ORANGE, thickness=2)
            _put_label(frame, label, x1, y1, COLOR_ORANGE)

        # ── HELMET CHECK → GREEN or RED box ───────────────────────────────
        elif dtype == "helmet_check":
            has_helmet  = det.get("has_helmet", True)
            helmet_conf = det.get("helmet_conf", conf)

            if has_helmet:
                color = COLOR_GREEN
                label = "WITH HELMET"
            else:
                color = COLOR_RED
                label = "NO HELMET"

            if show_conf and helmet_conf is not None:
                label += f"  {helmet_conf * 100:.0f}%"

            _glow_rect(frame, x1, y1, x2, y2, color, thickness=3)
            _put_label(frame, label, x1, y1, color, font_scale=0.6)

        # ── NUMBER PLATE → YELLOW box ─────────────────────────────────────
        elif dtype == "number_plate":
            plate_text = det.get("plate_text")
            plate_conf = det.get("plate_conf")

            _glow_rect(frame, x1, y1, x2, y2, COLOR_YELLOW, thickness=2)

            if plate_text:
                plate_label = plate_text
                if show_conf and plate_conf is not None:
                    plate_label += f"  {plate_conf * 100:.0f}%"
                _put_label(frame, plate_label, x1, y1, COLOR_YELLOW, font_scale=0.65)
                _put_text_below(frame, plate_text, x1, y2, COLOR_YELLOW, font_scale=0.48)
            else:
                _put_label(frame, "PLATE", x1, y1, COLOR_YELLOW)

        # ── FALLBACK — any other class ────────────────────────────────────
        else:
            color = (150, 150, 150)
            label = dtype.replace("_", " ").upper()
            if show_conf:
                label += f"  {conf * 100:.0f}%"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
            _put_label(frame, label, x1, y1, color, font_scale=0.45)

    # ── Frame-level summary overlay (top-right corner) ────────────────────
    rider_count   = sum(1 for d in detections if d.get("type") == "rider")
    plate_count   = sum(1 for d in detections if d.get("type") == "number_plate")
    violation_cnt = sum(
        1 for d in detections
        if d.get("type") == "helmet_check" and not d.get("has_helmet", True)
    )

    summary_lines = [
        f"RIDERS: {rider_count}",
        f"PLATES: {plate_count}",
        f"VIOLATIONS: {violation_cnt}",
    ]
    sx, sy = w - 175, 14
    for i, line in enumerate(summary_lines):
        col = COLOR_RED if (i == 2 and violation_cnt > 0) else (200, 200, 200)
        cv2.putText(frame, line, (sx, sy + i * 20),
                    FONT_SMALL, 0.48, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, line, (sx, sy + i * 20),
                    FONT_SMALL, 0.48, col, 1, cv2.LINE_AA)

    return frame