"""Tight watermark masks: refine a rough box into the actual icon/text shape, plus brush strokes."""
import cv2
import numpy as np

METHODS = ("Auto", "Color", "Threshold", "Edge")


def _fill(binary):
    out = np.zeros_like(binary)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, 255, -1)
    return out


def refine(img, box, method="Auto", grow=2):
    """Binary 0/255 mask (full frame) of the overlay inside box=(x0,y0,x1,y1)."""
    h, w = img.shape[:2]
    x0, x1 = sorted((int(round(box[0])), int(round(box[2]))))
    y0, y1 = sorted((int(round(box[1])), int(round(box[3]))))
    x0, x1 = max(0, x0), min(w, x1); y0, y1 = max(0, y0), min(h, y1)
    full = np.zeros((h, w), np.uint8)
    if x1 - x0 < 4 or y1 - y0 < 4: return full
    crop = img[y0:y1, x0:x1]
    method = (method or "Auto").capitalize()
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
    border = np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])
    diff = np.linalg.norm(lab - np.median(border, 0), axis=2)
    diff8 = np.clip(diff * (255.0 / max(1.0, np.percentile(diff, 99.5))), 0, 255).astype(np.uint8)
    votes = []
    if method in ("Auto", "Color"):
        _, m = cv2.threshold(diff8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU); votes.append(m)
    if method in ("Auto", "Threshold"):
        _, m = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        edge = np.concatenate([m[0], m[-1], m[:, 0], m[:, -1]])
        if edge.mean() > 127: m = 255 - m          # the border is background; the overlay is the other side
        votes.append(m)
    if method in ("Auto", "Edge"):
        e = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 40, 120)
        e = cv2.morphologyEx(e, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=2)
        votes.append(_fill(e) | e)
    m = votes[0] if len(votes) == 1 else ((np.sum([v > 0 for v in votes], 0) >= 2) * 255).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, lab_, stats, _ = cv2.connectedComponentsWithStats(m)
    keep = np.zeros_like(m); min_area = max(3, m.size // 2000)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area: keep[lab_ == i] = 255
    if grow: keep = cv2.dilate(keep, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1)))
    full[y0:y1, x0:x1] = keep
    return full


def brush(shape, strokes):
    m = np.zeros(shape, np.uint8)
    for x, y, r in strokes or []:
        cv2.circle(m, (int(round(x)), int(round(y))), max(1, int(round(r))), 255, -1)
    return m


def soft(mask, feather=2.0):
    """Anti-aliased 0..1 alpha: inside stays 1, edge ramps out over `feather` px."""
    if not mask.any(): return np.zeros(mask.shape, np.float32)
    outside = cv2.distanceTransform(255 - (mask > 0).astype(np.uint8) * 255, cv2.DIST_L2, 3)
    return np.clip(1.0 - outside / max(0.5, feather), 0, 1).astype(np.float32)


def scale_region(r, fx, fy):
    """Map a region drawn on one video onto another resolution."""
    if isinstance(r, dict):
        r = dict(r)
        if "points" in r: r["points"] = [[x * fx, y * fy] for x, y in r["points"]]
        if "box" in r: b = r["box"]; r["box"] = [b[0] * fx, b[1] * fy, b[2] * fx, b[3] * fy]
        if "strokes" in r: r["strokes"] = [[x * fx, y * fy, s * (fx + fy) / 2] for x, y, s in r["strokes"]]
        return r
    x0, y0, x1, y1, a, b = r
    return (x0 * fx, y0 * fy, x1 * fx, y1 * fy, a, b)
