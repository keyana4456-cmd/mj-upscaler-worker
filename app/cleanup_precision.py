"""Conservative mask geometry and shot boundaries, shared by preview and export."""
import cv2
import numpy as np

VERSION = 3


def cleanup_suffix(settings):
    import hashlib
    import json
    if not settings or not (settings.get('text') or settings.get('logo') or settings.get('boxes')):
        return ''
    tag = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:8]
    return f'_repair{VERSION}_{tag}'


def scene_ranges(paths):
    """Detect abrupt cuts; conservative heuristics cannot identify every dissolve."""
    starts = [0]
    previous = None
    for i, path in enumerate(paths):
        frame = cv2.imread(str(path))
        if frame is None:
            raise ValueError(f'Cannot read frame: {path}')
        small = cv2.resize(frame, (96, 54)).astype(np.float32)
        if previous is not None:
            delta = np.abs(small - previous)
            # Require widespread change, not just a moving caption or person.
            if delta.mean() > 32 and np.mean(delta.mean(axis=2) > 25) > .55:
                starts.append(i)
        previous = small
    return list(zip(starts, starts[1:] + [len(paths)]))


def manual_mask(shape, boxes, t, img=None):
    """Legacy rectangles, outlines {points}, brush {type: brush, strokes} and tight boxes {type: refine, box, method}."""
    h, w = shape
    mask = np.zeros(shape, np.uint8)
    for region in boxes or []:
        if isinstance(region, dict):
            ta, tb = region.get('start'), region.get('end')
            if not ((ta is None or t >= ta) and (tb is None or t <= tb)): continue
            kind = region.get('type', 'outline')
            if kind == 'brush':
                from mask_refine import brush
                mask |= brush(shape, region.get('strokes'))
            elif kind == 'refine':
                b = region.get('box') or [0, 0, 0, 0]
                if img is not None:
                    from mask_refine import refine
                    mask |= refine(img, b, region.get('method', 'Auto'))
                else:
                    xa, xb = np.clip(sorted((int(b[0]), int(b[2]))), 0, w); ya, yb = np.clip(sorted((int(b[1]), int(b[3]))), 0, h)
                    mask[ya:yb, xa:xb] = 255
            else:
                points = np.asarray(region.get('points', []), dtype=np.float32)
                if points.ndim == 2 and points.shape[1] == 2 and len(points) >= 3 and np.isfinite(points).all():
                    points[:, 0] = np.clip(points[:, 0], 0, w-1)
                    points[:, 1] = np.clip(points[:, 1], 0, h-1)
                    cv2.fillPoly(mask, [points.astype(np.int32)], 255)
            continue
        x0, y0, x1, y1, ta, tb = region
        if (ta is None or t >= ta) and (tb is None or t <= tb):
            xa, xb = np.clip(sorted((int(x0), int(x1))), 0, w)
            ya, yb = np.clip(sorted((int(y0), int(y1))), 0, h)
            mask[ya:yb, xa:xb] = 255
    return mask


def stabilize(raw, weak, radius=2):
    """Hold nearby supported detections, without filling boxes or bridging words."""
    result = []
    for i, current in enumerate(raw):
        near = raw[max(0, i-radius):min(len(raw), i+radius+1)]
        supported = (np.mean([m > 0 for m in near], axis=0) >= .6)
        m = current.copy()
        m[supported] = 255
        candidates = [v > 0 for k, v in weak.items() if abs(k-i) <= 6]
        if len(candidates) >= 3:
            stable = np.mean(candidates, axis=0) >= .75
            # Weak high-pass detections must agree with real text detections;
            # otherwise desk controls and background texture become watermarks.
            gate = cv2.dilate(m, np.ones((5, 5), np.uint8)) > 0
            m[stable & gate] = 255
        result.append(m)
    return result


def feather_inside(mask):
    """All pixels outside the approved mask remain unchanged before encoding."""
    distance = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    return np.minimum(distance / 1.5, 1.0).astype(np.float32)[:, :, None]


def chunks_for_scenes(scenes, chunk, overlap):
    for lo, hi in scenes:
        for start in range(lo, hi, chunk):
            yield start, min(hi, start+chunk), max(lo, start-overlap)
