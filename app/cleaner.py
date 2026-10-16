"""MJ Upscaler clean-up: remove burned-in subtitles/text, static logos and user-drawn areas
with ProPainter video inpainting. Only masked pixels are replaced."""
import os, sys, glob, shutil, subprocess, time, math
import numpy as np, cv2
from cleanup_precision import scene_ranges, manual_mask, stabilize, feather_inside, chunks_for_scenes

ROOT = os.path.dirname(os.path.abspath(__file__))
FFMPEG = os.environ.get("MJU_FFMPEG", r"C:\ytdlp\ffmpeg.exe")
PY = os.environ.get("MJU_PY", os.path.join(ROOT, "venv", "Scripts", "python.exe"))
PP = os.path.join(ROOT, "propainter")
DB_MODEL = os.path.join(ROOT, "models", "text_db_en.onnx")
NOWIN = 0x08000000 if os.name == "nt" else 0

def _run(cmd, log=None, cwd=None):
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                         errors="replace", creationflags=NOWIN, cwd=cwd, env=env)
    out = []
    for line in p.stdout:
        out.append(line)
        if log and line.strip() and "%|" not in line: log(line.rstrip()[-200:])
    p.wait()
    if p.returncode: raise RuntimeError(os.path.basename(cmd[0]) + " failed:\n" + "".join(out[-15:]))

# ---------------- masks ----------------
class TextDetector:
    """DB text detector tuned for old footage: runs on the frame and on a contrast-boosted copy,
    with low thresholds so faint / semi-transparent captions and watermarks are found too."""
    def __init__(self, sensitive=True):
        self.m = cv2.dnn.TextDetectionModel_DB(DB_MODEL)
        bt, pt = (0.25, 0.45) if sensitive else (0.3, 0.5)
        self.m.setBinaryThreshold(bt).setPolygonThreshold(pt).setUnclipRatio(1.35).setMaxCandidates(300)
        self.size = None; self.sensitive = sensitive
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    def _boosts(self, img):
        # faint / semi-transparent text: amplify fine detail (high-pass), both polarities
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hp = np.clip(cv2.subtract(g, cv2.GaussianBlur(g, (0, 0), 4)).astype(np.float32) * 8, 0, 255).astype(np.uint8)
        return [cv2.cvtColor(hp, cv2.COLOR_GRAY2BGR), cv2.cvtColor(255 - hp, cv2.COLOR_GRAY2BGR)]
    def _detect(self, img, W, H):
        polys, _ = self.m.detect(cv2.resize(img, (W, H), interpolation=cv2.INTER_CUBIC))
        return polys
    def __call__(self, img, zone="anywhere"):
        h, w = img.shape[:2]
        s = max(1.0, (960 if self.sensitive else 640) / max(w, h))  # small old footage: upsample so letters are big enough
        W, H = int(math.ceil(w * s / 32) * 32), int(math.ceil(h * s / 32) * 32)
        if self.size != (W, H):
            self.m.setInputParams(1.0 / 255.0, (W, H), (122.67891434, 116.66876762, 104.00698793), True); self.size = (W, H)
        return self._mask(list(self._detect(img, W, H)), w, h, W, H, zone)
    def weak(self, img, zone="anywhere"):
        """Extra detections on contrast-boosted frames (faint watermarks). Noisy on their own,
        so the caller keeps only spots that stay put over time and aren't on a face."""
        h, w = img.shape[:2]
        s = min(4.0, max(1.0, 1400 / max(w, h)))
        W, H = int(math.ceil(w * s / 32) * 32), int(math.ceil(h * s / 32) * 32)
        if self.size != (W, H):
            self.m.setInputParams(1.0 / 255.0, (W, H), (122.67891434, 116.66876762, 104.00698793), True); self.size = (W, H)
        polys = []
        for b in self._boosts(img): polys += list(self._detect(b, W, H))
        return self._mask(polys, w, h, W, H, zone)
    def _mask(self, polys, w, h, W, H, zone):
        mask = np.zeros((h, w), np.uint8)
        for p in polys:
            p = (np.array(p, np.float32) * [w / W, h / H]).astype(np.int32)
            x, y, bw, bh = cv2.boundingRect(p)
            if zone == "bottom" and y + bh / 2 < h * 0.6: continue
            if bh > h * 0.35 or bw * bh < 12: continue
            cx, cy = (x + bw / 2) / w, (y + bh / 2) / h
            central = 0.2 < cx < 0.8 and 0.2 < cy < 0.8
            if central and bw < 1.6 * bh and bh > h * 0.05: continue   # blob in the middle (face/hand), not a line of text
            cv2.fillPoly(mask, [p], 255)
        return mask

def static_overlay_mask(frames_dir, log=None, paths=None):
    """Logos/watermarks: edges that stay in the same place while the picture changes."""
    files = paths if paths is not None else sorted(glob.glob(os.path.join(frames_dir, "*.png")))
    pick = files[:: max(1, len(files) // 80)]
    if len(pick) < 10: return None
    imgs = [cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in pick]
    edges = np.mean([cv2.Canny(cv2.GaussianBlur(g, (3, 3), 0), 40, 120) > 0 for g in imgs], axis=0)
    std = np.std(np.stack(imgs).astype(np.float32), axis=0)
    motion = np.mean(np.abs(np.diff(np.stack(imgs).astype(np.float32), axis=0)))
    if motion < 4:  # picture barely changes; can't tell logo from scenery
        if log: log("Logo detection skipped: the shot is too static to tell a logo from the scene")
        return None
    m = ((edges > 0.4) & (std < 40)).astype(np.uint8) * 255
    h, w = m.shape
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    out = np.zeros_like(m)
    for i in range(1, n):
        x, y, bw, bh, a = st[i]
        if a < 30 or bw * bh > 0.12 * w * h: continue
        if x <= 1 or y <= 1 or x + bw >= w - 1 or y + bh >= h - 1: continue  # letterbox/frame edges
        out[lab == i] = 255  # retain component geometry; never fill its bounding rectangle
    if out.any():
        out = cv2.dilate(out, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        if np.mean(out > 0) > .05:
            if log: log('Automatic logo mask is ambiguous (over 5% of the picture); use an outline instead')
            return None
        if log: log(f"Logo/watermark found: {cv2.countNonZero(out) * 100 / (w * h):.1f}% of the picture")
        return out
    if log: log("No static logo found")
    return None

# ---------------- main ----------------
def clean_video(src, out_path, text=True, logo=True, boxes=None, text_zone="anywhere", final=True,
                chunk=120, overlap=12, log=print, progress=None, stop=None, work=None, limit_sec=0, keep_masks=None, output_filter=''):
    """Regions are legacy rectangles or {points, start, end} outline dictionaries."""
    work = work or os.path.join(os.environ.get("MJU_WORK", os.path.join(ROOT, "work")), "clean_" + str(int(time.time())))
    fr, mk = os.path.join(work, "frames"), os.path.join(work, "masks")
    for d in (fr, mk): shutil.rmtree(d, ignore_errors=True); os.makedirs(d)
    cap = cv2.VideoCapture(src); fps = cap.get(5) or 25; cap.release()
    t0 = time.time()
    log("Reading frames...")
    cmd = [FFMPEG, "-v", "error", "-y", "-i", src]
    if limit_sec: cmd += ["-t", str(limit_sec)]
    _run(cmd + [os.path.join(fr, "%06d.png")])
    files = sorted(os.listdir(fr)); n = len(files)
    h, w = cv2.imread(os.path.join(fr, files[0])).shape[:2]
    paths = [os.path.join(fr, f) for f in files]
    scenes = scene_ranges(paths)
    log(f"Precise cleanup: {len(scenes)} shot(s); masks stay within each shot")
    masks = build_masks(paths, fps, text, logo, boxes, text_zone, scenes, log, stop)
    cov = np.mean([m.mean() / 255 for m in masks]) * 100
    log(f"Mask covers {cov:.1f}% of the picture on average")
    for i, m in enumerate(masks): cv2.imwrite(os.path.join(mk, files[i]), m)
    # ProPainter in chunks with overlap
    res_ratio = min(1.0, 960 / max(w, h))
    outd = os.path.join(work, "clean"); os.makedirs(outd, exist_ok=True)
    px = (w * res_ratio) * (h * res_ratio)
    chunk = int(max(24, min(chunk, chunk * 76800 / px)))  # keep VRAM in budget at higher resolutions
    sub = str(min(80, chunk))
    for s, e, a in chunks_for_scenes(scenes, chunk, overlap):
        if stop and stop.is_set(): raise RuntimeError("Stopped")
        idx = range(a, e)
        if not any(masks[i].any() for i in range(s, e)):
            for i in range(s, e): shutil.copy(os.path.join(fr, files[i]), os.path.join(outd, files[i]))
            continue
        cf, cm, co = (os.path.join(work, x) for x in ("cf", "cm", "co"))
        for d in (cf, cm, co): shutil.rmtree(d, ignore_errors=True); os.makedirs(d)
        for j, i in enumerate(idx):
            shutil.copy(os.path.join(fr, files[i]), os.path.join(cf, f"{j:05d}.png"))
            shutil.copy(os.path.join(mk, files[i]), os.path.join(cm, f"{j:05d}.png"))
        # ProPainter optical flow requires at least two frames. For a one-frame
        # shot duplicate only that same frame, never borrow from another shot.
        if len(idx) == 1:
            shutil.copy(os.path.join(cf, "00000.png"), os.path.join(cf, "00001.png"))
            shutil.copy(os.path.join(cm, "00000.png"), os.path.join(cm, "00001.png"))
        log(f"Inpainting frames {s + 1}-{e} of {n}")
        _run([PY, "inference_propainter.py", "-i", cf, "-m", cm, "-o", co, "--save_frames", "--fp16",
              "--resize_ratio", f"{res_ratio:.3f}", "--mask_dilation", "2", "--subvideo_length", sub,
              "--neighbor_length", "10", "--ref_stride", "10", "--raft_iter", "12"], log, cwd=PP)
        res = sorted(glob.glob(os.path.join(co, "cf", "frames", "*.png")))
        for j, i in enumerate(idx):
            if i < s: continue
            orig = cv2.imread(os.path.join(fr, files[i])).astype(np.float32)
            inp = cv2.resize(cv2.imread(res[j]), (w, h), interpolation=cv2.INTER_CUBIC).astype(np.float32)
            a_ = feather_inside(masks[i])
            cv2.imwrite(os.path.join(outd, files[i]), (orig * (1 - a_) + inp * a_).astype(np.uint8))
        if progress: progress(e / n, (time.time() - t0) / e * (n - e))
        for d in (cf, cm, co): shutil.rmtree(d, ignore_errors=True)
    log(f"Clean-up speed: {n / (time.time() - t0):.1f} frames/s")
    enc = ["-c:v", "libx264", "-crf", "12" if final else "2", "-preset", "slow" if final else "veryfast", "-pix_fmt", "yuv420p"]
    if output_filter: enc = ['-vf', output_filter] + enc
    cmd = [FFMPEG, "-v", "error", "-y", "-framerate", str(fps), "-i", os.path.join(outd, "%06d.png")]
    if limit_sec: cmd += ["-t", str(limit_sec)]
    _run(cmd + ["-i", src, "-map", "0:v", "-map", "1:a?"] + enc + ["-c:a", "aac", "-b:a", "320k", "-shortest", out_path])
    if keep_masks: shutil.copytree(mk, keep_masks, dirs_exist_ok=True)
    shutil.rmtree(work, ignore_errors=True)
    return out_path

def build_masks(paths, fps, text=True, logo=True, boxes=None, text_zone="anywhere",
                scenes=None, log=print, stop=None, time_offset=0):
    """Use the same conservative geometry for render and preview."""
    scenes = scenes if scenes is not None else scene_ranges(paths)
    shape = cv2.imread(paths[0]).shape[:2]
    h, w = shape
    det = TextDetector() if text else None
    from logo_templates import LogoTemplates
    reference_detector = LogoTemplates() if logo else None
    recognized = set()
    fd = cv2.FaceDetectorYN.create(os.path.join(ROOT, "models", "yunet.onnx"), "", (w, h), .5) if text or logo else None
    masks = []
    for lo, hi in scenes:
        static = static_overlay_mask('', log, paths[lo:hi]) if logo else None
        raw, weak, reference_masks = [], {}, []
        for i in range(lo, hi):
            if stop and stop.is_set(): raise RuntimeError("Stopped")
            img = cv2.imread(paths[i])
            reference_mask = np.zeros(shape, np.uint8)
            if reference_detector is not None:
                reference_mask, matches = reference_detector.detect(img)
                for match in matches:
                    if match['name'] not in recognized:
                        log(f"Recognized complete watermark: {match['name']} (graphic and lettering)")
                        recognized.add(match['name'])
            reference_masks.append(reference_mask)
            m = det(img, text_zone) if det else np.zeros(shape, np.uint8)
            if det and (i-lo) % 3 == 0: weak[i-lo] = det.weak(img, text_zone)
            if static is not None: m |= static
            raw.append(m)
        stable = stabilize(raw, weak)
        # Small, resolution-aware allowance for antialiasing and outlines.
        radius = max(1, min(3, round(max(w, h) / 720)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))
        for local, m in enumerate(stable):
            i = lo + local
            m = cv2.dilate(m, kernel)
            if fd is not None and m.any():
                _, faces = fd.detect(cv2.imread(paths[i]))
                for face in (faces if faces is not None else []):
                    x, y, bw, bh = [int(v) for v in face[:4]]
                    m[max(0, y-bh//3):min(h, y+bh+bh//3), max(0, x-bw//4):min(w, x+bw+bw//4)] = 0
            # A verified reference identifies the overlaid logo even over a face.
            # Its approved graphic shape is not inferred by enlarging text boxes.
            m |= cv2.dilate(reference_masks[local], kernel)
            # Explicit rectangles still mean remove the entire selected area.
            # Apply after temporal smoothing so their time limits remain exact.
            needs_img = any(isinstance(b, dict) and b.get('type') == 'refine' for b in boxes or [])
            m |= manual_mask(shape, boxes, time_offset + i/fps, cv2.imread(paths[i]) if needs_img else None)
            masks.append(m)
    return masks


def preview_mask(src, t, text=True, logo=True, boxes=None, text_zone="anywhere"):
    """Preview actual mask geometry on a nearby frame window, split at cuts."""
    import tempfile
    cap = cv2.VideoCapture(src)
    try:
        fps = cap.get(5) or 25
        n = int(cap.get(7))
        t = max(0, min(t, max(0, (n-1)/fps)))
        start = max(0, t-15/fps)
        with tempfile.TemporaryDirectory(prefix='mju_mask_') as tmp:
            cmd = [FFMPEG, '-v', 'error', '-y']
            if start: cmd += ['-ss', str(start)]
            _run(cmd + ['-i', src, '-t', str(31/fps), os.path.join(tmp, '%06d.png')])
            paths = sorted(glob.glob(os.path.join(tmp, '*.png')))
            if not paths: raise ValueError('Cannot read preview frame')
            target = min(len(paths)-1, int(round((t-start)*fps)))
            masks = build_masks(paths, fps, text, logo, boxes, text_zone,
                                log=lambda _: None, time_offset=start)
            img, m = cv2.imread(paths[target]), masks[target]
    finally:
        cap.release()
    ov = img.copy(); ov[m > 0] = (0.45 * ov[m > 0] + 0.55 * np.array([65, 179, 224])).astype(np.uint8)
    return img, ov, m

if __name__ == "__main__":
    import argparse
    a = argparse.ArgumentParser()
    a.add_argument("src"); a.add_argument("out")
    a.add_argument("--notext", action="store_true"); a.add_argument("--nologo", action="store_true")
    a.add_argument("--box", action="append", default=[], help="x0,y0,x1,y1[,t0,t1]")
    a.add_argument("--zone", default="anywhere"); a.add_argument("--limit", type=float, default=0)
    a.add_argument("--keepmasks", default=None)
    o = a.parse_args()
    bx = []
    for b in o.box:
        v = [float(x) for x in b.split(",")]; bx.append(tuple(v[:4]) + ((v[4], v[5]) if len(v) == 6 else (None, None)))
    s = time.time()
    clean_video(o.src, o.out, not o.notext, not o.nologo, bx, o.zone, limit_sec=o.limit, keep_masks=o.keepmasks)
    print(f"TOTAL {time.time() - s:.1f}s")
