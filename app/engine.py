"""MJ Upscaler engine: chunked, resumable video upscaling for old footage."""
import os, sys, json, glob, math, shutil, subprocess, time, hashlib
import numpy as np, cv2
from color_tools import normalize as normalize_color, color_filter, color_suffix

ROOT = os.path.dirname(os.path.abspath(__file__))
FFMPEG = os.environ.get("MJU_FFMPEG", r"C:\ytdlp\ffmpeg.exe")
ESR = os.path.join(ROOT, "bin", "esr", "realesrgan-ncnn-vulkan.exe")
RIFE = os.path.join(ROOT, "bin", "rife", "rife-ncnn-vulkan.exe")
PY = os.environ.get("MJU_PY", os.path.join(ROOT, "venv", "Scripts", "python.exe"))
SEED_CLI = os.path.join(ROOT, "seedvr2", "inference_cli.py")
SEED_MODELS = os.path.join(ROOT, "models", "seedvr2")
os.environ["PYTHONUTF8"] = "1"; os.environ["PYTHONIOENCODING"] = "utf-8"
BIG_GPU = os.environ.get("MJU_BIG_GPU") == "1"  # cloud A100/H100: no offload, bigger batches
NOWIN = 0x08000000 if os.name == "nt" else 0

HEAVY = "Heavy restore (very bad footage)"
SEED_RES = {"Max (natural)": 720, "Max (slowest, 1080p)": 1080, HEAVY: 1080}  # SeedVR2 internal short side; final lanczos to target

def run(cmd, log=None, **kw):
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                         errors="replace", creationflags=NOWIN, **kw)
    out = []
    for line in p.stdout:
        out.append(line)
        if log and line.strip(): log(line.rstrip()[-200:])
    p.wait()
    if p.returncode != 0:
        raise RuntimeError(f"{os.path.basename(cmd[0])} failed:\n" + "".join(out[-15:]))
    return "".join(out)

def probe(path):
    cap = cv2.VideoCapture(path)
    w, h = int(cap.get(3)), int(cap.get(4)); fps = cap.get(5) or 25.0
    n = int(cap.get(7)); cap.release()
    if w == 0: raise RuntimeError("Can't read video: " + path)
    return dict(w=w, h=h, fps=fps, frames=n, dur=n / fps)

def crop_detect(src, dur):
    """Find black bars (most common cropdetect result over the clip). Returns (w, h, x, y) or None."""
    from collections import Counter
    import re
    try:
        out = subprocess.run([FFMPEG, "-hide_banner", "-t", f"{min(dur, 60):.1f}", "-i", src, "-map", "0:V:0",
                              "-vf", "cropdetect=24:2:0", "-f", "null", "-"], capture_output=True, text=True,
                             errors="replace", creationflags=NOWIN).stderr
        c = Counter(re.findall(r"crop=(\d+):(\d+):(\d+):(\d+)", out)).most_common(1)
        return tuple(int(v) for v in c[0][0]) if c else None
    except Exception:
        return None

def target_size(w, h, choice):
    th = 2160 if choice.startswith("4K") else 1080
    tw = int(round(th * w / h / 2) * 2)
    return tw, th

# ---------------- face restore (CodeFormer, high fidelity, low blend) ----------------
FFHQ = np.array([[192.98138, 239.94708], [318.90277, 240.1936], [256.63416, 314.01935],
                 [201.26117, 371.41043], [313.08905, 371.15118]], np.float32)
class FaceRestorer:
    def __init__(self, fidelity=0.8, strength=0.35):
        import onnxruntime as ort
        try: ort.preload_dlls()  # use the CUDA/cuDNN libraries that ship with PyTorch
        except Exception: pass
        providers = ['CPUExecutionProvider'] if os.environ.get('MJU_FACE_CPU') == '1' else [
            ('CUDAExecutionProvider', {'use_tf32': 1 if os.environ.get('MJU_FACE_TF32') == '1' else 0}), 'CPUExecutionProvider']
        self.sess = ort.InferenceSession(os.path.join(ROOT, "models", "codeformer.onnx"), providers=providers)
        self.inputs = {i.name: i for i in self.sess.get_inputs()}
        self.fid, self.strength = fidelity, strength
        self.det = None; self.mask = None
        self.processed_faces = 0
    def close(self):
        # Release ORT's CUDA allocator before the next SeedVR2 chunk.
        self.sess = None; self.det = None
    def __call__(self, img):
        h, w = img.shape[:2]
        if self.det is None or self.det_size != (w, h):
            self.det = cv2.FaceDetectorYN.create(os.path.join(ROOT, "models", "yunet.onnx"), "", (w, h), 0.8)
            self.det_size = (w, h)
        _, faces = self.det.detect(img)
        if faces is None: return img
        out = img.astype(np.float32)
        for f in faces:
            if f[2] < 48: continue  # too small to help
            lm = f[4:14].reshape(5, 2).astype(np.float32)
            M, _ = cv2.estimateAffinePartial2D(lm, FFHQ, method=cv2.LMEDS)
            crop = cv2.warpAffine(img, M, (512, 512), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            x = crop[:, :, ::-1].astype(np.float32) / 255.0
            x = ((x - 0.5) / 0.5).transpose(2, 0, 1)[None]
            feed = {}
            for name, inp in self.inputs.items():
                if len(inp.shape or []) == 4: feed[name] = x
                else: feed[name] = np.array([self.fid], dtype=np.float64 if "double" in inp.type else np.float32)
            y = self.sess.run(None, feed)[0][0].transpose(1, 2, 0)
            self.processed_faces += 1
            y = ((np.clip(y, -1, 1) + 1) / 2 * 255.0)[:, :, ::-1]
            if self.mask is None:
                mask = np.zeros((512, 512), np.float32)
                cv2.ellipse(mask, (256, 290), (170, 215), 0, 0, 360, 1, -1)
                self.mask = cv2.GaussianBlur(mask, (0, 0), 25) * self.strength
            IM = cv2.invertAffineTransform(M)
            # Paste back only inside the face's bounding box (the mask is 0 elsewhere): same pixels, much less work.
            pts = cv2.transform(np.array([[[0, 0], [511, 0], [0, 511], [511, 511]]], np.float32), IM)[0]
            x0, y0 = max(0, int(np.floor(pts[:, 0].min())) - 2), max(0, int(np.floor(pts[:, 1].min())) - 2)
            x1, y1 = min(w, int(np.ceil(pts[:, 0].max())) + 3), min(h, int(np.ceil(pts[:, 1].max())) + 3)
            if x1 <= x0 or y1 <= y0: continue
            T = IM.copy(); T[0, 2] -= x0; T[1, 2] -= y0
            yb = cv2.warpAffine(y, T, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR)
            mb = cv2.warpAffine(self.mask, T, (x1 - x0, y1 - y0))[:, :, None]
            out[y0:y1, x0:x1] = out[y0:y1, x0:x1] * (1 - mb) + yb * mb
        return np.clip(out, 0, 255).astype(np.uint8)


def post_frames(d_in, d_out, fast=False, face_restorer=None):
    """Shared postprocessing for the normal and isolated face paths."""
    for name in sorted(os.listdir(d_out)):
        p = os.path.join(d_out, name); up = cv2.imread(p)
        if fast:
            orig = cv2.imread(os.path.join(d_in, name))
            lz = cv2.resize(orig, (up.shape[1], up.shape[0]), interpolation=cv2.INTER_LANCZOS4)
            up = cv2.addWeighted(up, 0.5, lz, 0.5, 0)
        if face_restorer is not None: up = face_restorer(up)
        cv2.imwrite(p, up, [cv2.IMWRITE_PNG_COMPRESSION, 1])

# ---------------- main job ----------------
class Job:
    def __init__(self, src, out_dir, mode="Max (natural)", res="4K", codec="H.264",
                 deinterlace=False, denoise=True, face=False, grain=True, interp=False,
                 chunk_sec=8, limit_sec=0, log=print, progress=None, preview=None, stop=None, clean=None, chunk_ready=None, color=None):
        if mode == HEAVY: denoise, face, grain = False, True, True  # preset: own deblock/denoise pre-pass
        self.__dict__.update(locals()); del self.__dict__["self"]
        self.color = normalize_color(color)
        from cleanup_precision import cleanup_suffix
        self.cleanup_tag = cleanup_suffix(clean)
        self.info = probe(src)
        self.crop = None
        if mode == HEAVY:
            c = crop_detect(src, self.info["dur"])
            if c and c[0] >= 32 and c[1] >= 32 and (c[0] < self.info["w"] or c[1] < self.info["h"]):
                self.crop = c
            cw, ch = (self.crop[0], self.crop[1]) if self.crop else (self.info["w"], self.info["h"])
            self.info = dict(self.info, w=cw * 2, h=ch * 2)
        base = os.path.splitext(os.path.basename(src))[0]
        tag = {"Fast": "fast", "Max (natural)": "max", "Max (slowest, 1080p)": "max1080", HEAVY: "heavy"}[mode]
        ext = ".mov" if codec.startswith("ProRes") else ".mp4"
        os.makedirs(out_dir, exist_ok=True)
        self.out = os.path.join(out_dir, f"{base}_{tag}_{res.split()[0]}{'_60' if interp else ''}{f'_first{limit_sec:g}s' if limit_sec else ''}{color_suffix(self.color)}{self.cleanup_tag}{ext}")
        key_parts = [src, os.path.getsize(src), mode, res, codec, deinterlace, denoise, face, grain, interp, chunk_sec, limit_sec]
        if self.color: key_parts.append({'color': self.color})
        if clean and (clean.get('text') or clean.get('logo') or clean.get('boxes')):
            from cleanup_precision import VERSION
            key_parts.append({'cleanup_version': VERSION, 'clean': clean})
        key = json.dumps(key_parts)
        self.work = os.path.join(os.environ.get("MJU_WORK", os.path.join(ROOT, "work")), "jobs", hashlib.md5(key.encode()).hexdigest()[:10])
        os.makedirs(self.work, exist_ok=True)
        self.face_r = None
        self.seed_worker = None
        self.worker_disabled = False

    def _run_seed(self, command):
        if BIG_GPU or self.worker_disabled or os.environ.get('MJU_PERSISTENT_SEED', '1') == '0':
            return run(command, log=self.log, cwd=os.path.join(ROOT, 'seedvr2'))
        from seed_worker import SeedWorker
        try:
            if self.seed_worker is None:
                self.log('Starting reusable SeedVR2 worker (models cached in CPU RAM)')
                self.seed_worker = SeedWorker(PY, ROOT)
            return self.seed_worker.run(command[2:], log=self.log, stop=self.stop)
        except Exception as ex:
            if self.seed_worker is not None:
                self.seed_worker.close(force=True); self.seed_worker = None
            if self.stop is not None and self.stop.is_set(): raise RuntimeError('Stopped')
            # Preserve a reliable path if this upstream version cannot reuse a model.
            self.worker_disabled = True
            self.log(f'Reusable worker failed ({ex}); retrying this chunk with the original SeedVR2 CLI')
            output = command[command.index('--output') + 1]
            shutil.rmtree(output, ignore_errors=True)
            return run(command, log=self.log, cwd=os.path.join(ROOT, 'seedvr2'))

    def _pre_filter(self):
        f = []
        if self.crop: f.append("crop=%d:%d:%d:%d" % self.crop)
        if self.deinterlace: f.append("bwdif=mode=send_frame:deint=interlaced")
        if self.denoise: f.append("hqdn3d=1.2:1.2:2.5:2.5")  # gentle; keeps grain structure
        if self.mode == HEAVY:
            f += ["scale=iw*2:ih*2:flags=lanczos", "deblock=filter=strong:block=8", "hqdn3d=3:2:5:4"]
        return ",".join(f)

    def _upscale(self, d_in, d_out, n_frames):
        i = self.info
        if self.mode in SEED_RES:
            short = min(i["w"], i["h"])
            res = max(SEED_RES[self.mode], short)
            tile = "512" if res > 900 else "1024"
            vid = os.path.join(self.work, "seed_in.mp4")
            run([FFMPEG, "-v", "error", "-y", "-framerate", str(i["fps"]), "-i", os.path.join(d_in, "%06d.png"),
                 "-c:v", "libx264", "-preset", "ultrafast", "-qp", "0", "-pix_fmt", "yuv444p", vid])
            sd = os.path.join(self.work, "seed_out"); shutil.rmtree(sd, ignore_errors=True)
            self._run_seed([PY, SEED_CLI, vid, "--output", sd, "--output_format", "png", "--model_dir", SEED_MODELS,
                 "--dit_model", "seedvr2_ema_3b_fp8_e4m3fn.safetensors", "--resolution", str(res),
                 "--batch_size", "21", "--temporal_overlap", "4", "--color_correction", "lab",
                 *(["--batch_size", "45"] if BIG_GPU else ["--dit_offload_device", "cpu", "--vae_offload_device", "cpu"]),
                 "--vae_encode_tiled", "--vae_decode_tiled", "--vae_encode_tile_size", tile, "--vae_decode_tile_size", tile,
                 "--vae_encode_tile_overlap", "64", "--vae_decode_tile_overlap", "64"])
            pngs = sorted(glob.glob(os.path.join(sd, "**", "*.png"), recursive=True))
            if len(pngs) < n_frames: raise RuntimeError(f"SeedVR2 produced {len(pngs)}/{n_frames} frames")
            os.makedirs(d_out, exist_ok=True)
            for k, p in enumerate(pngs[:n_frames]): shutil.move(p, os.path.join(d_out, f"{k+1:06d}.png"))
            shutil.rmtree(sd, ignore_errors=True); os.remove(vid)
        else:
            os.makedirs(d_out, exist_ok=True)
            run([ESR, "-i", d_in, "-o", d_out, "-n", "realesr-animevideov3", "-s", "4", "-f", "png"])
        # post: natural blend / face
        if self.mode == "Fast" or self.face:
            if self.face and not BIG_GPU:
                # A separate process also releases CUDA's context/driver overhead;
                # closing an ORT session alone can leave that memory resident.
                run([PY, os.path.join(ROOT, 'face_worker.py'), d_in, d_out,
                     *(['--fast'] if self.mode == 'Fast' else [])], log=self.log)
                return
            if self.face and self.face_r is None:
                self.face_r = FaceRestorer()
                provider = self.face_r.sess.get_providers()[0]
                self.log(f'Face touch-up: {provider}')
            post_frames(d_in, d_out, self.mode == 'Fast', self.face_r)
            if self.face_r is not None:
                self.face_r.close(); self.face_r = None

    def _encode(self, d, fps, seg):
        tw, th = target_size(self.info["w"], self.info["h"], self.res)
        vf = f"scale={tw}:{th}:flags=lanczos"
        grade = color_filter(self.color)
        if grade: vf += ',' + grade
        if self.grain: vf += ",noise=c0s=5:c0f=t+u"
        if self.codec.startswith("ProRes"):
            enc = ["-vf", vf + ",format=yuv422p10le", "-c:v", "prores_ks", "-profile:v", "3"]
        else:
            enc = ["-vf", vf + ",format=yuv420p", "-c:v", "libx264", "-crf", "14", "-preset", "slow", "-profile:v", "high"]
        run([FFMPEG, "-v", "error", "-y", "-framerate", str(fps), "-i", os.path.join(d, "%06d.png")] + enc + [seg])

    def _publish_chunk(self, path, start, length, total):
        if self.chunk_ready:
            try: self.chunk_ready(path, start, length, total)
            except Exception as ex: self.log(f"Video preview unavailable for this chunk: {ex}")

    def run(self):
        try:
            return self._run()
        finally:
            if self.face_r is not None:
                self.face_r.close(); self.face_r = None
            if self.seed_worker is not None:
                self.seed_worker.close(); self.seed_worker = None

    def _run(self):
        c = self.clean or {}
        if c.get("text") or c.get("logo") or c.get("boxes"):
            import cleaner
            base = os.path.splitext(os.path.basename(self.src))[0]
            only = c.get("only")
            dst = os.path.join(self.out_dir, base + (f"_clean_first{self.limit_sec:g}s" if self.limit_sec else "_clean") + color_suffix(self.color) + self.cleanup_tag + '.mp4') if only else os.path.join(self.work, "cleaned.mp4")
            if not (os.path.exists(dst) and not only):
                self.log("Clean-up: removing text / logos / drawn areas")
                prog = (lambda p, r: self.progress(p * (1 if only else 0.15), r, 0)) if self.progress else None
                cleaner.clean_video(self.src, dst, c.get("text", False), c.get("logo", False), c.get("boxes"),
                                    c.get("zone", "anywhere"), final=bool(only), log=self.log, progress=prog,
                                    stop=self.stop, limit_sec=self.limit_sec,
                                    output_filter=color_filter(self.color) if only else '')
            if only:
                self.out = dst; self.log("Done -> " + dst); return dst
            self.src = dst
        i = self.info; fps = i["fps"]
        dur = min(i["dur"], self.limit_sec) if self.limit_sec else i["dur"]
        nchunks = max(1, math.ceil(dur / self.chunk_sec))
        segs, t0, done_frames = [], time.time(), 0
        ext = ".mov" if self.codec.startswith("ProRes") else ".mp4"
        for c in range(nchunks):
            if self.stop and self.stop.is_set(): raise RuntimeError("Stopped")
            seg = os.path.join(self.work, f"seg{c:04d}{ext}"); segs.append(seg)
            start = c * self.chunk_sec; length = min(self.chunk_sec, dur - start)
            if os.path.exists(seg) and os.path.getsize(seg) > 1000:
                self.log(f"Chunk {c+1}/{nchunks}: already done (resume)")
                self._publish_chunk(seg, start, length, dur)
                if self.progress: self.progress((c + 1) / nchunks, 0, 0)
                continue
            self.log(f"Chunk {c+1}/{nchunks}: {start:.0f}s - {start+length:.0f}s")
            d_in, d_up = os.path.join(self.work, "in"), os.path.join(self.work, "up")
            for d in (d_in, d_up): shutil.rmtree(d, ignore_errors=True)
            os.makedirs(d_in)
            nf = int(round(length * fps))
            cmd = [FFMPEG, "-v", "error", "-y", "-ss", f"{start:.4f}", "-i", self.src, "-map", "0:V:0", "-frames:v", str(nf)]
            pf = self._pre_filter()
            if pf: cmd += ["-vf", pf]
            run(cmd + ["-compression_level", "1", os.path.join(d_in, "%06d.png")])
            nf = len(os.listdir(d_in))
            cs = time.time()
            self._upscale(d_in, d_up, nf)
            d_fin, ofps = d_up, fps
            if self.interp:
                d_fin = os.path.join(self.work, "rife"); shutil.rmtree(d_fin, ignore_errors=True); os.makedirs(d_fin)
                run([RIFE, "-i", d_up, "-o", d_fin, "-m", os.path.join(os.path.dirname(RIFE), "rife-v4.6"),
                     "-n", str(nf * 2), "-f", "%06d.png"])
                ofps = fps * 2
            if self.preview:
                mid = f"{nf//2+1:06d}.png"
                self.preview(os.path.join(d_in, mid), os.path.join(d_up, mid))
            self._encode(d_fin, ofps, seg)
            self._publish_chunk(seg, start, length, dur)
            done_frames += nf
            spd = nf / (time.time() - cs)
            remain = (dur - (start + length)) * fps / max(spd, 1e-6)
            self.log(f"   {spd:.2f} frames/s")
            if self.progress: self.progress((c + 1) / nchunks, remain, spd)
            for d in (d_in, d_up, os.path.join(self.work, "rife")): shutil.rmtree(d, ignore_errors=True)
        lst = os.path.join(self.work, "list.txt")
        with open(lst, "w") as f:
            for s in segs: f.write(f"file '{s}'\n")
        acodec = ["-c:a", "pcm_s16le"] if ext == ".mov" else ["-c:a", "aac", "-b:a", "320k"]
        run([FFMPEG, "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-t", f"{dur:.3f}", "-i", self.src,
             "-map", "0:v", "-map", "1:a:0?", "-c:v", "copy"] + acodec + ["-shortest", self.out])
        shutil.rmtree(self.work, ignore_errors=True)
        self.log(f"Done in {(time.time()-t0)/60:.1f} min -> {self.out}")
        return self.out

def run_queue(files, settings, out_dir=None, log=print, progress=None, preview=None, stop=None, on_item=None, chunk_ready=None):
    """Process videos one after another; an error on one file is logged and the queue continues.
    Finished chunks are kept per job, so re-running the same queue resumes where it stopped."""
    results = []
    for k, src in enumerate(files):
        if stop and stop.is_set(): results.append((src, "skipped", "stopped")); continue
        if on_item: on_item(k, "running", "")
        try:
            od = out_dir or os.path.join(os.path.dirname(os.path.abspath(src)), "MJ Upscaled")
            j = Job(src, od, log=log, progress=(lambda p, r, s, k=k: (progress and progress(p, r, s), on_item and on_item(k, "progress", p))),
                    preview=preview, stop=stop, chunk_ready=chunk_ready, **settings)
            out = j.run(); results.append((src, "done", out))
            if on_item: on_item(k, "done", out)
        except Exception as ex:
            log(f"[{os.path.basename(src)}] failed: {ex}"); results.append((src, "error", str(ex).splitlines()[0][:200]))
            if on_item: on_item(k, "error", str(ex).splitlines()[0][:200])
    return results

if __name__ == "__main__":
    import argparse
    a = argparse.ArgumentParser()
    a.add_argument("src", nargs="+"); a.add_argument("--out", default=None)
    a.add_argument("--mode", default="Max (natural)"); a.add_argument("--res", default="4K")
    a.add_argument("--codec", default="H.264"); a.add_argument("--limit", type=float, default=0)
    a.add_argument("--face", action="store_true"); a.add_argument("--interp", action="store_true")
    a.add_argument("--deinterlace", action="store_true"); a.add_argument("--nodenoise", action="store_true")
    a.add_argument("--nograin", action="store_true"); a.add_argument("--chunk", type=float, default=8)
    a.add_argument("--cleantext", action="store_true"); a.add_argument("--cleanlogo", action="store_true")
    o = a.parse_args()
    clean = dict(text=o.cleantext, logo=o.cleanlogo)
    results = run_queue(o.src, dict(mode=o.mode, res=o.res, codec=o.codec, deinterlace=o.deinterlace, denoise=not o.nodenoise,
                        face=o.face, grain=not o.nograin, interp=o.interp, chunk_sec=o.chunk, limit_sec=o.limit, clean=clean),
                        out_dir=o.out, progress=lambda p, r, s: print(f"PROGRESS {p:.3f} eta {r:.0f}s speed {s:.2f}fps", flush=True))
    for src, st, info in results: print(f"QUEUE {st}: {os.path.basename(src)} -> {info}")
