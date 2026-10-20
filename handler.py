"""Runpod Serverless worker for MJ Upscaler: same pipeline as the Modal `process` function.
Network volume is mounted at /runpod-volume; /io and /models are symlinks into it.
Speed: all scratch work (frames, clean-up, SeedVR2 input/output) runs on the container's local disk; the input video and
models are copied to local disk once per worker; only finished chunks (seg*.mp4 / cleaned.mp4) are written to the volume.
The PC tells the worker which job folder to use ("work") and which chunks it already has ("skip"); with "no_concat" the
worker stops after the chunks and the PC joins them."""
import os, sys, json, time, hashlib, shutil, threading
sys.path.insert(0, "/opt/mju")
import runpod

VOL = "/runpod-volume"
LOCAL = os.environ.get("MJU_LOCAL", "/tmp/mju")
os.makedirs(os.path.join(LOCAL, "work"), exist_ok=True)
os.environ["MJU_WORK"] = os.path.join(LOCAL, "work")       # scratch on local disk (engine/cleaner read this at import/run)
_MODELS_READY = False

def T(msg, t0=[time.time()]):
    print(f"[{time.time() - t0[0]:7.1f}s] {msg}", flush=True)

def _prep():
    for d in ("io/in", "io/out", "io/work", "models/propainter_weights", "models/hf"):
        os.makedirs(os.path.join(VOL, d), exist_ok=True)

def _copy_tree(src, dst):
    n = 0
    for root, _, files in os.walk(src):
        for f in files:
            a = os.path.join(root, f); b = os.path.join(dst, os.path.relpath(a, src))
            if os.path.exists(b) and os.path.getsize(b) == os.path.getsize(a): continue
            os.makedirs(os.path.dirname(b), exist_ok=True); shutil.copyfile(a, b + ".part"); os.replace(b + ".part", b); n += os.path.getsize(a)
    return n

HF_URL = "https://huggingface.co/numz/SeedVR2_comfyUI/resolve/main/{}"
HF_REPO = {"seedvr2_ema_7b_fp8_e4m3fn_mixed_block35_fp16.safetensors": "AInVFX/SeedVR2_comfyUI"}
DIT_7B_FP8 = "seedvr2_ema_7b_fp8_e4m3fn_mixed_block35_fp16.safetensors"
_dl_lock = threading.Lock()

def _ensure_on_volume(fname):
    """Download a SeedVR2 weight straight from Hugging Face onto the network volume once (not via the PC)."""
    import requests
    dst = os.path.join(VOL, "models", "seedvr2", fname)
    with _dl_lock:
        if os.path.exists(dst) and os.path.getsize(dst) > 1 << 30: return dst
        t = time.time(); T(f"downloading {fname} from Hugging Face to the volume...")
        url = f"https://huggingface.co/{HF_REPO[fname]}/resolve/main/{fname}" if fname in HF_REPO else HF_URL.format(fname)
        with requests.get(url, stream=True, timeout=120, allow_redirects=True) as r:
            r.raise_for_status(); total = int(r.headers.get("Content-Length", 0)); got = 0
            with open(dst + ".part", "wb") as f:
                for b in r.iter_content(16 << 20): f.write(b); got += len(b)
        if total and got != total: raise RuntimeError(f"{fname} download incomplete ({got}/{total})")
        os.replace(dst + ".part", dst); T(f"{fname}: {got/2**30:.1f} GB in {time.time()-t:.0f}s")
    return dst

def _local_models(engine, dit):
    """Copy the SeedVR2 weights this job needs from the network volume to local disk once per worker."""
    dst = os.path.join(LOCAL, "models", "seedvr2"); os.makedirs(dst, exist_ok=True)
    t = time.time(); n = 0
    for f in os.listdir(os.path.join(VOL, "models", "seedvr2")):
        if f.endswith(".part") or (f.startswith("seedvr2_ema_") and f != dit): continue   # only the DiT in use + VAE/config
        a = os.path.join(VOL, "models", "seedvr2", f); b = os.path.join(dst, f)
        if os.path.isfile(a) and not (os.path.exists(b) and os.path.getsize(b) == os.path.getsize(a)):
            shutil.copyfile(a, b + ".part"); os.replace(b + ".part", b); n += os.path.getsize(a)
    if n: T(f"models on local disk ({n/2**30:.2f} GB copied in {time.time()-t:.0f}s)")
    engine.SEED_MODELS = dst

def _progress_path(name, settings):
    normalized = dict(settings)
    if normalized.get("mode") == "Fast": normalized["mode"] = "Max (natural)"
    tag = hashlib.sha256(json.dumps(normalized, sort_keys=True).encode()).hexdigest()[:12]
    return f"/io/out/{os.path.splitext(name)[0]}.{tag}.progress.json"

def handler(job):
    T0 = time.time(); _prep()
    jdir = os.path.join(LOCAL, "jobs", str(job.get("id", time.time())).replace("/", "_"))
    inp = job["input"]; name = inp["name"]; settings = dict(inp["settings"])
    import torch
    if not torch.cuda.is_available():
        msg = "No CUDA GPU visible in the Runpod worker (torch.cuda.is_available() is False); refusing to run on the CPU."
        print(msg, flush=True); return {"error": msg}
    vram = torch.cuda.get_device_properties(0).total_memory / 2**30
    gpu_name = torch.cuda.get_device_name(0)
    import engine
    engine.BIG_GPU = vram >= 40                  # 48 GB+: no offload + big batches
    engine.NO_OFFLOAD = vram >= 20               # 24/32 GB: no CPU offload (models fit), normal batch
    pro = str(settings.get("quality", "")).startswith("Pro")
    engine.CLOUD_SEED_RES = int(inp.get("seed_res") or 1080)
    if vram < 40 and engine.CLOUD_SEED_RES > 1080: engine.CLOUD_SEED_RES = 1080
    # --- speed v3: everything picked from the card's memory at runtime; job input can override for benchmarks ---
    big = vram >= 40
    if pro: engine.DIT_7B = engine.DIT_7B if big else DIT_7B_FP8        # 24-32 GB: 7B fp8 (~8.5 GB) fits, no block swapping
    engine.DIT_MODEL = engine.DIT_7B if pro else engine.DIT_3B
    engine.BLOCKS_TO_SWAP = int(inp.get("blocks_to_swap", 0))           # 0 everywhere now (H100/B200: fp16 fits; 4090: fp8 fits)
    engine.NO_OFFLOAD = engine.NO_OFFLOAD or engine.BLOCKS_TO_SWAP == 0
    engine.SEED_BATCH = int(inp.get("batch") or (45 if vram >= 120 else (33 if vram >= 70 else (21 if not pro else 13))))
    engine.TEMPORAL_OVERLAP = int(inp.get("overlap", 4 if not big else 3))
    engine.COLOR_CORRECTION = str(inp.get("color_correction", "lab"))   # "none" = skip LAB color transfer
    cc = torch.cuda.get_device_capability(0); attn = str(inp.get("attention", "auto"))
    if attn == "auto":
        try: from sageattention import sageattn_varlen; attn = "sageattn_2" if cc in ((8, 9), (9, 0)) else "sdpa"
        except Exception as ex:
            T(f"SageAttention unavailable; using SDPA: {ex}"); attn = "sdpa"
    engine.ATTENTION = attn
    T(f"GPU {gpu_name} {vram:.0f} GB, model {engine.DIT_MODEL}, SeedVR2 {engine.CLOUD_SEED_RES}p, batch {engine.SEED_BATCH}, "
      f"swap {engine.BLOCKS_TO_SWAP}, overlap {engine.TEMPORAL_OVERLAP}, attention {engine.ATTENTION}, color {engine.COLOR_CORRECTION}, torch {torch.__version__} cuda {torch.version.cuda}")
    try:
        import onnxruntime as ort; ort.preload_dlls(); prov = ort.get_available_providers(); T(f"ONNX providers: {prov}")
        if "CUDAExecutionProvider" not in prov: T("WARNING: ONNX face/text models will run on the CPU")
    except Exception as e: T(f"onnxruntime check failed: {e}")
    if settings.get("mode") == "Fast": settings["mode"] = "Max (natural)"
    if "face" in inp:
        if not isinstance(inp["face"], bool): return {"error": "face must be a JSON boolean"}
        settings["face_override"] = inp["face"]  # explicit override also applies to Heavy restore
    previews = []; pfile = _progress_path(name, settings)
    if inp.get("ptag"): pfile = pfile[:-len(".progress.json")] + f".{inp['ptag']}.progress.json"   # one file per parallel job
    runtime = dict(model=engine.DIT_MODEL, attention=engine.ATTENTION, batch=engine.SEED_BATCH,
                   blocks_to_swap=engine.BLOCKS_TO_SWAP, color_correction=engine.COLOR_CORRECTION,
                   overlap=engine.TEMPORAL_OVERLAP, seed_res=engine.CLOUD_SEED_RES)
    st = dict(state="running", progress=0, eta=-1, fps=0, phase="starting", gpu=gpu_name, runtime=runtime)
    lock = threading.Lock()
    def status(**kw):
        with lock:
            st.update(kw); tmp = pfile + ".tmp"
            with open(tmp, "w") as f: json.dump(dict(st, previews=previews, t=time.time(), elapsed=round(time.time() - T0)), f)
            os.replace(tmp, pfile)
    status()
    stop_hb = threading.Event()
    def heartbeat():
        while not stop_hb.wait(10):
            try: status()
            except Exception: pass
    threading.Thread(target=heartbeat, daemon=True).start()
    try:
        vsrc = f"/io/in/{name}"
        for _ in range(12):
            if os.path.exists(vsrc): break
            time.sleep(5)
        else:
            return {"error": f"Uploaded video not found on the Runpod volume: {name}. Run it again to re-upload."}
        status(phase="copying video + models to local disk")
        os.makedirs(os.path.join(LOCAL, "in"), exist_ok=True)
        src = os.path.join(LOCAL, "in", name)
        if not (os.path.exists(src) and os.path.getsize(src) == os.path.getsize(vsrc)):
            t = time.time(); shutil.copyfile(vsrc, src + ".part"); os.replace(src + ".part", src)
            T(f"video copied to local disk in {time.time()-t:.0f}s")
        if pro:
            status(phase="getting SeedVR2 7B (first time only)"); _ensure_on_volume(engine.DIT_7B) if pro else None
        _local_models(engine, engine.DIT_MODEL)
        remote = os.path.join(VOL, inp["work"]) if inp.get("work") else None; pre = set()
        def log(s):
            print(s, flush=True)
            if s.startswith("Chunk") or s.startswith("Clean-up") or "frames/s" in s: status(phase=s.strip()[:120])
        j = engine.Job(src, os.path.join(LOCAL, "out"), log=log, progress=None, **settings)
        j.work = jdir; os.makedirs(j.work, exist_ok=True)     # private scratch per job id (jobs never share or delete each other's files)
        if remote:
            os.makedirs(remote, exist_ok=True)
            for n in os.listdir(remote):     # finished chunks already on the volume (earlier Runpod run) + PC's cleaned.mp4
                if (n.startswith("seg") or n == "cleaned.mp4") and os.path.getsize(os.path.join(remote, n)) > 1000:
                    shutil.copyfile(os.path.join(remote, n), os.path.join(j.work, n)); pre.add(n)
        pass
        j.skip_chunks = set(int(x) for x in inp.get("skip", []))
        j.no_concat = bool(inp.get("no_concat"))
        tag = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:12]
        made = []; tchunk = [time.time()]
        def chunk_ready(path, start, length, total):
            if os.path.basename(path) in pre: return      # finished earlier: no copy, no preview
            if remote:
                stem = path[:-4]
                for pth in [p2 for p2 in (stem + "_head.mkv", stem + "_tail.mkv") if os.path.exists(p2)] + [path]:   # overlap pieces first, body last
                    b = os.path.basename(pth); dst = os.path.join(remote, b)
                    if not os.path.exists(dst):
                        shutil.copyfile(pth, dst + ".part"); os.replace(dst + ".part", dst); made.append(b)
                cl = os.path.join(j.work, "cleaned.mp4")
                if os.path.exists(cl) and not os.path.exists(os.path.join(remote, "cleaned.mp4")):
                    shutil.copyfile(cl, os.path.join(remote, "cleaned.mp4.part")); os.replace(os.path.join(remote, "cleaned.mp4.part"), os.path.join(remote, "cleaned.mp4"))
            T(f"chunk at {start:.0f}s ready ({time.time()-tchunk[0]:.0f}s since previous)"); tchunk[0] = time.time()
            try:
                from preview_media import make_proxy
                dest = f"/io/out/{os.path.splitext(name)[0]}.{tag}.preview/{start:012.4f}.mp4"
                make_proxy(path, dest, engine.FFMPEG)
                previews.append(dict(path=dest.replace("/io/", "", 1), start=start, length=length, total=total))
            except Exception as ex: T(f"preview failed: {ex}")
        def prog(p, r, s):
            print(f"PROGRESS {p:.3f} eta {r:.0f}s speed {s:.2f}fps", flush=True)
            status(progress=p, eta=r, fps=s)
            try: runpod.serverless.progress_update(job, {"progress": round(p, 4), "eta": r, "fps": s})
            except Exception: pass
        j.progress = prog; j.chunk_ready = chunk_ready
        try:
            out = j.run()
        except Exception as e:
            status(state="error", error=str(e)[:500]); return {"error": str(e)[:1000]}
        if j.no_concat:
            segs = sorted(n for n in os.listdir(remote) if n.startswith("seg")) if remote else []
            status(state="done", progress=1, eta=0, fps=0, phase="chunks done")
            T(f"job done in {time.time()-T0:.0f}s, {len(made)} new chunk(s)")
            return {"segs": segs, "new": made, "work": inp.get("work"), "gpu": gpu_name, "seconds": round(time.time() - T0), "runtime": runtime}
        vout = f"/io/out/{os.path.splitext(name)[0]}/" + os.path.basename(out)
        os.makedirs(os.path.dirname(vout), exist_ok=True); shutil.copyfile(out, vout + ".part"); os.replace(vout + ".part", vout)
        status(state="done", progress=1, eta=0, fps=0, out=vout.replace("/io/", "", 1))
        T(f"job done in {time.time()-T0:.0f}s")
        return {"out": vout.replace("/io/", "", 1), "gpu": gpu_name, "seconds": round(time.time() - T0), "runtime": runtime}
    finally:
        stop_hb.set()
        shutil.rmtree(jdir, ignore_errors=True)

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
