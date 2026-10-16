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

def _local_models(engine):
    """Copy SeedVR2 weights from the network volume to local disk once per worker (loads much faster every chunk)."""
    global _MODELS_READY
    dst = os.path.join(LOCAL, "models", "seedvr2")
    if not _MODELS_READY:
        t = time.time(); n = _copy_tree(os.path.join(VOL, "models", "seedvr2"), dst)
        T(f"models on local disk ({n/2**30:.2f} GB copied in {time.time()-t:.0f}s)"); _MODELS_READY = True
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
    T(f"GPU {gpu_name} {vram:.0f} GB, big_gpu={engine.BIG_GPU}, no_offload={engine.NO_OFFLOAD}, torch {torch.__version__} cuda {torch.version.cuda}")
    try:
        import onnxruntime as ort; ort.preload_dlls(); prov = ort.get_available_providers(); T(f"ONNX providers: {prov}")
        if "CUDAExecutionProvider" not in prov: T("WARNING: ONNX face/text models will run on the CPU")
    except Exception as e: T(f"onnxruntime check failed: {e}")
    if settings.get("mode") == "Fast": settings["mode"] = "Max (natural)"
    previews = []; pfile = _progress_path(name, settings)
    if inp.get("ptag"): pfile = pfile[:-len(".progress.json")] + f".{inp['ptag']}.progress.json"   # one file per parallel job
    st = dict(state="running", progress=0, eta=-1, fps=0, phase="starting", gpu=gpu_name)
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
        _local_models(engine)
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
                b = os.path.basename(path); dst = os.path.join(remote, b)
                if not os.path.exists(dst):
                    shutil.copyfile(path, dst + ".part"); os.replace(dst + ".part", dst); made.append(b)
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
            return {"segs": segs, "new": made, "work": inp.get("work"), "gpu": gpu_name, "seconds": round(time.time() - T0)}
        vout = f"/io/out/{os.path.splitext(name)[0]}/" + os.path.basename(out)
        os.makedirs(os.path.dirname(vout), exist_ok=True); shutil.copyfile(out, vout + ".part"); os.replace(vout + ".part", vout)
        status(state="done", progress=1, eta=0, fps=0, out=vout.replace("/io/", "", 1))
        T(f"job done in {time.time()-T0:.0f}s")
        return {"out": vout.replace("/io/", "", 1), "gpu": gpu_name, "seconds": round(time.time() - T0)}
    finally:
        stop_hb.set()
        shutil.rmtree(jdir, ignore_errors=True)

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
