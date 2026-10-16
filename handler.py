"""Runpod Serverless worker for MJ Upscaler: same pipeline as the Modal `process` function.
Network volume is mounted at /runpod-volume; /io and /models are symlinks into it, so paths and
job keys match the Modal worker exactly (inputs /io/in, outputs /io/out, chunks /io/work/jobs/<key>)."""
import os, sys, json, time, hashlib
sys.path.insert(0, "/opt/mju")
import runpod

VOL = "/runpod-volume"

def _prep():
    for d in ("io/in", "io/out", "io/work", "models/propainter_weights", "models/hf"):
        os.makedirs(os.path.join(VOL, d), exist_ok=True)

def _progress_path(name, settings):
    normalized = dict(settings)
    if normalized.get("mode") == "Fast": normalized["mode"] = "Max (natural)"
    tag = hashlib.sha256(json.dumps(normalized, sort_keys=True).encode()).hexdigest()[:12]
    return f"/io/out/{os.path.splitext(name)[0]}.{tag}.progress.json"

def handler(job):
    _prep()
    inp = job["input"]; name = inp["name"]; settings = dict(inp["settings"])
    import engine
    try:
        import torch
        vram = torch.cuda.get_device_properties(0).total_memory / 2**30
        engine.BIG_GPU = vram >= 40          # L40S/A100: no offload + big batches; 24 GB cards use the home-PC path
        print(f"GPU {torch.cuda.get_device_name(0)} {vram:.0f} GB, big_gpu={engine.BIG_GPU}", flush=True)
    except Exception as e: print("GPU check failed:", e, flush=True)
    try:
        import onnxruntime as ort; ort.preload_dlls(); print("ONNX providers:", ort.get_available_providers(), flush=True)
    except Exception as e: print("onnxruntime check failed:", e, flush=True)
    src = f"/io/in/{name}"
    for _ in range(12):
        if os.path.exists(src): break
        time.sleep(5)
    else:
        return {"error": f"Uploaded video not found on the Runpod volume: {name}. Run it again to re-upload."}
    if settings.get("mode") == "Fast": settings["mode"] = "Max (natural)"
    previews = []; pfile = _progress_path(name, settings)
    def status(**kw):
        tmp = pfile + ".tmp"
        with open(tmp, "w") as f: json.dump(dict(kw, previews=previews, t=time.time()), f)
        os.replace(tmp, pfile)
    def chunk_ready(path, start, length, total):
        from preview_media import make_proxy
        tag = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:12]
        dest = f"/io/out/{os.path.splitext(name)[0]}.{tag}.preview/{start:012.4f}.mp4"
        make_proxy(path, dest, engine.FFMPEG)
        previews.append(dict(path=dest.replace("/io/", "", 1), start=start, length=length, total=total))
    def prog(p, r, s):
        print(f"PROGRESS {p:.3f} eta {r:.0f}s speed {s:.2f}fps", flush=True)
        status(state="running", progress=p, eta=r, fps=s)
        try: runpod.serverless.progress_update(job, {"progress": round(p, 4), "eta": r, "fps": s})
        except Exception: pass
    status(state="running", progress=0, eta=-1, fps=0)
    j = engine.Job(src, f"/io/out/{os.path.splitext(name)[0]}", log=lambda s: print(s, flush=True),
                   progress=prog, chunk_ready=chunk_ready, **settings)
    try:
        out = j.run()
    except Exception as e:
        status(state="error", error=str(e)[:500]); return {"error": str(e)[:1000]}
    status(state="done", progress=1, eta=0, fps=0, out=out.replace("/io/", "", 1))
    return {"out": out.replace("/io/", "", 1)}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
