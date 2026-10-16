"""One local SeedVR2 process per video, with CPU model caches between chunks."""
import gc
import json
import os
import queue
import subprocess
import sys
import threading
import time

RESULT = '@@MJU_RESULT@@'


class SeedWorker:
    def __init__(self, python, root):
        self.process = subprocess.Popen([python, '-u', os.path.abspath(__file__)],
            cwd=os.path.join(root, 'seedvr2'), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace', bufsize=1,
            env=dict(os.environ, PYTHONUTF8='1', PYTHONIOENCODING='utf-8'),
            creationflags=0x08000000 if os.name == 'nt' else 0)
        self.lines = queue.Queue()
        def read():
            try:
                for line in self.process.stdout: self.lines.put(line)
            finally: self.lines.put(None)
        threading.Thread(target=read, daemon=True).start()

    def run(self, arguments, log=None, stop=None):
        if self.process.poll() is not None: raise RuntimeError('SeedVR2 worker exited')
        self.process.stdin.write(json.dumps({'args': arguments}) + '\n'); self.process.stdin.flush()
        tail = []
        while True:
            if stop is not None and stop.is_set():
                self.close(force=True); raise RuntimeError('Stopped')
            try: line = self.lines.get(timeout=.25)
            except queue.Empty: continue
            if line is None: raise RuntimeError('SeedVR2 worker exited:\n' + ''.join(tail[-12:]))
            if line.startswith(RESULT):
                result = json.loads(line[len(RESULT):])
                if not result['ok']: raise RuntimeError(result['error'])
                return result
            tail.append(line)
            if len(tail) > 30: tail.pop(0)
            if log and line.strip(): log(line.rstrip()[-200:])

    def close(self, force=False):
        process = self.process
        if process.poll() is None and not force:
            try:
                process.stdin.write(json.dumps({'close': True}) + '\n'); process.stdin.flush()
                process.wait(timeout=8)
            except (OSError, subprocess.TimeoutExpired): force = True
        if force and process.poll() is None:
            # Windows venv launchers can own another Python process. Stop only
            # descendants of our own worker, never other Python applications.
            import psutil
            try: children = psutil.Process(process.pid).children(recursive=True)
            except psutil.NoSuchProcess: children = []
            for child in reversed(children):
                try: child.kill()
                except psutil.NoSuchProcess: pass
            process.kill(); process.wait(timeout=5)
        if process.stdin: process.stdin.close()


def serve():
    root = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(root, 'seedvr2'))
    import inference_cli as cli
    import torch
    cache = {}
    validated = False
    print('SeedVR2 worker ready; models will stay cached in CPU RAM between chunks.', flush=True)
    for line in sys.stdin:
        request = json.loads(line)
        if request.get('close'): break
        start = time.perf_counter()
        try:
            sys.argv = [cli.__file__, *request['args'], '--cache_dit', '--cache_vae']
            args = cli.parse_arguments()
            cli.debug.enabled = args.debug
            if args.dit_offload_device != 'cpu' or args.vae_offload_device != 'cpu':
                raise ValueError('Persistent local worker requires CPU offload for both models')
            if not validated:
                if not cli.download_weight(dit_model=args.dit_model, vae_model=cli.DEFAULT_VAE,
                                           model_dir=args.model_dir, debug=cli.debug):
                    raise RuntimeError('Model validation failed')
                validated = True
            count = cli.process_single_file(args.input, args, ['0'], args.output, runner_cache=cache)
            # Drop per-clip tensors while preserving upstream CPU model caches.
            context = cache.get('ctx', {})
            keep = {'dit_device', 'vae_device', 'dit_offload_device', 'vae_offload_device',
                    'tensor_offload_device', 'compute_dtype'}
            for key in list(context):
                if key not in keep: del context[key]
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.synchronize(); torch.cuda.empty_cache()
            response = dict(ok=True, frames=count, seconds=time.perf_counter()-start,
                            gpu_allocated_mib=torch.cuda.memory_allocated()/2**20,
                            gpu_reserved_mib=torch.cuda.memory_reserved()/2**20)
        except Exception as ex:
            import traceback
            traceback.print_exc()
            response = dict(ok=False, error=str(ex))
        print(RESULT + json.dumps(response), flush=True)
        if not response['ok']: break


if __name__ == '__main__': serve()
