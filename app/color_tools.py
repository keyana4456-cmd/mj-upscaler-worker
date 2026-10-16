"""Optional, fixed color grading shared by previews and exports."""
import hashlib
import io
import json
import math
import os
import subprocess

PRESETS = {
    'Neutral / reset': dict(warmth=0, tint=0, saturation=1),
    'Reduce pink / magenta cast': dict(warmth=0, tint=-30, saturation=1),
    'Reduce green cast': dict(warmth=0, tint=25, saturation=1),
    'Warm up blue cast': dict(warmth=25, tint=0, saturation=1),
    'Cool down yellow cast': dict(warmth=-25, tint=0, saturation=1),
    'Revive faded colors': dict(warmth=0, tint=0, saturation=1.15),
}


def normalize(options):
    if not options or not options.get('enabled', False): return {}
    result = {}
    for key, default, low, high in [('warmth', 0, -100, 100), ('tint', 0, -100, 100), ('saturation', 1, .5, 1.5)]:
        value = float(options.get(key, default))
        if not math.isfinite(value): raise ValueError(f'Invalid color setting: {key}')
        result[key] = round(max(low, min(high, value)), 3)
    if result == PRESETS['Neutral / reset']: return {}
    return dict(enabled=True, **result)


def color_filter(options):
    options = normalize(options)
    if not options: return ''
    filters = []
    warmth, tint = options['warmth'] * .0012, options['tint'] * .0012
    if warmth or tint:
        red, green, blue = warmth + tint/2, -tint, -warmth + tint/2
        filters.append(f'colorbalance=rm={red:.6f}:gm={green:.6f}:bm={blue:.6f}'
                       f':rh={red/2:.6f}:gh={green/2:.6f}:bh={blue/2:.6f}:pl=1')
    if options['saturation'] != 1:
        filters.append(f"eq=saturation={options['saturation']:.3f}")
    return ','.join(filters)


def color_suffix(options):
    options = normalize(options)
    return '_color_' + hashlib.sha256(json.dumps(options, sort_keys=True).encode()).hexdigest()[:8] if options else ''


def preview_image(source, second, options, ffmpeg):
    """Read one source frame and return a before/after image without saving footage."""
    from PIL import Image
    grade = color_filter(options) or 'null'
    scale = "scale=w='min(480,iw)':h='min(360,ih)':force_original_aspect_ratio=decrease:force_divisible_by=2"
    graph = f'[0:V:0]{scale},format=gbrp,split=2[a][b];[b]{grade},format=gbrp[c];[a][c]hstack=inputs=2,format=rgb24[out]'
    command = [ffmpeg, '-v', 'error', '-nostdin', '-ss', f'{max(0,second):.4f}', '-i', source,
               '-filter_complex', graph, '-map', '[out]', '-frames:v', '1', '-an',
               '-c:v', 'png', '-f', 'image2pipe', '-']
    result = subprocess.run(command, capture_output=True, timeout=30,
                            creationflags=0x08000000 if os.name == 'nt' else 0)
    if result.returncode or not result.stdout:
        raise RuntimeError(result.stderr.decode(errors='replace')[-500:] or 'Cannot read this frame')
    with Image.open(io.BytesIO(result.stdout)) as image: return image.copy()
