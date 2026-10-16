"""Small, silent review copies of completed exports; originals are never changed."""
import os
import subprocess


def make_proxy(source, destination, ffmpeg):
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    if os.path.isfile(destination) and os.path.getsize(destination) > 1000:
        return destination
    temporary = destination + '.part.mp4'
    command = [ffmpeg, '-v', 'error', '-nostdin', '-y', '-i', source,
               '-an', '-vf', "scale=w='min(960,iw)':h='min(540,ih)':force_original_aspect_ratio=decrease:force_divisible_by=2",
               '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '23',
               '-pix_fmt', 'yuv420p', '-movflags', '+faststart', temporary]
    result = subprocess.run(command, capture_output=True, timeout=180,
                            creationflags=0x08000000 if os.name == 'nt' else 0)
    if result.returncode:
        raise RuntimeError(result.stderr.decode(errors='replace')[-400:])
    os.replace(temporary, destination)
    return destination


def contiguous_end(segments):
    end = 0.0
    for segment in sorted(segments, key=lambda s: s['start']):
        if segment['start'] > end + .05:
            break
        end = max(end, segment['start'] + segment['length'])
    return end


def locate(segments, second):
    for segment in sorted(segments, key=lambda s: s['start'], reverse=True):
        if segment['start'] <= second < segment['start'] + segment['length'] + .001:
            return segment, max(0.0, min(second - segment['start'], segment['length'] - .001))
    return None, 0.0
