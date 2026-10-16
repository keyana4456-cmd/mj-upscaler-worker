"""Run a chunk's face processing, then exit to free the entire CUDA context."""
import argparse
from engine import FaceRestorer, post_frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input'); parser.add_argument('output')
    parser.add_argument('--fast', action='store_true')
    args = parser.parse_args()
    restorer = FaceRestorer()
    print('Face touch-up: ' + restorer.sess.get_providers()[0], flush=True)
    try:
        post_frames(args.input, args.output, args.fast, restorer)
        print(f'Face touch-up: restored {restorer.processed_faces} faces in this chunk', flush=True)
    finally: restorer.close()


if __name__ == '__main__': main()
