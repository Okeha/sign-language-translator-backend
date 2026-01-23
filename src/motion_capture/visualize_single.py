"""Utility: create an overlay debug video for a single input video.

Usage:
    python visualize_single.py /path/to/video.mp4 /path/to/output.mp4 --indices

If `output` is omitted, an output file with suffix `_overlay.mp4` is created
next to the input video.
"""
import argparse
from pathlib import Path
import sys

from extract_motion import MotionExtractor


def main():
    parser = argparse.ArgumentParser(description="Create an overlay video with MediaPipe landmarks")
    parser.add_argument("video", help="Path to input video file")
    parser.add_argument("output", nargs="?", help="Output video path (optional)")
    parser.add_argument("--from-json", dest="from_json", help="Path to exported motion JSON to draw instead of detections")
    parser.add_argument("--indices", action="store_true", help="Overlay landmark indices")
    parser.add_argument("--fps", type=int, default=30, help="Target FPS for visualization (default: 30)")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Video not found: {video_path}")
        sys.exit(2)

    if args.output:
        output_path = Path(args.output)
    else:
        out_dir = Path(__file__).parent / "motion_library" / "overlayed_video"
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / (video_path.stem + "_overlay.mp4")

    print(f"Input:  {video_path}")
    print(f"Output: {output_path}")

    extractor = MotionExtractor(target_fps=args.fps, verbose=True)
    try:
        extractor.visualize_tracking(str(video_path), str(output_path), overlay_indices=args.indices, overlay_json=(args.from_json if args.from_json else None))
        print(f"Saved overlay video: {output_path}")
    finally:
        extractor.close()


if __name__ == "__main__":
    main()
