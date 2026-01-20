"""
Validation script to verify hand detection and motion extraction quality.
Tests extraction on a single video and generates detailed diagnostics.
"""

import sys
import json
import cv2
import numpy as np
from pathlib import Path
from extract_motion import MotionExtractor

def draw_landmarks_on_frame(frame, pose_landmarks, hand_landmarks_left, hand_landmarks_right, face_landmarks):
    """Draw MediaPipe landmarks on frame for visualization."""
    h, w, _ = frame.shape
    vis_frame = frame.copy()
    
    # Draw pose landmarks (body)
    if pose_landmarks:
        for landmark in pose_landmarks:
            x, y = int(landmark.x * w), int(landmark.y * h)
            cv2.circle(vis_frame, (x, y), 3, (0, 255, 0), -1)
    
    # Draw left hand landmarks
    if hand_landmarks_left:
        for landmark in hand_landmarks_left:
            x, y = int(landmark.x * w), int(landmark.y * h)
            cv2.circle(vis_frame, (x, y), 2, (255, 0, 0), -1)
    
    # Draw right hand landmarks
    if hand_landmarks_right:
        for landmark in hand_landmarks_right:
            x, y = int(landmark.x * w), int(landmark.y * h)
            cv2.circle(vis_frame, (x, y), 2, (0, 0, 255), -1)
    
    # Draw face landmarks (subset for clarity)
    if face_landmarks:
        # Only draw key facial features (lips, eyebrows)
        key_indices = [61, 291, 0, 17, 61, 185, 40, 39, 37, 267, 269, 270, 409]
        for idx in key_indices:
            if idx < len(face_landmarks):
                landmark = face_landmarks[idx]
                x, y = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(vis_frame, (x, y), 2, (255, 255, 0), -1)
    
    return vis_frame


def validate_video(video_path: str, output_dir: str = "validation_output"):
    """
    Validate motion extraction on a single video.
    
    Args:
        video_path: Path to video file
        output_dir: Directory to save validation outputs
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"VALIDATING: {video_path.name}")
    print(f"{'='*60}\n")
    
    # Initialize extractor
    print("Initializing MediaPipe extractor...")
    extractor = MotionExtractor()
    
    # Load and resample video
    print(f"Loading video from: {video_path}")
    cap, total_frames, duration, original_fps = extractor.load_video(str(video_path))
    print(f"  Original: {total_frames} frames @ {original_fps:.2f} fps")
    
    resampled_frames = extractor.resample_frames(cap, total_frames, duration)
    cap.release()
    print(f"  Resampled: {len(resampled_frames)} frames @ 30 fps\n")
    
    # Process frames and collect stats
    print("Processing frames...")
    detection_stats = {
        "total_frames": len(resampled_frames),
        "body_detected": 0,
        "left_hand_detected": 0,
        "right_hand_detected": 0,
        "face_detected": 0,
        "finger_angles": {"mcp": [], "pip": [], "dip": []}
    }
    
    frame_data_list = []
    sample_visualizations = [0, len(resampled_frames)//4, len(resampled_frames)//2, len(resampled_frames)-1]
    
    for i, frame in enumerate(resampled_frames):
        # Process frame - returns dict with all data
        frame_data = extractor.process_frame(frame, i)
        
        # Extract components
        body_data = frame_data.get("body")
        hands_data = frame_data.get("hands") or {"left": None, "right": None}
        face_data = frame_data.get("face")
        
        # Update stats
        if body_data:
            detection_stats["body_detected"] += 1
        if hands_data.get("left"):
            detection_stats["left_hand_detected"] += 1
            # Collect finger angle samples
            for finger in ["thumb", "index", "middle", "ring", "pinky"]:
                if finger in hands_data["left"] and hands_data["left"][finger]:
                    finger_data = hands_data["left"][finger]
                    detection_stats["finger_angles"]["mcp"].append(finger_data["mcp"])
                    detection_stats["finger_angles"]["pip"].append(finger_data["pip"])
                    detection_stats["finger_angles"]["dip"].append(finger_data["dip"])
        if hands_data.get("right"):
            detection_stats["right_hand_detected"] += 1
        if face_data:
            detection_stats["face_detected"] += 1
        
        # Store frame data
        frame_data_list.append(frame_data)
        
        # Save visualization for sample frames
        if i in sample_visualizations:
            # Note: Visualization removed since we don't have raw landmark objects anymore
            # Just print stats
            print(f"  Frame {i:3d}: Body={body_data is not None}, "
                  f"L_Hand={hands_data.get('left') is not None}, "
                  f"R_Hand={hands_data.get('right') is not None}, "
                  f"Face={face_data is not None}")
    
    # Print detection statistics
    print(f"\n{'='*60}")
    print("DETECTION STATISTICS")
    print(f"{'='*60}")
    print(f"Total Frames:      {detection_stats['total_frames']}")
    print(f"Body Detected:     {detection_stats['body_detected']} "
          f"({detection_stats['body_detected']/detection_stats['total_frames']*100:.1f}%)")
    print(f"Left Hand:         {detection_stats['left_hand_detected']} "
          f"({detection_stats['left_hand_detected']/detection_stats['total_frames']*100:.1f}%)")
    print(f"Right Hand:        {detection_stats['right_hand_detected']} "
          f"({detection_stats['right_hand_detected']/detection_stats['total_frames']*100:.1f}%)")
    print(f"Face Detected:     {detection_stats['face_detected']} "
          f"({detection_stats['face_detected']/detection_stats['total_frames']*100:.1f}%)")
    
    # Print finger angle statistics
    if detection_stats["finger_angles"]["mcp"]:
        print(f"\n{'='*60}")
        print("FINGER ANGLE STATISTICS (degrees)")
        print(f"{'='*60}")
        for joint in ["mcp", "pip", "dip"]:
            angles = detection_stats["finger_angles"][joint]
            print(f"{joint.upper()} angles: "
                  f"min={min(angles):.1f}°, "
                  f"max={max(angles):.1f}°, "
                  f"mean={np.mean(angles):.1f}°, "
                  f"std={np.std(angles):.1f}°")
    
    # Save full JSON output
    motion_data = {
        "gloss": video_path.stem,
        "fps": 30,
        "duration": len(frame_data_list) / 30.0,
        "frames": frame_data_list
    }
    
    json_output_path = output_dir / f"{video_path.stem}_motion.json"
    with open(json_output_path, 'w') as f:
        json.dump(motion_data, f, indent=2)
    print(f"\nFull JSON saved to: {json_output_path}")
    
    # Print sample frame data
    print(f"\n{'='*60}")
    print("SAMPLE FRAME DATA (Frame 0)")
    print(f"{'='*60}")
    sample_frame = frame_data_list[0]
    print(json.dumps(sample_frame, indent=2)[:1000] + "...")
    
    # Save visualizations info
    print(f"\n{'='*60}")
    print("VISUALIZATIONS")
    print(f"{'='*60}")
    print(f"Saved {len(sample_visualizations)} visualized frames to: {frames_dir}")
    for i in sample_visualizations:
        print(f"  - frame_{i:04d}.jpg")
    
    # Final verdict
    print(f"\n{'='*60}")
    print("VALIDATION VERDICT")
    print(f"{'='*60}")
    
    hand_detection_rate = (detection_stats['left_hand_detected'] + detection_stats['right_hand_detected']) / (2 * detection_stats['total_frames']) * 100
    
    if hand_detection_rate > 70:
        print("✅ EXCELLENT: Hand detection rate > 70%")
    elif hand_detection_rate > 40:
        print("⚠️  MODERATE: Hand detection rate 40-70% (may need threshold adjustment)")
    else:
        print("❌ POOR: Hand detection rate < 40% (check video quality or thresholds)")
    
    if detection_stats["finger_angles"]["mcp"]:
        angle_variance = np.std(detection_stats["finger_angles"]["mcp"])
        if angle_variance > 10:
            print("✅ GOOD: Finger angles show variance (handshapes are captured)")
        else:
            print("⚠️  WARNING: Low finger angle variance (may indicate static pose)")
    else:
        print("❌ CRITICAL: No hand data extracted!")
    
    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python validate_extraction.py <video_path> [output_dir]")
        print("\nExample:")
        print("  python validate_extraction.py raw_videos/69241.mp4")
        print("  python validate_extraction.py raw_videos/69241.mp4 my_output")
        sys.exit(1)
    
    video_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "validation_output"
    
    if not Path(video_path).exists():
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)
    
    validate_video(video_path, output_dir)
