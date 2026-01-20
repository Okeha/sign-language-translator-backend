"""Quick test to verify backend fixes for quaternions and finger angles."""

import sys
import json
from pathlib import Path
from extract_motion import MotionExtractor

def test_extraction():
    print("Testing updated extraction with quaternions and corrected finger angles...\n")
    
    # Initialize extractor
    extractor = MotionExtractor(verbose=True)
    
    # Test video
    video_path = "src/model/finetune/data_engineering/raw_videos/69241.mp4"
    
    print(f"Processing: {video_path}\n")
    
    # Extract motion
    motion_data = extractor.extract_motion(video_path, gloss="TEST_BOOK")
    
    # Check first frame with hands detected
    print("="*60)
    print("CHECKING FRAME WITH HANDS DETECTED")
    print("="*60)
    
    for frame in motion_data["frames"]:
        if frame["hands"]["left"] and frame["hands"]["right"]:
            print(f"\n✅ Found frame at {frame['timestamp']}s with both hands\n")
            
            # Check body quaternions
            print("🔹 BODY DATA:")
            print(f"  Left shoulder quaternion: {frame['body'].get('left_shoulder_quat', 'MISSING')}")
            print(f"  Left elbow quaternion: {frame['body'].get('left_elbow_quat', 'MISSING')}")
            print(f"  Right shoulder quaternion: {frame['body'].get('right_shoulder_quat', 'MISSING')}")
            print(f"  Right elbow quaternion: {frame['body'].get('right_elbow_quat', 'MISSING')}")
            
            # Check hand wrist rotation
            print(f"\n🔹 HAND WRIST ROTATIONS:")
            print(f"  Left wrist rotation: {frame['hands']['left'].get('wrist_rotation', 'MISSING')}")
            print(f"  Right wrist rotation: {frame['hands']['right'].get('wrist_rotation', 'MISSING')}")
            
            # Check finger angles (should be 0-90° not 90-180°)
            print(f"\n🔹 FINGER ANGLES (should be 0-90° for natural poses):")
            left_index = frame['hands']['left']['index']
            print(f"  Left index MCP: {left_index['mcp']:.1f}° (0° = straight)")
            print(f"  Left index PIP: {left_index['pip']:.1f}°")
            print(f"  Left index DIP: {left_index['dip']:.1f}°")
            
            print(f"\n🔹 VALIDATION:")
            # Quaternions should be present
            has_quats = all([
                'left_shoulder_quat' in frame['body'],
                'left_elbow_quat' in frame['body'],
                'wrist_rotation' in frame['hands']['left']
            ])
            print(f"  ✅ Quaternions present: {has_quats}")
            
            # Finger angles should be inverted (smaller for straighter fingers)
            avg_angle = (left_index['mcp'] + left_index['pip'] + left_index['dip']) / 3
            reasonable_range = 0 <= avg_angle <= 90
            print(f"  ✅ Finger angles in natural range (0-90°): {reasonable_range} (avg={avg_angle:.1f}°)")
            
            break
    else:
        print("⚠️  No frame with both hands detected")
    
    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)

if __name__ == "__main__":
    test_extraction()
