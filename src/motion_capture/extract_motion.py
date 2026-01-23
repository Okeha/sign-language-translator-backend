"""
Motion Extractor for Sign Language Videos

Extracts skeletal, hand, and facial motion data from sign language videos using MediaPipe Holistic.
Outputs JSON files with Three.js-compatible coordinates for 3D avatar animation.
"""

import cv2
import json
import mediapipe as mp
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import tempfile
from motion_utils import (
    calculate_euler_angles,
    calculate_finger_curl,
    calculate_finger_joint_angles,
    calculate_blendshape_from_distance,
    convert_to_threejs_coords,
    exponential_decay_interpolate,
    calculate_quaternion_from_direction,
    calculate_quaternion_from_direction_rpm,
    calculate_wrist_quaternion
)


class MotionExtractor:
    """Extracts motion data from sign language videos using MediaPipe Holistic."""
    
    # MediaPipe pose landmark indices
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    
    # Hand landmark indices
    WRIST = 0
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_MCP = 5
    INDEX_PIP = 6
    INDEX_DIP = 7
    INDEX_TIP = 8
    MIDDLE_MCP = 9
    MIDDLE_PIP = 10
    MIDDLE_DIP = 11
    MIDDLE_TIP = 12
    RING_MCP = 13
    RING_PIP = 14
    RING_DIP = 15
    RING_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20
    
    # Face landmark indices
    UPPER_LIP = 13
    LOWER_LIP = 14
    MOUTH_LEFT = 61
    MOUTH_RIGHT = 291
    LEFT_EYEBROW_INNER = 70
    LEFT_EYE_TOP = 159
    RIGHT_EYEBROW_INNER = 300
    RIGHT_EYE_TOP = 386
    
    def __init__(self, target_fps: int = 30, verbose: bool = True):
        """
        Initialize Motion Extractor.
        
        Args:
            target_fps: Target frame rate for motion data (default 30)
            verbose: Enable logging
        """
        self.target_fps = target_fps
        self.verbose = verbose
        
        # Initialize separate MediaPipe landmarkers (new API doesn't have HolisticLandmarker)
        # Pose Landmarker
        pose_model = self._download_model('pose')
        pose_options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=pose_model),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_segmentation_masks=False
        )
        # Global timestamp offset (ms) to ensure monotonic timestamps across videos
        self._global_ts_offset = 0
        self.pose_landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(pose_options)
        
        # Hand Landmarker
        hand_model = self._download_model('hand')
        hand_options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=hand_model),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.hand_landmarker = mp.tasks.vision.HandLandmarker.create_from_options(hand_options)
        
        # Face Landmarker
        face_model = self._download_model('face')
        face_options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=face_model),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False
        )
        self.face_landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(face_options)
        
        # Tracking state for missing detections
        self.prev_left_hand = None
        self.prev_right_hand = None
        self.left_hand_missing_frames = 0
        self.right_hand_missing_frames = 0
    
    def _download_model(self, model_type: str) -> str:
        """Download MediaPipe model if needed."""
        import urllib.request
        import os
        
        model_urls = {
            'pose': 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task',
            'hand': 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task',
            'face': 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task'
        }
        
        model_names = {
            'pose': 'pose_landmarker_heavy.task',
            'hand': 'hand_landmarker.task',
            'face': 'face_landmarker.task'
        }
        
        model_path = Path.home() / ".mediapipe" / model_names[model_type]
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        if not model_path.exists():
            if self.verbose:
                print(f"📥 Downloading MediaPipe {model_type} model (first time only)...")
            
            urllib.request.urlretrieve(model_urls[model_type], model_path)
            
            if self.verbose:
                print(f"   ✅ {model_type.capitalize()} model downloaded")
        
        return str(model_path)
        self.right_hand_missing_frames = 0
        
    def load_video(self, video_path: str) -> Tuple[cv2.VideoCapture, int, float, int]:
        """
        Load video and extract metadata.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Tuple of (capture, total_frames, duration, fps)
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        
        return cap, total_frames, duration, fps
    
    def resample_frames(self, cap: cv2.VideoCapture, total_frames: int, 
                       duration: float, orig_fps: float, global_offset_ms: int = 0) -> List[Tuple[np.ndarray, int]]:
        """
        Resample video to target FPS.
        
        Args:
            cap: OpenCV VideoCapture object
            total_frames: Total number of frames in video
            duration: Video duration in seconds
            
        Returns:
            List of resampled frames
        """
        target_frame_count = int(duration * self.target_fps)
        frame_indices = np.linspace(0, total_frames - 1, target_frame_count, dtype=int)

        frames = []  # list of (frame, timestamp_ms)
        current_idx = 0
        frame_count = 0
        
        last_ts = -1
        for target_idx in frame_indices:
            # Seek to target frame
            while current_idx <= target_idx:
                ret, frame = cap.read()
                if not ret:
                    break
                if current_idx == target_idx:
                    # Timestamp based on original video fps (ms)
                    timestamp_ms = int(round((target_idx / (orig_fps if orig_fps > 0 else 1.0)) * 1000))
                    # Apply global offset and ensure strictly increasing timestamps
                    timestamp_ms = timestamp_ms + int(global_offset_ms)
                    if timestamp_ms <= last_ts:
                        timestamp_ms = last_ts + 1
                    last_ts = timestamp_ms
                    frames.append((frame, timestamp_ms))
                    frame_count += 1
                current_idx += 1
        
        cap.release()
        
        if self.verbose:
            print(f"   Resampled {len(frames)} frames at {self.target_fps}fps")
        
        return frames
    
    def extract_body_data(self, pose_landmarks) -> Optional[Dict]:
        """Export raw landmarks for Kalidokit processing."""
        if not pose_landmarks:
            return None
        
        landmarks = pose_landmarks if isinstance(pose_landmarks, list) else pose_landmarks
        
        # Export ALL 33 pose landmarks (Kalidokit needs them)
        world_landmarks = []
        for lm in landmarks:
            world_landmarks.append({
                "x": float(lm.x),
                "y": float(lm.y),
                "z": float(lm.z),
                "visibility": float(lm.visibility) if hasattr(lm, 'visibility') else 1.0
            })
        
        return {
            "worldLandmarks": world_landmarks
        }
    def extract_hand_data(self, hand_landmarks) -> Optional[Dict]:
        """Export raw hand landmarks for Kalidokit."""
        if not hand_landmarks:
            return None
        
        landmarks = hand_landmarks if isinstance(hand_landmarks, list) else hand_landmarks
        
        # Export all 21 hand landmarks as a list
        landmarks_list = []
        for lm in landmarks:
            landmarks_list.append({
                "x": float(lm.x),
                "y": float(lm.y),
                "z": float(lm.z)
            })
        
        # Return as dict so exponential_decay_interpolate still works
        return {
            "landmarks": landmarks_list
        }

    def extract_face_data(self, face_landmarks) -> Optional[Dict]:
        """Extract facial blendshapes from face landmarks."""
        if not face_landmarks:
            return None
        
        # In new API, face_landmarks is a list, not an object with .landmark
        landmarks = face_landmarks if isinstance(face_landmarks, list) else face_landmarks
        
        # Jaw open (vertical distance between lips)
        jaw_open = calculate_blendshape_from_distance(
            landmarks[self.UPPER_LIP],
            landmarks[self.LOWER_LIP],
            min_distance=0.01,  # Closed mouth
            max_distance=0.08   # Fully open
        )
        
        # Mouth smile (horizontal distance between mouth corners)
        mouth_smile = calculate_blendshape_from_distance(
            landmarks[self.MOUTH_LEFT],
            landmarks[self.MOUTH_RIGHT],
            min_distance=0.05,  # Neutral
            max_distance=0.10   # Wide smile
        )
        
        # Eyebrow raise left
        eyebrow_raise_l = calculate_blendshape_from_distance(
            landmarks[self.LEFT_EYE_TOP],
            landmarks[self.LEFT_EYEBROW_INNER],
            min_distance=0.03,  # Neutral
            max_distance=0.05   # Raised
        )
        
        # Eyebrow raise right
        eyebrow_raise_r = calculate_blendshape_from_distance(
            landmarks[self.RIGHT_EYE_TOP],
            landmarks[self.RIGHT_EYEBROW_INNER],
            min_distance=0.03,  # Neutral
            max_distance=0.05   # Raised
        )
        
        return {
            "jawOpen": jaw_open,
            "mouthSmile": mouth_smile,
            "eyeBrowRaise_L": eyebrow_raise_l,
            "eyeBrowRaise_R": eyebrow_raise_r
        }
    

    def reset_landmarkers(self):
        """Reset MediaPipe landmarkers to handle new video timestamps."""
        # Close existing landmarkers
        self.pose_landmarker.close()
        self.hand_landmarker.close()
        self.face_landmarker.close()
        
        # Reinitialize them
        pose_model = self._download_model('pose')
        pose_options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=pose_model),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_segmentation_masks=False
        )
        self.pose_landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(pose_options)
        
        hand_model = self._download_model('hand')
        hand_options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=hand_model),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.hand_landmarker = mp.tasks.vision.HandLandmarker.create_from_options(hand_options)
        
        face_model = self._download_model('face')
        face_options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=face_model),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False
        )
        self.face_landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(face_options)
    def process_frame(self, frame: np.ndarray, frame_idx: int, timestamp_ms: Optional[int] = None) -> Dict:
        """
        Process single frame with MediaPipe and extract all motion data.
        
        Args:
            frame: BGR video frame
            frame_idx: Frame index for timestamp calculation
            
        Returns:
            Dict with timestamp, body, hands, and face data
        """
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Process with separate landmarkers. If running in VIDEO mode, pass timestamp when available.
        if timestamp_ms is not None:
            # Prefer video-mode API when available. Only fall back to image-mode
            # detect if the video-method is not present (AttributeError).
            try:
                pose_results = self.pose_landmarker.detect_for_video(mp_image, timestamp_ms)
            except AttributeError:
                pose_results = self.pose_landmarker.detect(mp_image)

            try:
                hand_results = self.hand_landmarker.detect_for_video(mp_image, timestamp_ms)
            except AttributeError:
                hand_results = self.hand_landmarker.detect(mp_image)

            try:
                face_results = self.face_landmarker.detect_for_video(mp_image, timestamp_ms)
            except AttributeError:
                face_results = self.face_landmarker.detect(mp_image)
        else:
            pose_results = self.pose_landmarker.detect(mp_image)
            hand_results = self.hand_landmarker.detect(mp_image)
            face_results = self.face_landmarker.detect(mp_image)
        
        # Extract body data from pose_world_landmarks
        body_data = None
        if pose_results.pose_world_landmarks:
            body_data = self.extract_body_data(pose_results.pose_world_landmarks[0] if pose_results.pose_world_landmarks else None)
        
        # Extract hand data with missing detection handling
        # Hand results contain list of detected hands with handedness
        left_hand_data = None
        right_hand_data = None
        
        if hand_results.hand_landmarks and hand_results.handedness:
            for hand_landmarks, handedness in zip(hand_results.hand_landmarks, hand_results.handedness):
                # Check if it's left or right hand
                hand_label = handedness[0].category_name.lower()
                hand_data = self.extract_hand_data(hand_landmarks)
                
                if hand_label == 'left':
                    left_hand_data = hand_data
                elif hand_label == 'right':
                    right_hand_data = hand_data
        
        # Apply exponential decay for missing hands
        if left_hand_data is None:
            self.left_hand_missing_frames += 1
            if self.left_hand_missing_frames < 5:
                left_hand_data = exponential_decay_interpolate(None, self.prev_left_hand)
        else:
            self.left_hand_missing_frames = 0
            self.prev_left_hand = left_hand_data
        
        if right_hand_data is None:
            self.right_hand_missing_frames += 1
            if self.right_hand_missing_frames < 5:
                right_hand_data = exponential_decay_interpolate(None, self.prev_right_hand)
        else:
            self.right_hand_missing_frames = 0
            self.prev_right_hand = right_hand_data
        
        # Extract face data
        face_data = None
        if face_results.face_landmarks:
            face_data = self.extract_face_data(face_results.face_landmarks[0] if face_results.face_landmarks else None)
        
        # Calculate timestamp (keep output schema as seconds/frame index)
        timestamp = frame_idx / self.target_fps
        
        return {
            "timestamp": round(timestamp, 3),
            "body": body_data,
            "hands": {
                "left": left_hand_data,
                "right": right_hand_data
            },
            "face": face_data
        }
    
def extract_motion(self, video_path: str, gloss: str) -> Dict:
    """
    Extract complete motion data from video.
    
    Args:
        video_path: Path to video file
        gloss: Sign language word/gloss
        
    Returns:
        Dict with motion data including metadata and frames
    """
    if self.verbose:
        print(f"\n🎬 Processing: {gloss.upper()}")
        print(f"   Video: {video_path}")
    
    # Reset tracking state
    self.prev_left_hand = None
    self.prev_right_hand = None
    self.left_hand_missing_frames = 0
    self.right_hand_missing_frames = 0
    
    # Reset landmarkers to handle fresh timestamps
    self.reset_landmarkers()
    
    # Load and resample video
    cap, total_frames, duration, fps = self.load_video(video_path)
    
    if self.verbose:
        print(f"   Original: {total_frames} frames at {fps:.2f}fps, {duration:.2f}s")
    
    # Reset global timestamp offset for this video
    self._global_ts_offset = 0
    
    frames = self.resample_frames(cap, total_frames, duration, fps, global_offset_ms=0)
    
    # Process each frame
    motion_frames = []
    for idx, (frame, ts_ms) in enumerate(frames):
        frame_data = self.process_frame(frame, idx, timestamp_ms=ts_ms)
        motion_frames.append(frame_data)
        
        if self.verbose and (idx + 1) % 10 == 0:
            print(f"   Processed {idx + 1}/{len(frames)} frames", end='\r')
    
    if self.verbose:
        print(f"   ✅ Completed {len(motion_frames)} frames")
    
    return {
        "gloss": gloss,
        "fps": self.target_fps,
        "duration": round(duration, 3),
        "frame_count": len(motion_frames),
        "frames": motion_frames
    }


    def save_motion_data(self, motion_data: Dict, output_path: Path):
        """Save motion data to JSON file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(motion_data, f, indent=2)
        
        if self.verbose:
            print(f"   💾 Saved to: {output_path}")
    
    def close(self):
        """Release MediaPipe resources."""
        self.pose_landmarker.close()
        self.hand_landmarker.close()
        self.face_landmarker.close()

    def visualize_tracking(self, video_path: str, output_path: Optional[str] = None, overlay_indices: bool = False, overlay_json: Optional[str] = None):
        """
        Create a new video with detected hand/pose landmarks overlaid for debugging.

        Args:
            video_path: input video file path
            output_path: output video path to write (mp4)
            overlay_indices: draw landmark index numbers next to points
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or self.target_fps

        # Default output path: save under motion_library/overlayed_video
        base_dir = Path(__file__).parent
        default_out_dir = base_dir / "motion_library" / "overlayed_video"
        default_out_dir.mkdir(parents=True, exist_ok=True)

        if output_path is None:
            output_path = default_out_dir / (Path(video_path).stem + "_overlay.mp4")
        else:
            output_path = Path(output_path)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        # If overlay_json provided, load exported motion JSON to draw same landmarks
        motion_json = None
        if overlay_json:
            try:
                with open(overlay_json, 'r', encoding='utf-8') as jf:
                    motion_json = json.load(jf)
            except Exception:
                motion_json = None

        frame_idx = 0
        last_ts = -1

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            timestamp_ms = int(round((frame_idx / (fps if fps > 0 else 1.0)) * 1000)) + int(self._global_ts_offset)
            if timestamp_ms <= last_ts:
                timestamp_ms = last_ts + 1
            last_ts = timestamp_ms

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # Safe defaults for detector results when drawing from JSON
            hand_results = None
            pose_results = None
            face_results = None

            # If motion JSON provided, draw landmarks from exported JSON for this frame index
            if motion_json and 'frames' in motion_json and frame_idx < len(motion_json['frames']):
                js_frame = motion_json['frames'][frame_idx]
                hands = js_frame.get('hands', {})

                for side in ('left', 'right'):
                    hand = hands.get(side)
                    if not hand:
                        continue

                    landmarks = hand.get('landmarks')
                    # landmarks in exported JSON are Three.js coords or normalized coords.
                    if not landmarks:
                        continue

                    # Heuristic: decide if x is normalized (0..1) or world meters
                    sample_x = landmarks[0].get('x', 0.0)
                    is_normalized = (-0.5 <= sample_x <= 1.5)

                    color = (0, 255, 0) if side == 'left' else (0, 128, 255)
                    for i, lm in enumerate(landmarks):
                        lx = lm.get('x', 0.0)
                        ly = lm.get('y', 0.0)
                        if is_normalized:
                            x_px = int(lx * width)
                            y_px = int(ly * height)
                        else:
                            # Fallback: treat Three.js X/Y as normalized around center
                            # Map X from [-0.5,0.5] -> [0,width]
                            x_px = int((0.5 + lx) * width)
                            y_px = int((0.5 + ly) * height)

                        cv2.circle(frame, (x_px, y_px), 3, color, -1)
                        if overlay_indices:
                            cv2.putText(frame, str(i), (x_px + 4, y_px - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

                    # Draw label
                    try:
                        wrist = landmarks[0]
                        wx = int((wrist.get('x', 0.0) * width) if is_normalized else (0.5 + wrist.get('x', 0.0)) * width)
                        wy = int((wrist.get('y', 0.0) * height) if is_normalized else (0.5 + wrist.get('y', 0.0)) * height)
                        cv2.putText(frame, side, (wx - 10, wy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
                    except Exception:
                        pass

            else:
                # Run detectors in video mode when available
                try:
                    hand_results = self.hand_landmarker.detect_for_video(mp_image, timestamp_ms)
                except AttributeError:
                    hand_results = self.hand_landmarker.detect(mp_image)

                try:
                    pose_results = self.pose_landmarker.detect_for_video(mp_image, timestamp_ms)
                except AttributeError:
                    pose_results = self.pose_landmarker.detect(mp_image)

                # Draw hands from detections
                if hand_results and getattr(hand_results, 'hand_landmarks', None):
                    for hand_idx, hand_landmarks in enumerate(hand_results.hand_landmarks):
                        # handedness label if present
                        label = None
                        if getattr(hand_results, 'handedness', None) and len(hand_results.handedness) > hand_idx:
                            try:
                                label = hand_results.handedness[hand_idx][0].category_name
                            except Exception:
                                label = None

                        # Draw landmarks (assume normalized x/y in [0,1])
                        for i, lm in enumerate(hand_landmarks):
                            try:
                                x_px = int(lm.x * width)
                                y_px = int(lm.y * height)
                            except Exception:
                                # fallback: skip if coordinates not available
                                continue

                            color = (0, 255, 0) if label and label.lower() == 'left' else (0, 128, 255)
                            cv2.circle(frame, (x_px, y_px), 3, color, -1)
                            if overlay_indices:
                                cv2.putText(frame, str(i), (x_px + 4, y_px - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

                        # Put handedness text
                        if label:
                            # place label near wrist if available
                            try:
                                wrist = hand_landmarks[0]
                                wx = int(wrist.x * width)
                                wy = int(wrist.y * height)
                                cv2.putText(frame, label, (wx - 10, wy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
                            except Exception:
                                pass

            # Draw pose keypoints (optional small circles)
            if pose_results and getattr(pose_results, 'pose_world_landmarks', None):
                # pose_world_landmarks may not have image coordinates; try pose_landmarks instead
                if getattr(pose_results, 'pose_landmarks', None):
                    pl = pose_results.pose_landmarks[0]
                    for lm in pl:
                        try:
                            x_px = int(lm.x * width)
                            y_px = int(lm.y * height)
                            cv2.circle(frame, (x_px, y_px), 2, (255, 0, 0), -1)
                        except Exception:
                            continue

            out.write(frame)
            frame_idx += 1

        cap.release()
        out.release()
        return output_path
