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
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_segmentation_masks=False
        )
        self.pose_landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(pose_options)
        
        # Hand Landmarker
        hand_model = self._download_model('hand')
        hand_options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=hand_model),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
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
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
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
                       duration: float) -> List[np.ndarray]:
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
        
        frames = []
        current_idx = 0
        frame_count = 0
        
        for target_idx in frame_indices:
            # Seek to target frame
            while current_idx <= target_idx:
                ret, frame = cap.read()
                if not ret:
                    break
                if current_idx == target_idx:
                    frames.append(frame)
                    frame_count += 1
                current_idx += 1
        
        cap.release()
        
        if self.verbose:
            print(f"   Resampled {len(frames)} frames at {self.target_fps}fps")
        
        return frames
    
    def extract_body_data(self, pose_landmarks) -> Optional[Dict]:
        """Extract body joint positions and rotations from pose landmarks."""
        if not pose_landmarks:
            return None
        
        # In new API, pose_landmarks is a list, not an object with .landmark
        landmarks = pose_landmarks if isinstance(pose_landmarks, list) else pose_landmarks
        
        # Extract joint positions (Three.js coordinates)
        left_shoulder = convert_to_threejs_coords(landmarks[self.LEFT_SHOULDER])
        left_elbow = convert_to_threejs_coords(landmarks[self.LEFT_ELBOW])
        left_wrist = convert_to_threejs_coords(landmarks[self.LEFT_WRIST])
        
        right_shoulder = convert_to_threejs_coords(landmarks[self.RIGHT_SHOULDER])
        right_elbow = convert_to_threejs_coords(landmarks[self.RIGHT_ELBOW])
        right_wrist = convert_to_threejs_coords(landmarks[self.RIGHT_WRIST])
        
        # Calculate Euler angles
        left_yaw, left_pitch, left_roll = calculate_euler_angles(
            landmarks[self.LEFT_SHOULDER],
            landmarks[self.LEFT_ELBOW],
            landmarks[self.LEFT_WRIST]
        )
        
        right_yaw, right_pitch, right_roll = calculate_euler_angles(
            landmarks[self.RIGHT_SHOULDER],
            landmarks[self.RIGHT_ELBOW],
            landmarks[self.RIGHT_WRIST]
        )
        
        # Calculate quaternions for bone rotations
        import numpy as np
        
        # Left shoulder quaternion (shoulder -> elbow direction)
        left_shoulder_dir = np.array([
            landmarks[self.LEFT_ELBOW].x - landmarks[self.LEFT_SHOULDER].x,
            landmarks[self.LEFT_ELBOW].y - landmarks[self.LEFT_SHOULDER].y,
            -(landmarks[self.LEFT_ELBOW].z - landmarks[self.LEFT_SHOULDER].z)  # Negate Z for Three.js
        ])
        left_shoulder_quat = calculate_quaternion_from_direction(left_shoulder_dir)
        
        # Left elbow quaternion (elbow -> wrist direction)
        left_elbow_dir = np.array([
            landmarks[self.LEFT_WRIST].x - landmarks[self.LEFT_ELBOW].x,
            landmarks[self.LEFT_WRIST].y - landmarks[self.LEFT_ELBOW].y,
            -(landmarks[self.LEFT_WRIST].z - landmarks[self.LEFT_ELBOW].z)
        ])
        left_elbow_quat = calculate_quaternion_from_direction(left_elbow_dir)
        
        # Right shoulder quaternion
        right_shoulder_dir = np.array([
            landmarks[self.RIGHT_ELBOW].x - landmarks[self.RIGHT_SHOULDER].x,
            landmarks[self.RIGHT_ELBOW].y - landmarks[self.RIGHT_SHOULDER].y,
            -(landmarks[self.RIGHT_ELBOW].z - landmarks[self.RIGHT_SHOULDER].z)
        ])
        right_shoulder_quat = calculate_quaternion_from_direction(right_shoulder_dir)
        
        # Right elbow quaternion
        right_elbow_dir = np.array([
            landmarks[self.RIGHT_WRIST].x - landmarks[self.RIGHT_ELBOW].x,
            landmarks[self.RIGHT_WRIST].y - landmarks[self.RIGHT_ELBOW].y,
            -(landmarks[self.RIGHT_WRIST].z - landmarks[self.RIGHT_ELBOW].z)
        ])
        right_elbow_quat = calculate_quaternion_from_direction(right_elbow_dir)
        
        return {
            "left_shoulder": {"x": left_shoulder[0], "y": left_shoulder[1], "z": left_shoulder[2]},
            "left_elbow": {"x": left_elbow[0], "y": left_elbow[1], "z": left_elbow[2]},
            "left_wrist": {"x": left_wrist[0], "y": left_wrist[1], "z": left_wrist[2]},
            "left_arm_rotation": {"yaw": left_yaw, "pitch": left_pitch, "roll": left_roll},
            "left_shoulder_quat": left_shoulder_quat,
            "left_elbow_quat": left_elbow_quat,
            "right_shoulder": {"x": right_shoulder[0], "y": right_shoulder[1], "z": right_shoulder[2]},
            "right_elbow": {"x": right_elbow[0], "y": right_elbow[1], "z": right_elbow[2]},
            "right_wrist": {"x": right_wrist[0], "y": right_wrist[1], "z": right_wrist[2]},
            "right_arm_rotation": {"yaw": right_yaw, "pitch": right_pitch, "roll": right_roll},
            "right_shoulder_quat": right_shoulder_quat,
            "right_elbow_quat": right_elbow_quat
        }
    
    def extract_hand_data(self, hand_landmarks) -> Optional[Dict]:
        """Extract finger joint angles from hand landmarks."""
        if not hand_landmarks:
            return None
        
        # In new API, hand_landmarks is a list, not an object with .landmark
        landmarks = hand_landmarks if isinstance(hand_landmarks, list) else hand_landmarks
        
        # Calculate joint angles for each finger (3 joints per finger)
        # Thumb (special case - uses CMC as base)
        thumb_angles = calculate_finger_joint_angles(
            landmarks[self.THUMB_MCP],      # CMC (base)
            landmarks[self.THUMB_MCP],      # MCP
            landmarks[self.THUMB_IP],       # IP
            landmarks[self.THUMB_IP],       # IP (reused)
            landmarks[self.THUMB_TIP]       # Tip
        )
        
        # Index finger
        index_angles = calculate_finger_joint_angles(
            landmarks[self.WRIST],          # Base reference
            landmarks[self.INDEX_MCP],      # Knuckle
            landmarks[self.INDEX_PIP],      # Middle joint
            landmarks[self.INDEX_DIP],      # Near tip
            landmarks[self.INDEX_TIP]       # Tip
        )
        
        # Middle finger
        middle_angles = calculate_finger_joint_angles(
            landmarks[self.WRIST],
            landmarks[self.MIDDLE_MCP],
            landmarks[self.MIDDLE_PIP],
            landmarks[self.MIDDLE_DIP],
            landmarks[self.MIDDLE_TIP]
        )
        
        # Ring finger
        ring_angles = calculate_finger_joint_angles(
            landmarks[self.WRIST],
            landmarks[self.RING_MCP],
            landmarks[self.RING_PIP],
            landmarks[self.RING_DIP],
            landmarks[self.RING_TIP]
        )
        
        # Pinky finger
        pinky_angles = calculate_finger_joint_angles(
            landmarks[self.WRIST],
            landmarks[self.PINKY_MCP],
            landmarks[self.PINKY_PIP],
            landmarks[self.PINKY_DIP],
            landmarks[self.PINKY_TIP]
        )
        
        # Calculate wrist orientation from hand plane
        wrist_quat = calculate_wrist_quaternion(
            landmarks[self.WRIST],
            landmarks[self.INDEX_MCP],
            landmarks[self.MIDDLE_MCP],
            landmarks[self.PINKY_MCP]
        )
        
        return {
            "wrist_rotation": wrist_quat,
            "thumb": thumb_angles,
            "index": index_angles,
            "middle": middle_angles,
            "ring": ring_angles,
            "pinky": pinky_angles
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
    
    def process_frame(self, frame: np.ndarray, frame_idx: int) -> Dict:
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
        
        # Process with separate landmarkers (IMAGE mode - no timestamp needed)
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
        
        # Calculate timestamp
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
        
        # Load and resample video
        cap, total_frames, duration, fps = self.load_video(video_path)
        
        if self.verbose:
            print(f"   Original: {total_frames} frames at {fps:.2f}fps, {duration:.2f}s")
        
        frames = self.resample_frames(cap, total_frames, duration)
        
        # Process each frame
        motion_frames = []
        for idx, frame in enumerate(frames):
            frame_data = self.process_frame(frame, idx)
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
