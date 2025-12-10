"""
Preprocess videos into fixed-frame tensors to speed up training.

This script scans `data_engineering/raw_videos`, samples `NUM_FRAMES` evenly
from each video using OpenCV, converts them to (T, C, H, W) float16 tensors, 
and saves them as `.pt` files under `preprocessed_frames/`.

Run this once before training or whenever `NUM_FRAMES` changes.
"""

import os
import torch
import cv2
import numpy as np
from tqdm import tqdm

# Configuration
RAW_DIR = "data_engineering/raw_videos"
SAVE_DIR = "preprocessed_frames" # Updated to match train.py expectation
NUM_FRAMES = 8 # Increased to 8

# Ensure save directory exists
os.makedirs(SAVE_DIR, exist_ok=True)

# Loop through all videos in RAW_DIR
video_files = [f for f in os.listdir(RAW_DIR) if f.endswith(".mp4")]

for file_name in tqdm(video_files, desc="Processing Videos"):
    video_path = os.path.join(RAW_DIR, file_name)
    save_path = os.path.join(SAVE_DIR, file_name.replace(".mp4", ".pt"))

    try:
        # Use OpenCV to read video (same logic as sample_frames)
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"⚠️  Could not open video: {file_name}")
            continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames <= 0:
            print(f"⚠️  Video has 0 frames, skipping: {file_name}")
            cap.release()
            continue

        # Sample frame indices evenly
        indices = np.linspace(0, total_frames - 1, NUM_FRAMES).astype(int)
        frames = []
        
        for idx in indices:
            # Seek to specific frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Convert BGR (OpenCV default) to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            else:
                # If read fails, duplicate last successful frame
                if len(frames) > 0:
                    frames.append(frames[-1])
        
        cap.release()
        
        # Handle cases where video was too short or corrupt
        if len(frames) == 0:
            print(f"⚠️  Could not read any frames from: {file_name}")
            continue
        
        # Pad if we couldn't read enough frames
        while len(frames) < NUM_FRAMES:
            frames.append(frames[-1])
        
        # Convert to numpy array: (T, H, W, C), uint8 - SAME as sample_frames
        frames_array = np.array(frames)  # (T, H, W, C), uint8
        
        # Convert to tensor WITHOUT permuting - keep (T, H, W, C) to match sample_frames exactly
        frames_tensor = torch.from_numpy(frames_array).to(torch.float16)  # (T, H, W, C), float16

        # Save
        torch.save(frames_tensor, save_path)

    except Exception as e:
        print(f"❌ Failed to process {file_name}: {e}")

print(f"\n✅ Preprocessing complete! Saved {len(video_files)} videos to {SAVE_DIR}")