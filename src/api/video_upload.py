"""Bounded-memory helpers for uploaded-video inference."""

from typing import BinaryIO, Callable, List, TypeVar

import cv2
import numpy as np
from fastapi import UploadFile

UPLOAD_CHUNK_SIZE = 1024 * 1024
Frame = TypeVar("Frame")


async def copy_upload_in_chunks(
    upload: UploadFile,
    destination: BinaryIO,
    chunk_size: int = UPLOAD_CHUNK_SIZE,
) -> None:
    """Copy an UploadFile to disk without materializing the full upload in memory."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1 byte")

    while True:
        chunk = await upload.read(chunk_size)
        if not chunk:
            break
        destination.write(chunk)


def _uniform_sample_indices(total_frames: int, num_frames: int) -> np.ndarray:
    if total_frames < 1:
        raise ValueError("At least one source frame is required")
    if num_frames < 1:
        raise ValueError("num_frames must be at least 1")

    return np.linspace(0, total_frames - 1, num_frames).astype(int)


def _count_decodable_frames(
    video_path: str,
    capture_factory: Callable[[str], object],
) -> int:
    """Count frames that can actually be decoded without retaining their arrays."""
    capture = capture_factory(video_path)
    try:
        total_frames = 0
        while capture.isOpened():
            ok, _ = capture.read()
            if not ok:
                break
            total_frames += 1
        return total_frames
    finally:
        capture.release()


def extract_uniform_frames(
    video_path: str,
    num_frames: int,
    capture_factory: Callable[[str], object] = cv2.VideoCapture,
    convert_to_rgb: Callable[[Frame], Frame] = lambda frame: cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB,
    ),
) -> List[Frame]:
    """
    Return the legacy uniform sample sequence while retaining only sampled frames.

    The first pass decodes frames only to count the readable prefix and immediately
    discards them. The second pass retains only unique target positions, keeping
    decoded-frame memory bounded by the model sample count instead of video length.
    """
    total_frames = _count_decodable_frames(video_path, capture_factory)
    if total_frames == 0:
        return []

    indices = _uniform_sample_indices(total_frames, num_frames)
    unique_indices = list(dict.fromkeys(indices.tolist()))
    target_indices = set(unique_indices)
    frames_by_index = {}

    capture = capture_factory(video_path)
    try:
        frame_index = 0
        while capture.isOpened() and len(frames_by_index) < len(unique_indices):
            ok, frame = capture.read()
            if not ok:
                break

            if frame_index in target_indices:
                frames_by_index[frame_index] = convert_to_rgb(frame)

            frame_index += 1
    finally:
        capture.release()

    missing_indices = [
        index for index in unique_indices if index not in frames_by_index
    ]
    if missing_indices:
        raise ValueError(
            "Video ended before all uniformly sampled frames could be decoded"
        )

    return [frames_by_index[index] for index in indices]
