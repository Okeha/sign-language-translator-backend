"""Sampling helpers shared by VideoMAE inference paths."""

from typing import Callable, List, Sequence, TypeVar

import numpy as np

EncodedFrame = TypeVar("EncodedFrame")
DecodedFrame = TypeVar("DecodedFrame")


def uniform_sample_indices(total_frames: int, num_frames: int) -> np.ndarray:
    """Return the same uniform integer sample positions used by the legacy path."""
    if total_frames < 1:
        raise ValueError("At least one source frame is required")
    if num_frames < 1:
        raise ValueError("num_frames must be at least 1")

    return np.linspace(0, total_frames - 1, num_frames).astype(int)


def decode_uniformly_sampled_frames(
    encoded_frames: Sequence[EncodedFrame],
    num_frames: int,
    decoder: Callable[[List[EncodedFrame]], List[DecodedFrame]],
) -> List[DecodedFrame]:
    """Decode only unique source frames referenced by the uniform sample sequence."""
    indices = uniform_sample_indices(len(encoded_frames), num_frames)

    # dict preserves first-seen order and removes duplicate positions produced when
    # the source contains fewer frames than the model requires.
    unique_indices = list(dict.fromkeys(indices.tolist()))
    selected_encoded_frames = [encoded_frames[index] for index in unique_indices]
    decoded_frames = decoder(selected_encoded_frames)

    if len(decoded_frames) != len(unique_indices):
        raise ValueError("Decoder returned an unexpected number of frames")

    decoded_by_index = dict(zip(unique_indices, decoded_frames))
    return [decoded_by_index[index] for index in indices]
