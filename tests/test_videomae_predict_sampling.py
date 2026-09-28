import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import torch

from src.api.videomae.model_service import VideoMAEService


class DeterministicProcessor:
    def __init__(self):
        self.last_frames = None
        self.last_pixel_values = None

    def __call__(self, frames, return_tensors):
        self.last_frames = list(frames)
        pixel_values = torch.zeros(
            (1, len(frames), 3, 224, 224),
            dtype=torch.float32,
        )
        for index, frame in enumerate(frames):
            pixel_values[0, index, 0, 0, 0] = float(frame[0, 0, 0])
        self.last_pixel_values = pixel_values
        return {"pixel_values": pixel_values}


class DeterministicModel:
    def __call__(self, pixel_values):
        sample_signature = int(pixel_values[:, :, 0, 0, 0].sum().item())
        predicted_index = sample_signature % 5
        logits = torch.zeros((1, 5), dtype=torch.float32)
        logits[0, predicted_index] = 10.0
        return SimpleNamespace(logits=logits)


class VideoMAEPredictSamplingTests(unittest.TestCase):
    def make_service(self):
        service = object.__new__(VideoMAEService)
        service.processor = DeterministicProcessor()
        service.model = DeterministicModel()
        service.device = torch.device("cpu")
        service.id2label = {index: f"label-{index}" for index in range(5)}
        service.decode_base64_frames = MagicMock(
            side_effect=lambda frames: [
                np.full((2, 2, 3), int(frame), dtype=np.uint8)
                for frame in frames
            ]
        )
        return service

    def test_predict_matches_legacy_sampling_fixture_and_processor_shape(self):
        service = self.make_service()
        encoded_frames = [str(index) for index in range(60)]
        legacy_indices = np.linspace(0, 59, 16).astype(int)
        expected_selected = [encoded_frames[index] for index in legacy_indices]
        expected_prediction_index = int(legacy_indices.sum()) % 5

        prediction = service.predict(encoded_frames)

        service.decode_base64_frames.assert_called_once_with(expected_selected)
        self.assertEqual(
            [int(frame[0, 0, 0]) for frame in service.processor.last_frames],
            legacy_indices.tolist(),
        )
        self.assertEqual(
            tuple(service.processor.last_pixel_values.shape),
            (1, 16, 3, 224, 224),
        )
        self.assertEqual(
            prediction.gloss,
            f"label-{expected_prediction_index}",
        )


if __name__ == "__main__":
    unittest.main()
