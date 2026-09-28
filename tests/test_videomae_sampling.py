import unittest
from unittest.mock import Mock

import numpy as np

from src.api.videomae.sampling import (
    decode_uniformly_sampled_frames,
    uniform_sample_indices,
)


class VideoMAESamplingTests(unittest.TestCase):
    def test_sample_indices_match_legacy_policy_across_boundaries(self):
        for total_frames in (1, 8, 16, 60, 120):
            with self.subTest(total_frames=total_frames):
                expected = np.linspace(0, total_frames - 1, 16).astype(int)
                actual = uniform_sample_indices(total_frames, 16)
                np.testing.assert_array_equal(actual, expected)

    def test_decoder_receives_only_unique_selected_source_frames(self):
        encoded_frames = [f"frame-{index}" for index in range(60)]
        expected_indices = np.linspace(0, 59, 16).astype(int)
        expected_unique_indices = list(dict.fromkeys(expected_indices.tolist()))
        decoder = Mock(
            side_effect=lambda frames: [f"decoded:{frame}" for frame in frames]
        )

        sampled = decode_uniformly_sampled_frames(
            encoded_frames,
            16,
            decoder,
        )

        decoder.assert_called_once_with(
            [encoded_frames[index] for index in expected_unique_indices]
        )
        self.assertEqual(
            sampled,
            [f"decoded:{encoded_frames[index]}" for index in expected_indices],
        )
        self.assertEqual(len(decoder.call_args.args[0]), 16)

    def test_short_input_decodes_each_source_at_most_once_and_repeats_samples(self):
        encoded_frames = [f"frame-{index}" for index in range(8)]
        expected_indices = np.linspace(0, 7, 16).astype(int)
        decoder = Mock(side_effect=lambda frames: [object() for _ in frames])

        sampled = decode_uniformly_sampled_frames(
            encoded_frames,
            16,
            decoder,
        )

        decoded_source_frames = decoder.call_args.args[0]
        self.assertEqual(decoded_source_frames, encoded_frames)
        self.assertEqual(len(sampled), 16)

        # Duplicate legacy sample positions must reuse the already-decoded object.
        for left, right in zip(expected_indices[:-1], expected_indices[1:]):
            if left == right:
                matching_positions = np.where(expected_indices == left)[0]
                first = sampled[matching_positions[0]]
                self.assertTrue(
                    all(sampled[position] is first for position in matching_positions)
                )

    def test_maximum_120_frame_batch_decodes_only_16_images(self):
        encoded_frames = [f"frame-{index}" for index in range(120)]
        decoder = Mock(side_effect=lambda frames: list(frames))

        sampled = decode_uniformly_sampled_frames(
            encoded_frames,
            16,
            decoder,
        )

        self.assertEqual(len(sampled), 16)
        self.assertEqual(len(decoder.call_args.args[0]), 16)
        self.assertLessEqual(len(decoder.call_args.args[0]), min(120, 16))

    def test_unselected_invalid_frame_is_not_decoded(self):
        encoded_frames = [f"frame-{index}" for index in range(120)]
        encoded_frames[1] = "corrupt"

        def decoder(frames):
            if "corrupt" in frames:
                raise ValueError("invalid frame")
            return list(frames)

        sampled = decode_uniformly_sampled_frames(
            encoded_frames,
            16,
            decoder,
        )

        self.assertEqual(len(sampled), 16)
        self.assertNotIn("corrupt", sampled)

    def test_selected_invalid_frame_propagates_decoder_failure(self):
        encoded_frames = [f"frame-{index}" for index in range(120)]
        # Index 7 is selected by np.linspace(0, 119, 16).astype(int).
        encoded_frames[7] = "corrupt"

        def decoder(frames):
            if "corrupt" in frames:
                raise ValueError("invalid frame")
            return list(frames)

        with self.assertRaisesRegex(ValueError, "invalid frame"):
            decode_uniformly_sampled_frames(
                encoded_frames,
                16,
                decoder,
            )

    def test_decoder_output_count_must_match_unique_selected_frames(self):
        encoded_frames = [f"frame-{index}" for index in range(60)]
        decoder = Mock(side_effect=lambda frames: list(frames[:-1]))

        with self.assertRaisesRegex(
            ValueError,
            "Decoder returned an unexpected number of frames",
        ):
            decode_uniformly_sampled_frames(
                encoded_frames,
                16,
                decoder,
            )

    def test_invalid_frame_counts_are_rejected(self):
        with self.assertRaises(ValueError):
            uniform_sample_indices(0, 16)
        with self.assertRaises(ValueError):
            uniform_sample_indices(1, 0)


if __name__ == "__main__":
    unittest.main()
