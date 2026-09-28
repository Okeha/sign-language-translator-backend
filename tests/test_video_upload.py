import asyncio
import io
import unittest

import numpy as np

from src.api.video_upload import copy_upload_in_chunks, extract_uniform_frames


class FakeUpload:
    def __init__(self, payload: bytes):
        self._payload = io.BytesIO(payload)
        self.read_sizes = []

    async def read(self, size: int) -> bytes:
        self.read_sizes.append(size)
        return self._payload.read(size)


class FakeCapture:
    def __init__(self, frames, fail_read_at=None, stop_read_at=None):
        self.frames = list(frames)
        self.position = 0
        self.fail_read_at = fail_read_at
        self.stop_read_at = stop_read_at
        self.released = False

    def isOpened(self):
        return not self.released

    def grab(self):
        if self.position >= len(self.frames):
            return False
        self.position += 1
        return True

    def read(self):
        if self.fail_read_at is not None and self.position == self.fail_read_at:
            raise RuntimeError("decode failed")
        if self.stop_read_at is not None and self.position >= self.stop_read_at:
            return False, None
        if self.position >= len(self.frames):
            return False, None
        frame = self.frames[self.position]
        self.position += 1
        return True, frame

    def release(self):
        self.released = True


class CaptureFactory:
    def __init__(
        self,
        frames,
        fail_second_read_at=None,
        stop_read_at=None,
    ):
        self.frames = list(frames)
        self.fail_second_read_at = fail_second_read_at
        self.stop_read_at = stop_read_at
        self.instances = []

    def __call__(self, _video_path):
        fail_at = self.fail_second_read_at if len(self.instances) == 1 else None
        capture = FakeCapture(
            self.frames,
            fail_read_at=fail_at,
            stop_read_at=self.stop_read_at,
        )
        self.instances.append(capture)
        return capture


class VideoUploadTests(unittest.TestCase):
    def test_upload_is_copied_using_bounded_reads(self):
        payload = b"abcdefghijklmnopqrstuvwxyz"
        upload = FakeUpload(payload)
        destination = io.BytesIO()

        asyncio.run(copy_upload_in_chunks(upload, destination, chunk_size=8))

        self.assertEqual(destination.getvalue(), payload)
        self.assertTrue(upload.read_sizes)
        self.assertTrue(all(size == 8 for size in upload.read_sizes))

    def test_long_video_retains_only_uniform_sample_positions(self):
        frames = list(range(60))
        factory = CaptureFactory(frames)
        expected_indices = np.linspace(0, 59, 16).astype(int)

        sampled = extract_uniform_frames(
            "video.mp4",
            16,
            capture_factory=factory,
            convert_to_rgb=lambda frame: frame,
        )

        self.assertEqual(sampled, [frames[index] for index in expected_indices])
        self.assertEqual(len(sampled), 16)
        self.assertEqual(len(factory.instances), 2)
        self.assertTrue(all(capture.released for capture in factory.instances))

    def test_short_video_preserves_repeat_frame_sampling(self):
        frames = list(range(8))
        factory = CaptureFactory(frames)
        expected_indices = np.linspace(0, 7, 16).astype(int)

        sampled = extract_uniform_frames(
            "short.mp4",
            16,
            capture_factory=factory,
            convert_to_rgb=lambda frame: frame,
        )

        self.assertEqual(sampled, [frames[index] for index in expected_indices])
        self.assertEqual(len(sampled), 16)

    def test_empty_video_releases_capture_and_returns_no_frames(self):
        factory = CaptureFactory([])

        sampled = extract_uniform_frames(
            "empty.mp4",
            16,
            capture_factory=factory,
            convert_to_rgb=lambda frame: frame,
        )

        self.assertEqual(sampled, [])
        self.assertEqual(len(factory.instances), 1)
        self.assertTrue(factory.instances[0].released)

    def test_decode_exception_releases_both_capture_passes(self):
        factory = CaptureFactory(range(20), fail_second_read_at=3)

        with self.assertRaisesRegex(RuntimeError, "decode failed"):
            extract_uniform_frames(
                "broken.mp4",
                16,
                capture_factory=factory,
                convert_to_rgb=lambda frame: frame,
            )

        self.assertEqual(len(factory.instances), 2)
        self.assertTrue(all(capture.released for capture in factory.instances))

    def test_counting_matches_legacy_readable_prefix_when_grab_outlives_read(self):
        frames = list(range(20))
        factory = CaptureFactory(frames, stop_read_at=10)
        expected_indices = np.linspace(0, 9, 16).astype(int)

        sampled = extract_uniform_frames(
            "truncated.mp4",
            16,
            capture_factory=factory,
            convert_to_rgb=lambda frame: frame,
        )

        self.assertEqual(sampled, [frames[index] for index in expected_indices])
        self.assertEqual(len(factory.instances), 2)
        self.assertTrue(all(capture.released for capture in factory.instances))


if __name__ == "__main__":
    unittest.main()
