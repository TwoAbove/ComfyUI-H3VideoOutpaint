import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch


MODULE_PATH = Path(__file__).parents[1] / "nodes.py"
SPEC = importlib.util.spec_from_file_location("h3_video_outpaint_nodes", MODULE_PATH)
h3_nodes = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(h3_nodes)


class FakeClip:
    def __init__(self):
        self.prompts = []
        self.image_batches = []

    def tokenize(self, prompt, images=None):
        self.prompts.append(prompt)
        self.image_batches.append(images)
        return prompt

    def encode_from_tokens_scheduled(self, tokens):
        return [[torch.zeros((1, 1, 1)), {}]]


class FakeModel:
    def clone(self):
        return FakeModel()

    def add_object_patch(self, name, value):
        raise AssertionError(f"unexpected model patch: {name}")


class FakeVAE:
    def __init__(self):
        self.encoded = []
        self.decoded = []

    def encode(self, frames):
        self.encoded.append(frames.clone())
        frame_count, latent_t, _ = h3_nodes.temporal_shape(int(frames.shape[0]))
        assert frame_count == frames.shape[0]
        return torch.full(
            (1, 24, latent_t, frames.shape[1] // 16, frames.shape[2] // 16),
            float(len(self.encoded)),
        )

    def decode(self, latent):
        self.decoded.append(latent.clone())
        frame_count = sum(
            h3_nodes.comfy.ldm.minimax.model.FRAME_PER_TOKEN[
                index % len(h3_nodes.comfy.ldm.minimax.model.FRAME_PER_TOKEN)
            ]
            for index in range(latent.shape[2])
        )
        return torch.full(
            (frame_count, latent.shape[-2] * 16, latent.shape[-1] * 16, 3),
            0.25,
        )


class FakeLazyVideo(h3_nodes.Input.Video):
    def __init__(self, path, width, height, frame_count, frame_rate):
        self.path = path
        self.width = width
        self.height = height
        self.frame_count = frame_count
        self.frame_rate = h3_nodes.Fraction(frame_rate)
        self.components_requested = False

    def get_components(self):
        self.components_requested = True
        raise AssertionError("streaming source was materialized")

    def save_to(self, *args, **kwargs):
        raise AssertionError("source save should not be called")

    def as_trimmed(self, *args, **kwargs):
        return None

    def get_stream_source(self):
        return self.path

    def get_dimensions(self):
        return self.width, self.height

    def get_frame_count(self):
        return self.frame_count

    def get_frame_rate(self):
        return self.frame_rate

    def get_duration(self):
        return self.frame_count / float(self.frame_rate)


class WindowingTests(unittest.TestCase):
    def test_sliding_windows_are_phase_aligned_and_cover_timeline(self):
        frame_count = 634
        window_frames = 73
        starts = h3_nodes._global_window_starts(frame_count, window_frames)

        self.assertEqual(starts[0], 0)
        self.assertEqual(starts[-1], frame_count - window_frames)
        self.assertEqual(len(starts), 18)
        self.assertTrue(all(start % h3_nodes.CONTEXT_FRAMES == 0 for start in starts))
        covered = {
            frame_index
            for start in starts
            for frame_index in range(start, start + window_frames)
        }
        self.assertEqual(covered, set(range(frame_count)))
        self.assertTrue(
            all(
                previous + window_frames > current
                for previous, current in zip(starts, starts[1:])
            )
        )

    def test_sliding_sampler_freezes_overlap_and_commits_only_new_tokens(self):
        frame_count = 107
        specs, window_video_t, window_audio_t = h3_nodes._global_window_specs(
            frame_count, 73
        )
        _, global_video_t, global_audio_t = h3_nodes.temporal_shape(frame_count)
        global_video = torch.zeros((1, 1, global_video_t, 1, 1))
        global_audio = torch.zeros((1, 1, 2, global_audio_t))
        video_mask = torch.ones((1, 1, global_video_t, 1, 1))
        audio_mask = torch.ones((1, 1, 2, global_audio_t))
        latent = {
            "samples": h3_nodes.comfy.nested_tensor.NestedTensor(
                (global_video, global_audio)
            ),
            "noise_mask": h3_nodes.comfy.nested_tensor.NestedTensor(
                (video_mask, audio_mask)
            ),
        }
        source_video = torch.full((1, 1, global_video_t, 1, 1), 2.0)
        conditioning = [[torch.zeros((1, 1, 1)), {}]]
        calls = []

        def sample(*args, **kwargs):
            window_video, window_audio = args[8].unbind()
            sampled_video = torch.full_like(window_video, 10.0 * (len(calls) + 1))
            sampled_audio = torch.full_like(window_audio, 10.0 * (len(calls) + 1))
            calls.append(
                {
                    "mask": kwargs["noise_mask"],
                    "keyframes": args[6][0][1]["minimax_keyframes"],
                }
            )
            return h3_nodes.comfy.nested_tensor.NestedTensor(
                (sampled_video, sampled_audio)
            )

        with patch.object(h3_nodes.comfy.sample, "sample", side_effect=sample):
            sampled = h3_nodes._sample_sliding_latent(
                FakeModel(),
                conditioning,
                conditioning,
                latent,
                source_video,
                [
                    (1, 1, window_video_t, 1, 1),
                    (1, 1, 2, window_audio_t),
                ],
                specs,
                global_video_t,
                0,
                0,
                7,
                1,
                "res_multistep",
                "simple",
            )

        self.assertEqual(len(calls), 2)
        second_video_mask, second_audio_mask = calls[1]["mask"].unbind()
        self.assertTrue(torch.all(second_video_mask[:, :, :12] == 0))
        self.assertTrue(torch.all(second_video_mask[:, :, 12:] == 1))
        self.assertTrue(torch.all(second_audio_mask[..., :66] == 0))
        self.assertTrue(torch.all(second_audio_mask[..., 66:] == 1))
        overlap_keyframe, source_keyframe = calls[1]["keyframes"]
        self.assertEqual(overlap_keyframe["latent"].shape[2], 12)
        self.assertTrue(torch.all(overlap_keyframe["latent"] == 10))
        self.assertEqual(overlap_keyframe["audio_latent"].shape[3], 66)
        self.assertTrue(torch.all(overlap_keyframe["audio_latent"] == 10))
        self.assertEqual(source_keyframe["latent"].shape[2], window_video_t)

        sampled_video, sampled_audio = sampled["samples"].unbind()
        self.assertTrue(torch.all(sampled_video[:, :, :window_video_t] == 10))
        self.assertTrue(torch.all(sampled_video[:, :, window_video_t:] == 20))
        self.assertTrue(torch.all(sampled_audio[..., :window_audio_t] == 10))
        self.assertTrue(torch.all(sampled_audio[..., window_audio_t:] == 20))


class MaskOwnershipTests(unittest.TestCase):
    def test_cc58_mask_generates_only_outside_the_source(self):
        mask = h3_nodes._spatial_generation_mask(
            latent_h=60,
            latent_w=78,
            source_h=704,
            source_w=1248,
            left=0,
            top=128,
            device=torch.device("cpu"),
        )[0, 0]

        self.assertTrue(torch.all((mask == 0) | (mask == 1)))
        self.assertTrue(torch.all(mask[:8] == 1))
        self.assertTrue(torch.all(mask[8:52] == 0))
        self.assertTrue(torch.all(mask[52:] == 1))

    def test_621_source_frames_leave_four_future_tokens_for_generation(self):
        self.assertEqual(h3_nodes.temporal_shape(621), (634, 187, 1057))
        self.assertEqual(h3_nodes._observed_video_tokens(621, 187), 183)

    def test_keyframes_use_native_conditioning_metadata(self):
        source_video = torch.randn((1, 24, 5, 2, 4))
        existing = {"resolved_frame_index": 9, "latent": torch.zeros(1)}
        keyframe = {
            "resolved_frame_index": 0,
            "latent": source_video,
            "latent_y": 8,
            "latent_x": 2,
        }
        conditioning = [
            [torch.zeros((1, 1, 1)), {"minimax_keyframes": [existing]}]
        ]

        conditioned = h3_nodes._conditioning_with_keyframes(
            conditioning, [keyframe]
        )

        self.assertIs(
            conditioned[0][1]["minimax_keyframes"][0],
            existing,
        )
        self.assertIs(
            conditioned[0][1]["minimax_keyframes"][1],
            keyframe,
        )
        self.assertIs(
            conditioned[0][1]["minimax_keyframes"][1]["latent"],
            source_video,
        )
        self.assertIsNot(conditioned[0][1], conditioning[0][1])

    def test_spatial_keyframe_rows_share_target_coordinates(self):
        layout = h3_nodes.comfy.ldm.minimax.model.PackedLayout(
            text_len=1,
            latent_t=2,
            latent_h=6,
            latent_w=8,
            audio_t=1,
            keyframes=[
                {
                    "resolved_frame_index": 0,
                    "latent": torch.zeros((1, 24, 2, 2, 4)),
                    "latent_y": 2,
                    "latent_x": 2,
                }
            ],
        )
        cond_start, cond_stop, _ = next(
            segment for segment in layout.segments if segment[2] == "cond"
        )
        video_start, _, _ = next(
            segment for segment in layout.segments if segment[2] == "video"
        )
        target_rows = torch.tensor([5, 6, 17, 18]) + video_start

        self.assertTrue(
            torch.equal(
                layout.position_ids[cond_start:cond_stop],
                layout.position_ids[target_rows],
            )
        )
        self.assertTrue(
            torch.all(~layout.img_update[: cond_stop - cond_start])
        )

    def test_global_latent_copies_one_source_encoding_into_target(self):
        source_count = 103
        aligned_count = h3_nodes.temporal_shape(source_count)[0]
        source = torch.full(
            (aligned_count, 288, 32, 3),
            127,
            dtype=torch.uint8,
        )
        vae = FakeVAE()
        spatial_mask = h3_nodes._spatial_generation_mask(
            20,
            2,
            288,
            32,
            0,
            16,
            torch.device("cpu"),
        )

        (
            latent,
            source_video,
            window_shapes,
            specs,
            observed,
            source_latent_y,
            source_latent_x,
        ) = h3_nodes._assemble_global_latent(
            source,
            source_count,
            vae,
            0,
            16,
            0,
            16,
            spatial_mask,
            73,
        )
        video_mask = latent["noise_mask"].unbind()[0]
        video = latent["samples"].unbind()[0]

        self.assertEqual(len(vae.encoded), 1)
        self.assertEqual(
            vae.encoded[0].shape,
            (aligned_count, 288, 32, 3),
        )
        self.assertTrue(torch.all(video[:, :, :, :1] == 0))
        self.assertTrue(torch.equal(video[:, :, :, 1:19], source_video))
        self.assertTrue(torch.all(video[:, :, :, 19:] == 0))
        self.assertTrue(
            torch.all(video_mask[:, :, :observed, 1:19] == 0)
        )
        self.assertTrue(
            torch.all(video_mask[:, :, :observed, :1] == 1)
        )
        self.assertTrue(
            torch.all(video_mask[:, :, :observed, 19:] == 1)
        )
        self.assertTrue(torch.all(video_mask[:, :, observed:] == 1))
        self.assertEqual(
            window_shapes[0][2],
            h3_nodes.temporal_shape(73)[1],
        )
        self.assertEqual(specs, [(0, 0, 0), (34, 10, 56)])
        self.assertEqual((source_latent_y, source_latent_x), (1, 0))
        self.assertEqual(source_video.shape, (1, 24, 32, 18, 2))
        self.assertLess(observed, source_video.shape[2])



class GeometryTests(unittest.TestCase):
    def test_auto_caps_at_validated_window_without_rewriting_manual_values(self):
        self.assertEqual(h3_nodes._auto_denoise_window_frames(64, 64), 107)
        source = FakeLazyVideo("/tmp/not-read.mp4", 1254, 720, 192, 30)
        for requested, expected in (
            ("124", 124),
            ("158", 158),
            ("192", 192),
            ("global", None),
        ):
            video = h3_nodes._StreamingH3Video(
                source_video=source,
                model=FakeModel(),
                conditioning=None,
                video_vae=FakeVAE(),
                skip_first_frames=0,
                frame_load_cap=192,
                target_aspect="9:12 portrait",
                generation_megapixels=1.0,
                minimum_source_megapixels=0.7,
                max_upscale=1.5,
                seed=7,
                steps=20,
                sampler_name="res_multistep",
                scheduler="simple",
                temporal_window_frames=requested,
            )
            self.assertEqual(video.denoise_window_frames, expected)
            self.assertEqual(video.frame_rate, h3_nodes.Fraction(30, 1))
            self.assertEqual(video.frame_count, 192)

    def test_cc58_geometry_crops_and_places_on_h3_boundaries(self):
        geometry = h3_nodes._aligned_crop_geometry(1254, 720)
        self.assertEqual(geometry, (1248, 704, 3, 8, 3, 8))

        canvas = h3_nodes._best_effort_canvas(
            1248,
            704,
            "9:12 portrait",
            1.0,
            0.7,
            1.5,
        )
        self.assertEqual(canvas, (1248, 960, 0, 128, 0, 128))


class StreamingContractTests(unittest.TestCase):
    def test_save_emits_generated_alignment_tail_at_source_fps(self):
        source_count = 18
        aligned_count = h3_nodes.temporal_shape(source_count)[0]
        output_values = []
        sampler_masks = []
        sampler_latents = []

        def sample_sliding(*args, **kwargs):
            latent = args[3]
            sampler_masks.append(latent["noise_mask"].unbind()[0].clone())
            video_latent = latent["samples"].unbind()[0]
            source_latent = args[4]
            sampler_latents.append(
                (video_latent.clone(), source_latent.clone(), args[8], args[9])
            )
            return {"samples": latent["samples"]}

        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "source.mp4"
            output_path = Path(temp_dir) / "output.mp4"
            with h3_nodes.av.open(source_path, mode="w") as container:
                stream = container.add_stream("h264", rate=30)
                stream.width = 70
                stream.height = 48
                stream.pix_fmt = "yuv420p"
                x = torch.arange(70, dtype=torch.int16).view(1, 70)
                y = torch.arange(48, dtype=torch.int16).view(48, 1)
                for index in range(source_count):
                    image = torch.empty((48, 70, 3), dtype=torch.uint8)
                    image[..., 0] = ((3 * x + index) % 256).to(torch.uint8)
                    image[..., 1] = ((5 * y + 2 * index) % 256).to(torch.uint8)
                    image[..., 2] = ((x + y + 3 * index) % 256).to(torch.uint8)
                    frame = h3_nodes.av.VideoFrame.from_ndarray(
                        image.numpy(), format="rgb24"
                    )
                    for packet in stream.encode(frame):
                        container.mux(packet)
                for packet in stream.encode(None):
                    container.mux(packet)

            source_video = FakeLazyVideo(
                source_path,
                width=70,
                height=48,
                frame_count=source_count,
                frame_rate=30,
            )
            vae = FakeVAE()
            clip = FakeClip()
            video, width, height, fps, length, info = (
                h3_nodes.MiniMaxH3SimpleVideoOutpaint().outpaint(
                    model=FakeModel(),
                    clip=clip,
                    video_vae=vae,
                    source_video=source_video,
                    skip_first_frames=0,
                    frame_load_cap=0,
                    target_aspect="9:12 portrait",
                    generation_megapixels=0,
                    minimum_source_megapixels=0,
                    max_upscale=1.5,
                    seed=7,
                    steps=1,
                    sampler_name="res_multistep",
                    scheduler="simple",
                )
            )
            original_encode = video._encode_output_frame

            def tracked_encode(output, stream, frame):
                output_values.append(
                    frame[
                        video.top + video.source_height // 2,
                        video.left + video.source_width // 2,
                        0,
                    ].item()
                )
                return original_encode(output, stream, frame)

            with (
                patch.object(
                    h3_nodes,
                    "_sample_sliding_latent",
                    side_effect=sample_sliding,
                ),
                patch.object(
                    video,
                    "_encode_output_frame",
                    side_effect=tracked_encode,
                ),
            ):
                video.save_to(output_path)

            with h3_nodes.av.open(output_path, mode="r") as container:
                output_stream = container.streams.video[0]
                decoded_count = sum(1 for _ in container.decode(output_stream))
                output_fps = float(output_stream.average_rate)
                output_size = (output_stream.width, output_stream.height)

        self.assertFalse(source_video.components_requested)
        self.assertEqual(clip.prompts, [""])
        self.assertEqual(clip.image_batches, [None])
        self.assertEqual((width, height, fps, length), (64, 96, 30.0, aligned_count))
        self.assertEqual(output_size, (64, 96))
        self.assertEqual(output_fps, 30.0)
        self.assertEqual(decoded_count, aligned_count)
        self.assertEqual(len(vae.encoded), 1)
        self.assertEqual(vae.encoded[0].shape, (aligned_count, 32, 64, 3))
        target_latent, source_latent, source_y, source_x = sampler_latents[0]
        self.assertTrue(
            torch.equal(
                target_latent[
                    :,
                    :,
                    :,
                    source_y : source_y + source_latent.shape[-2],
                    source_x : source_x + source_latent.shape[-1],
                ],
                source_latent,
            )
        )
        self.assertEqual(len(vae.decoded), 1)
        self.assertEqual(len(output_values), aligned_count)
        self.assertTrue(
            torch.allclose(
                torch.tensor(output_values),
                torch.full((aligned_count,), 0.25),
            )
        )
        self.assertEqual(len(sampler_masks), 1)
        mask = sampler_masks[0]
        observed = h3_nodes._observed_video_tokens(source_count, mask.shape[2])
        expected_mask = torch.ones_like(mask[:, :, :observed])
        expected_mask[
            :,
            :,
            :,
            source_y : source_y + source_latent.shape[-2],
            source_x : source_x + source_latent.shape[-1],
        ] = 0.0
        self.assertTrue(torch.equal(mask[:, :, :observed], expected_mask))
        self.assertTrue(torch.all(mask[:, :, observed:] == 1))
        self.assertIn(
            f"{source_count} source frames -> {aligned_count} H3-aligned output frames",
            info,
        )
        self.assertIn("internal 24 fps, delivered at 30 fps", info)
        self.assertNotIn("MiniMaxH3VideoOutpaintToSize", h3_nodes.NODE_CLASS_MAPPINGS)


if __name__ == "__main__":
    unittest.main()
