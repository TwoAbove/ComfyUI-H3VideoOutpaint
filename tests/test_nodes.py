import importlib.util
import math
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
            (1, frame_count, latent.shape[-2] * 16, latent.shape[-1] * 16, 3),
            0.25,
        )

class FakeAudioVAE:
    def __init__(self):
        self.encoded = []
        self.decoded = []

    def encode(self, waveform):
        self.encoded.append(waveform.clone())
        latent_t = math.ceil(waveform.shape[1] / 800)
        return torch.full((1, 32, 2, latent_t), 0.5)

    def decode(self, latent):
        self.decoded.append(latent.clone())
        waveform = latent.mean(dim=1).repeat_interleave(800, dim=-1)
        return waveform.movedim(1, -1)


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
        starts = h3_nodes._global_window_starts(634, 73)

        self.assertEqual(starts[0], 0)
        self.assertEqual(starts[-1], 561)
        self.assertTrue(
            all(start % h3_nodes.CONTEXT_FRAMES == 0 for start in starts)
        )
        self.assertEqual(
            {
                frame
                for start in starts
                for frame in range(start, start + 73)
            },
            set(range(634)),
        )

    def test_completed_overlap_is_reencoded_hard_input_to_next_window(self):
        video = torch.zeros((1, 24, 32, 4, 2))
        video[:, :, :, 1:3, 0] = 7
        audio = torch.zeros((1, 32, 2, 18))
        video_mask = torch.ones((1, 1, 32, 4, 2))
        video_mask[:, :, :, 1:3, 0] = 0
        audio_mask = torch.ones((1, 1, 2, 18))
        latent = {
            "samples": h3_nodes.comfy.nested_tensor.NestedTensor(
                (video, audio)
            ),
            "noise_mask": h3_nodes.comfy.nested_tensor.NestedTensor(
                (video_mask, audio_mask)
            ),
            "source_keyframe": {
                "latent": torch.full((1, 24, 32, 2, 1), 7.0),
                "audio_latent": None,
                "latent_y": 1,
                "latent_x": 0,
                "resolved_frame_index": 0,
            },
        }
        window_shapes = [(1, 24, 22, 4, 2), (1, 32, 2, 12)]
        specs = [(0, 0, 0), (34, 10, 6)]
        conditioning = [[torch.zeros((1, 1, 1)), {}]]
        calls = []

        class MarkingVAE(FakeVAE):
            def encode(self, frames):
                latent = super().encode(frames)
                return torch.full_like(latent, 10.0 + len(self.encoded))

        vae = MarkingVAE()

        def sample(*args, **kwargs):
            local_video, local_audio = args[8].unbind()
            local_video_mask, local_audio_mask = kwargs[
                "noise_mask"
            ].unbind()
            keyframes = args[6][0][1]["minimax_keyframes"]
            self.assertIs(args[6], args[7])
            calls.append(
                {
                    "video": local_video.clone(),
                    "video_mask": local_video_mask.clone(),
                    "keyframes": keyframes,
                }
            )
            value = float(len(calls) * 2)
            sampled_video = torch.where(
                local_video_mask.bool(),
                torch.full_like(local_video, value),
                local_video,
            )
            sampled_audio = torch.where(
                local_audio_mask.bool(),
                torch.full_like(local_audio, value),
                local_audio,
            )
            return h3_nodes.comfy.nested_tensor.NestedTensor(
                (sampled_video, sampled_audio)
            )

        zero_noise = h3_nodes.comfy.nested_tensor.NestedTensor(
            (torch.zeros_like(video), torch.zeros_like(audio))
        )
        with (
            patch.object(
                h3_nodes.comfy.sample,
                "prepare_noise",
                return_value=zero_noise,
            ),
            patch.object(
                h3_nodes.comfy.sample,
                "sample",
                side_effect=sample,
            ),
        ):
            result = h3_nodes._sample_sliding_latent(
                FakeModel(),
                [conditioning, conditioning],
                latent,
                window_shapes,
                specs,
                vae,
                1,
                20,
                "res_multistep",
                "simple",
            )

        result_video, result_audio = result["samples"].unbind()
        first, second = calls

        # window 1: pinned source tokens, standalone source keyframe
        self.assertTrue(torch.all(first["video"][:, :, :, 1:3, 0] == 7))
        self.assertTrue(torch.all(first["video_mask"][:, :, :, 1:3, 0] == 0))
        self.assertTrue(torch.all(first["video_mask"][:, :, :, :, 1] == 1))
        self.assertEqual(len(first["keyframes"]), 1)
        self.assertTrue(torch.all(first["keyframes"][0]["latent"] == 7))

        # generated tokens are VAE round-tripped before commit; pinned ones are not
        self.assertEqual(len(vae.decoded), 2)
        self.assertEqual(tuple(vae.decoded[0].shape), (1, 24, 22, 4, 2))

        # window 2: overlap tokens are the re-encoded output of window 1 and
        # arrive both pinned and as a keyframe
        self.assertTrue(torch.all(result_video[:, :, :22, 1:3, 0] == 7))
        self.assertTrue(torch.all(result_video[:, :, :22, :, 1] == 11))
        self.assertTrue(torch.all(second["video"][:, :, :12, :, 1] == 11))
        self.assertTrue(torch.all(second["video_mask"][:, :, :12] == 0))
        self.assertEqual(len(second["keyframes"]), 2)
        self.assertTrue(torch.all(second["keyframes"][1]["latent"][:, :, :, :, 1] == 11))
        self.assertTrue(torch.all(result_video[:, :, 22:, :, 1] == 12))
        self.assertTrue(torch.all(result_audio[..., :12] == 2))
        self.assertTrue(torch.all(result_audio[..., 12:] == 4))

    def test_spatial_keyframe_uses_its_source_grid_rows(self):
        source = torch.zeros((1, 24, 22, 44, 78))
        layout = h3_nodes.comfy.ldm.minimax.model.PackedLayout(
            1,
            22,
            60,
            78,
            122,
            keyframes=[
                {
                    "latent": source,
                    "latent_y": 8,
                    "latent_x": 0,
                    "resolved_frame_index": 0,
                }
            ],
        )

        cond_start, cond_stop, kind = layout.segments[1]
        self.assertEqual(kind, "cond")
        self.assertEqual(cond_stop - cond_start, 22 * 22 * 39)


class MaskOwnershipTests(unittest.TestCase):
    def test_mask_leaves_band_facing_edge_rows_free(self):
        mask = h3_nodes._spatial_generation_mask(
            60, 78, 720, 1248, 0, 112, torch.device("cpu")
        )[0, 0]

        # source rows are tokens 7:52; the encoder folds the band context into
        # rows 7 and 51, so only 8:51 are pinned; the columns touch the canvas
        self.assertTrue(torch.all((mask == 0) | (mask == 1)))
        self.assertTrue(torch.all(mask[:8] == 1))
        self.assertTrue(torch.all(mask[8:51] == 0))
        self.assertTrue(torch.all(mask[51:] == 1))
        self.assertEqual(h3_nodes._pinned_span(0, 78, 78), (0, 78))

    def test_band_tone_is_matched_to_source_and_composite_feathers_the_free_rows(self):
        # decode: top band and its 16 free rows +0.1, bottom band and its 32
        # free rows -0.2 red; the interior decodes close to the source
        source = torch.full((2, 64, 64, 3), 0.5)
        frames = torch.full((2, 128, 64, 3), 0.5)
        frames[:, :48] += 0.1
        frames[:, 64:, :, 0] -= 0.2

        h3_nodes._match_band_tone(frames, source, 32, 96, 16, 32, 1)
        frames[:, 32:96].lerp_(source, h3_nodes._source_weight(64, 64, 16, 32, 0, 0))

        self.assertTrue(torch.equal(frames[:, 48:64], source[:, 16:32]))
        self.assertTrue(torch.allclose(frames[:, 31:97], torch.full((2, 66, 64, 3), 0.5), atol=1e-6))
        self.assertTrue(torch.all((frames[:, 30] - 0.5).abs() < 0.01))
        self.assertTrue(torch.all((frames[:, 97, :, 0] - 0.5).abs() < 0.01))
        # the far edge keeps most of the band's own tone
        self.assertTrue(torch.all(frames[:, 0] > 0.59))
        self.assertTrue(torch.all(frames[:, 127, :, 0] < 0.31))

    def test_seam_detail_does_not_extend_as_stripes_through_band(self):
        source = torch.full((2, 64, 64, 3), 0.5)
        source[:, -4:, 24:40] += 0.2
        frames = torch.full((2, 128, 64, 3), 0.6)

        h3_nodes._match_band_tone(frames, source, 32, 96, 16, 16, 1)

        self.assertTrue(torch.allclose(frames[:, 104], frames[:, 104, :1].expand(-1, 64, -1)))

    def test_source_weight_fades_only_toward_bands(self):
        weight = h3_nodes._source_weight(48, 40, 0, 32, 16, 0)[..., 0]
        self.assertTrue(torch.all(weight[:16, 16:] == 1))
        self.assertTrue(torch.all(weight[-1] == 0))
        self.assertTrue(torch.all(weight[:, 0] == 0))
        self.assertTrue(torch.all(weight[1:, 20] <= weight[:-1, 20]))
        self.assertTrue(torch.all(weight[0, 1:] >= weight[0, :-1]))

    def test_621_source_frames_leave_four_future_tokens_for_generation(self):
        self.assertEqual(h3_nodes.temporal_shape(621), (634, 187, 1057))
        self.assertEqual(h3_nodes._observed_video_tokens(621, 187), 183)

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
            20, 2, 288, 32, 0, 16, torch.device("cpu")
        )

        latent, window_shapes, _ = h3_nodes._assemble_global_latent(
            source,
            source_count,
            vae,
            0,
            16,
            0,
            16,
            spatial_mask,
            None,
            73,
        )
        video_mask = latent["noise_mask"].unbind()[0]
        video = latent["samples"].unbind()[0]
        observed = h3_nodes._observed_video_tokens(
            source_count, video_mask.shape[2]
        )

        # standalone source encode copied into the target; the edge rows 1 and
        # 18 are copied but left free in the mask
        self.assertEqual(len(vae.encoded), 1)
        self.assertEqual(vae.encoded[0].shape, (aligned_count, 288, 32, 3))
        self.assertTrue(torch.all(video[:, :, :, :1] == 0))
        self.assertTrue(torch.all(video[:, :, :, 1:19] == 1))
        self.assertTrue(torch.all(video[:, :, :, 19:] == 0))
        self.assertTrue(torch.all(video_mask[:, :, :observed, 2:18] == 0))
        self.assertTrue(torch.all(video_mask[:, :, :observed, :2] == 1))
        self.assertTrue(torch.all(video_mask[:, :, :observed, 18:] == 1))
        self.assertTrue(torch.all(video_mask[:, :, observed:] == 1))
        self.assertEqual(
            window_shapes[0][2],
            h3_nodes.temporal_shape(73)[1],
        )
        # pinned rows are 2:18, already 2x2-patch aligned, so the keyframe is
        # exactly those rows
        keyframe = latent["source_keyframe"]
        self.assertEqual(keyframe["latent_y"], 2)
        self.assertEqual(tuple(keyframe["latent"].shape), (1, 24, observed, 16, 2))
        self.assertTrue(torch.all(keyframe["latent"] == 1))

    def test_global_latent_preserves_source_audio_and_generates_tail(self):
        source_count = 18
        aligned_count, _, global_audio_t = h3_nodes.temporal_shape(source_count)
        source = torch.zeros((aligned_count, 32, 64, 3), dtype=torch.uint8)
        spatial_mask = h3_nodes._spatial_generation_mask(
            6, 4, 32, 64, 0, 32, torch.device("cpu")
        )
        source_audio = torch.full((1, 32, 2, 30), 0.5)

        latent, _, _ = h3_nodes._assemble_global_latent(
            source,
            source_count,
            FakeVAE(),
            0,
            32,
            0,
            32,
            spatial_mask,
            source_audio,
            aligned_count,
        )
        audio = latent["samples"].unbind()[1]
        audio_mask = latent["noise_mask"].unbind()[1]

        self.assertEqual(audio.shape[-1], global_audio_t)
        self.assertTrue(torch.all(audio[..., :30] == 0.5))
        self.assertTrue(torch.all(audio[..., 30:] == 0))
        self.assertTrue(torch.all(audio_mask[..., :30] == 0))
        self.assertTrue(torch.all(audio_mask[..., 30:] == 1))

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
                clip=FakeClip(),
                prompt="",
                video_vae=FakeVAE(),
                audio_vae=None,
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

    def test_1254x720_geometry_keeps_every_source_row_on_the_expanded_axis(self):
        canvas = h3_nodes._best_effort_canvas(
            1254,
            720,
            "9:12 portrait",
            1.0,
            0.7,
            1.5,
        )
        # width is fixed and must sit on the 32 px canvas grid; height gains
        # bands and keeps all 720 rows on the 16 px latent grid
        self.assertEqual(canvas, (1248, 960, 1248, 720, 0, 112, 0, 128))

    def test_fixed_axis_is_cropped_to_the_canvas_grid(self):
        canvas = h3_nodes._best_effort_canvas(
            1264,
            720,
            "9:12 portrait",
            1.0,
            0.7,
            1.5,
        )
        self.assertEqual(canvas[:4], (1248, 960, 1248, 720))


class StreamingContractTests(unittest.TestCase):
    def test_save_emits_generated_alignment_tail_at_source_fps(self):
        source_count = 18
        aligned_count = h3_nodes.temporal_shape(source_count)[0]
        output_values = []
        sampler_masks = []
        sampler_latents = []

        def sample_sliding(*args, **kwargs):
            latent = args[2]
            self.assertEqual(len(args[1]), len(args[4]))
            sampler_masks.append(latent["noise_mask"].unbind()[0].clone())
            sampler_latents.append(latent["samples"].unbind()[0].clone())
            return {"samples": latent["samples"]}

        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "source.mp4"
            output_path = Path(temp_dir) / "output.mp4"
            with h3_nodes.av.open(source_path, mode="w") as container:
                stream = container.add_stream("h264", rate=30)
                stream.width = 134
                stream.height = 96
                stream.pix_fmt = "yuv420p"
                x = torch.arange(134, dtype=torch.int16).view(1, 134)
                y = torch.arange(96, dtype=torch.int16).view(96, 1)
                for index in range(source_count):
                    image = torch.empty((96, 134, 3), dtype=torch.uint8)
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
                width=134,
                height=96,
                frame_count=source_count,
                frame_rate=30,
            )
            vae = FakeVAE()
            audio_vae = FakeAudioVAE()
            clip = FakeClip()
            video, width, height, fps, length, info = (
                h3_nodes.MiniMaxH3SimpleVideoOutpaint().outpaint(
                    model=FakeModel(),
                    clip=clip,
                    video_vae=vae,
                    audio_vae=audio_vae,
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
                    prompt="lush meadow",
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

            source_frames = h3_nodes._load_aligned_source_frames(
                source_video,
                0,
                0,
                source_count,
                video.source_width,
                video.source_height,
                crop=(
                    video.crop_left,
                    video.crop_top,
                    video.source_width,
                    video.source_height,
                ),
            )

            with h3_nodes.av.open(output_path, mode="r") as container:
                output_stream = container.streams.video[0]
                decoded_count = sum(1 for _ in container.decode(output_stream))
                output_fps = float(output_stream.average_rate)
                output_size = (output_stream.width, output_stream.height)
                audio_stream_count = len(container.streams.audio)

        self.assertFalse(source_video.components_requested)
        # one Qwen context per window, each carrying that window's opening
        # source frame as <Picture 1>
        self.assertEqual(clip.prompts, ["lush meadow"])
        self.assertEqual(len(clip.image_batches), 1)
        picture = clip.image_batches[0][0]
        self.assertEqual(tuple(picture.shape), (1, 96, 128, 3))
        self.assertTrue(torch.allclose(picture[0], vae.encoded[0][0]))
        self.assertEqual((width, height, fps, length), (128, 160, 30.0, aligned_count))
        self.assertEqual(output_size, (128, 160))
        self.assertEqual(output_fps, 30.0)
        self.assertEqual(decoded_count, aligned_count)
        self.assertEqual(audio_stream_count, 0)
        self.assertEqual(audio_vae.encoded, [])
        self.assertEqual(audio_vae.decoded, [])
        self.assertEqual(len(vae.encoded), 1)
        self.assertEqual(vae.encoded[0].shape, (aligned_count, 96, 128, 3))
        target_latent = sampler_latents[0]
        source_y = video.top // 16
        source_h = video.source_height // 16
        self.assertTrue(torch.all(target_latent[:, :, :, source_y : source_y + source_h] == 1))
        self.assertTrue(torch.all(target_latent[:, :, :, :source_y] == 0))
        self.assertTrue(torch.all(target_latent[:, :, :, source_y + source_h :] == 0))
        self.assertEqual(len(vae.decoded), 1)
        self.assertEqual(len(output_values), aligned_count)
        source_center = (
            source_frames[
                :source_count,
                video.source_height // 2,
                video.source_width // 2,
                0,
            ].to(torch.float32)
            / 255.0
        )
        self.assertTrue(
            torch.allclose(torch.tensor(output_values[:source_count]), source_center)
        )
        self.assertTrue(
            torch.allclose(
                torch.tensor(output_values[source_count:]),
                torch.full((aligned_count - source_count,), 0.25),
            )
        )
        self.assertEqual(len(sampler_masks), 1)
        mask = sampler_masks[0]
        observed = h3_nodes._observed_video_tokens(source_count, mask.shape[2])
        expected_mask = torch.ones_like(mask[:, :, :observed])
        pinned = h3_nodes._pinned_span(source_y, source_h, mask.shape[-2])
        self.assertEqual(pinned, (3, 7))
        expected_mask[:, :, :, pinned[0] : pinned[1]] = 0.0
        self.assertTrue(torch.equal(mask[:, :, :observed], expected_mask))
        self.assertTrue(torch.all(mask[:, :, observed:] == 1))
        self.assertIn(
            f"{source_count} source frames -> {aligned_count} H3-aligned output frames",
            info,
        )
        self.assertIn("internal 24 fps, delivered at 30 fps", info)
        self.assertNotIn("MiniMaxH3VideoOutpaintToSize", h3_nodes.NODE_CLASS_MAPPINGS)


    def test_save_conditions_on_source_audio_and_muxes_generated_tail(self):
        source_count = 18
        frame_rate = 30
        aligned_count = h3_nodes.temporal_shape(source_count)[0]
        audio_masks = []

        def sample_sliding(*args, **kwargs):
            latent = args[2]
            audio_masks.append(latent["noise_mask"].unbind()[1].clone())
            return {"samples": latent["samples"]}

        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "source_audio.mp4"
            output_path = Path(temp_dir) / "output_audio.mp4"
            with h3_nodes.av.open(source_path, mode="w") as container:
                video_stream = container.add_stream("h264", rate=frame_rate)
                video_stream.width = 64
                video_stream.height = 32
                video_stream.pix_fmt = "yuv420p"
                audio_stream = container.add_stream(
                    "aac",
                    rate=h3_nodes.AUDIO_SAMPLE_RATE,
                )
                audio_stream.layout = "stereo"
                for index in range(source_count):
                    image = torch.full(
                        (32, 64, 3),
                        index,
                        dtype=torch.uint8,
                    )
                    frame = h3_nodes.av.VideoFrame.from_ndarray(
                        image.numpy(),
                        format="rgb24",
                    )
                    for packet in video_stream.encode(frame):
                        container.mux(packet)
                for packet in video_stream.encode(None):
                    container.mux(packet)

                sample_count = round(
                    source_count
                    / frame_rate
                    * h3_nodes.AUDIO_SAMPLE_RATE
                )
                time = torch.arange(sample_count) / h3_nodes.AUDIO_SAMPLE_RATE
                waveform = (
                    0.1
                    * torch.sin(2 * math.pi * 440 * time)
                    .repeat(2, 1)
                    .to(torch.float32)
                )
                for start in range(0, sample_count, 1024):
                    frame = h3_nodes.av.AudioFrame.from_ndarray(
                        waveform[:, start : start + 1024].numpy(),
                        format="fltp",
                        layout="stereo",
                    )
                    frame.sample_rate = h3_nodes.AUDIO_SAMPLE_RATE
                    frame.pts = start
                    frame.time_base = h3_nodes.Fraction(
                        1,
                        h3_nodes.AUDIO_SAMPLE_RATE,
                    )
                    for packet in audio_stream.encode(frame):
                        container.mux(packet)
                for packet in audio_stream.encode(None):
                    container.mux(packet)

            source_video = FakeLazyVideo(
                source_path,
                width=64,
                height=32,
                frame_count=source_count,
                frame_rate=frame_rate,
            )
            video_vae = FakeVAE()
            audio_vae = FakeAudioVAE()
            video = h3_nodes.MiniMaxH3SimpleVideoOutpaint().outpaint(
                model=FakeModel(),
                clip=FakeClip(),
                video_vae=video_vae,
                audio_vae=audio_vae,
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
            )[0]

            with patch.object(
                h3_nodes,
                "_sample_sliding_latent",
                side_effect=sample_sliding,
            ):
                video.save_to(output_path)

            with h3_nodes.av.open(output_path, mode="r") as container:
                self.assertEqual(len(container.streams.audio), 1)
                audio_stream = container.streams.audio[0]
                output_audio = torch.cat(
                    [
                        torch.from_numpy(frame.to_ndarray())
                        for frame in container.decode(audio_stream)
                    ],
                    dim=-1,
                )
                audio_duration = float(
                    audio_stream.duration * audio_stream.time_base
                )

        expected_model_samples = round(
            source_count / h3_nodes.FPS * h3_nodes.AUDIO_SAMPLE_RATE
        )
        self.assertFalse(source_video.components_requested)
        self.assertEqual(
            audio_vae.encoded[0].shape,
            (1, expected_model_samples, 2),
        )
        self.assertEqual(len(audio_vae.decoded), 1)
        self.assertEqual(len(audio_masks), 1)
        self.assertTrue(torch.all(audio_masks[0][..., :30] == 0))
        self.assertTrue(torch.all(audio_masks[0][..., 30:] == 1))
        self.assertGreater(output_audio.abs().max().item(), 0.01)
        self.assertAlmostEqual(
            audio_duration,
            aligned_count / frame_rate,
            delta=0.05,
        )

if __name__ == "__main__":
    unittest.main()
