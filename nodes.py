"""Training-free latent-masked video outpainting for MiniMax H3 in ComfyUI."""

import gc
import io
import json
import math
import os
from fractions import Fraction

import av
import torch

import comfy.ldm.minimax.model
import comfy.model_management
import comfy.sample
import comfy.utils
import comfy.nested_tensor
import comfy.samplers
from comfy_api.latest import Input, Types
from comfy_extras.nodes_minimax_h3 import (
    AUDIO_LATENT_FPS,
    CANVAS_MULTIPLE,
    FPS,
    temporal_shape,
)


CONTEXT_FRAMES = 17
CONTEXT_LATENT_T = 5
WINDOW_PIXEL_BUDGET = 90_000_000
WINDOW_STRIDE_FRAMES = 2 * CONTEXT_FRAMES
SOURCE_CONTEXT = 128
DENOISE_WINDOW_FRAMES = (56, 73, 90, 107, 124, 141, 158, 175, 192)
AUTO_DENOISE_WINDOW_FRAMES = DENOISE_WINDOW_FRAMES[:4]


def _align_spatial(value):
    return max(CANVAS_MULTIPLE, math.ceil(value / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)


def _aligned_crop_geometry(width, height):
    cropped_width = width // CANVAS_MULTIPLE * CANVAS_MULTIPLE
    cropped_height = height // CANVAS_MULTIPLE * CANVAS_MULTIPLE
    if cropped_width < CANVAS_MULTIPLE or cropped_height < CANVAS_MULTIPLE:
        raise ValueError(
            f"Source video {width}x{height} is smaller than one "
            f"{CANVAS_MULTIPLE}-pixel H3 spatial chunk."
        )
    left = (width - cropped_width) // 2
    top = (height - cropped_height) // 2
    return (
        cropped_width,
        cropped_height,
        left,
        top,
        width - cropped_width - left,
        height - cropped_height - top,
    )


def _best_effort_canvas(
    source_width,
    source_height,
    target_aspect,
    generation_megapixels,
    minimum_source_megapixels,
    max_upscale,
):
    if source_width % CANVAS_MULTIPLE or source_height % CANVAS_MULTIPLE:
        raise ValueError(
            f"H3 source geometry must be divisible by {CANVAS_MULTIPLE}, got "
            f"{source_width}x{source_height}."
        )
    width = source_width
    height = source_height
    if target_aspect == "source":
        target_ratio = source_width / source_height
    elif target_aspect == "9:12 portrait":
        target_ratio = 9.0 / 12.0
    else:
        raise ValueError(f"Unknown target aspect: {target_aspect}")

    candidates = [(width, height)]
    if width / height > target_ratio:
        target_height = _align_spatial(math.ceil(width / target_ratio))
        candidates.extend(
            (width, candidate_height)
            for candidate_height in range(
                height + CANVAS_MULTIPLE, target_height + 1, CANVAS_MULTIPLE
            )
        )
    elif width / height < target_ratio:
        target_width = _align_spatial(math.ceil(height * target_ratio))
        candidates.extend(
            (candidate_width, height)
            for candidate_width in range(
                width + CANVAS_MULTIPLE, target_width + 1, CANVAS_MULTIPLE
            )
        )

    allowed = []
    for candidate_width, candidate_height in candidates:
        if (candidate_width - source_width) % (2 * CANVAS_MULTIPLE) or (
            candidate_height - source_height
        ) % (2 * CANVAS_MULTIPLE):
            continue
        model_width, model_height = _model_canvas(
            candidate_width,
            candidate_height,
            generation_megapixels,
        )
        resize = max(
            candidate_width / model_width,
            candidate_height / model_height,
        )
        model_source_width, model_source_height, *_ = _scaled_source_geometry(
            source_width,
            source_height,
            (candidate_width - source_width) // 2,
            (candidate_height - source_height) // 2,
            candidate_width,
            candidate_height,
            model_width,
            model_height,
        )
        source_megapixels = model_source_width * model_source_height / 1_000_000
        if (
            max_upscale <= 0 or resize <= max_upscale
        ) and source_megapixels >= minimum_source_megapixels:
            allowed.append((candidate_width, candidate_height))

    if allowed:
        width, height = min(
            allowed,
            key=lambda size: (
                abs(size[0] / size[1] - target_ratio),
                size[0] * size[1],
            ),
        )

    left = (width - source_width) // 2
    top = (height - source_height) // 2
    return (
        width,
        height,
        left,
        top,
        width - source_width - left,
        height - source_height - top,
    )


def _model_canvas(width, height, generation_megapixels):
    max_pixels = round(float(generation_megapixels) * 1_000_000)
    if max_pixels <= 0 or width * height <= max_pixels:
        return width, height

    scale = math.sqrt(max_pixels / float(width * height))
    model_width = max(
        CANVAS_MULTIPLE, math.floor(width * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE
    )
    model_height = max(
        CANVAS_MULTIPLE, math.floor(height * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE
    )
    while model_width * model_height > max_pixels:
        if (
            model_width / width >= model_height / height
            and model_width > CANVAS_MULTIPLE
        ):
            model_width -= CANVAS_MULTIPLE
        elif model_height > CANVAS_MULTIPLE:
            model_height -= CANVAS_MULTIPLE
        else:
            break
    return model_width, model_height


def _scaled_source_geometry(
    source_width,
    source_height,
    source_x,
    source_y,
    canvas_width,
    canvas_height,
    model_width,
    model_height,
):
    latent_multiple = CANVAS_MULTIPLE // 2
    left = (
        round(source_x * model_width / canvas_width / latent_multiple) * latent_multiple
    )
    top = (
        round(source_y * model_height / canvas_height / latent_multiple)
        * latent_multiple
    )
    right = (
        round((source_x + source_width) * model_width / canvas_width / latent_multiple)
        * latent_multiple
    )
    bottom = (
        round(
            (source_y + source_height) * model_height / canvas_height / latent_multiple
        )
        * latent_multiple
    )
    left = min(max(0, left), model_width - CANVAS_MULTIPLE)
    top = min(max(0, top), model_height - CANVAS_MULTIPLE)
    right = max(left + CANVAS_MULTIPLE, min(model_width, right))
    bottom = max(top + CANVAS_MULTIPLE, min(model_height, bottom))
    return (
        right - left,
        bottom - top,
        left,
        top,
        model_width - right,
        model_height - bottom,
    )


def _resize_frames(frames, width, height):
    if frames.shape[1] == height and frames.shape[2] == width:
        return frames[..., :3]
    return comfy.utils.common_upscale(
        frames[..., :3].movedim(-1, 1),
        width,
        height,
        "lanczos",
        "disabled",
    ).movedim(1, -1)


def _iter_video_frames(video, skip_first_frames, frame_load_cap, crop=None):
    source = video.get_stream_source()
    if isinstance(source, io.BytesIO):
        source.seek(0)
    start_time, duration = video.get_active_trim_window()
    with av.open(source, mode="r") as container:
        if not container.streams.video:
            raise ValueError("The source contains no video stream.")
        stream = container.streams.video[0]
        start_pts = int(start_time / stream.time_base)
        end_pts = int((start_time + duration) / stream.time_base) if duration else None
        if start_pts:
            container.seek(start_pts, stream=stream)

        skipped = 0
        emitted = 0
        for frame in container.decode(stream):
            if frame.pts is not None and frame.pts < start_pts:
                continue
            if end_pts is not None and frame.pts is not None and frame.pts >= end_pts:
                break
            if skipped < skip_first_frames:
                skipped += 1
                continue
            if frame_load_cap and emitted >= frame_load_cap:
                break
            emitted += 1
            image = frame.to_ndarray(format="rgb24")
            if crop is not None:
                left, top, width, height = crop
                image = image[top : top + height, left : left + width]
            yield image


def _scan_video_frames(video, skip_first_frames, frame_load_cap, crop=None):
    first = None
    last = None
    count = 0
    for frame in _iter_video_frames(
        video, skip_first_frames, frame_load_cap, crop=crop
    ):
        if first is None:
            first = torch.from_numpy(frame).to(torch.float32).div_(255.0)
        last = torch.from_numpy(frame).to(torch.float32).div_(255.0)
        count += 1
    if first is None:
        raise ValueError("The selected source video contains no frames.")
    return first.unsqueeze(0), last.unsqueeze(0), count


def _auto_denoise_window_frames(width, height):
    pixels = width * height
    eligible = [
        frame_count
        for frame_count in AUTO_DENOISE_WINDOW_FRAMES
        if frame_count * pixels <= WINDOW_PIXEL_BUDGET
    ]
    if not eligible:
        raise ValueError(
            f"Native H3 canvas {width}x{height} exceeds the automatic "
            "temporal-window budget."
        )
    return max(eligible)


def _global_window_starts(frame_count, window_frames):
    if frame_count == window_frames:
        return [0]
    if window_frames <= WINDOW_STRIDE_FRAMES:
        raise ValueError(
            f"H3 window size {window_frames} cannot preserve temporal overlap."
        )
    final_start = frame_count - window_frames
    if final_start < 0 or final_start % CONTEXT_FRAMES:
        raise ValueError(
            f"H3 global frame count {frame_count} cannot be covered by "
            f"{window_frames}-frame phase-aligned windows."
        )
    return sorted({0, final_start, *range(0, final_start + 1, WINDOW_STRIDE_FRAMES)})


def _global_window_specs(frame_count, window_frames):
    _, window_video_t, window_audio_t = temporal_shape(window_frames)
    _, _, global_audio_t = temporal_shape(frame_count)
    final_start = frame_count - window_frames
    specs = []
    for frame_start in _global_window_starts(frame_count, window_frames):
        video_start = frame_start // CONTEXT_FRAMES * CONTEXT_LATENT_T
        audio_start = (
            global_audio_t - window_audio_t
            if frame_start == final_start
            else round(frame_start / FPS * AUDIO_LATENT_FPS)
        )
        specs.append((frame_start, video_start, audio_start))
    return specs, window_video_t, window_audio_t


def _inpaint_canvas(frames, left, top, right, bottom):
    height = int(frames.shape[1]) + top + bottom
    width = int(frames.shape[2]) + left + right
    canvas = frames.new_full((frames.shape[0], height, width, 3), 0.5)
    canvas[
        :,
        top : top + frames.shape[1],
        left : left + frames.shape[2],
    ].copy_(frames[..., :3])
    return canvas


def _spatial_generation_mask(
    latent_h,
    latent_w,
    canvas_h,
    canvas_w,
    source_h,
    source_w,
    left,
    top,
    right,
    bottom,
    device,
):
    ys = (torch.arange(latent_h, device=device, dtype=torch.float32) + 0.5) * (
        canvas_h / latent_h
    )
    xs = (torch.arange(latent_w, device=device, dtype=torch.float32) + 0.5) * (
        canvas_w / latent_w
    )
    inside_y = (ys >= top) & (ys < top + source_h)
    inside_x = (xs >= left) & (xs < left + source_w)
    inside = inside_y[:, None] & inside_x[None, :]
    mask = (~inside).to(torch.float32)

    def collar(distance):
        return (distance < SOURCE_CONTEXT).to(torch.float32)

    if top:
        mask = torch.maximum(mask, collar(ys - top)[:, None] * inside)
    if bottom:
        mask = torch.maximum(mask, collar(top + source_h - ys)[:, None] * inside)
    if left:
        mask = torch.maximum(mask, collar(xs - left)[None, :] * inside)
    if right:
        mask = torch.maximum(mask, collar(left + source_w - xs)[None, :] * inside)
    return mask[None, None]


def _observed_video_tokens(frame_count, latent_t):
    observed_frames = 0
    for token_index in range(latent_t):
        span = comfy.ldm.minimax.model.FRAME_PER_TOKEN[
            token_index % len(comfy.ldm.minimax.model.FRAME_PER_TOKEN)
        ]
        if observed_frames + span > frame_count:
            return token_index
        observed_frames += span
    return latent_t


def _conditioning_with_keyframes(conditioning, keyframes):
    conditioned = []
    for cross_attn, options in conditioning:
        options = options.copy()
        options["minimax_keyframes"] = [
            *options.get("minimax_keyframes", ()),
            *keyframes,
        ]
        conditioned.append([cross_attn, options])
    return conditioned


def _load_aligned_source_frames(
    video, skip_first_frames, frame_load_cap, frame_count, width, height, crop=None
):
    aligned_count, _, _ = temporal_shape(frame_count)
    frames = torch.empty(
        (aligned_count, height, width, 3), dtype=torch.uint8, device="cpu"
    )
    iterator = _iter_video_frames(video, skip_first_frames, frame_load_cap, crop=crop)
    for index in range(frame_count):
        try:
            frame = next(iterator)
        except StopIteration as error:
            raise RuntimeError(
                f"Video decoding ended after {index} of {frame_count} requested frames."
            ) from error
        if frame.shape[:2] != (height, width):
            raise ValueError(
                f"Video frame changed dimensions from {width}x{height} to "
                f"{frame.shape[1]}x{frame.shape[0]}."
            )
        frames[index].copy_(torch.from_numpy(frame))
    if aligned_count > frame_count:
        frames[frame_count:].copy_(frames[frame_count - 1])
    return frames


def _assemble_global_latent(
    source_frames,
    source_frame_count,
    video_vae,
    left,
    top,
    right,
    bottom,
    spatial_mask,
    denoise_window_frames,
):
    aligned_count = int(source_frames.shape[0])
    _, global_video_t, global_audio_t = temporal_shape(aligned_count)
    specs, window_video_t, window_audio_t = _global_window_specs(
        aligned_count, denoise_window_frames
    )

    latent_height = (source_frames.shape[1] + top + bottom) // 16
    latent_width = (source_frames.shape[2] + left + right) // 16
    video_shape = (1, 24, global_video_t, latent_height, latent_width)
    audio_shape = (1, 32, 2, global_audio_t)
    window_shapes = [
        (1, 24, window_video_t, latent_height, latent_width),
        (1, 32, 2, window_audio_t),
    ]
    accumulate_device = comfy.model_management.intermediate_device()

    canvas_frames = _inpaint_canvas(
        source_frames.to(torch.float32).div_(255.0),
        left,
        top,
        right,
        bottom,
    )
    video = video_vae.encode(canvas_frames)
    if tuple(video.shape) != video_shape:
        raise RuntimeError(
            f"H3 VAE produced {tuple(video.shape)}, expected {video_shape}."
        )
    video = video.to(accumulate_device)
    del canvas_frames

    observed_video_t = _observed_video_tokens(source_frame_count, global_video_t)
    source_video_shape = (
        1,
        24,
        global_video_t,
        source_frames.shape[1] // 16,
        source_frames.shape[2] // 16,
    )
    source_input = source_frames.to(torch.float32).div_(255.0)
    source_video = video_vae.encode(source_input)
    if tuple(source_video.shape) != source_video_shape:
        raise RuntimeError(
            f"H3 source VAE produced {tuple(source_video.shape)}, "
            f"expected {source_video_shape}."
        )
    source_video = source_video.to(accumulate_device)
    del source_input

    audio = torch.zeros(audio_shape, dtype=torch.float32, device=accumulate_device)
    video_mask = (
        spatial_mask.to(accumulate_device)
        .unsqueeze(2)
        .expand(1, 1, global_video_t, -1, -1)
        .clone()
    )
    video_mask[:, :, observed_video_t:] = 1.0
    audio_mask = torch.ones(
        (1, 1, 2, global_audio_t), dtype=torch.float32, device=accumulate_device
    )
    latent = {
        "samples": comfy.nested_tensor.NestedTensor((video, audio)),
        "noise_mask": comfy.nested_tensor.NestedTensor((video_mask, audio_mask)),
    }
    return (
        latent,
        source_video,
        window_shapes,
        specs,
        observed_video_t,
        top // 16,
        left // 16,
    )


def _sample_sliding_latent(
    model,
    positive,
    negative,
    latent,
    source_video,
    window_shapes,
    window_specs,
    observed_video_t,
    source_latent_y,
    source_latent_x,
    seed,
    steps,
    sampler_name,
    scheduler,
):
    global_video, global_audio = latent["samples"].unbind()
    global_video_mask, global_audio_mask = latent["noise_mask"].unbind()
    global_video_noise, global_audio_noise = comfy.sample.prepare_noise(
        latent["samples"], seed
    ).unbind()
    window_video_t = window_shapes[0][2]
    window_audio_t = window_shapes[1][3]
    committed_video_stop = 0
    committed_audio_stop = 0

    for _, video_start, audio_start in window_specs:
        video_stop = video_start + window_video_t
        audio_stop = audio_start + window_audio_t
        if video_start > committed_video_stop or audio_start > committed_audio_stop:
            raise RuntimeError(
                "H3 sliding windows left a gap in the latent trajectory."
            )

        video_overlap = committed_video_stop - video_start
        audio_overlap = committed_audio_stop - audio_start
        window_video = global_video[:, :, video_start:video_stop].contiguous()
        window_audio = global_audio[..., audio_start:audio_stop].contiguous()
        window_video_mask = global_video_mask[:, :, video_start:video_stop].contiguous()
        window_audio_mask = global_audio_mask[..., audio_start:audio_stop].contiguous()
        window_video_mask[:, :, :video_overlap] = 0.0
        window_audio_mask[..., :audio_overlap] = 0.0

        keyframes = []
        if video_overlap:
            keyframes.append(
                {
                    "resolved_frame_index": 0,
                    "latent": global_video[
                        :, :, video_start:committed_video_stop
                    ].contiguous(),
                    "audio_latent": global_audio[
                        ..., audio_start:committed_audio_stop
                    ].contiguous(),
                }
            )
        source_stop = min(video_stop, observed_video_t)
        if source_stop > video_start:
            keyframes.append(
                {
                    "resolved_frame_index": 0,
                    "latent": source_video[:, :, video_start:source_stop].contiguous(),
                    "latent_y": source_latent_y,
                    "latent_x": source_latent_x,
                }
            )

        window_latent = comfy.nested_tensor.NestedTensor((window_video, window_audio))
        window_mask = comfy.nested_tensor.NestedTensor(
            (window_video_mask, window_audio_mask)
        )
        window_noise = comfy.nested_tensor.NestedTensor(
            (
                global_video_noise[:, :, video_start:video_stop].contiguous(),
                global_audio_noise[..., audio_start:audio_stop].contiguous(),
            )
        )
        window_positive = _conditioning_with_keyframes(positive, keyframes)
        window_negative = _conditioning_with_keyframes(negative, keyframes)
        sampled = comfy.sample.sample(
            model,
            window_noise,
            steps,
            1.0,
            sampler_name,
            scheduler,
            window_positive,
            window_negative,
            window_latent,
            denoise=1.0,
            noise_mask=window_mask,
            disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED,
            seed=seed,
        )
        sampled_video, sampled_audio = sampled.unbind()

        video_commit_start = max(video_start, committed_video_stop)
        audio_commit_start = max(audio_start, committed_audio_stop)
        global_video[:, :, video_commit_start:video_stop].copy_(
            sampled_video[:, :, video_commit_start - video_start :]
        )
        global_audio[..., audio_commit_start:audio_stop].copy_(
            sampled_audio[..., audio_commit_start - audio_start :]
        )
        committed_video_stop = video_stop
        committed_audio_stop = audio_stop

    if (
        committed_video_stop != global_video.shape[2]
        or committed_audio_stop != global_audio.shape[3]
    ):
        raise RuntimeError("H3 sliding windows did not complete the latent trajectory.")
    return {"samples": comfy.nested_tensor.NestedTensor((global_video, global_audio))}


class _StreamingH3Video(Input.Video):
    def __init__(
        self,
        source_video,
        model,
        visual_conditioning,
        video_vae,
        skip_first_frames,
        frame_load_cap,
        target_aspect,
        generation_megapixels,
        minimum_source_megapixels,
        max_upscale,
        seed,
        steps,
        sampler_name,
        scheduler,
        temporal_window_frames="auto",
    ):
        self.source_video = source_video
        self.model = model.clone()
        self.visual_conditioning = visual_conditioning
        self.video_vae = video_vae
        self.skip_first_frames = int(skip_first_frames)
        self.frame_load_cap = int(frame_load_cap)
        self.target_aspect = target_aspect
        self.generation_megapixels = float(generation_megapixels)
        self.minimum_source_megapixels = float(minimum_source_megapixels)
        self.max_upscale = float(max_upscale)
        self.seed = int(seed)
        self.steps = int(steps)
        self.sampler_name = sampler_name
        self.scheduler = scheduler

        self.input_width, self.input_height = source_video.get_dimensions()
        (
            self.source_width,
            self.source_height,
            self.crop_left,
            self.crop_top,
            self.crop_right,
            self.crop_bottom,
        ) = _aligned_crop_geometry(self.input_width, self.input_height)
        source_count = source_video.get_frame_count() - self.skip_first_frames
        if self.frame_load_cap:
            source_count = min(source_count, self.frame_load_cap)
        if source_count < 1:
            raise ValueError("The selected source video contains no frames.")
        self.source_frame_count = source_count
        self.source_frame_rate = Fraction(source_video.get_frame_rate())
        self.frame_count = temporal_shape(source_count)[0]
        self.frame_rate = self.source_frame_rate
        (
            self.width,
            self.height,
            self.left,
            self.top,
            self.right,
            self.bottom,
        ) = _best_effort_canvas(
            self.source_width,
            self.source_height,
            self.target_aspect,
            self.generation_megapixels,
            self.minimum_source_megapixels,
            self.max_upscale,
        )
        self.model_width = self.width
        self.model_height = self.height
        self.generated_upscale = 1.0
        if temporal_window_frames == "auto":
            self.denoise_window_frames = _auto_denoise_window_frames(
                self.model_width, self.model_height
            )
        elif temporal_window_frames == "global":
            self.denoise_window_frames = None
        else:
            self.denoise_window_frames = int(temporal_window_frames)
            if self.denoise_window_frames not in DENOISE_WINDOW_FRAMES:
                raise ValueError(
                    "Unsupported H3 temporal window: "
                    f"{self.denoise_window_frames} frames."
                )
        self.model_source_width = self.source_width
        self.model_source_height = self.source_height
        self.model_left = self.left
        self.model_top = self.top
        self.model_right = self.right
        self.model_bottom = self.bottom
        self.model_source_megapixels = (
            self.model_source_width * self.model_source_height / 1_000_000
        )

    def prepare_visual_conditioning(self, clip):
        first_frame, last_frame, actual_count = _scan_video_frames(
            self.source_video,
            self.skip_first_frames,
            self.frame_load_cap,
            crop=(
                self.crop_left,
                self.crop_top,
                self.source_width,
                self.source_height,
            ),
        )
        self.source_frame_count = actual_count
        self.frame_count = temporal_shape(actual_count)[0]
        conditioning_images = [
            _resize_frames(
                first_frame,
                self.model_source_width,
                self.model_source_height,
            )
        ]
        if actual_count > 1:
            conditioning_images.append(
                _resize_frames(
                    last_frame,
                    self.model_source_width,
                    self.model_source_height,
                )
            )
        self.visual_conditioning = clip.encode_from_tokens_scheduled(
            clip.tokenize("", images=conditioning_images)
        )

    def get_components(self):
        raise RuntimeError(
            "Streaming H3 outpaint must be connected directly to Save Video."
        )

    def get_dimensions(self):
        return self.width, self.height

    def get_bit_depth(self):
        return 8

    def get_duration(self):
        return self.frame_count / float(self.frame_rate)

    def get_frame_count(self):
        return self.frame_count

    def get_frame_rate(self):
        return self.frame_rate

    def as_trimmed(
        self,
        start_time=None,
        duration=None,
        strict_duration=False,
    ):
        return None

    def _encode_output_frame(self, output, stream, frame):
        image = (
            frame[..., :3]
            .mul(255)
            .clamp(0, 255)
            .to(device="cpu", dtype=torch.uint8)
            .numpy()
        )
        video_frame = av.VideoFrame.from_ndarray(image, format="rgb24")
        for packet in stream.encode(video_frame):
            output.mux(packet)

    def save_to(
        self,
        path,
        format=Types.VideoContainer.AUTO,
        codec=Types.VideoCodec.AUTO,
        metadata=None,
        bit_depth=None,
        crf=None,
    ):
        format_value = getattr(format, "value", format)
        codec_value = getattr(codec, "value", codec)
        if format_value not in ("auto", "mp4"):
            raise ValueError("Streaming H3 outpaint currently saves MP4 only.")
        if codec_value not in ("auto", "h264"):
            raise ValueError("Streaming H3 outpaint currently encodes H.264 only.")
        if bit_depth is not None and bit_depth > 8:
            raise ValueError(
                "Streaming H3 outpaint currently encodes 8-bit video only."
            )

        actual_count = self.source_frame_count
        source_frames = _load_aligned_source_frames(
            self.source_video,
            self.skip_first_frames,
            self.frame_load_cap,
            actual_count,
            self.source_width,
            self.source_height,
            crop=(
                self.crop_left,
                self.crop_top,
                self.source_width,
                self.source_height,
            ),
        )
        aligned_count = int(source_frames.shape[0])
        window_frames = (
            aligned_count
            if self.denoise_window_frames is None
            else min(self.denoise_window_frames, aligned_count)
        )
        spatial_mask = _spatial_generation_mask(
            self.model_height // 16,
            self.model_width // 16,
            self.model_height,
            self.model_width,
            self.model_source_height,
            self.model_source_width,
            self.model_left,
            self.model_top,
            self.model_right,
            self.model_bottom,
            comfy.model_management.intermediate_device(),
        )
        (
            latent,
            source_video,
            window_shapes,
            window_specs,
            observed_video_t,
            source_latent_y,
            source_latent_x,
        ) = _assemble_global_latent(
            source_frames,
            actual_count,
            self.video_vae,
            self.model_left,
            self.model_top,
            self.model_right,
            self.model_bottom,
            spatial_mask,
            window_frames,
        )
        del source_frames
        sampled = _sample_sliding_latent(
            self.model,
            self.visual_conditioning,
            self.visual_conditioning,
            latent,
            source_video,
            window_shapes,
            window_specs,
            observed_video_t,
            source_latent_y,
            source_latent_x,
            self.seed,
            self.steps,
            self.sampler_name,
            self.scheduler,
        )
        sampled_video = sampled["samples"].unbind()[0]
        comfy.model_management.unload_all_models()
        self.model = None
        self.visual_conditioning = None
        del sampled, latent, source_video, spatial_mask
        gc.collect()
        comfy.model_management.soft_empty_cache()

        decoded_frames = self.video_vae.decode(sampled_video)
        if decoded_frames.ndim == 5:
            if decoded_frames.shape[0] != 1:
                raise RuntimeError(
                    f"H3 VAE decoded batch {decoded_frames.shape[0]}; expected 1."
                )
            decoded_frames = decoded_frames[0]
        if decoded_frames.shape[0] != aligned_count:
            raise RuntimeError(
                f"H3 VAE decoded {decoded_frames.shape[0]} frames; "
                f"expected {aligned_count}."
            )
        decoded_frames = decoded_frames.to(device="cpu", dtype=torch.float32)
        del sampled_video

        open_options = {
            "mode": "w",
            "options": {
                "movflags": (
                    "use_metadata_tags+faststart"
                    if isinstance(path, (str, os.PathLike))
                    else "use_metadata_tags"
                )
            },
        }
        if isinstance(path, io.BytesIO):
            open_options["format"] = "mp4"

        encoded_count = 0
        with av.open(path, **open_options) as output:
            if metadata:
                for key, value in metadata.items():
                    output.metadata[key] = (
                        value if isinstance(value, str) else json.dumps(value)
                    )
            output_stream = output.add_stream("h264", rate=self.frame_rate)
            output_stream.width = self.width
            output_stream.height = self.height
            output_stream.pix_fmt = "yuv420p"
            if crf is not None:
                output_stream.options = {"crf": str(crf)}

            for generated_frame in decoded_frames:
                self._encode_output_frame(output, output_stream, generated_frame)
                encoded_count += 1

            for packet in output_stream.encode(None):
                output.mux(packet)

        if encoded_count != aligned_count:
            raise RuntimeError(
                f"Encoded {encoded_count} frames for an "
                f"{aligned_count}-frame H3 trajectory."
            )


class MiniMaxH3SimpleVideoOutpaint:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "video_vae": ("VAE",),
                "source_video": ("VIDEO",),
                "skip_first_frames": (
                    "INT",
                    {"default": 0, "min": 0, "max": 999999, "step": 1},
                ),
                "frame_load_cap": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 999999,
                        "step": 1,
                        "tooltip": "Maximum source frames; 0 processes the remaining video.",
                    },
                ),
                "target_aspect": (
                    ["9:12 portrait", "source"],
                    {"default": "9:12 portrait"},
                ),
                "generation_megapixels": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 64.0,
                        "step": 0.1,
                        "tooltip": "H3 sampling canvas budget; 0 samples at full output resolution.",
                    },
                ),
                "minimum_source_megapixels": (
                    "FLOAT",
                    {
                        "default": 0.7,
                        "min": 0.0,
                        "max": 64.0,
                        "step": 0.05,
                        "tooltip": "Minimum H3 canvas area reserved for the native source. The target aspect stops before source context falls below this quality floor.",
                    },
                ),
                "max_upscale": (
                    "FLOAT",
                    {
                        "default": 1.5,
                        "min": 1.0,
                        "max": 8.0,
                        "step": 0.05,
                        "tooltip": "Maximum linear resize from the H3 canvas to the native output. The target aspect is best effort within this limit.",
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "control_after_generate": True,
                    },
                ),
                "steps": ("INT", {"default": 20, "min": 1, "max": 100}),
                "sampler_name": (
                    comfy.samplers.KSampler.SAMPLERS,
                    {"default": "res_multistep"},
                ),
                "scheduler": (
                    comfy.samplers.KSampler.SCHEDULERS,
                    {"default": "simple"},
                ),
            },
            "optional": {
                "temporal_window_frames": (
                    ["auto", "107", "124", "158", "192", "global"],
                    {
                        "default": "auto",
                        "tooltip": "Transformer context per denoiser call. Auto stays at or below the validated 107-frame policy; manual values are unchanged. Matching validation found expanded-region artifacts at 192 for the cc58 source.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("VIDEO", "INT", "INT", "FLOAT", "INT", "STRING")
    RETURN_NAMES = ("video", "width", "height", "fps", "length", "info")
    FUNCTION = "outpaint"
    CATEGORY = "MiniMax H3/Outpaint"
    DESCRIPTION = "Prompt-free H3 outpainting with phase-stable temporal continuation."

    def outpaint(
        self,
        model,
        clip,
        video_vae,
        source_video,
        skip_first_frames,
        frame_load_cap,
        target_aspect,
        generation_megapixels,
        minimum_source_megapixels,
        max_upscale,
        seed,
        steps,
        sampler_name,
        scheduler,
        temporal_window_frames="auto",
    ):
        video = _StreamingH3Video(
            source_video=source_video,
            model=model,
            visual_conditioning=None,
            video_vae=video_vae,
            skip_first_frames=skip_first_frames,
            frame_load_cap=frame_load_cap,
            target_aspect=target_aspect,
            generation_megapixels=generation_megapixels,
            minimum_source_megapixels=minimum_source_megapixels,
            max_upscale=max_upscale,
            seed=seed,
            steps=steps,
            sampler_name=sampler_name,
            scheduler=scheduler,
            temporal_window_frames=temporal_window_frames,
        )
        video.prepare_visual_conditioning(clip)
        denoise_window = (
            "global"
            if video.denoise_window_frames is None
            else video.denoise_window_frames
        )
        info = (
            f"Sliding H3 outpaint: {video.input_width}x{video.input_height} cropped to "
            f"{video.source_width}x{video.source_height} -> {video.width}x{video.height}; "
            f"{video.source_frame_count} source frames -> "
            f"{video.frame_count} H3-aligned output frames; "
            f"internal {FPS} fps, delivered at {float(video.source_frame_rate):g} fps; "
            f"H3 canvas={video.model_width}x{video.model_height}, "
            f"source={video.model_source_megapixels:.3f} MP "
            f"(floor {float(minimum_source_megapixels):g} MP), "
            f"resize={video.generated_upscale:.3f}x (limit {float(max_upscale):g}x), "
            f"denoiser window={denoise_window} frames. "
            "Each completed overlap conditions the next window; Save Video encodes "
            "the assembled latent once."
        )
        return (
            video,
            video.width,
            video.height,
            float(video.frame_rate),
            video.frame_count,
            info,
        )


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3SimpleVideoOutpaint": MiniMaxH3SimpleVideoOutpaint,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3SimpleVideoOutpaint": "MiniMax H3 Video Outpaint (Prompt Free)",
}
