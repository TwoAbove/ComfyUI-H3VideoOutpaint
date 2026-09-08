"""Training-free latent-masked video outpainting for MiniMax H3 in ComfyUI."""

import gc
import io
import json
import math
import os
from fractions import Fraction

import av
import torch
import torchaudio

import comfy.ldm.minimax.model
import comfy.model_management
import comfy.sample
import comfy.utils
import comfy.nested_tensor
import comfy.samplers
import node_helpers
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
AUDIO_SAMPLE_RATE = 32000
DENOISE_WINDOW_FRAMES = (56, 73, 90, 107, 124, 141, 158, 175, 192)
LATENT_MULTIPLE = CANVAS_MULTIPLE // 2
AUTO_DENOISE_WINDOW_FRAMES = DENOISE_WINDOW_FRAMES[:4]
SEAM_TONE_ROWS = 4


def _align_spatial(value):
    return max(CANVAS_MULTIPLE, math.ceil(value / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)


def _split_band(total):
    near = total // 2 // LATENT_MULTIPLE * LATENT_MULTIPLE
    return near, total - near


def _best_effort_canvas(
    input_width,
    input_height,
    target_aspect,
    generation_megapixels,
    minimum_source_megapixels,
    max_upscale,
):
    # The source is cropped only as far as the grids force it: the axis that
    # gains bands keeps every row down to the 16 px latent grid, the fixed
    # axis has to sit on the 32 px canvas grid.
    source_width = input_width // LATENT_MULTIPLE * LATENT_MULTIPLE
    source_height = input_height // LATENT_MULTIPLE * LATENT_MULTIPLE
    if source_width < CANVAS_MULTIPLE or source_height < CANVAS_MULTIPLE:
        raise ValueError(
            f"Source video {input_width}x{input_height} is smaller than one "
            f"{CANVAS_MULTIPLE}-pixel H3 spatial chunk."
        )
    if target_aspect == "source":
        target_ratio = source_width / source_height
    elif target_aspect == "9:12 portrait":
        target_ratio = 9.0 / 12.0
    else:
        raise ValueError(f"Unknown target aspect: {target_aspect}")
    expand_height = source_width / source_height > target_ratio
    expand_width = source_width / source_height < target_ratio
    if not expand_width:
        source_width = source_width // CANVAS_MULTIPLE * CANVAS_MULTIPLE
    if not expand_height:
        source_height = source_height // CANVAS_MULTIPLE * CANVAS_MULTIPLE
    width = source_width
    height = source_height

    candidates = [(width, height)]
    if expand_height:
        target_height = _align_spatial(math.ceil(width / target_ratio))
        candidates.extend(
            (width, candidate_height)
            for candidate_height in range(
                _align_spatial(height + 2 * CANVAS_MULTIPLE),
                target_height + 1,
                CANVAS_MULTIPLE,
            )
        )
    elif expand_width:
        target_width = _align_spatial(math.ceil(height * target_ratio))
        candidates.extend(
            (candidate_width, height)
            for candidate_width in range(
                _align_spatial(width + 2 * CANVAS_MULTIPLE),
                target_width + 1,
                CANVAS_MULTIPLE,
            )
        )

    allowed = []
    for candidate_width, candidate_height in candidates:
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
            _split_band(candidate_width - source_width)[0],
            _split_band(candidate_height - source_height)[0],
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

    left, right = _split_band(width - source_width)
    top, bottom = _split_band(height - source_height)
    return width, height, source_width, source_height, left, top, right, bottom


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
    left = (
        round(source_x * model_width / canvas_width / LATENT_MULTIPLE) * LATENT_MULTIPLE
    )
    top = (
        round(source_y * model_height / canvas_height / LATENT_MULTIPLE)
        * LATENT_MULTIPLE
    )
    right = (
        round((source_x + source_width) * model_width / canvas_width / LATENT_MULTIPLE)
        * LATENT_MULTIPLE
    )
    bottom = (
        round(
            (source_y + source_height) * model_height / canvas_height / LATENT_MULTIPLE
        )
        * LATENT_MULTIPLE
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


def _load_source_audio(video, start_time, duration, sample_rate):
    source = video.get_stream_source()
    if isinstance(source, io.BytesIO):
        source.seek(0)
    sample_count = round(duration * sample_rate)
    with av.open(source, mode="r") as container:
        stream = next(
            (
                stream
                for stream in reversed(container.streams.audio)
                if stream.codec_context is not None
            ),
            None,
        )
        if stream is None:
            return None

        waveform = torch.zeros((1, 2, sample_count), dtype=torch.float32)
        resampler = av.AudioResampler(
            format="fltp",
            layout="stereo",
            rate=sample_rate,
        )
        start_pts = int(start_time / stream.time_base)
        if start_pts:
            container.seek(start_pts, stream=stream)
        cursor = 0

        def copy_frame(frame):
            nonlocal cursor
            if frame.pts is not None:
                cursor = round(
                    (float(frame.pts * frame.time_base) - start_time)
                    * sample_rate
                )
            samples = torch.from_numpy(frame.to_ndarray())
            source_start = max(0, -cursor)
            target_start = max(0, cursor)
            length = min(
                samples.shape[-1] - source_start,
                sample_count - target_start,
            )
            if length > 0:
                waveform[
                    0,
                    :,
                    target_start : target_start + length,
                ].copy_(
                    samples[
                        :,
                        source_start : source_start + length,
                    ]
                )
            cursor += samples.shape[-1]

        for packet in container.demux(stream):
            for decoded in packet.decode():
                for frame in resampler.resample(decoded):
                    copy_frame(frame)
            if cursor >= sample_count:
                break
        for frame in resampler.resample(None):
            copy_frame(frame)
        return waveform


def _fit_waveform(waveform, sample_count):
    if waveform.shape[-1] == sample_count:
        return waveform
    fitted = waveform.new_zeros((*waveform.shape[:-1], sample_count))
    copied = min(waveform.shape[-1], sample_count)
    fitted[..., :copied].copy_(waveform[..., :copied])
    return fitted


def _audio_to_model_timeline(waveform, frame_rate):
    if frame_rate == FPS:
        return waveform
    return torchaudio.functional.resample(
        waveform,
        FPS * frame_rate.denominator,
        frame_rate.numerator,
    )


def _audio_from_model_timeline(waveform, frame_rate):
    if frame_rate == FPS:
        return waveform
    return torchaudio.functional.resample(
        waveform,
        frame_rate.numerator,
        FPS * frame_rate.denominator,
    )


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


def _count_video_frames(video, skip_first_frames, frame_load_cap):
    count = sum(
        1
        for _ in _iter_video_frames(video, skip_first_frames, frame_load_cap)
    )
    if count == 0:
        raise ValueError("The selected source video contains no frames.")
    return count


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
    return sorted(
        {0, final_start, *range(0, final_start + 1, WINDOW_STRIDE_FRAMES)}
    )


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


def _pinned_span(offset, source_tokens, latent_tokens):
    # The encoder folds whatever lies past the source edge into the outermost
    # latent row, so pinning a band-facing edge row makes the model render that
    # context (a border, or replicated rows). Leave it free; its real pixels
    # are feathered back after decode.
    start = offset + (offset > 0)
    stop = offset + source_tokens - (offset + source_tokens < latent_tokens)
    return start, stop


def _spatial_generation_mask(latent_h, latent_w, source_h, source_w, left, top, device):
    mask = torch.ones((1, 1, latent_h, latent_w), dtype=torch.float32, device=device)
    y0, y1 = _pinned_span(top // 16, source_h // 16, latent_h)
    x0, x1 = _pinned_span(left // 16, source_w // 16, latent_w)
    mask[:, :, y0:y1, x0:x1] = 0.0
    return mask


def _seam_offset(generated, source):
    # One RGB correction per frame; column-wise offsets imprint seam detail
    # through the band as stripes.
    return generated.mean((1, 2), keepdim=True) - source.mean((1, 2), keepdim=True)


def _match_band_tone(frames, source, start, stop, free_start, free_stop, axis):
    # The model renders each band with its own tone. Measure the step between
    # the real source rows and the generated rows across the seam and ramp it
    # out of the band; the free rows are corrected in full since the feathered
    # composite hands over to them.
    f = frames.movedim(axis, 1)
    s = source.movedim(axis, 1)
    rows = SEAM_TONE_ROWS
    if start > 0:
        offset = _seam_offset(f[:, start - rows : start], s[:, :rows])
        ramp = torch.cat(
            (torch.arange(1, start + 1, dtype=f.dtype).div_(start), torch.ones(free_start))
        )
        f[:, : start + free_start].sub_(offset * ramp.view(1, -1, 1, 1))
    tail = f.shape[1] - stop
    if tail > 0:
        offset = _seam_offset(f[:, stop : stop + rows], s[:, -rows:])
        ramp = torch.cat(
            (torch.ones(free_stop), torch.arange(tail, 0, -1, dtype=f.dtype).div_(tail))
        )
        f[:, stop - free_stop :].sub_(offset * ramp.view(1, -1, 1, 1))


def _source_weight(height, width, top, bottom, left, right):
    # Real source pixels replace the decode, except on the free rows facing a
    # band, where they fade into the decode so the seam stays a single
    # decoder-continuous image.
    rows = torch.ones(height)
    cols = torch.ones(width)
    if top:
        rows[:top] = torch.linspace(0, 1, top)
    if bottom:
        rows[-bottom:] = torch.linspace(1, 0, bottom)
    if left:
        cols[:left] = torch.linspace(0, 1, left)
    if right:
        cols[-right:] = torch.linspace(1, 0, right)
    rows = rows.square() * (3 - 2 * rows)
    cols = cols.square() * (3 - 2 * cols)
    return (rows.view(-1, 1) * cols.view(1, -1)).unsqueeze(-1)


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
    source_audio_latent,
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

    source_y = top // 16
    source_x = left // 16
    source_input = source_frames.to(torch.float32).div_(255.0)
    source_latent = video_vae.encode(source_input)
    del source_input

    observed_video_t = _observed_video_tokens(source_frame_count, global_video_t)
    source_shape = (
        1,
        24,
        global_video_t,
        source_frames.shape[1] // 16,
        source_frames.shape[2] // 16,
    )
    if tuple(source_latent.shape) != source_shape:
        raise RuntimeError(
            f"H3 source VAE produced {tuple(source_latent.shape)}, "
            f"expected {source_shape}."
        )
    source_latent = source_latent.to(accumulate_device)
    video = source_latent.new_zeros(video_shape)
    video[
        :,
        :,
        :,
        source_y : source_y + source_latent.shape[-2],
        source_x : source_x + source_latent.shape[-1],
    ].copy_(source_latent)

    audio = torch.zeros(audio_shape, dtype=torch.float32, device=accumulate_device)
    source_audio_condition = None
    observed_audio_t = 0
    if source_audio_latent is not None:
        if tuple(source_audio_latent.shape[:-1]) != audio_shape[:-1]:
            raise RuntimeError(
                f"H3 audio VAE produced {tuple(source_audio_latent.shape)}, "
                f"expected [1, 32, 2, T]."
            )
        observed_audio_t = min(source_audio_latent.shape[-1], global_audio_t)
        audio[..., :observed_audio_t].copy_(
            source_audio_latent[..., :observed_audio_t].to(accumulate_device)
        )
        source_audio_condition = source_audio_latent[
            ..., :observed_audio_t
        ].to(accumulate_device)

    def temporal_mask(mask):
        mask = (
            mask.to(accumulate_device)
            .unsqueeze(2)
            .expand(1, 1, global_video_t, -1, -1)
            .clone()
        )
        mask[:, :, observed_video_t:] = 1.0
        return mask

    audio_mask = torch.ones(
        (1, 1, 2, global_audio_t),
        dtype=torch.float32,
        device=accumulate_device,
    )
    audio_mask[..., :observed_audio_t] = 0.0
    # The DiT patchifies 2x2 latent tokens, so the keyframe is the
    # patch-aligned interior of the pinned source rows.
    y0, y1 = _pinned_span(source_y, source_latent.shape[-2], latent_height)
    x0, x1 = _pinned_span(source_x, source_latent.shape[-1], latent_width)
    key_y0 = (y0 + 1) // 2 * 2 - source_y
    key_x0 = (x0 + 1) // 2 * 2 - source_x
    key_y1 = y1 // 2 * 2 - source_y
    key_x1 = x1 // 2 * 2 - source_x
    latent = {
        "samples": comfy.nested_tensor.NestedTensor((video, audio)),
        "noise_mask": comfy.nested_tensor.NestedTensor(
            (temporal_mask(spatial_mask), audio_mask)
        ),
        "source_keyframe": {
            "latent": source_latent[
                :, :, :observed_video_t, key_y0:key_y1, key_x0:key_x1
            ].contiguous(),
            "audio_latent": source_audio_condition,
            "latent_y": source_y + key_y0,
            "latent_x": source_x + key_x0,
            "resolved_frame_index": 0,
        },
    }
    return latent, window_shapes, specs


def _reencode_generated(video_vae, sampled_video, window_video_mask):
    # Committed latents feed the next window as hard ground truth. Model
    # output drifts off the encoder manifold and that drift compounds window
    # to window, so round-trip the generated tokens through the VAE first.
    frames = video_vae.decode(sampled_video)[0]
    clean = video_vae.encode(frames).to(
        device=sampled_video.device, dtype=sampled_video.dtype
    )
    return torch.where(window_video_mask > 0, clean, sampled_video)


def _sample_sliding_latent(
    model,
    conditionings,
    latent,
    window_shapes,
    window_specs,
    video_vae,
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
    source_keyframe = latent["source_keyframe"]
    source_video = source_keyframe["latent"]
    source_audio = source_keyframe["audio_latent"]
    window_video_t = window_shapes[0][2]
    window_audio_t = window_shapes[1][3]
    committed_video_stop = 0
    committed_audio_stop = 0

    for conditioning, (_, video_start, audio_start) in zip(
        conditionings, window_specs, strict=True
    ):
        video_stop = video_start + window_video_t
        audio_stop = audio_start + window_audio_t
        video_overlap = max(0, committed_video_stop - video_start)
        audio_overlap = max(0, committed_audio_stop - audio_start)
        source_video_stop = min(video_stop, source_video.shape[2])
        source_audio_stop = (
            min(audio_stop, source_audio.shape[-1])
            if source_audio is not None else audio_start
        )
        window_video = global_video[:, :, video_start:video_stop].contiguous()
        window_audio = global_audio[..., audio_start:audio_stop].contiguous()
        window_video_mask = global_video_mask[:, :, video_start:video_stop].clone()
        window_audio_mask = global_audio_mask[..., audio_start:audio_stop].clone()
        window_noise = comfy.nested_tensor.NestedTensor(
            (
                global_video_noise[:, :, video_start:video_stop].contiguous(),
                global_audio_noise[..., audio_start:audio_stop].contiguous(),
            )
        )
        keyframes = []
        if video_start < source_video_stop or (
            source_audio is not None and audio_start < source_audio_stop
        ):
            keyframes.append(
                {
                    "latent": source_video[:, :, video_start:source_video_stop].contiguous()
                    if video_start < source_video_stop else None,
                    "audio_latent": source_audio[..., audio_start:source_audio_stop].contiguous()
                    if source_audio is not None and audio_start < source_audio_stop
                    else None,
                    "latent_y": source_keyframe["latent_y"],
                    "latent_x": source_keyframe["latent_x"],
                    "resolved_frame_index": 0,
                }
            )
        if video_overlap or audio_overlap:
            window_video_mask[:, :, :video_overlap] = 0.0
            window_audio_mask[..., :audio_overlap] = 0.0
            keyframes.append(
                {
                    "latent": global_video[
                        :, :, video_start:committed_video_stop
                    ].contiguous(),
                    "audio_latent": global_audio[
                        ..., audio_start:committed_audio_stop
                    ].contiguous(),
                    "resolved_frame_index": 0,
                }
            )
        conditioned = node_helpers.conditioning_set_values(
            conditioning, {"minimax_keyframes": keyframes}
        )
        sampled = comfy.sample.sample(
            model,
            window_noise,
            steps,
            1.0,
            sampler_name,
            scheduler,
            conditioned,
            conditioned,
            comfy.nested_tensor.NestedTensor((window_video, window_audio)),
            denoise=1.0,
            noise_mask=comfy.nested_tensor.NestedTensor(
                (window_video_mask, window_audio_mask)
            ),
            seed=seed,
        )
        sampled_video, sampled_audio = sampled.unbind()
        sampled_video = _reencode_generated(
            video_vae, sampled_video, window_video_mask
        )
        commit_video_start = max(committed_video_stop, video_start)
        commit_audio_start = max(committed_audio_stop, audio_start)
        global_video[:, :, commit_video_start:video_stop].copy_(
            sampled_video[:, :, commit_video_start - video_start:]
        )
        global_audio[..., commit_audio_start:audio_stop].copy_(
            sampled_audio[..., commit_audio_start - audio_start:]
        )
        committed_video_stop = video_stop
        committed_audio_stop = audio_stop

    return {
        "samples": comfy.nested_tensor.NestedTensor(
            (global_video, global_audio)
        )
    }


class _StreamingH3Video(Input.Video):
    def __init__(
        self,
        source_video,
        model,
        clip,
        prompt,
        video_vae,
        audio_vae,
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
        source_pixels="exact",
    ):
        self.source_video = source_video
        self.model = model.clone()
        self.clip = clip
        self.prompt = prompt
        self.video_vae = video_vae
        self.audio_vae = audio_vae
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
            self.source_width,
            self.source_height,
            self.left,
            self.top,
            self.right,
            self.bottom,
        ) = _best_effort_canvas(
            self.input_width,
            self.input_height,
            self.target_aspect,
            self.generation_megapixels,
            self.minimum_source_megapixels,
            self.max_upscale,
        )
        self.crop_left = (self.input_width - self.source_width) // 2
        self.crop_top = (self.input_height - self.source_height) // 2
        self.composite_source = source_pixels == "exact"
        if temporal_window_frames == "auto":
            self.denoise_window_frames = _auto_denoise_window_frames(
                self.width, self.height
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

    def prepare_conditioning(self):
        actual_count = _count_video_frames(
            self.source_video,
            self.skip_first_frames,
            self.frame_load_cap,
        )
        self.source_frame_count = actual_count
        self.frame_count = temporal_shape(actual_count)[0]

    def _window_conditionings(self, source_frames, window_frames):
        # fl2va shows every anchored keyframe to Qwen as <Picture 1> as well
        # as to the DiT, so each window's text context carries its own
        # opening source frame; the prompt may be empty.
        conditionings = []
        for frame_start in _global_window_starts(
            int(source_frames.shape[0]), window_frames
        ):
            frame = source_frames[frame_start : frame_start + 1].to(torch.float32).div_(255.0)
            conditionings.append(
                self.clip.encode_from_tokens_scheduled(
                    self.clip.tokenize(self.prompt, images=[frame])
                )
            )
        return conditionings

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

    def _composite_source(self, frames, source_frames):
        # Source frames are ground truth; the decoder only approximates them.
        source = source_frames.to(torch.float32).div_(255.0)
        source_stop_y = self.top + self.source_height
        source_stop_x = self.left + self.source_width
        pinned_y = _pinned_span(
            self.top // 16, self.source_height // 16, self.height // 16
        )
        pinned_x = _pinned_span(
            self.left // 16, self.source_width // 16, self.width // 16
        )
        free_top = pinned_y[0] * 16 - self.top
        free_bottom = source_stop_y - pinned_y[1] * 16
        free_left = pinned_x[0] * 16 - self.left
        free_right = source_stop_x - pinned_x[1] * 16
        _match_band_tone(
            frames, source, self.top, source_stop_y, free_top, free_bottom, 1
        )
        _match_band_tone(
            frames, source, self.left, source_stop_x, free_left, free_right, 2
        )
        frames[:, self.top : source_stop_y, self.left : source_stop_x].lerp_(
            source,
            _source_weight(
                self.source_height,
                self.source_width,
                free_top,
                free_bottom,
                free_left,
                free_right,
            ),
        )

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

    def _encode_output_audio(self, output, stream, waveform):
        for start in range(0, waveform.shape[-1], 1024):
            samples = waveform[0, :, start : start + 1024].numpy()
            audio_frame = av.AudioFrame.from_ndarray(
                samples,
                format="fltp",
                layout="stereo",
            )
            audio_frame.sample_rate = AUDIO_SAMPLE_RATE
            audio_frame.pts = start
            audio_frame.time_base = Fraction(1, AUDIO_SAMPLE_RATE)
            for packet in stream.encode(audio_frame):
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
        source_audio = _load_source_audio(
            self.source_video,
            self.source_video.get_active_trim_window()[0]
            + self.skip_first_frames / float(self.source_frame_rate),
            actual_count / float(self.source_frame_rate),
            AUDIO_SAMPLE_RATE,
        )
        source_audio_latent = None
        if source_audio is not None:
            model_audio = _audio_to_model_timeline(
                source_audio,
                self.source_frame_rate,
            )
            model_audio = _fit_waveform(
                model_audio,
                round(actual_count / FPS * AUDIO_SAMPLE_RATE),
            )
            source_audio_latent = self.audio_vae.encode(
                model_audio.movedim(1, -1)
            )
            del model_audio
        window_frames = (
            aligned_count
            if self.denoise_window_frames is None
            else min(self.denoise_window_frames, aligned_count)
        )
        spatial_mask = _spatial_generation_mask(
            self.height // 16,
            self.width // 16,
            self.source_height,
            self.source_width,
            self.left,
            self.top,
            comfy.model_management.intermediate_device(),
        )
        latent, window_shapes, window_specs = _assemble_global_latent(
            source_frames,
            actual_count,
            self.video_vae,
            self.left,
            self.top,
            self.right,
            self.bottom,
            spatial_mask,
            source_audio_latent,
            window_frames,
        )
        sampled = _sample_sliding_latent(
            self.model,
            self._window_conditionings(source_frames, window_frames),
            latent,
            window_shapes,
            window_specs,
            self.video_vae,
            self.seed,
            self.steps,
            self.sampler_name,
            self.scheduler,
        )
        sampled_video, sampled_audio = sampled["samples"].unbind()
        comfy.model_management.unload_all_models()
        self.model = None
        self.clip = None
        del sampled, latent, spatial_mask, source_audio_latent
        gc.collect()
        comfy.model_management.soft_empty_cache()

        decoded_frames = self.video_vae.decode(sampled_video)[0]
        if decoded_frames.shape[0] != aligned_count:
            raise RuntimeError(
                f"H3 VAE decoded {decoded_frames.shape[0]} frames; "
                f"expected {aligned_count}."
            )
        decoded_frames = decoded_frames.to(device="cpu", dtype=torch.float32)
        del sampled_video
        if self.composite_source:
            self._composite_source(decoded_frames[:actual_count], source_frames[:actual_count])
        del source_frames
        output_audio = None
        if source_audio is not None:
            output_sample_count = round(
                aligned_count
                / float(self.source_frame_rate)
                * AUDIO_SAMPLE_RATE
            )
            if aligned_count == actual_count:
                output_audio = _fit_waveform(
                    source_audio,
                    output_sample_count,
                )
            else:
                output_audio = self.audio_vae.decode(sampled_audio)
                output_audio = output_audio.movedim(-1, 1).to(
                    device="cpu",
                    dtype=torch.float32,
                )
                output_audio = _audio_from_model_timeline(
                    output_audio,
                    self.source_frame_rate,
                )
                output_audio = _fit_waveform(
                    output_audio,
                    output_sample_count,
                )
                source_samples = source_audio.shape[-1]
                transition_samples = min(
                    source_samples,
                    round(
                        800
                        * FPS
                        / float(self.source_frame_rate)
                    ),
                )
                transition_start = source_samples - transition_samples
                output_audio[..., :transition_start].copy_(
                    source_audio[..., :transition_start]
                )
                weight = torch.linspace(
                    0.0,
                    1.0,
                    transition_samples,
                    dtype=output_audio.dtype,
                )
                output_audio[
                    ...,
                    transition_start:source_samples,
                ] = torch.lerp(
                    source_audio[
                        ...,
                        transition_start:source_samples,
                    ],
                    output_audio[
                        ...,
                        transition_start:source_samples,
                    ],
                    weight,
                )
        self.audio_vae = None
        del sampled_audio, source_audio

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
            video_stream = output.add_stream("h264", rate=self.frame_rate)
            video_stream.width = self.width
            video_stream.height = self.height
            video_stream.pix_fmt = "yuv420p"
            if crf is not None:
                video_stream.options = {"crf": str(crf)}

            audio_stream = None
            if output_audio is not None:
                audio_stream = output.add_stream("aac", rate=AUDIO_SAMPLE_RATE)
                audio_stream.layout = "stereo"
                audio_stream.bit_rate = 192000

            for generated_frame in decoded_frames:
                self._encode_output_frame(output, video_stream, generated_frame)
                encoded_count += 1

            for packet in video_stream.encode(None):
                output.mux(packet)
            if audio_stream is not None:
                self._encode_output_audio(output, audio_stream, output_audio)
                for packet in audio_stream.encode(None):
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
                "audio_vae": (
                    "VAE",
                    {
                        "tooltip": "MiniMax H3 audio VAE. Conditions on source audio and generates any aligned tail."
                    },
                ),
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
                "prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "dynamicPrompts": True,
                        "tooltip": "Optional MiniMax H3 scene description: setting, lighting, subjects, camera, and what is beyond each edge (for example a purple wall above, a wooden floor below). Each window also sees its opening source frame, so an empty prompt works; the prompt steers band content and does not correct band tone. Do not write that the frame continues above and below: on uniform, texture-like scenes the model then repeats the source in the bands.",
                    },
                ),
                "temporal_window_frames": (
                    ["auto", "107", "124", "158", "192", "global"],
                    {
                        "default": "auto",
                        "tooltip": "Transformer context per denoiser call. auto picks the largest window up to 107 frames that fits the canvas; larger windows cost memory and destabilise the bands.",
                    },
                ),
                "source_pixels": (
                    ["exact", "decoded"],
                    {
                        "default": "exact",
                        "tooltip": "exact: the real source pixels are composited over the decode with a per-frame RGB tone match at each seam and a 16 px fade at each band edge. decoded: the whole frame is the VAE decode, so the seam is decoder-continuous but the source is a VAE reconstruction (about 3 px MAE softer).",
                    },
                ),
            },
        }

    RETURN_TYPES = ("VIDEO", "INT", "INT", "FLOAT", "INT", "STRING")
    RETURN_NAMES = ("video", "width", "height", "fps", "length", "info")
    FUNCTION = "outpaint"
    CATEGORY = "MiniMax H3/Outpaint"
    DESCRIPTION = "H3 video outpainting with optional text guidance and phase-stable temporal continuation."

    def outpaint(
        self,
        model,
        clip,
        video_vae,
        audio_vae,
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
        prompt="",
        temporal_window_frames="auto",
        source_pixels="exact",
    ):
        video = _StreamingH3Video(
            source_video=source_video,
            model=model,
            clip=clip,
            prompt=prompt,
            video_vae=video_vae,
            audio_vae=audio_vae,
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
            source_pixels=source_pixels,
        )
        video.prepare_conditioning()
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
            f"denoiser window={denoise_window} frames; "
            f"source pixels {source_pixels}."
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
    "MiniMaxH3SimpleVideoOutpaint": "MiniMax H3 Video Outpaint",
}
