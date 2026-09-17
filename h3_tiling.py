"""Spatial tiling of H3 high-resolution model evaluations, for the two-stage split.

Ported line by line from slmonker/selflift-Avatar's `h3_tiling.py` (v0.1.2),
which is itself a snapshot of facok/comfyui-SelfLift's `h3_tiling.py` plus a
mask-compatibility guard. See PROVENANCE.json and the README for source hashes
and licensing notes.

Upstream implementation notes (unchanged by this port):
  * Two ModelPatcher wrappers, so it works with any sampler:
      - WrappersMP.DIFFUSION_MODEL  -> _tiled_forward
      - WrappersMP.PREPARE_SAMPLING -> _prepare_tiled_sampling
  * Splits along the longer 2x2 patch axis, rebuilds a PackedLayout per tile and
    copies the matching rows of position_ids out of the full layout, so every
    tile keeps the correct global coordinates.
  * Tile outputs accumulate on CPU and are fused with a normalized linear window;
    only the first tile's audio prediction is kept.
  * The sampling memory budget is estimated for the largest tile.

Local adaptations for the split architecture:
  * The node takes MODEL + the high-res stage LATENT and is wired only in front
    of the high-res sampler.
  * `tiles` can force the tile count (0 = automatic). The automatic search stays
    as upstream, but this machine reports MIN_WEIGHT_MEMORY_RATIO = 0.0, so its
    estimate is optimistic and a manual override is worth keeping.
  * Wrapper keys carry a Swan_Bits prefix so both plugins can coexist.
"""

from functools import partial
import inspect
import logging
import math

import torch

import comfy.ldm.common_dit
from comfy.ldm.minimax.model import PackedLayout
import comfy.model_base
import comfy.model_management
import comfy.patcher_extension
import comfy.sample
import comfy.sampler_helpers

from .h3_transition import _streams

WRAPPER_KEY = "swan_bits_high_resolution_tiling"


def _regions(length, tile_count=2):
    patches = (length + 1) // 2
    tile_count = max(1, min(tile_count, max(1, patches // 2)))
    if patches < 4 or tile_count <= 1:
        return [(0, length)]
    overlap = min(4, max(1, patches // 8))
    regions = []
    for index in range(tile_count):
        start = max(0, index * patches // tile_count - overlap) * 2
        end = min(patches, (index + 1) * patches // tile_count + overlap) * 2
        end = min(end, length)
        regions.append((start, end))
    regions[-1] = (regions[-1][0], length)
    return regions


def _packed_layout(signature, payload):
    options = {"keyframes": payload.get("keyframes"), "refs": payload.get("refs")}
    if "frame_count" in inspect.signature(PackedLayout).parameters:
        options["frame_count"] = payload.get("frame_count")
    return PackedLayout(*signature, **options)


def _tile_payload(payload, context, video, audio, axis, start, end):
    height, width = video.shape[-2:]
    padded_height, padded_width = (height + 1) // 2 * 2, (width + 1) // 2 * 2
    full_layout = payload.get("layout")
    signature = (context.shape[1], video.shape[2], padded_height, padded_width, audio.shape[-1])
    if full_layout is None or full_layout.signature != signature:
        full_layout = _packed_layout(signature, payload)
    tiled = payload.copy()
    if payload.get("keyframes"):
        keyframes = []
        for keyframe in payload["keyframes"]:
            latent = keyframe["latent"]
            if latent.shape[-2:] != (height, width):
                raise ValueError("Swan_Bits H3 tiling: tiled H3 keyframes must match the target latent height and width")
            region = latent.narrow(axis, start, end - start)
            keyframes.append({**keyframe, "latent": comfy.ldm.common_dit.pad_to_patch_size(
                region, (1, 2, 2)).contiguous()})
        tiled["keyframes"] = keyframes
        if not payload.get("refs"):
            tiled["cond_video_latents"] = [keyframe["latent"] for keyframe in keyframes]
    tile_height = end - start if axis == 3 else height
    tile_width = end - start if axis == 4 else width
    layout = _packed_layout((context.shape[1], video.shape[2], (tile_height + 1) // 2 * 2,
                             (tile_width + 1) // 2 * 2, audio.shape[-1]), tiled)
    for (source_start, source_end, kind), (target_start, target_end, _) in zip(full_layout.segments, layout.segments):
        positions = full_layout.position_ids[source_start:source_end]
        if kind in ("cond", "video"):
            positions = positions.reshape(-1, padded_height // 2, padded_width // 2, 3)
            positions = positions.narrow(axis - 2, start // 2, (end - start + 1) // 2).reshape(-1, 3)
        layout.position_ids[target_start:target_end].copy_(positions)
    tiled["layout"] = layout
    return tiled


def _tiled_forward(executor, streams, timestep, context, transformer_options, minimax_payload=None,
                   n_tiles=2, plan=None, **kwargs):
    video, audio = streams
    # Only the fully generated video / fully preserved audio combination is supported.
    # A redundant full-frame all-one video condition must not reach a smaller tile.
    video_mask = kwargs.get("denoise_mask")
    if video_mask is not None:
        if not bool((video_mask == 1).all()):
            raise ValueError("Swan_Bits H3 tiling: tiled video preservation is not supported")
        kwargs = {**kwargs, "denoise_mask": None}
    audio_mask = kwargs.get("audio_denoise_mask")
    if audio_mask is not None and not bool((audio_mask == 0).all()):
        raise ValueError("Swan_Bits H3 tiling: tiled audio conditioning requires a fully preserved audio mask")
    axis = 3 if (video.shape[3] + 1) // 2 >= (video.shape[4] + 1) // 2 else 4
    length = video.shape[axis]
    regions = _regions(length, plan['tiles'] if plan is not None else n_tiles)
    if len(regions) == 1:
        return executor(streams, timestep, context, transformer_options, minimax_payload=minimax_payload, **kwargs)
    if kwargs.get("control") is not None:
        raise ValueError("Swan_Bits H3 tiling: high-resolution H3 tiling does not support ControlNet")
    # Keep the stitched accumulator on CPU so previous tiles do not remain on
    # the accelerator while the next tile is evaluated.
    video_output = torch.zeros(video.shape, dtype=torch.float32, device="cpu")
    audio_output = None
    weights = torch.zeros(length, dtype=torch.float32, device="cpu")
    for index, (start, end) in enumerate(regions):
        tile = video.narrow(axis, start, end - start).contiguous()
        payload = _tile_payload(minimax_payload or {}, context, video, audio, axis, start, end)
        predicted_video, predicted_audio = executor(
            [tile, audio], timestep, context, transformer_options.copy(), minimax_payload=payload, **kwargs)
        window = torch.ones(end - start, device=video.device, dtype=torch.float32)
        if index > 0:
            overlap = min(end, regions[index - 1][1]) - start
            window[:overlap] *= (torch.arange(overlap, device=video.device, dtype=torch.float32) + 0.5) / overlap
        if index + 1 < len(regions):
            overlap = end - regions[index + 1][0]
            window[-overlap:] *= 1.0 - (torch.arange(overlap, device=video.device, dtype=torch.float32) + 0.5) / overlap
        predicted_video = predicted_video.float().cpu()
        window = window.cpu()
        weights[start:end].add_(window)
        window_shape = [1] * video.ndim
        window_shape[axis] = end - start
        video_output.narrow(axis, start, end - start).addcmul_(predicted_video.float(), window.view(window_shape))
        if audio_output is None:
            audio_output = predicted_audio.float().clone()
        del tile, payload, predicted_video, predicted_audio, window
    window_shape[axis] = length
    video_output.div_(weights.view(window_shape))
    return [video_output.to(device=video.device, dtype=video.dtype), audio_output.to(audio.dtype)]


def _condition_elements(condition, tile_height, tile_width, channels):
    text = condition.get("cross_attn")
    elements = text.shape[-2] * channels * 4 if text is not None else 0
    for keyframe in condition.get("minimax_keyframes") or []:
        shape = keyframe["latent"].shape
        elements += shape[1] * shape[2] * tile_height * tile_width
    for reference in condition.get("minimax_refs") or []:
        latent = reference.get("latent")
        if latent is not None:
            shape = latent.shape
            elements += shape[1] * shape[2] * ((shape[3] + 1) // 2 * 2) * ((shape[4] + 1) // 2 * 2)
        audio = reference.get("audio_latent")
        if audio is not None:
            elements += math.prod(audio.shape[1:])
    return elements


def _budget(model, noise_shape, conds, latent_shapes, regions, axis):
    video_shape, audio_shape = latent_shapes
    full_elements = math.prod(video_shape[1:]) + math.prod(audio_shape[1:])
    tile_shape = list(video_shape)
    tile_shape[axis] = max(end - start for start, end in regions)
    tile_shape[3] = (tile_shape[3] + 1) // 2 * 2
    tile_shape[4] = (tile_shape[4] + 1) // 2 * 2
    tile_elements = math.prod(tile_shape[1:]) + math.prod(audio_shape[1:])
    condition_elements = max((_condition_elements(condition, tile_shape[3], tile_shape[4], video_shape[1])
                              for group in conds.values() for condition in (group or [])), default=0)
    buffer_bytes = full_elements * 4 * 8
    bytes_per_element = model.model.memory_required((1, 1, 1))
    budget_elements = tile_elements + condition_elements + math.ceil(buffer_bytes / bytes_per_element)
    budget_shape = (noise_shape[0], 1, budget_elements)
    preferred, minimum = comfy.sampler_helpers.estimate_memory(model, budget_shape, conds)
    return budget_shape, tuple(tile_shape), buffer_bytes, preferred, minimum


def _prepare_tiled_sampling(executor, model, noise_shape, conds, model_options=None,
                            force_full_load=False, force_offload=False, *, latent_shapes, plan=None):
    video_shape, audio_shape = latent_shapes
    axis = 3 if (video_shape[3] + 1) // 2 >= (video_shape[4] + 1) // 2 else 4
    full_elements = math.prod(video_shape[1:]) + math.prod(audio_shape[1:])
    if tuple(noise_shape) != (video_shape[0], 1, full_elements):
        raise ValueError("Swan_Bits H3 tiling: tiled memory planning received a different latent shape than the sampling input")
    forced = plan.get('force_tiles', 0) if plan is not None else 0
    if forced >= 1:
        regions = _regions(video_shape[axis], forced)
        count = len(regions)
        plan['tiles'] = count
        logging.info("[Swan_Bits tiling plan] forced axis=%s tiles=%d",
                     'H' if axis == 3 else 'W', count)
    else:
        count = 2
        if plan is not None:
            available = _available_workspace(model)
            for count in range(1, 9):
                regions = _regions(video_shape[axis], count)
                _, _, _, _, minimum = _budget(model, noise_shape, conds, latent_shapes, regions, axis)
                if minimum <= available:
                    break
            count = len(regions)
            plan['tiles'] = count
            logging.info("[Swan_Bits tiling plan] axis=%s tiles=%d target=%.2f MiB estimate_fits=%s",
                         'H' if axis == 3 else 'W', plan['tiles'], available / 2**20, minimum <= available)
        regions = _regions(video_shape[axis], count)
    if len(regions) == 1 or force_offload:
        return executor(model, noise_shape, conds, model_options=model_options,
                        force_full_load=force_full_load, force_offload=force_offload)
    budget_shape, tile_shape, buffer_bytes, preferred, minimum = _budget(
        model, noise_shape, conds, latent_shapes, regions, axis)
    logging.info("[Swan_Bits tiling memory] largest_tile=%s full_audio=%s full_state_buffers=%.2f MiB "
                 "minimum=%.2f MiB preferred=%.2f MiB (ComfyUI estimates; additional models and reserves excluded)",
                 tuple(tile_shape), audio_shape, buffer_bytes * noise_shape[0] / 2**20,
                 minimum / 2**20, preferred / 2**20)
    return executor(model, budget_shape, conds, model_options=model_options,
                    force_full_load=force_full_load, force_offload=force_offload)


def _available_workspace(model):
    manager = comfy.model_management
    free = manager.get_free_memory(model.load_device)
    reclaimable = 0
    seen = set()
    for loaded in manager.loaded_models():
        patcher = loaded if callable(getattr(loaded, "loaded_size", None)) else loaded.model
        load_device = getattr(patcher, "load_device", None)
        identity = id(patcher.model)
        if load_device == model.load_device and identity not in seen:
            seen.add(identity)
            size_fn = getattr(patcher, "loaded_size", None)
            if callable(size_fn):
                reclaimable += size_fn()
    pool = min(manager.get_total_memory(model.load_device), free + reclaimable)
    weights = min(model.model_size(), pool * manager.MIN_WEIGHT_MEMORY_RATIO)
    available = max(0, pool - weights - manager.minimum_inference_memory())
    logging.info("[Swan_Bits tiling capacity] free=%.2f MiB reclaimable_weights=%.2f MiB "
                 "weight_allowance=%.2f MiB workspace=%.2f MiB",
                 free / 2**20, reclaimable / 2**20, weights / 2**20, available / 2**20)
    return available


def tiled_model(model, latent_shapes, force_tiles=0):
    if not isinstance(model.model, comfy.model_base.MiniMaxH3):
        raise ValueError("Swan_Bits H3 tiling: high-resolution tiling requires a MiniMax H3 model")
    if len(latent_shapes) != 2 or len(latent_shapes[0]) != 5 or len(latent_shapes[1]) != 4:
        raise ValueError("Swan_Bits H3 tiling: high-resolution tiling requires H3 video and audio latent streams")
    plan = {"force_tiles": int(force_tiles)}
    patched = model.clone()
    patched.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
                                 WRAPPER_KEY, partial(_tiled_forward, plan=plan))
    patched.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.PREPARE_SAMPLING,
                                 WRAPPER_KEY,
                                 partial(_prepare_tiled_sampling, latent_shapes=tuple(tuple(shape) for shape in latent_shapes),
                                         plan=plan))
    return patched


def _stage_latent_shapes(model, latent):
    """Latent shapes of the high-res sampling input, normalized like the sampler does."""
    samples = comfy.sample.fix_empty_latent_channels(
        model, latent["samples"],
        latent.get("downscale_ratio_spacial", None),
        latent.get("downscale_ratio_temporal", None))
    streams, _ = _streams(samples)
    return [tuple(stream.shape) for stream in streams]


class SWAN_H3TiledModel:
    """Wrap the high-res stage MODEL so each evaluation runs on spatial tiles."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {
                    "tooltip": "Model used by the high-res SamplerCustom. Leave the low-res stage on the unwrapped model."
                }),
                "stage_latent": ("LATENT", {
                    "tooltip": "The latent the high-res sampler samples (the transition lift output, or the target Empty H3 AV Latent). Its shape must match."
                }),
                "tiles": ("INT", {
                    "default": 0, "min": 0, "max": 8, "step": 1,
                    "tooltip": "0 = pick the largest tile count that fits the estimated workspace. 1 = tiling off (identical to the unwrapped model). 2-8 = force that count."
                }),
                "enable_tiling": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch"
    CATEGORY = "SwanBits/H3"
    DESCRIPTION = (
        "Spatial tiling for the high-res stage of a two-stage H3 split. Splits each model "
        "evaluation along the longer 2x2-patch axis, forwards the complete audio stream to "
        "every tile, and fuses tiles with an overlap window. Masks are accepted only as "
        "video=1 everywhere / audio=0 everywhere (see the keep_audio option on the lift)."
    )

    def patch(self, model, stage_latent, tiles, enable_tiling):
        if not enable_tiling or tiles == 1:
            return (model,)
        shapes = _stage_latent_shapes(model, stage_latent)
        return (tiled_model(model, shapes, force_tiles=tiles),)
