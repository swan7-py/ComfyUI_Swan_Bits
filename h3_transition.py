"""Two-stage progressive sampling support: load the learned H3 latent upscaler,
split a sigma schedule so the low-res stage runs to zero, and lift the low-res
result to the target grid between two independent samplers.

Graph contract:
  low-res SamplerCustom (split schedule, ends at sigma 0)
    -> SWAN_TransitionLift -> clean target-size latent
    -> high-res SamplerCustom (tail schedule, add_noise = True re-noises natively).

The loader output follows the H3_LATENT_UPSCALER provider contract used by
MiniMax-H3-Flow-Aligned-Regenerate (api_version=1, kind
"minimax_h3_learned_latent_upscaler", callable upscale_clean_video), so the same
handle also plugs into that plugin's learned_upscaler input.

The lift implements the SelfLift-zero transition (arXiv:2609.02036, Eq. 3-10
minus the NFE-reuse trick): paired direct/pixel-VAE lifts of the clean endpoint,
residual risk map, top-rho artifact-aware correction.
"""
import os

import torch
import torch.nn.functional as F

import folder_paths
from .h3_upscaler_model import load_model as _load_upscaler_model, make_norm_tensors

_LATENT_UPSCALE_FOLDER = "latent_upscale_models"
if _LATENT_UPSCALE_FOLDER not in folder_paths.folder_names_and_paths:
    folder_paths.add_model_folder_path(
        _LATENT_UPSCALE_FOLDER,
        os.path.join(folder_paths.models_dir, _LATENT_UPSCALE_FOLDER),
    )

try:
    import comfy.nested_tensor
    _NESTED_CLS = getattr(comfy.nested_tensor, "NestedTensor", None)
except ImportError:
    _NESTED_CLS = None

H3_UPSCALER_API_VERSION = 1
H3_UPSCALER_KIND = "minimax_h3_learned_latent_upscaler"
_PRECISION_DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def _scan_models():
    """List checkpoints with H3 upscalers first, so the combo default is a valid H3 file."""
    try:
        names = folder_paths.get_filename_list(_LATENT_UPSCALE_FOLDER)
    except Exception:
        names = []
    models = [n for n in names if os.path.splitext(n)[1].lower() in (".pth", ".safetensors")]
    models.sort(key=lambda n: ("h3" not in n.lower(), n))
    return models if models else ["(place checkpoints in models/latent_upscale_models)"]


def _streams(samples):
    """Split a latent sample into per-stream tensors; H3 AV latents are nested."""
    if _NESTED_CLS is not None and isinstance(samples, _NESTED_CLS) and samples.is_nested:
        return list(samples.unbind()), True
    return [samples], False


def _pack(streams, nested):
    if nested and _NESTED_CLS is not None:
        return _NESTED_CLS(streams)
    return streams[0]


def _video_stream(streams):
    for index, stream in enumerate(streams):
        if stream.ndim in (4, 5):
            return index, stream
    raise ValueError("SwanBits transition lift: no video/image stream found in the latent")


class SwanH3UpscalerProvider:
    """Provider handle matching the Flow H3_LATENT_UPSCALER attribute contract."""

    api_version = H3_UPSCALER_API_VERSION
    kind = H3_UPSCALER_KIND

    def __init__(self, model_name, device, precision, offload_after_upscale):
        self.model_name = model_name
        self.device = device
        self.precision = precision
        self.offload_after_upscale = bool(offload_after_upscale)
        self.dtype = _PRECISION_DTYPES[precision]
        if "h3" not in model_name.lower():
            self._verify_resizer_arch(model_name)
        self.model = _load_upscaler_model(model_name, torch.device(device), precision)

    def _verify_resizer_arch(self, model_name):
        """Fail with a clear message when the picked checkpoint is not a LatentResizer3D."""
        from .h3_upscaler_model import _load_raw_sd, _extract_upscaler_sd
        path = folder_paths.get_full_path_or_raise(_LATENT_UPSCALE_FOLDER, model_name)
        up_sd = _extract_upscaler_sd(_load_raw_sd(path))
        if "conv_in.weight" not in up_sd:
            raise ValueError(
                f"'{model_name}' is not a MiniMax H3 3D latent upscaler (no conv_in.* weights). "
                "Pick a checkpoint such as minimax_h3_latent_upscaler_3d_fp16.safetensors."
            )

    @property
    def inference_device(self):
        return self.device

    def upscale_clean_video(self, video, *, target_h, target_w):
        """Lift one clean [B,24,T,H,W] (or [B,24,H,W]) video latent to the exact target grid."""
        dev = torch.device(self.device)
        model = self.model.to(dev)
        was_4d = video.ndim == 4
        s = video.to(device=dev, dtype=self.dtype, copy=True)
        if was_4d:
            s = s.unsqueeze(2)
        b, c, t, h_in, w_in = s.shape
        scale = ((target_w / w_in) + (target_h / h_in)) / 2.0
        mean, std = make_norm_tensors(dev, self.dtype)
        with torch.inference_mode():
            s_norm = (s - mean) / std
            out = model(s_norm, scale=scale, target_size=(t, target_h, target_w),
                        enable_chunking=True)
            out = out * std + mean
        if was_4d:
            out = out.squeeze(2)
        if self.offload_after_upscale:
            model.to("cpu")
        return out.to(device=video.device, dtype=video.dtype)


class SWAN_H3UpscalerLoader:
    """Load a learned H3 latent-upscaler checkpoint once and pass the handle around."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (_scan_models(), {
                    "tooltip": "Checkpoint from models/latent_upscale_models."
                }),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
                "precision": (["fp16", "fp32", "bf16"], {"default": "fp16"}),
                "offload_after_upscale": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Move the cached upscaler to CPU after every lift. Leave off for repeated chunks/runs when VRAM allows."
                }),
            }
        }

    RETURN_TYPES = ("H3_LATENT_UPSCALER",)
    RETURN_NAMES = ("learned_upscaler",)
    FUNCTION = "load"
    CATEGORY = "SwanBits/H3"
    DESCRIPTION = (
        "Loads a MiniMax H3 learned latent-upscaler checkpoint and outputs a provider "
        "handle compatible with the Flow-Aligned Regenerate learned_upscaler input "
        "and the Swan transition lift."
    )

    def load(self, model_name, device, precision, offload_after_upscale):
        if model_name.startswith("("):
            raise ValueError(f"No upscaler checkpoint available: {model_name}")
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        provider = SwanH3UpscalerProvider(model_name, device, precision, offload_after_upscale)
        return (provider,)


class SWAN_SigmasLowZero:
    """Split one schedule into a low stage that runs fully to zero and a high tail.

    low_sigmas  = [sigma_0 ... sigma_k, 0]   (low-res sampler, add_noise = True)
    high_sigmas = [sigma_k ... 0]            (high-res sampler, add_noise = True re-noises the lifted clean latent)

    The trailing zero costs one extra NFE versus a fused progressive sampler:
    the last low-res interval completes at low resolution so the lift receives a
    clean x0 instead of a partially noised state.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sigmas": ("SIGMAS",),
                "low_steps": ("INT", {
                    "default": 5,
                    "min": 1,
                    "max": 10000,
                    "step": 1,
                    "tooltip": "Denoiser evaluations executed at low resolution. The high stage receives the remaining intervals."
                }),
            }
        }

    RETURN_TYPES = ("SIGMAS", "SIGMAS")
    RETURN_NAMES = ("low_sigmas", "high_sigmas")
    FUNCTION = "split"
    CATEGORY = "SwanBits/H3"
    DESCRIPTION = (
        "Splits a sigma schedule for two-stage sampling. The low stage ends at zero "
        "so the transition lift gets a fully denoised latent; the high tail starts "
        "at the split sigma and re-noises the lifted latent via SamplerCustom add_noise."
    )

    def split(self, sigmas, low_steps):
        if sigmas.ndim != 1 or sigmas.numel() < 4:
            raise ValueError("Swan sigmas split needs a 1D schedule with at least 3 intervals")
        total_intervals = sigmas.numel() - 1
        if low_steps < 1 or low_steps > total_intervals - 1:
            raise ValueError(
                f"low_steps {low_steps} out of range: schedule has {total_intervals} intervals "
                "and the high stage needs at least one"
            )
        low = torch.cat([sigmas[:low_steps], sigmas.new_zeros(1)])
        high = sigmas[low_steps:].clone()
        return (low, high)


def _learned_lift(video, provider, target_hw):
    """Learned 3D-conv lift through the provider handle (exact target grid)."""
    h_out, w_out = target_hw
    lifted = provider.upscale_clean_video(video, target_h=h_out, target_w=w_out)
    if lifted.shape[-2:] != (h_out, w_out):
        raise ValueError(
            f"learned upscaler returned {tuple(lifted.shape[-2:])}, expected {(h_out, w_out)}"
        )
    return lifted


def _interp_lift(video, target_hw, method):
    h_out, w_out = target_hw
    if video.ndim == 4:
        if method == "nearest":
            return F.interpolate(video.float(), size=(h_out, w_out), mode="nearest").to(video.dtype)
        return F.interpolate(video.float(), size=(h_out, w_out), mode="bilinear", align_corners=False).to(video.dtype)
    size = (video.shape[2], h_out, w_out)
    if method == "nearest":
        return F.interpolate(video.float(), size=size, mode="nearest").to(video.dtype)
    return F.interpolate(video.float(), size=size, mode="trilinear", align_corners=False).to(video.dtype)


def _pixel_anchor(video, vae, target_hw):
    """VAE decode -> chunked bicubic upscale -> re-encode (SelfLift-zero Eq. 5)."""
    h_out, w_out = target_hw
    vae_dtype = getattr(vae, "vae_dtype", None)
    work_dtype = vae_dtype if vae_dtype in (torch.float16, torch.bfloat16, torch.float32) else torch.float32
    if vae.device.type == "cpu":
        work_dtype = torch.float32

    def anchor_single(sample):
        frames = vae.decode(sample)
        if frames.ndim == 5:
            frames = frames.reshape(-1, frames.shape[-3], frames.shape[-2], frames.shape[-1])
        ratio = frames.shape[1] // sample.shape[-2]
        n, hp, wp = frames.shape[0], h_out * ratio, w_out * ratio
        up = torch.empty((n, hp, wp, frames.shape[-1]), dtype=work_dtype)
        for i in range(0, n, 32):
            chunk = frames[i:i + 32].movedim(-1, 1).to(device=vae.device, dtype=work_dtype)
            chunk = F.interpolate(chunk, size=(hp, wp), mode="bicubic", antialias=True)
            up[i:i + 32] = chunk.movedim(1, -1).to(up.device)
            del chunk
        del frames
        return vae.encode(up).float()

    if video.shape[0] == 1:
        return anchor_single(video)
    return torch.cat([anchor_single(sample) for sample in video.split(1)], dim=0)


def _consistency_lift(z_lat, z_pix, rho, w_min, w_max):
    """SelfLift-zero correction: move the top-rho risky locations toward the pixel anchor."""
    if rho <= 0.0:
        return z_lat
    if rho >= 1.0 and w_min >= 1.0 and w_max >= 1.0:
        return z_pix
    delta = z_pix - z_lat
    s = delta.abs().mean(dim=1)
    view = (-1,) + (1,) * (s.ndim - 1)
    flat = s.flatten(1)
    thr = torch.quantile(flat, 1.0 - rho, dim=1).view(view)
    mask = s >= thr
    s_min = s.masked_fill(~mask, float("inf")).flatten(1).amin(dim=1).view(view)
    s_max = s.masked_fill(~mask, float("-inf")).flatten(1).amax(dim=1).view(view)
    w = w_min + (w_max - w_min) * (s - s_min) / (s_max - s_min + 1e-8)
    w = torch.where(mask, w, torch.zeros_like(w)).unsqueeze(1)
    return z_lat + w * delta


class SWAN_TransitionLift:
    """SelfLift-zero transition between a low-res and a high-res sampler.

    Builds paired lifts of the low-res clean endpoint - direct latent lift
    (learned upscaler when connected, otherwise interpolated) plus a VAE
    decode/upscale/re-encode pixel anchor - and corrects the top-rho
    inconsistent locations toward the anchor (arXiv:2609.02036, mode=zero).
    rho = 0 skips the pixel route and keeps the direct lift only.

    Output is a clean target-size latent; the high-res SamplerCustom re-noises it
    through add_noise with the tail schedule.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lowres_latent": ("LATENT", {
                    "tooltip": "Fully denoised low-resolution latent (the low-res SamplerCustom output)."
                }),
                "target_scale": ("FLOAT", {
                    "default": 1.43,
                    "min": 1.0,
                    "max": 4.0,
                    "step": 0.01,
                    "tooltip": "Linear width/height lift factor. Ignored when highres_latent is connected."
                }),
                "direct_lift": (["learned", "nearest", "bilinear"], {
                    "default": "learned",
                    "tooltip": "learned requires the connected upscaler handle, otherwise falls back to nearest."
                }),
                "rho": ("FLOAT", {
                    "default": 0.3,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Fraction of highest-risk locations corrected toward the VAE pixel anchor. H3 testing suggests 0.6 with w=1/1 as a starting point; 0 skips the pixel route."
                }),
                "w_min": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "w_max": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            },
            "optional": {
                "highres_latent": ("LATENT", {
                    "tooltip": "Optional empty target latent; its H/W define the output grid exactly."
                }),
                "vae": ("VAE", {
                    "tooltip": "Required for rho > 0 (pixel-anchor route). Use the VAE of the sampled model."
                }),
                "upscaler": ("H3_LATENT_UPSCALER", {
                    "tooltip": "Provider handle from the Swan H3 upscaler loader node."
                }),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("highres_latent",)
    FUNCTION = "lift"
    CATEGORY = "SwanBits/H3"
    DESCRIPTION = (
        "SelfLift-zero transition: paired direct/pixel lifts of the low-res clean "
        "endpoint with artifact-aware correction. Re-noising happens in the next "
        "SamplerCustom via add_noise."
    )

    def lift(self, lowres_latent, target_scale, direct_lift, rho, w_min, w_max,
             highres_latent=None, vae=None, upscaler=None):
        src = lowres_latent["samples"]
        streams, nested = _streams(src)
        v_idx, video = _video_stream(streams)

        if len(streams) > 1:
            shapes = [tuple(s.shape) for s in streams]
            bad = [(i, sh) for i, (s, sh) in enumerate(zip(streams, shapes))
                   if i != v_idx and s.ndim != 4]
            if bad:
                raise ValueError(
                    "SwanBits transition lift: non-video stream(s) have wrong rank "
                    f"{bad}; H3 audio must stay [B, 32, 2, T] (4D) next to video "
                    "[B, 24, T, H, W]. Do not split/merge the AV latent with generic "
                    "latent nodes - feed the nested AV latent directly, and use a "
                    "second Empty MiniMax H3 AV Latent (same frame count, smaller "
                    "width/height) for the low-res stage."
                )

        if highres_latent is not None:
            ref_streams, _ = _streams(highres_latent["samples"])
            _, ref = _video_stream(ref_streams)
            h_out, w_out = ref.shape[-2], ref.shape[-1]
        else:
            # round half up to even latent dims (32px alignment); Python round() is
            # banker's rounding and silently shaved 81->80 / 45->44 (1280x704 drift)
            h_out = max(2, int(video.shape[-2] * target_scale / 2 + 0.5) * 2)
            w_out = max(2, int(video.shape[-1] * target_scale / 2 + 0.5) * 2)

        if (h_out, w_out) == (video.shape[-2], video.shape[-1]):
            return (lowres_latent,)
        target_hw = (h_out, w_out)

        use_pixel = rho > 0.0
        if use_pixel and vae is None:
            raise ValueError("rho > 0 requires a VAE for the pixel-anchor route")
        use_learned = direct_lift == "learned" and upscaler is not None
        if direct_lift == "learned" and upscaler is None:
            print("[SwanBits] transition lift: no upscaler connected, falling back to nearest")

        if use_learned:
            z_lat = _learned_lift(video, upscaler, target_hw)
        else:
            z_lat = _interp_lift(video, target_hw, direct_lift)

        if use_pixel:
            z_pix = _pixel_anchor(video, vae, target_hw).to(device=z_lat.device, dtype=z_lat.dtype)
            lifted = _consistency_lift(z_lat, z_pix, rho, w_min, w_max)
        else:
            lifted = z_lat

        out_streams = list(streams)
        out_streams[v_idx] = lifted.to(device=video.device, dtype=video.dtype)
        return ({"samples": _pack(out_streams, nested)},)


NODE_CLASS_MAPPINGS = {
    "SWAN_H3UpscalerLoader": SWAN_H3UpscalerLoader,
    "SWAN_SigmasLowZero": SWAN_SigmasLowZero,
    "SWAN_TransitionLift": SWAN_TransitionLift,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SWAN_H3UpscalerLoader": "Swan Load H3 Latent Upscaler Model",
    "SWAN_SigmasLowZero": "Swan Sigmas Split: Low Runs to Zero",
    "SWAN_TransitionLift": "Swan SelfLift Transition Lift (H3)",
}
