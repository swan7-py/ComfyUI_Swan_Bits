import numpy as np
import torch

import comfy.model_management
from comfy_extras.sam3d_body.rasterizer import render_pose_data_torch

from .sam3d_track_color import color_inputs, color_spec, resolve_colors


class SWAN_SAM3DBodySolid:
    """Render SAM3D pose data with one flat colour per tracked person.

    The built-in renderer only offers default/normals/rainbow/depth presets, so a
    single-colour body wasn't reachable. It does take an unexposed `base_color`
    though, which is the albedo of the lit 'default' shader. This renders each
    person on their own with a white albedo, keeps the resulting shaded grey as
    the per-pixel mask, then tints it with that person's colour and paints far to
    near so occlusion survives.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_data": ("MHR_POSE_DATA", {
                    "tooltip": "MHR pose data from SAM3DBody Predict.",
                }),
                "width": ("INT", {
                    "default": 0, "min": 0, "max": 16384, "step": 8,
                    "tooltip": "Output width. 0 = use the pose data's native size.",
                }),
                "height": ("INT", {
                    "default": 0, "min": 0, "max": 16384, "step": 8,
                    "tooltip": "Output height. 0 = use the pose data's native size.",
                }),
                **color_inputs(),
            },
            "optional": {
                "background": ("IMAGE", {
                    "tooltip": "Plate to draw over, one frame per input frame. Omitted = black.",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "render"
    CATEGORY = "SwanBits/3D"
    DESCRIPTION = (
        "Render SAM3D pose data with a solid colour per person, keeping the "
        "shading and the source camera motion. Pairs with Swan SAM3D Track Colors "
        "so the video and the exported GLB use the same palette."
    )

    def render(self, pose_data, width, height, background=None, **colors):
        frames = pose_data["frames"]
        native_h, native_w = pose_data["image_size"]
        out_w = int(width) or int(native_w)
        out_h = int(height) or int(native_h)

        # Colour index must follow the person slot, not the per-frame count, or a
        # frame with fewer people would reassign everyone's colour.
        total = max((len(people) for people in frames), default=0)
        if total == 0:
            raise ValueError("Swan SAM3DBody Solid Colors: pose data has no people.")
        tint = [torch.tensor(c, dtype=torch.float32) for c in
                resolve_colors(color_spec(colors), total, to_linear=False)]

        device = comfy.model_management.get_torch_device()
        cache = {}
        rendered = []
        for frame_idx, people in enumerate(frames):
            canvas = torch.zeros((out_h, out_w, 3), device=device, dtype=torch.float32)
            if background is not None:
                plate = background[min(frame_idx, background.shape[0] - 1)]
                canvas = plate.to(device=device, dtype=torch.float32)

            # pred_cam_t Z is depth into the scene: draw the far ones first so the
            # near ones paint over them.
            order = sorted(range(len(people)),
                           key=lambda i: -float(np.asarray(people[i]["pred_cam_t"]).reshape(-1)[2]))
            for person_idx in order:
                single = {
                    "frames": [[people[person_idx]]],
                    "faces": pose_data["faces"],
                    "image_size": pose_data["image_size"],
                }
                shaded = render_pose_data_torch(
                    single, 0, out_w, out_h,
                    background=None,
                    composite="over",
                    opacity=1.0,
                    shader_preset="default",
                    base_color=(1.0, 1.0, 1.0),
                    cache=cache,
                )
                covered = shaded.sum(-1) > 1e-4
                paint = shaded * tint[person_idx].to(device=device)
                canvas = torch.where(covered.unsqueeze(-1), paint, canvas)

            rendered.append(canvas.clamp(0.0, 1.0))

        return (torch.stack(rendered, dim=0),)
