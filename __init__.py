from .audio_drive import SWAN_MiniMaxH3AudioDrive
from .audio_info import SWAN_AudioInfo
from .h3_tiling import SWAN_H3TiledModel
from .h3_transition import (
    SWAN_H3UpscalerLoader,
    SWAN_SigmasLowZero,
    SWAN_TransitionLift,
    SWAN_KeyframeResize,
)
from .sam3d_track_color import SWAN_SAM3DTrackColor
from .sam3d_body_solid import SWAN_SAM3DBodySolid
from .sam3d_palette import SWAN_SAM3DTrackPalette

NODE_CLASS_MAPPINGS = {
    "SWAN_MiniMaxH3AudioDrive": SWAN_MiniMaxH3AudioDrive,
    "SWAN_AudioInfo": SWAN_AudioInfo,
    "SWAN_H3UpscalerLoader": SWAN_H3UpscalerLoader,
    "SWAN_SigmasLowZero": SWAN_SigmasLowZero,
    "SWAN_TransitionLift": SWAN_TransitionLift,
    "SWAN_KeyframeResize": SWAN_KeyframeResize,
    "SWAN_H3TiledModel": SWAN_H3TiledModel,
    "SWAN_SAM3DTrackColor": SWAN_SAM3DTrackColor,
    "SWAN_SAM3DBodySolid": SWAN_SAM3DBodySolid,
    "SWAN_SAM3DTrackPalette": SWAN_SAM3DTrackPalette,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SWAN_MiniMaxH3AudioDrive": "Swan MiniMax H3 Audio Drive",
    "SWAN_AudioInfo": "Swan Audio Info",
    "SWAN_H3UpscalerLoader": "Swan Load H3 Latent Upscaler Model",
    "SWAN_SigmasLowZero": "Swan Sigmas Split: Low Runs to Zero",
    "SWAN_TransitionLift": "Swan SelfLift Transition Lift (H3)",
    "SWAN_KeyframeResize": "Swan Resize H3 Keyframes (Conditioning)",
    "SWAN_H3TiledModel": "Swan H3 Tiled Model (High Stage)",
    "SWAN_SAM3DTrackColor": "Swan SAM3D Track Colors",
    "SWAN_SAM3DBodySolid": "Swan SAM3DBody Solid Colors",
    "SWAN_SAM3DTrackPalette": "Swan SAM3D Track Palette",
}

WEB_DIRECTORY = "./web"
