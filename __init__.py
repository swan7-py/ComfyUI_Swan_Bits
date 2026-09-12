from .audio_drive import SWAN_MiniMaxH3AudioDrive
from .audio_info import SWAN_AudioInfo
from .h3_transition import (
    SWAN_H3UpscalerLoader,
    SWAN_SigmasLowZero,
    SWAN_TransitionLift,
)

NODE_CLASS_MAPPINGS = {
    "SWAN_MiniMaxH3AudioDrive": SWAN_MiniMaxH3AudioDrive,
    "SWAN_AudioInfo": SWAN_AudioInfo,
    "SWAN_H3UpscalerLoader": SWAN_H3UpscalerLoader,
    "SWAN_SigmasLowZero": SWAN_SigmasLowZero,
    "SWAN_TransitionLift": SWAN_TransitionLift,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SWAN_MiniMaxH3AudioDrive": "Swan MiniMax H3 Audio Drive",
    "SWAN_AudioInfo": "Swan Audio Info",
    "SWAN_H3UpscalerLoader": "Swan Load H3 Latent Upscaler Model",
    "SWAN_SigmasLowZero": "Swan Sigmas Split: Low Runs to Zero",
    "SWAN_TransitionLift": "Swan H3 Transition Lift (Flow Handoff / SelfLift)",
}
