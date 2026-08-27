import torch


class SWAN_AudioInfo:
    """Extract metadata from a ComfyUI AUDIO object (e.g. from the official Load Audio node).

    The official Load Audio node outputs a dict of the form
    {"waveform": torch.Tensor [batch, channels, samples], "sample_rate": int}.
    This node reports duration (in seconds), sampling frequency and the corresponding
    number of video frames at the given fps.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {
                    "tooltip": "Audio loaded by the official Load Audio node: "
                               "{'waveform': [B, C, samples], 'sample_rate': int}."
                }),
                "fps": ("INT", {
                    "default": 24,
                    "min": 1,
                    "step": 1,
                    "tooltip": "Frame rate used to derive the video frame count from the audio duration."
                }),
            }
        }

    RETURN_TYPES = ("STRING", "FLOAT", "INT", "INT")
    RETURN_NAMES = ("info", "duration_s", "sample_rate", "num_frames")
    FUNCTION = "extract"
    CATEGORY = "SwanBits/Audio"
    DESCRIPTION = (
        "Reports duration (seconds), sample rate and the video frame count "
        "(derived from duration x fps) of a ComfyUI AUDIO input."
    )

    def extract(self, audio, fps):
        if not isinstance(audio, dict):
            raise ValueError("Swan Audio Info expects a ComfyUI AUDIO dict.")

        waveform = audio.get("waveform")
        sample_rate = audio.get("sample_rate")
        if waveform is None or sample_rate is None:
            raise ValueError("The AUDIO input is missing 'waveform' or 'sample_rate'.")
        if not isinstance(waveform, torch.Tensor) or waveform.ndim != 3:
            raise ValueError("Expected waveform tensor of shape [batch, channels, samples].")

        num_samples = int(waveform.shape[-1])
        sample_rate = int(sample_rate)
        fps = int(fps)

        duration_s = num_samples / float(sample_rate)
        num_frames = int(round(duration_s * fps))

        info = (
            f"duration:     {duration_s:.6f} s\n"
            f"sample rate:  {sample_rate} Hz\n"
            f"fps:          {fps}\n"
            f"video frames: {num_frames}"
        )
        return (info, duration_s, sample_rate, num_frames)
