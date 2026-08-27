# ComfyUI_Swan_Bits

个人零散节点收录，开放给大家用。/ A small personal collection of ComfyUI custom nodes. Free to use.

## 节点 / Nodes

### Swan MiniMax H3 Audio Drive
锁死源音频、只生成视频，用于 MiniMax H3 音频驱动视频。/ Locks the source audio and generates video only, for MiniMax H3 audio-driven video.

基于 [comfyui-vrgamedevgirl](https://github.com/vrgamegirl19/comfyui-vrgamedevgirl) 的 `VRGDG MiniMax H3 Audio Drive`（作者 VRGameDevGirl），感谢原作者。/ Based on `VRGDG MiniMax H3 Audio Drive` from [comfyui-vrgamedevgirl](https://github.com/vrgamegirl19/comfyui-vrgamedevgirl) by **VRGameDevGirl**. Thanks to the original author.

- 输入 / Inputs: `av_latent` (LATENT), `source_audio` (AUDIO), `audio_vae` (VAE)
- 输出 / Outputs: `audio_driven_av_latent` (LATENT), `original_audio` (AUDIO)

### Swan Audio Info
读取官方 `Load Audio` 节点输出，返回音频相关信息。/ Reads the official `Load Audio` node output and reports audio details.

- 输入 / Inputs: `audio` (AUDIO), `fps` (INT, 默认 24 / default 24)
- 输出 / Outputs: `info` (STRING), `duration_s` (FLOAT, 秒 / seconds), `sample_rate` (INT, 单位 Hz), `num_frames` (INT, 由 duration × fps 得出 / derived from duration × fps)

## 安装 / Install
把本文件夹复制到 `ComfyUI/custom_nodes/` 并重启 ComfyUI。/ Copy this folder into `ComfyUI/custom_nodes/` and restart ComfyUI.
