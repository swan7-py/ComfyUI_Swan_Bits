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

### Swan Load H3 Latent Upscaler Model
加载 MiniMax H3 学习式 latent 上采样权重（`models/latent_upscale_models/`），输出 `H3_LATENT_UPSCALER` Provider 句柄。该句柄既可接本包的 Transition Lift 节点，也兼容 [MiniMax-H3-Flow-Aligned-Regenerate](https://github.com/xmarre/MiniMax-H3-Flow-Aligned-Regenerate) 的 `learned_upscaler` 输入。/ Loads a learned H3 latent-upscaler checkpoint and outputs an `H3_LATENT_UPSCALER` provider handle, compatible with both the Swan transition lift and the `learned_upscaler` input of MiniMax-H3-Flow-Aligned-Regenerate.

- 输入 / Inputs: `model_name` (COMBO), `device` (cuda/cpu), `precision` (fp16/fp32/bf16), `offload_after_upscale` (BOOLEAN)
- 输出 / Outputs: `learned_upscaler` (H3_LATENT_UPSCALER)

### Swan Sigmas Split: Low Runs to Zero
把一条 sigma 日程拆成两段，用于"低清采样 → 提升 → 高清采样"的两段式渐进工作流：低清段以补零收尾（完全去噪，得到干净 x0），高清尾段从拆分点 sigma 继续。总评估次数 N+1。/ Splits one sigma schedule for a two-stage progressive workflow: the low-res stage ends at zero (fully denoised, handing a clean x0 to the lift), and the high tail resumes from the split sigma. Total NFE becomes N + 1.

- 输入 / Inputs: `sigmas` (SIGMAS), `low_steps` (INT)
- 输出 / Outputs: `low_sigmas` (SIGMAS), `high_sigmas` (SIGMAS)

### Swan SelfLift Transition Lift (H3)
两段式采样之间的 SelfLift-zero 过渡修复节点：对低清干净端点做双路提升——直接 latent 提升（可接学习式上采样，否则 nearest/bilinear 插值）+ VAE 像素重编码锚——以两者残差为伪影风险图，把 top-rho 高风险位置向像素锚修正（arXiv:2609.02036）。`rho=0` 时跳过像素路线，仅做直接提升。只提升视频流，音频流原样直通。输出干净的目标尺寸 latent，重加噪由下游 SamplerCustom 的 `add_noise` 完成。/ SelfLift-zero transition between two samplers: paired direct lift (learned upscaler or interpolation) plus a VAE decode→upscale→re-encode pixel anchor; the residual becomes an artifact-risk map and the top-rho risky locations are corrected toward the anchor. `rho=0` keeps the direct lift only. Video stream is lifted, audio passes through untouched. Re-noising happens in the next SamplerCustom via `add_noise`.

- 输入 / Inputs: `lowres_latent` (LATENT), `target_scale` (FLOAT), `direct_lift` (learned/nearest/bilinear), `rho` (FLOAT), `w_min` / `w_max` (FLOAT), 可选 / optional: `highres_latent` (LATENT), `vae` (VAE), `upscaler` (H3_LATENT_UPSCALER)
- 输出 / Outputs: `highres_latent` (LATENT)

参考接法 / Reference wiring:
```
SamplerCustom (低清 / low-res, low_sigmas, add_noise=True)
  → Swan SelfLift Transition Lift → SamplerCustom (高清 / high-res, high_sigmas, add_noise=True)
```

## 鸣谢 / Acknowledgements

- [SelfLift: Accelerating Few-Step Diffusion via Self-Recovering Resolution Transition](https://arxiv.org/abs/2609.02036) —— SelfLift-zero 过渡算法来源；ComfyUI 实现参考 [facok/comfyui-SelfLift](https://github.com/facok/comfyui-SelfLift)，感谢原作者。/ Source of the SelfLift-zero transition algorithm; ComfyUI implementation reference.
- [LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler) —— H3 学习式上采样权重与网络架构来源，本包内的模型加载代码移植自该项目（自包含、无运行时依赖），感谢原作者。/ Source of the H3 learned upscaler weights and network architecture; the model-loading code in this package is a self-contained port of that project (no runtime dependency). Thanks to the original author.

## 安装 / Install
把本文件夹复制到 `ComfyUI/custom_nodes/` 并重启 ComfyUI。/ Copy this folder into `ComfyUI/custom_nodes/` and restart ComfyUI.
