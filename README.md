# moss-soundeffect-mlx-swift

MLX-Swift port of [MOSS-SoundEffect-v2.0](https://huggingface.co/OpenMOSS-Team/MOSS-SoundEffect-v2.0)
(OpenMOSS) — text → sound effects (foley / ambience / creature / action), 48 kHz, ≤ 30 s —
for Apple Silicon. Apache-2.0.

Split from [xocialize/moss-soundeffect-mlx](https://github.com/xocialize/moss-soundeffect-mlx)
(the Python parity oracle + conversion tooling + golden fixtures); every module here is
validated against that oracle's golden tensors on the CPU stream.

Weights: [mlx-community/MOSS-SoundEffect-v2.0-bf16](https://huggingface.co/mlx-community/MOSS-SoundEffect-v2.0-bf16) ·
[mlx-community/MOSS-SoundEffect-v2.0-4bit](https://huggingface.co/mlx-community/MOSS-SoundEffect-v2.0-4bit)

```swift
import MossSoundEffectMLX

// directory = a downloaded mlx-community/MOSS-SoundEffect-v2.0-{bf16,4bit} snapshot.
// It must contain model_index.json + mlx/{dit,vae}.safetensors + text_encoder/ + tokenizer/.
// Quantization (for -4bit) is auto-detected from model_index.json — no extra args.
let pipe = try await MossSoundEffectPipeline.load(from: directory)
let audio = try pipe.generate(prompt: "a heavy wooden door creaks open slowly", seconds: 5)
// audio: (1, 1, samples) MLXArray at 48 kHz, in [-1, 1]
```

`load(from:)` is `async throws`; `generate(prompt:seconds:…)` is synchronous `throws`
(it accepts `negativePrompt`, `numInferenceSteps`, `cfgScale`, `sigmaShift`, `seed`,
`appendDurationSuffix`, and an `onStep` cancellation hook). A lower-level
`generate(context:contextNega:noise:…)` overload denoises pre-encoded contexts.

Components (Swift sources): `WanAudioModel` (1.3B flow-matching DiT, `WanAudioModel.swift`),
`DAC` (continuous 128-d VAE, 50 Hz latents — `decode` + `encodeMode`, `DACVAEDecoder.swift`),
`Qwen3TextEncoder` + `WanPrompter` (tokenization ships in-package via swift-transformers),
`FlowMatchScheduler`, plus the attention stack (`SelfAttention`/`CrossAttention`/`DiTBlock`/`Head`).
The MLXEngine capability package wrapping this is
[xocialize/mlx-moss-soundeffect-swift](https://github.com/xocialize/mlx-moss-soundeffect-swift).

## Dependencies

- [mlx-swift](https://github.com/ml-explore/mlx-swift) (MLX / MLXNN / MLXFast)
- [swift-transformers](https://github.com/huggingface/swift-transformers) (Qwen3 tokenizer, in-package)

Platforms: macOS 14+, iOS 16+.

## Tests

```bash
swift test    # fast tests run on the CPU stream (no metallib needed)
```

The heavy golden-parity tests are opt-in: set `MOSS_SFX_RUN_SLOW=1`, and point
`MOSS_SFX_REPO` at the Python repo (for fixtures) and `MOSS_SFX_MLX_WEIGHTS_DIR`
at a converted weights snapshot.
