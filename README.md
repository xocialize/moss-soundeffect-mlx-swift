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

// directory = a downloaded mlx-community/MOSS-SoundEffect-v2.0-{bf16,4bit} snapshot
let pipe = try await MossSoundEffectPipeline.load(from: directory)
let audio = try pipe.generate(prompt: "a heavy wooden door creaks open slowly", seconds: 5)
// audio: (1, 1, samples) MLXArray at 48 kHz, in [-1, 1]
```

Components: `WanAudioModel` (1.3B flow-matching DiT), `DAC` (continuous 128-d VAE, 50 Hz
latents), `Qwen3TextEncoder` + `WanPrompter` (tokenization ships in-package via
swift-transformers), `FlowMatchScheduler`. The MLXEngine capability package wrapping this
is `xocialize/mlx-moss-soundeffect-swift`.

Tests run on the CPU stream (no metallib needed): `swift test`. The heavy golden-parity
tests need the fixtures from the Python repo (`MOSS_SFX_REPO` env var) and the weights
snapshot (`MOSS_SFX_MLX_WEIGHTS_DIR`).
