// Swift parity tests vs the MLX-Python golden fixtures.
//
// Everything runs on the CPU stream: no metallib needed (works under plain
// `swift test`), and CPU fp32 is bitwise-stable vs the Python oracle —
// GPU fp32 matmul is tf32-like and would mask real bugs.
//
// Inputs come from tests/fixtures/swift_goldens.safetensors (export via
// scripts/export_swift_fixtures.py); weights from the original checkpoint
// (DiT, fp32 parity master) and the converted vae.safetensors (fp32).
//
// The full denoise-loop test is ~22 DiT passes on CPU (15-20 min) — gated
// behind MOSS_SFX_RUN_SLOW=1.

import Foundation
import MLX
import XCTest

@testable import MossSoundEffectMLX

final class ParityTests: XCTestCase {
    /// Root of the Python oracle repo (moss-soundeffect-mlx) — golden fixtures live there.
    static let repoRoot: URL = {
        if let env = ProcessInfo.processInfo.environment["MOSS_SFX_REPO"] {
            return URL(fileURLWithPath: env)
        }
        // …/Tests/MossSoundEffectMLXTests/ParityTests.swift -> this repo root,
        // then the sibling oracle repo (dev-machine layout).
        return URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("moss-soundeffect-mlx")
    }()

    static let weightsDir: URL = {
        if let env = ProcessInfo.processInfo.environment["MOSS_SFX_MLX_WEIGHTS_DIR"] {
            return URL(fileURLWithPath: env)
        }
        return URL(fileURLWithPath: "/Volumes/DEV_ARCHIVE/weights/MOSS-SoundEffect-v2.0")
    }()

    static var goldens: [String: MLXArray] = [:]

    override class func setUp() {
        super.setUp()
        Device.setDefault(device: Device(.cpu))
        let url = repoRoot
            .appendingPathComponent("tests/fixtures/swift_goldens.safetensors")
        // loadArrays is LAZY and mmap-backed: rewriting this file while tests
        // run poisons unevaluated tensors (observed: NaN + 10-100x slowdown
        // from denormal garbage). Never re-export fixtures during a test run.
        goldens = (try? loadArrays(url: url)) ?? [:]
    }

    private func requireGoldens() throws {
        try XCTSkipIf(Self.goldens.isEmpty, "run scripts/export_swift_fixtures.py first")
    }

    private func maxAbsDiff(_ a: MLXArray, _ b: MLXArray) -> Float {
        MLX.abs(a - b).max().item(Float.self)
    }

    // MARK: - Scheduler (pure math, no weights)

    func testSchedulerSigmaSchedule() {
        let scheduler = FlowMatchScheduler(
            numInferenceSteps: 10, shift: 5.0, sigmaMin: 0.0, extraOneStep: true)
        // Expected values from the parity-locked Python oracle:
        // sigmas = linspace(1, 0, 11)[:-1]; s = 5s/(1+4s)
        let expected: [Float] = [
            1.0, 0.9783, 0.9524, 0.9211, 0.8824, 0.8333, 0.7692, 0.6818, 0.5556, 0.3571,
        ]
        XCTAssertEqual(scheduler.sigmas.count, expected.count)
        for (got, want) in zip(scheduler.sigmas, expected) {
            XCTAssertEqual(got, want, accuracy: 1e-4)
        }
    }

    // MARK: - VAE decode (golden latent -> golden audio)

    func testVAEDecodeGoldenLatent() throws {
        try requireGoldens()
        let vae = DAC()
        try WeightLoading.loadConverted(
            vae, url: Self.weightsDir.appendingPathComponent("mlx/vae.safetensors"),
            dtype: .float32)

        let audio = vae.decode(Self.goldens["golden_final_latent"]!)
        eval(audio)
        XCTAssertEqual(audio.shape, [1, 1, 1_440_000])
        let diff = maxAbsDiff(audio, Self.goldens["golden_audio"]!)
        XCTAssertLessThan(diff, 1e-2, "VAE decode diverges: max_abs=\(diff)")
    }

    // MARK: - DiT velocity (golden noise + context -> golden velocity)

    func testDiTVelocityParity() throws {
        try requireGoldens()
        let dit = WanAudioModel()
        try WeightLoading.loadDiTFromOriginal(
            dit,
            url: Self.weightsDir.appendingPathComponent(
                "transformer/diffusion_pytorch_model.safetensors"))

        let x = Self.goldens["golden_noise"]!
        let ctx = Self.goldens["golden_context"]!

        for (tVal, goldenKey) in [(Float(1000), "golden_velocity_t1000"), (Float(500), "golden_velocity_t500")] {
            let v = dit(x, timestep: MLXArray([tVal]), context: ctx)
            eval(v)
            let diff = maxAbsDiff(v, Self.goldens[goldenKey]!)
            XCTAssertLessThan(diff, 1e-2, "velocity t=\(tVal) diverges: max_abs=\(diff)")
        }
    }

    // MARK: - Qwen3 text encoder (golden ids -> golden context)

    func testQwen3EncoderParity() throws {
        try requireGoldens()
        try XCTSkipIf(
            Self.goldens["golden_ids"] == nil,
            "re-run scripts/export_swift_fixtures.py to add token fixtures")

        let encoder = Qwen3TextEncoder()
        try encoder.loadWeights(
            from: Self.weightsDir.appendingPathComponent("text_encoder"),
            dtype: .float32)

        let ids = Self.goldens["golden_ids"]!
        let mask = Self.goldens["golden_mask"]!
        let validLen = mask.sum().item(Int.self)

        var emb = encoder(ids)
        emb = Qwen3TextEncoder.zeroPads(emb, validLengths: [validLen])
        eval(emb)

        // Same gate as the Python oracle test: Qwen3's massive activations
        // (~1.2e4 mid-stack) put the fp32 accumulation floor near 1e-3.
        let diff = maxAbsDiff(emb, Self.goldens["golden_context"]!)
        XCTAssertLessThan(diff, 2e-3, "text encoder diverges: max_abs=\(diff)")
    }

    // MARK: - Full denoise loop + decode (slow; MOSS_SFX_RUN_SLOW=1)

    func testDenoiseLoopAndDecodeParity() throws {
        try requireGoldens()
        try XCTSkipIf(
            ProcessInfo.processInfo.environment["MOSS_SFX_RUN_SLOW"] != "1",
            "set MOSS_SFX_RUN_SLOW=1 (~45 min on CPU; resumable)")

        // Resumable: the latent is checkpointed after every denoise step
        // (~4 min each on the CPU stream), so a killed run continues from the
        // last completed step instead of restarting the whole loop.
        // Gate: final latent vs golden < 1e-2 fp32. The decode seam is gated
        // separately (testVAEDecodeGoldenLatent decodes the golden latent at
        // < 1e-2, and the staged VAE test is bitwise-exact vs the oracle).
        let steps = 10  // golden_meta.json: steps=10, cfg=4.0, shift=5.0
        let cfg: Float = 4.0
        let checkpointDir = URL(fileURLWithPath: "/tmp/moss-sfx-e2e-checkpoints")
        try FileManager.default.createDirectory(
            at: checkpointDir, withIntermediateDirectories: true)
        func checkpointURL(_ step: Int) -> URL {
            checkpointDir.appendingPathComponent("latent_after_step_\(step).safetensors")
        }

        var startStep = 0
        var latents = Self.goldens["golden_noise"]!
        for step in stride(from: steps - 1, through: 0, by: -1) {
            if let resumed = try? loadArrays(url: checkpointURL(step)),
                let arr = resumed["latents"]
            {
                latents = arr
                startStep = step + 1
                print("resuming after step \(step)")
                break
            }
        }

        if startStep < steps {
            let dit = WanAudioModel()
            try WeightLoading.loadDiTFromOriginal(
                dit,
                url: Self.weightsDir.appendingPathComponent(
                    "transformer/diffusion_pytorch_model.safetensors"))

            let scheduler = FlowMatchScheduler(
                numInferenceSteps: steps, shift: 5.0, sigmaMin: 0.0, extraOneStep: true)
            let ctx = Self.goldens["golden_context"]!
            let ctxNega = Self.goldens["golden_context_nega"]!

            for i in startStep ..< steps {
                let t = scheduler.timesteps[i]
                let timestep = MLXArray([t])
                let vPosi = dit(latents, timestep: timestep, context: ctx)
                let vNega = dit(latents, timestep: timestep, context: ctxNega)
                let v = vNega + cfg * (vPosi - vNega)
                latents = scheduler.step(v, timestep: t, sample: latents)
                eval(latents)
                try save(arrays: ["latents": latents], url: checkpointURL(i))
                print("step \(i + 1)/\(steps) checkpointed")
            }
        }

        let diff = maxAbsDiff(latents, Self.goldens["golden_final_latent"]!)
        XCTAssertLessThan(diff, 1e-2, "final latent diverges: max_abs=\(diff)")
        // Green: clean up so future runs start fresh.
        try? FileManager.default.removeItem(at: checkpointDir)
    }
}
