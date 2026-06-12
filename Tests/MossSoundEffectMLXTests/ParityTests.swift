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
    static let repoRoot: URL = {
        if let env = ProcessInfo.processInfo.environment["MOSS_SFX_REPO"] {
            return URL(fileURLWithPath: env)
        }
        // …/swift/Tests/MossSoundEffectMLXTests/ParityTests.swift -> repo root
        return URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
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
            "set MOSS_SFX_RUN_SLOW=1 (~20 min on CPU)")

        let dit = WanAudioModel()
        try WeightLoading.loadDiTFromOriginal(
            dit,
            url: Self.weightsDir.appendingPathComponent(
                "transformer/diffusion_pytorch_model.safetensors"))
        let vae = DAC()
        try WeightLoading.loadConverted(
            vae, url: Self.weightsDir.appendingPathComponent("mlx/vae.safetensors"),
            dtype: .float32)

        let pipeline = MossSoundEffectPipeline(dit: dit, vae: vae)
        // golden_meta.json: steps=10, cfg=4.0, shift=5.0
        let audio = pipeline.generate(
            context: Self.goldens["golden_context"]!,
            contextNega: Self.goldens["golden_context_nega"]!,
            noise: Self.goldens["golden_noise"]!,
            numInferenceSteps: 10,
            cfgScale: 4.0,
            sigmaShift: 5.0)

        let diff = maxAbsDiff(audio, Self.goldens["golden_audio"]!)
        XCTAssertLessThan(diff, 1e-2, "e2e audio diverges: max_abs=\(diff)")
    }
}
