// Stage-by-stage VAE decode bisection vs Python-oracle intermediates.
// Run: swift test --filter testVAEDecodeStages
// Fixture: tests/fixtures/vae_debug_stages.safetensors (tiny T=5 latent).

import Foundation
import MLX
import XCTest

@testable import MossSoundEffectMLX

final class VAEDebugTests: XCTestCase {
    func testVAEDecodeStages() throws {
        Device.setDefault(device: Device(.cpu))
        let fixtures = ParityTests.repoRoot
            .appendingPathComponent("tests/fixtures/vae_debug_stages.safetensors")
        let stages = try loadArrays(url: fixtures)

        let vae = DAC()
        try WeightLoading.loadConverted(
            vae,
            url: ParityTests.weightsDir.appendingPathComponent("mlx/vae.safetensors"),
            dtype: .float32)

        var h = stages["input"]!.transposed(0, 2, 1)
        h = vae.post_quant_conv(h)
        eval(h)
        var diff = MLX.abs(h - stages["post_quant_conv"]!).max().item(Float.self)
        print("post_quant_conv: max_abs=\(diff) |h|max=\(MLX.abs(h).max().item(Float.self))")

        for (i, layer) in vae.decoder.model.layers.enumerated() {
            h = layer(h)
            eval(h)
            let ref = stages["stage_\(i)"]!
            if h.shape != ref.shape {
                print("stage_\(i): SHAPE MISMATCH swift=\(h.shape) ref=\(ref.shape)")
                XCTFail("shape mismatch at stage \(i)")
                return
            }
            diff = MLX.abs(h - ref).max().item(Float.self)
            let hmax = MLX.abs(h).max().item(Float.self)
            print("stage_\(i): max_abs=\(diff) |h|max=\(hmax)")
        }
        XCTAssertLessThan(diff, 1e-3)
    }
}
