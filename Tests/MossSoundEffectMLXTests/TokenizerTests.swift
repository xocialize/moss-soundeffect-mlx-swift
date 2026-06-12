// Tokenizer parity: swift-transformers Qwen3 tokenizer vs the golden ids
// produced by HF transformers (padding='max_length', max_length=512).

import Foundation
import MLX
import XCTest

@testable import MossSoundEffectMLX

final class TokenizerTests: XCTestCase {
    func testTokenizerMatchesGoldenIds() async throws {
        Device.setDefault(device: Device(.cpu))
        let goldens = try loadArrays(
            url: ParityTests.repoRoot.appendingPathComponent(
                "tests/fixtures/swift_goldens.safetensors"))
        guard let goldenIds = goldens["golden_ids"], let goldenMask = goldens["golden_mask"]
        else {
            throw XCTSkip("re-run scripts/export_swift_fixtures.py")
        }

        let tokenizer = try await WanPrompter.loadTokenizer(
            from: ParityTests.weightsDir.appendingPathComponent("tokenizer"))

        // golden_meta: prompt + " duration: 10.0s" suffix (the trained format).
        let prompt = "a heavy wooden door creaks open slowly duration: 10.0s"
        var ids = tokenizer.encode(text: whitespaceClean(prompt))
        let validLen = goldenMask.sum().item(Int.self)
        XCTAssertEqual(ids.count, validLen, "token count differs from HF")

        ids += Array(repeating: 151_643, count: 512 - ids.count)
        let goldenRow: [Int32] = goldenIds[0].asArray(Int32.self)
        XCTAssertEqual(ids.map(Int32.init), goldenRow, "token ids differ from HF")

        // "" -> zero valid tokens (no BOS) — the all-zero negative context.
        XCTAssertEqual(tokenizer.encode(text: "").count, 0)
    }
}
