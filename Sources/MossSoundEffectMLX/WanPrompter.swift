// WanPrompter — Swift transpose of moss_sfx_mlx/prompter.py (parity-locked).
//
// Tokenization via swift-transformers (MLXEngine has no internal tokenizer,
// so the Qwen3 tokenizer ships with this package). Behavior contract:
//   * pad/truncate to text_len=512; Qwen3 pad token id 151643
//   * "" -> 0 valid tokens (Qwen3 adds no BOS) -> all-zero context
//   * pad-position embeddings zeroed after encoding
//
// Cleaning: upstream runs ftfy.fix_text + html.unescape(x2) + whitespace
// collapse. HTML unescape + whitespace collapse are ported; ftfy (mojibake
// repair) is NOT — a no-op for well-formed UTF-8 prompts, documented as a
// limitation for mojibake input.

import Foundation
import MLX
import Tokenizers

func whitespaceClean(_ text: String) -> String {
    let unescaped = htmlUnescape(htmlUnescape(text))
    let collapsed = unescaped.replacingOccurrences(
        of: "\\s+", with: " ", options: .regularExpression)
    return collapsed.trimmingCharacters(in: .whitespacesAndNewlines)
}

func htmlUnescape(_ text: String) -> String {
    guard text.contains("&") else { return text }
    var result = text
    for (entity, char) in [
        ("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"),
        ("&quot;", "\""), ("&#39;", "'"), ("&apos;", "'"), ("&nbsp;", " "),
    ] {
        result = result.replacingOccurrences(of: entity, with: char)
    }
    return result
}

public final class WanPrompter {
    public let textLen: Int
    public let padTokenId: Int
    let tokenizer: Tokenizer
    let textEncoder: Qwen3TextEncoder

    public init(
        tokenizer: Tokenizer, textEncoder: Qwen3TextEncoder,
        textLen: Int = 512, padTokenId: Int = 151_643
    ) {
        self.tokenizer = tokenizer
        self.textEncoder = textEncoder
        self.textLen = textLen
        self.padTokenId = padTokenId
    }

    /// Load the tokenizer from the HF snapshot's tokenizer/ directory.
    /// swift-transformers requires a config.json beside the tokenizer files
    /// (diffusers-style snapshots keep it in the component dirs instead), so
    /// stage the files with a minimal config in a temp dir when absent.
    public static func loadTokenizer(from directory: URL) async throws -> Tokenizer {
        let fm = FileManager.default
        if fm.fileExists(atPath: directory.appendingPathComponent("config.json").path) {
            return try await AutoTokenizer.from(modelFolder: directory)
        }
        let staged = fm.temporaryDirectory
            .appendingPathComponent("moss-sfx-tokenizer-\(directory.path.hashValue)")
        if !fm.fileExists(atPath: staged.path) {
            try fm.createDirectory(at: staged, withIntermediateDirectories: true)
            for file in try fm.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil) {
                try? fm.copyItem(at: file, to: staged.appendingPathComponent(file.lastPathComponent))
            }
            // tokenizer_config.json's tokenizer_class drives the choice; the
            // model config only needs to exist and name the family.
            try Data(#"{"model_type": "qwen3"}"#.utf8)
                .write(to: staged.appendingPathComponent("config.json"))
        }
        return try await AutoTokenizer.from(modelFolder: staged)
    }

    /// Tokenize with max_length padding/truncation — HF `padding='max_length',
    /// truncation=True, max_length=512` equivalent. Returns (ids, validLengths).
    public func tokenize(_ prompts: [String]) -> (ids: MLXArray, validLengths: [Int]) {
        var rows = [Int32]()
        var validLengths = [Int]()
        for prompt in prompts {
            var ids = tokenizer.encode(text: whitespaceClean(prompt))
            if ids.count > textLen { ids = Array(ids.prefix(textLen)) }
            validLengths.append(ids.count)
            ids += Array(repeating: padTokenId, count: textLen - ids.count)
            rows += ids.map(Int32.init)
        }
        return (MLXArray(rows, [prompts.count, textLen]), validLengths)
    }

    /// Full upstream encode_prompt: clean -> tokenize -> Qwen3 -> zero pads.
    public func encodePrompt(_ prompts: [String]) -> MLXArray {
        let (ids, validLengths) = tokenize(prompts)
        let emb = textEncoder(ids)
        return Qwen3TextEncoder.zeroPads(emb, validLengths: validLengths)
    }
}
