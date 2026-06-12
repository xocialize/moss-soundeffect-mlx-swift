// Qwen3 text encoder — Swift transpose of mlx-lm's qwen3.py (the oracle our
// parity-locked moss_sfx_mlx/model/qwen3_text_encoder.py wraps).
//
// Returns post-final-norm last-layer hidden states (== HF hidden_states[-1]).
// No LM head — Qwen3-1.7B ties embeddings and the SFX pipeline never decodes.
//
// Behavior contract (parity-locked, docs/upstream-findings.md):
//   * per-head RMSNorm on Q/K (head_dim 128) — NOT full-dim like the DiT
//   * GQA 16 query heads / 8 KV heads; plain RoPE theta=1e6, no scaling
//   * causal mask; right-padding is harmless pre-zeroing (pads can't attend
//     backward into real tokens' outputs)
//   * caller zeroes pad-position embeddings ("" -> all-zero context: the
//     Qwen3 tokenizer emits no BOS, so an empty prompt has 0 valid tokens)
//
// Weight keys match the HF shards directly (model.layers.N.self_attn.q_proj…).

import Foundation
import MLX
import MLXFast
import MLXNN

public struct Qwen3Config: Sendable {
    public var hiddenSize = 2048
    public var numHiddenLayers = 28
    public var intermediateSize = 6144
    public var numAttentionHeads = 16
    public var numKeyValueHeads = 8
    public var headDim = 128
    public var rmsNormEps: Float = 1e-6
    public var vocabSize = 151936
    public var ropeTheta: Float = 1_000_000

    public init() {}
}

final class Qwen3Attention: Module {
    let nHeads: Int
    let nKVHeads: Int
    let scale: Float

    @ModuleInfo var q_proj: Linear
    @ModuleInfo var k_proj: Linear
    @ModuleInfo var v_proj: Linear
    @ModuleInfo var o_proj: Linear
    @ModuleInfo var q_norm: RMSNorm
    @ModuleInfo var k_norm: RMSNorm
    let rope: RoPE

    init(_ config: Qwen3Config) {
        self.nHeads = config.numAttentionHeads
        self.nKVHeads = config.numKeyValueHeads
        self.scale = Float(1.0 / Double(config.headDim).squareRoot())
        let dim = config.hiddenSize
        self.q_proj = Linear(dim, nHeads * config.headDim, bias: false)
        self.k_proj = Linear(dim, nKVHeads * config.headDim, bias: false)
        self.v_proj = Linear(dim, nKVHeads * config.headDim, bias: false)
        self.o_proj = Linear(nHeads * config.headDim, dim, bias: false)
        // Per-head RMSNorm over head_dim (Qwen3 signature feature).
        self.q_norm = RMSNorm(dimensions: config.headDim, eps: config.rmsNormEps)
        self.k_norm = RMSNorm(dimensions: config.headDim, eps: config.rmsNormEps)
        self.rope = RoPE(dimensions: config.headDim, traditional: false, base: config.ropeTheta)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray?) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)

        var queries = q_norm(q_proj(x).reshaped(B, L, nHeads, -1)).transposed(0, 2, 1, 3)
        var keys = k_norm(k_proj(x).reshaped(B, L, nKVHeads, -1)).transposed(0, 2, 1, 3)
        let values = v_proj(x).reshaped(B, L, nKVHeads, -1).transposed(0, 2, 1, 3)

        queries = rope(queries)
        keys = rope(keys)

        let output = MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values, scale: scale, mask: mask)
        return o_proj(output.transposed(0, 2, 1, 3).reshaped(B, L, -1))
    }
}

final class Qwen3MLP: Module, UnaryLayer {
    @ModuleInfo var gate_proj: Linear
    @ModuleInfo var down_proj: Linear
    @ModuleInfo var up_proj: Linear

    init(dim: Int, hiddenDim: Int) {
        self.gate_proj = Linear(dim, hiddenDim, bias: false)
        self.down_proj = Linear(hiddenDim, dim, bias: false)
        self.up_proj = Linear(dim, hiddenDim, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        down_proj(silu(gate_proj(x)) * up_proj(x))
    }
}

final class Qwen3TransformerBlock: Module {
    @ModuleInfo var self_attn: Qwen3Attention
    @ModuleInfo var mlp: Qwen3MLP
    @ModuleInfo var input_layernorm: RMSNorm
    @ModuleInfo var post_attention_layernorm: RMSNorm

    init(_ config: Qwen3Config) {
        self.self_attn = Qwen3Attention(config)
        self.mlp = Qwen3MLP(dim: config.hiddenSize, hiddenDim: config.intermediateSize)
        self.input_layernorm = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self.post_attention_layernorm = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray?) -> MLXArray {
        let h = x + self_attn(input_layernorm(x), mask: mask)
        return h + mlp(post_attention_layernorm(h))
    }
}

final class Qwen3Model: Module {
    @ModuleInfo var embed_tokens: Embedding
    @ModuleInfo var layers: [Qwen3TransformerBlock]
    @ModuleInfo var norm: RMSNorm

    init(_ config: Qwen3Config) {
        self.embed_tokens = Embedding(embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
        self.layers = (0 ..< config.numHiddenLayers).map { _ in Qwen3TransformerBlock(config) }
        self.norm = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        super.init()
    }

    func callAsFunction(_ inputs: MLXArray) -> MLXArray {
        var h = embed_tokens(inputs)
        let L = h.dim(1)
        let mask: MLXArray? =
            L > 1
            ? MultiHeadAttention.createAdditiveCausalMask(L).asType(h.dtype)
            : nil
        for layer in layers {
            h = layer(h, mask: mask)
        }
        return norm(h)  // post-final-norm == HF hidden_states[-1]
    }
}

public final class Qwen3TextEncoder: Module {
    @ModuleInfo var model: Qwen3Model

    public init(config: Qwen3Config = Qwen3Config()) {
        self.model = Qwen3Model(config)
        super.init()
    }

    /// Load the original HF bf16 shards (model*.safetensors) from the
    /// text_encoder directory; no conversion needed.
    public func loadWeights(from directory: URL, dtype: DType = .bfloat16) throws {
        let fm = FileManager.default
        let shards = try fm.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
            .filter { $0.lastPathComponent.hasPrefix("model") && $0.pathExtension == "safetensors" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        var weights = [String: MLXArray]()
        for shard in shards {
            for (k, v) in try loadArrays(url: shard) {
                weights[k] = v.asType(dtype)
            }
        }
        weights.removeValue(forKey: "lm_head.weight")  // tied; unused
        try update(parameters: ModuleParameters.unflattened(weights), verify: .noUnusedKeys)
        eval(self)
    }

    /// ids: (B, L) token ids -> (B, L, hidden) last-layer hidden states.
    /// Pad-position zeroing is the caller's job (mirrors WanPrompter).
    public func callAsFunction(_ ids: MLXArray) -> MLXArray {
        model(ids)
    }

    /// WanPrompter.encode_prompt tail: zero embeddings past each row's valid length.
    public static func zeroPads(_ embeddings: MLXArray, validLengths: [Int]) -> MLXArray {
        let L = embeddings.dim(1)
        let positions = MLXArray(0 ..< L).reshaped(1, L, 1)
        let valid = MLXArray(validLengths.map { Int32($0) }).reshaped(validLengths.count, 1, 1)
        return embeddings * (positions .< valid).asType(embeddings.dtype)
    }
}
