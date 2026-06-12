// WanAudioModel — Swift transpose of moss_sfx_mlx/model/wan_audio_dit.py
// (parity-locked MLX-Python oracle, itself isomorphic to upstream PyTorch).
//
// Weight keys match the converted dit.safetensors directly:
//   patch_embedding.{weight,bias}            (Conv1d, MLX (O,K,I) layout)
//   text_embedding.layers.{0,2}.{weight,bias}
//   time_embedding.layers.{0,2}.{weight,bias}
//   time_projection.layers.1.{weight,bias}
//   blocks.<i>.…                              (see Attention.swift)
//   head.head.{weight,bias}, head.modulation
//
// RoPE tables are NOT stored as MLXArray properties — every stored MLXArray
// becomes a tracked parameter and would break update(parameters:verify:).
// They live in a plain (non-Module) cache class instead.

import Foundation
import MLX
import MLXNN

public struct WanAudioModelConfig: Sendable {
    public var dim = 1536
    public var inDim = 128
    public var ffnDim = 8960
    public var outDim = 128
    public var textDim = 2048
    public var freqDim = 256
    public var eps: Float = 1e-6
    public var patchSize = 1
    public var numHeads = 12
    public var numLayers = 30

    public init() {}
}

/// Plain class (deliberately not a Module): holds the precomputed RoPE tables
/// outside parameter tracking.
final class RoPEFreqsCache {
    // Oracle: precompute_freqs_cis_1d(head_dim) -> torch.chunk(3) -> re-concat
    // in forward. chunk-then-concat restores the original, so the net table is
    // plain full-head_dim 1-D RoPE; the 3-way split is kept upstream only as a
    // 3D-video vestige. We store the already-concatenated table.
    let cos: MLXArray
    let sin: MLXArray

    init(headDim: Int, end: Int = 16384) {
        let (c, s) = precomputeFreqsCis(dim: headDim, end: end)
        self.cos = c
        self.sin = s
    }

    /// (cos, sin) sliced to f positions, shaped [f, 1, headDim/2].
    func sliced(_ f: Int) -> (cos: MLXArray, sin: MLXArray) {
        (
            cos[0 ..< f].reshaped(f, 1, -1),
            sin[0 ..< f].reshaped(f, 1, -1)
        )
    }
}

public class WanAudioModel: Module {
    let config: WanAudioModelConfig
    private let freqsCache: RoPEFreqsCache

    @ModuleInfo var patch_embedding: Conv1d
    @ModuleInfo var text_embedding: Sequential
    @ModuleInfo var time_embedding: Sequential
    @ModuleInfo var time_projection: Sequential
    @ModuleInfo var blocks: [DiTBlock]
    @ModuleInfo var head: Head

    public init(_ config: WanAudioModelConfig = WanAudioModelConfig()) {
        self.config = config
        self.freqsCache = RoPEFreqsCache(headDim: config.dim / config.numHeads)

        self.patch_embedding = Conv1d(
            inputChannels: config.inDim, outputChannels: config.dim,
            kernelSize: config.patchSize, stride: config.patchSize)
        self.text_embedding = Sequential(layers: [
            Linear(config.textDim, config.dim),
            GELU(approximation: .precise),
            Linear(config.dim, config.dim),
        ])
        self.time_embedding = Sequential(layers: [
            Linear(config.freqDim, config.dim),
            SiLU(),
            Linear(config.dim, config.dim),
        ])
        self.time_projection = Sequential(layers: [
            SiLU(),
            Linear(config.dim, config.dim * 6),
        ])
        self.blocks = (0 ..< config.numLayers).map { _ in
            DiTBlock(dim: config.dim, numHeads: config.numHeads, ffnDim: config.ffnDim, eps: config.eps)
        }
        self.head = Head(dim: config.dim, outDim: config.outDim, patchSize: config.patchSize, eps: config.eps)
        super.init()
    }

    /// x: (B, C, T) channel-first like the oracle; conv runs channel-last.
    func patchify(_ x: MLXArray) -> (MLXArray, Int) {
        let h = patch_embedding(x.transposed(0, 2, 1))  // (B, f, dim)
        return (h, h.dim(1))
    }

    func unpatchify(_ x: MLXArray, f: Int) -> MLXArray {
        // rearrange 'b f (p c) -> b c (f p)'
        let p = config.patchSize
        let b = x.dim(0)
        let h = x.reshaped(b, f, p, -1)         // (b, f, p, c)
            .transposed(0, 3, 1, 2)             // (b, c, f, p)
        return h.reshaped(b, h.dim(1), f * p)   // (b, c, f*p)
    }

    /// x: (B, 128, T) latents; timestep: (B,); context: (B, 512, 2048).
    /// Returns velocity (B, 128, T).
    public func callAsFunction(
        _ x: MLXArray, timestep: MLXArray, context: MLXArray
    ) -> MLXArray {
        let t = time_embedding(sinusoidalEmbedding1d(config.freqDim, timestep))
        let tMod = time_projection(t).reshaped(t.dim(0), 6, config.dim)
        let ctx = text_embedding(context)

        var (h, f) = patchify(x)
        let freqs = freqsCache.sliced(f)

        for block in blocks {
            h = block(h, context: ctx, tMod: tMod, freqs: freqs)
        }

        h = head(h, tMod: t)
        return unpatchify(h, f: f)
    }
}
