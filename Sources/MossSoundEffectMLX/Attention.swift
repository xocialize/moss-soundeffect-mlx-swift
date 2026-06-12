// DiT building blocks — Swift transpose of moss_sfx_mlx/model/wan_video_dit.py
// (parity-locked MLX-Python oracle). Property names are snake_case on purpose:
// they must produce the exact flattened keys of the converted safetensors
// (e.g. "blocks.0.self_attn.norm_q.weight", "ffn.layers.0.weight").
//
// RoPE is carried as a (cos, sin) pair of fp32 arrays — interleaved even/odd
// pair rotation in fp32, cast back (oracle wan_video_dit.py rope_apply).

import Foundation
import MLX
import MLXFast
import MLXNN

func flashAttention(q: MLXArray, k: MLXArray, v: MLXArray, numHeads: Int) -> MLXArray {
    let b = q.dim(0)
    let s = q.dim(1)
    let qh = q.reshaped(b, q.dim(1), numHeads, -1).transposed(0, 2, 1, 3)
    let kh = k.reshaped(b, k.dim(1), numHeads, -1).transposed(0, 2, 1, 3)
    let vh = v.reshaped(b, v.dim(1), numHeads, -1).transposed(0, 2, 1, 3)
    let scale = Float(1.0 / Double(qh.dim(-1)).squareRoot())
    let x = MLXFast.scaledDotProductAttention(
        queries: qh, keys: kh, values: vh, scale: scale, mask: nil)
    return x.transposed(0, 2, 1, 3).reshaped(b, s, -1)
}

func modulate(_ x: MLXArray, shift: MLXArray, scale: MLXArray) -> MLXArray {
    x * (1 + scale) + shift
}

func sinusoidalEmbedding1d(_ dim: Int, _ position: MLXArray) -> MLXArray {
    // Oracle computes fp32 (PyTorch reference used fp64; parity-validated).
    let half = dim / 2
    let freqs = MLX.pow(
        MLXArray(Float(10000)),
        -MLXArray(0 ..< half).asType(.float32) / Float(half))
    let sinusoid = position.asType(.float32).expandedDimensions(axis: 1) * freqs.expandedDimensions(axis: 0)
    return MLX.concatenated([MLX.cos(sinusoid), MLX.sin(sinusoid)], axis: 1)
}

/// 1-D RoPE table: cos/sin of outer(pos, theta^(-2i/dim)), computed in Double
/// like the oracle's float64 precompute, stored fp32.
func precomputeFreqsCis(dim: Int, end: Int = 16384, theta: Double = 10000.0, s: Double = 1.0)
    -> (cos: MLXArray, sin: MLXArray)
{
    let half = dim / 2
    var cosVals = [Float](repeating: 0, count: end * half)
    var sinVals = [Float](repeating: 0, count: end * half)
    for i in 0 ..< half {
        let freq = 1.0 / pow(theta, Double(2 * i) / Double(dim))
        for p in 0 ..< end {
            let angle = Double(p) * s * freq
            cosVals[p * half + i] = Float(cos(angle))
            sinVals[p * half + i] = Float(sin(angle))
        }
    }
    return (MLXArray(cosVals, [end, half]), MLXArray(sinVals, [end, half]))
}

func ropeApply(_ x: MLXArray, freqs: (cos: MLXArray, sin: MLXArray), numHeads: Int) -> MLXArray {
    // Interleaved even/odd pairs, fp32 compute, cast back — oracle rope_apply.
    let outDtype = x.dtype
    let b = x.dim(0)
    let s = x.dim(1)
    var xh = x.reshaped(b, s, numHeads, -1).asType(.float32)
    let pairCount = xh.dim(-1) / 2
    xh = xh.reshaped(b, s, numHeads, pairCount, 2)
    let xEven = xh[.ellipsis, 0]
    let xOdd = xh[.ellipsis, 1]
    let cos = freqs.cos  // [s, 1, d/2]
    let sin = freqs.sin
    let out = MLX.stacked(
        [xEven * cos - xOdd * sin, xEven * sin + xOdd * cos], axis: -1
    ).reshaped(b, s, -1)
    return out.asType(outDtype)
}

public class SelfAttention: Module {
    let num_heads: Int

    @ModuleInfo var q: Linear
    @ModuleInfo var k: Linear
    @ModuleInfo var v: Linear
    @ModuleInfo var o: Linear
    @ModuleInfo var norm_q: RMSNorm
    @ModuleInfo var norm_k: RMSNorm

    init(dim: Int, numHeads: Int, eps: Float = 1e-6) {
        self.num_heads = numHeads
        // RMSNorm over the FULL model dim before the head split (oracle note).
        self.q = Linear(dim, dim)
        self.k = Linear(dim, dim)
        self.v = Linear(dim, dim)
        self.o = Linear(dim, dim)
        self.norm_q = RMSNorm(dimensions: dim, eps: eps)
        self.norm_k = RMSNorm(dimensions: dim, eps: eps)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, freqs: (cos: MLXArray, sin: MLXArray)) -> MLXArray {
        var qx = norm_q(q(x))
        var kx = norm_k(k(x))
        let vx = v(x)
        qx = ropeApply(qx, freqs: freqs, numHeads: num_heads)
        kx = ropeApply(kx, freqs: freqs, numHeads: num_heads)
        let attn = flashAttention(q: qx, k: kx, v: vx, numHeads: num_heads)
        return o(attn)
    }
}

public class CrossAttention: Module {
    let num_heads: Int

    @ModuleInfo var q: Linear
    @ModuleInfo var k: Linear
    @ModuleInfo var v: Linear
    @ModuleInfo var o: Linear
    @ModuleInfo var norm_q: RMSNorm
    @ModuleInfo var norm_k: RMSNorm

    init(dim: Int, numHeads: Int, eps: Float = 1e-6) {
        // has_image_input branches omitted — false for MOSS-SoundEffect-v2.0.
        self.num_heads = numHeads
        self.q = Linear(dim, dim)
        self.k = Linear(dim, dim)
        self.v = Linear(dim, dim)
        self.o = Linear(dim, dim)
        self.norm_q = RMSNorm(dimensions: dim, eps: eps)
        self.norm_k = RMSNorm(dimensions: dim, eps: eps)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, _ y: MLXArray) -> MLXArray {
        let qx = norm_q(q(x))
        let kx = norm_k(k(y))
        let vx = v(y)
        let attn = flashAttention(q: qx, k: kx, v: vx, numHeads: num_heads)
        return o(attn)
    }
}

public class DiTBlock: Module {
    @ModuleInfo var self_attn: SelfAttention
    @ModuleInfo var cross_attn: CrossAttention
    @ModuleInfo var norm1: LayerNorm
    @ModuleInfo var norm2: LayerNorm
    @ModuleInfo var norm3: LayerNorm
    @ModuleInfo var ffn: Sequential
    @ParameterInfo var modulation: MLXArray

    init(dim: Int, numHeads: Int, ffnDim: Int, eps: Float = 1e-6) {
        self.self_attn = SelfAttention(dim: dim, numHeads: numHeads, eps: eps)
        self.cross_attn = CrossAttention(dim: dim, numHeads: numHeads, eps: eps)
        self.norm1 = LayerNorm(dimensions: dim, eps: eps, affine: false)
        self.norm2 = LayerNorm(dimensions: dim, eps: eps, affine: false)
        self.norm3 = LayerNorm(dimensions: dim, eps: eps)
        self.ffn = Sequential(layers: [
            Linear(dim, ffnDim),
            GELU(approximation: .precise),  // upstream GELU(approximate='tanh')
            Linear(ffnDim, dim),
        ])
        self.modulation = MLXRandom.normal([1, 6, dim]) / Float(Double(dim).squareRoot())
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, context: MLXArray, tMod: MLXArray,
        freqs: (cos: MLXArray, sin: MLXArray)
    ) -> MLXArray {
        // Inference path: tMod is (B, 6, dim); chunk on axis 1 (oracle chunk_dim=1).
        let chunks = MLX.split(modulation.asType(tMod.dtype) + tMod, parts: 6, axis: 1)
        let (shiftMsa, scaleMsa, gateMsa) = (chunks[0], chunks[1], chunks[2])
        let (shiftMlp, scaleMlp, gateMlp) = (chunks[3], chunks[4], chunks[5])

        var h = x
        let inputX = modulate(norm1(h), shift: shiftMsa, scale: scaleMsa)
        h = h + gateMsa * self_attn(inputX, freqs: freqs)
        h = h + cross_attn(norm3(h), context)
        let inputX2 = modulate(norm2(h), shift: shiftMlp, scale: scaleMlp)
        h = h + gateMlp * ffn(inputX2)
        return h
    }
}

public class Head: Module {
    let patchSize: Int

    @ModuleInfo var norm: LayerNorm
    @ModuleInfo var head: Linear
    @ParameterInfo var modulation: MLXArray

    init(dim: Int, outDim: Int, patchSize: Int, eps: Float) {
        self.patchSize = patchSize
        self.norm = LayerNorm(dimensions: dim, eps: eps, affine: false)
        self.head = Linear(dim, outDim * patchSize)
        self.modulation = MLXRandom.normal([1, 2, dim]) / Float(Double(dim).squareRoot())
        super.init()
    }

    func callAsFunction(_ x: MLXArray, tMod: MLXArray) -> MLXArray {
        // Inference path: tMod is (B, dim) — the oracle's `else` branch.
        let chunks = MLX.split(
            modulation.asType(tMod.dtype) + tMod.expandedDimensions(axis: 1),
            parts: 2, axis: 1)
        let (shift, scale) = (chunks[0], chunks[1])
        return head(norm(x) * (1 + scale) + shift)
    }
}
