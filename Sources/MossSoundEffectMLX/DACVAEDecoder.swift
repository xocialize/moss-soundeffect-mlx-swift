// DAC continuous VAE — Swift transpose of moss_sfx_mlx/model/dac_vae.py
// (parity-locked MLX-Python oracle). Channel-last internally, (B, C, T) at
// the public encode/decode boundaries, weight-norm pre-fused at conversion.
//
// Weight keys match the converted vae.safetensors:
//   encoder.block.layers.<i>…   decoder.model.layers.<i>…
//   quant_conv.{weight,bias}    post_quant_conv.{weight,bias}
//   …Snake "alpha" is (1, 1, C)
//
// NO latent scale constant — post_quant_conv is the learned equivalent.

import Foundation
import MLX
import MLXNN

public struct DACConfig: Sendable {
    // Pickled ctor kwargs of vae_128d_48k.pth (NOT dac-package defaults).
    public var encoderDim = 128
    public var encoderRates = [2, 3, 4, 5, 8]
    public var latentDim = 128
    public var decoderDim = 2048
    public var decoderRates = [8, 5, 4, 3, 2]
    public var sampleRate = 48000

    public var hopLength: Int { encoderRates.reduce(1, *) }

    public init() {}
}

func snake(_ x: MLXArray, alpha: MLXArray) -> MLXArray {
    x + (1.0 / (alpha + 1e-9)) * MLX.pow(MLX.sin(alpha * x), 2)
}

final class Snake1d: Module, UnaryLayer {
    @ParameterInfo var alpha: MLXArray

    init(channels: Int) {
        self.alpha = MLXArray.ones([1, 1, channels])  // channel-last
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        snake(x, alpha: alpha)
    }
}

/// ConvTranspose1d with explicit outputPadding (decoder upsamplers use
/// output_padding = stride % 2). Weight layout (O, K, I) — matches the
/// converted safetensors directly.
final class WNConvTranspose1d: Module, UnaryLayer {
    let stride: Int
    let padding: Int
    let outputPadding: Int

    @ParameterInfo var weight: MLXArray
    @ParameterInfo var bias: MLXArray

    init(
        inputChannels: Int, outputChannels: Int, kernelSize: Int,
        stride: Int = 1, padding: Int = 0, outputPadding: Int = 0
    ) {
        self.stride = stride
        self.padding = padding
        self.outputPadding = outputPadding
        let scale = Float(1.0 / Double(inputChannels * kernelSize).squareRoot())
        self.weight = MLXRandom.uniform(
            low: -scale, high: scale, [outputChannels, kernelSize, inputChannels])
        self.bias = MLXArray.zeros([outputChannels])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let y = convTransposed1d(
            x, weight, stride: stride, padding: padding,
            dilation: 1, outputPadding: outputPadding, groups: 1)
        return y + bias
    }
}

final class ResidualUnit: Module, UnaryLayer {
    @ModuleInfo var block: Sequential

    init(dim: Int, dilation: Int) {
        let pad = ((7 - 1) * dilation) / 2
        self.block = Sequential(layers: [
            Snake1d(channels: dim),
            Conv1d(
                inputChannels: dim, outputChannels: dim, kernelSize: 7,
                stride: 1, padding: pad, dilation: dilation),
            Snake1d(channels: dim),
            Conv1d(inputChannels: dim, outputChannels: dim, kernelSize: 1),
        ])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let y = block(x)
        // Time axis is 1 (channel-last); crop only in padding-disabled mode.
        let pad = (x.dim(1) - y.dim(1)) / 2
        let xc = pad > 0 ? x[0..., pad ..< (x.dim(1) - pad)] : x
        return xc + y
    }
}

final class EncoderBlock: Module, UnaryLayer {
    @ModuleInfo var block: Sequential

    init(dim: Int, stride: Int) {
        self.block = Sequential(layers: [
            ResidualUnit(dim: dim / 2, dilation: 1),
            ResidualUnit(dim: dim / 2, dilation: 3),
            ResidualUnit(dim: dim / 2, dilation: 9),
            Snake1d(channels: dim / 2),
            Conv1d(
                inputChannels: dim / 2, outputChannels: dim,
                kernelSize: 2 * stride, stride: stride,
                padding: Int(ceil(Double(stride) / 2))),
        ])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { block(x) }
}

final class Encoder: Module, UnaryLayer {
    @ModuleInfo var block: Sequential

    init(dModel: Int, strides: [Int], dLatent: Int) {
        var d = dModel
        var layers: [UnaryLayer] = [
            Conv1d(inputChannels: 1, outputChannels: d, kernelSize: 7, padding: 3)
        ]
        for stride in strides {
            d *= 2
            layers.append(EncoderBlock(dim: d, stride: stride))
        }
        layers.append(Snake1d(channels: d))
        layers.append(Conv1d(inputChannels: d, outputChannels: dLatent, kernelSize: 3, padding: 1))
        self.block = Sequential(layers: layers)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { block(x) }
}

final class DecoderBlock: Module, UnaryLayer {
    @ModuleInfo var block: Sequential

    init(inputDim: Int, outputDim: Int, stride: Int) {
        self.block = Sequential(layers: [
            Snake1d(channels: inputDim),
            WNConvTranspose1d(
                inputChannels: inputDim, outputChannels: outputDim,
                kernelSize: 2 * stride, stride: stride,
                padding: Int(ceil(Double(stride) / 2)),
                outputPadding: stride % 2),
            ResidualUnit(dim: outputDim, dilation: 1),
            ResidualUnit(dim: outputDim, dilation: 3),
            ResidualUnit(dim: outputDim, dilation: 9),
        ])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { block(x) }
}

final class Decoder: Module, UnaryLayer {
    @ModuleInfo var model: Sequential

    init(inputChannel: Int, channels: Int, rates: [Int], dOut: Int = 1) {
        var layers: [UnaryLayer] = [
            Conv1d(inputChannels: inputChannel, outputChannels: channels, kernelSize: 7, padding: 3)
        ]
        var outputDim = channels
        for (i, stride) in rates.enumerated() {
            let inputDim = channels / (1 << i)
            outputDim = channels / (1 << (i + 1))
            layers.append(DecoderBlock(inputDim: inputDim, outputDim: outputDim, stride: stride))
        }
        layers.append(Snake1d(channels: outputDim))
        layers.append(Conv1d(inputChannels: outputDim, outputChannels: dOut, kernelSize: 7, padding: 3))
        layers.append(Tanh())
        self.model = Sequential(layers: layers)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { model(x) }
}

public class DAC: Module {
    public let config: DACConfig

    @ModuleInfo var encoder: Encoder
    @ModuleInfo var decoder: Decoder
    @ModuleInfo var quant_conv: Conv1d
    @ModuleInfo var post_quant_conv: Conv1d

    public init(_ config: DACConfig = DACConfig()) {
        self.config = config
        self.encoder = Encoder(
            dModel: config.encoderDim, strides: config.encoderRates,
            dLatent: config.latentDim)
        self.quant_conv = Conv1d(
            inputChannels: config.latentDim, outputChannels: 2 * config.latentDim, kernelSize: 1)
        self.post_quant_conv = Conv1d(
            inputChannels: config.latentDim, outputChannels: config.latentDim, kernelSize: 1)
        self.decoder = Decoder(
            inputChannel: config.latentDim, channels: config.decoderDim,
            rates: config.decoderRates)
        super.init()
    }

    /// audioData: (B, 1, T) -> posterior mean (B, D, T_lat) (inference `.mode()`).
    public func encodeMode(_ audioData: MLXArray) -> MLXArray {
        var z = encoder(audioData.transposed(0, 2, 1))  // (B, T_lat, D)
        z = quant_conv(z)                               // (B, T_lat, 2D)
        let mean = MLX.split(z, parts: 2, axis: -1)[0]
        return mean.transposed(0, 2, 1)                 // (B, D, T_lat)
    }

    /// z: (B, D, T_lat) raw DiT latents -> (B, 1, T) waveform in [-1, 1].
    /// NO scale constant — do not add one (docs/upstream-findings.md §4).
    public func decode(_ z: MLXArray) -> MLXArray {
        var h = z.transposed(0, 2, 1)                   // (B, T_lat, D)
        h = post_quant_conv(h)
        let audio = decoder(h)                          // (B, T, 1)
        return audio.transposed(0, 2, 1)                // (B, 1, T)
    }
}
