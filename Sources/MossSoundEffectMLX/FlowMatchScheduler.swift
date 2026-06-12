// FlowMatchScheduler — Swift transpose of moss_sfx_mlx/scheduler.py
// (itself parity-locked against the upstream PyTorch reference).
//
// Pure Float math on Swift arrays; only step()/addNoise() touch MLXArray.
// Pipeline config: FlowMatchScheduler(shift: 5.0, sigmaMin: 0.0, extraOneStep: true)

import Foundation
import MLX

public final class FlowMatchScheduler {
    public let numTrainTimesteps: Int
    public var shift: Float
    public let sigmaMax: Float
    public let sigmaMin: Float
    public let inverseTimesteps: Bool
    public let extraOneStep: Bool
    public let reverseSigmas: Bool

    public private(set) var sigmas: [Float] = []
    public private(set) var timesteps: [Float] = []

    public init(
        numInferenceSteps: Int = 100,
        numTrainTimesteps: Int = 1000,
        shift: Float = 3.0,
        sigmaMax: Float = 1.0,
        sigmaMin: Float = 0.003 / 1.002,
        inverseTimesteps: Bool = false,
        extraOneStep: Bool = false,
        reverseSigmas: Bool = false
    ) {
        self.numTrainTimesteps = numTrainTimesteps
        self.shift = shift
        self.sigmaMax = sigmaMax
        self.sigmaMin = sigmaMin
        self.inverseTimesteps = inverseTimesteps
        self.extraOneStep = extraOneStep
        self.reverseSigmas = reverseSigmas
        setTimesteps(numInferenceSteps)
    }

    private static func linspace(_ start: Float, _ stop: Float, _ num: Int) -> [Float] {
        guard num > 1 else { return [start] }
        let step = (stop - start) / Float(num - 1)
        return (0 ..< num).map { start + step * Float($0) }
    }

    public func setTimesteps(
        _ numInferenceSteps: Int = 100,
        denoisingStrength: Float = 1.0,
        shift: Float? = nil
    ) {
        if let shift { self.shift = shift }
        let sigmaStart = sigmaMin + (sigmaMax - sigmaMin) * denoisingStrength
        if extraOneStep {
            sigmas = Array(Self.linspace(sigmaStart, sigmaMin, numInferenceSteps + 1).dropLast())
        } else {
            sigmas = Self.linspace(sigmaStart, sigmaMin, numInferenceSteps)
        }
        if inverseTimesteps { sigmas.reverse() }
        // Classic flow-match shift formula.
        sigmas = sigmas.map { self.shift * $0 / (1 + (self.shift - 1) * $0) }
        if reverseSigmas { sigmas = sigmas.map { 1 - $0 } }
        timesteps = sigmas.map { $0 * Float(numTrainTimesteps) }
    }

    private func nearestTimestepId(_ timestep: Float) -> Int {
        // Upstream looks the timestep up by nearest match, not by index.
        var bestId = 0
        var bestDist = Float.greatestFiniteMagnitude
        for (i, t) in timesteps.enumerated() {
            let d = abs(t - timestep)
            if d < bestDist {
                bestDist = d
                bestId = i
            }
        }
        return bestId
    }

    public func step(
        _ modelOutput: MLXArray, timestep: Float, sample: MLXArray, toFinal: Bool = false
    ) -> MLXArray {
        let timestepId = nearestTimestepId(timestep)
        let sigma = sigmas[timestepId]
        let sigmaNext: Float
        if toFinal || timestepId + 1 >= timesteps.count {
            // Last step: jump straight to the boundary.
            sigmaNext = (inverseTimesteps || reverseSigmas) ? 1 : 0
        } else {
            sigmaNext = sigmas[timestepId + 1]
        }
        return sample + modelOutput * (sigmaNext - sigma)
    }

    public func addNoise(
        _ originalSamples: MLXArray, noise: MLXArray, timestep: Float
    ) -> MLXArray {
        let sigma = sigmas[nearestTimestepId(timestep)]
        // x_t = (1 - sigma) * x_0 + sigma * eps
        return (1 - sigma) * originalSamples + sigma * noise
    }
}
