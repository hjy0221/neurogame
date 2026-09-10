import Foundation

struct BrainConfiguration {
    let neuronCount: Int
    let inputCount: Int
    let outputCount: Int
    let recurrentFanout: Int
    let inhibitoryRatio: Double

    static let v01 = BrainConfiguration(
        neuronCount: 500,
        inputCount: SensorEncoder.inputCount,
        outputCount: ActionDecoder.outputCount,
        recurrentFanout: 8,
        inhibitoryRatio: 0.2
    )
}

struct NeuralConnection {
    let source: Int
    let target: Int
    let weight: Double
}

struct BrainActivitySnapshot {
    let membrane: [Double]
    let spikes: [Bool]
    let outputTraces: [Double]
    let recentSpikeCount: Int
    var connections: [NeuralConnection] = []
    var activity: [Double] = []
    var inputs: [Double] = []
    var inputConnections: [NeuralConnection] = []

    static func empty(neuronCount: Int) -> BrainActivitySnapshot {
        BrainActivitySnapshot(
            membrane: Array(repeating: 0, count: neuronCount),
            spikes: Array(repeating: false, count: neuronCount),
            outputTraces: Array(repeating: 0, count: ActionDecoder.outputCount),
            recentSpikeCount: 0
        )
    }
}

final class BrainSimulator {
    private struct Synapse {
        let target: Int
        let weight: Double
    }

    private let configuration: BrainConfiguration
    private var random = SeededRandom(seed: 1001)
    private var membrane: [Double]
    private var adaptiveThreshold: [Double]
    private var spikes: [Bool]
    private var spikeTrace: [Double]
    private var outputTraces: [Double]
    private var recurrent: [[Synapse]]
    private var inputWeights: [[Synapse]]
    private var outputPools: [[Int]]
    private var recentSpikeCount = 0
    private var connections: [NeuralConnection] = []
    private var inputConnections: [NeuralConnection] = []
    private var latestInputs: [Double] = []

    init(configuration: BrainConfiguration) {
        self.configuration = configuration
        self.membrane = Array(repeating: 0, count: configuration.neuronCount)
        self.adaptiveThreshold = Array(repeating: 1, count: configuration.neuronCount)
        self.spikes = Array(repeating: false, count: configuration.neuronCount)
        self.spikeTrace = Array(repeating: 0, count: configuration.neuronCount)
        self.outputTraces = Array(repeating: 0, count: configuration.outputCount)
        self.recurrent = Array(repeating: [], count: configuration.neuronCount)
        self.inputWeights = Array(repeating: [], count: configuration.inputCount)
        self.outputPools = []
        buildNetwork()
    }

    func step(inputs: [Double], dt: Double) -> [Double] {
        latestInputs = inputs
        var currents = Array(repeating: 0.0, count: configuration.neuronCount)

        for inputIndex in 0..<min(inputs.count, inputWeights.count) {
            let value = inputs[inputIndex]
            guard value > 0 else { continue }
            for synapse in inputWeights[inputIndex] {
                currents[synapse.target] += value * synapse.weight
            }
        }

        for source in 0..<configuration.neuronCount where spikes[source] {
            for synapse in recurrent[source] {
                currents[synapse.target] += synapse.weight
            }
        }

        recentSpikeCount = 0
        for index in 0..<configuration.neuronCount {
            spikeTrace[index] *= exp(-dt * 8.0)
            adaptiveThreshold[index] += (1.0 - adaptiveThreshold[index]) * dt * 1.4

            let leak = -membrane[index] * 5.5
            let noise = random.nextDouble(in: -0.012...0.012)
            membrane[index] += (leak + currents[index] + noise) * dt

            if membrane[index] >= adaptiveThreshold[index] {
                spikes[index] = true
                membrane[index] = -0.25
                adaptiveThreshold[index] += 0.32
                spikeTrace[index] = 1.0
                recentSpikeCount += 1
            } else {
                spikes[index] = false
            }
        }

        for outputIndex in 0..<outputPools.count {
            let pool = outputPools[outputIndex]
            let sum = pool.reduce(0.0) { $0 + spikeTrace[$1] }
            let normalized = min(1, sum / Double(max(pool.count, 1)) * 4.2)
            outputTraces[outputIndex] = outputTraces[outputIndex] * 0.78 + normalized * 0.22
        }

        return outputTraces
    }

    func snapshot() -> BrainActivitySnapshot {
        BrainActivitySnapshot(
            membrane: membrane,
            spikes: spikes,
            outputTraces: outputTraces,
            recentSpikeCount: recentSpikeCount,
            connections: connections,
            activity: spikeTrace,
            inputs: latestInputs,
            inputConnections: inputConnections
        )
    }

    private func buildNetwork() {
        for inputIndex in 0..<configuration.inputCount {
            for _ in 0..<18 {
                let target = random.nextInt(0..<configuration.neuronCount)
                inputWeights[inputIndex].append(Synapse(target: target, weight: random.nextDouble(in: 9...16)))
            }
        }

        for source in 0..<configuration.neuronCount {
            let inhibitory = Double(source) / Double(configuration.neuronCount) < configuration.inhibitoryRatio
            for _ in 0..<configuration.recurrentFanout {
                let localOffset = random.nextInt(-35...35)
                let target = positiveModulo(source + localOffset + random.nextInt(0..<configuration.neuronCount), configuration.neuronCount)
                let magnitude = random.nextDouble(in: 0.75...2.7)
                recurrent[source].append(Synapse(target: target, weight: inhibitory ? -magnitude * 1.2 : magnitude))
            }
        }

        let poolSize = max(12, configuration.neuronCount / 32)
        for outputIndex in 0..<configuration.outputCount {
            let start = configuration.neuronCount - ((outputIndex + 1) * poolSize)
            let pool = (0..<poolSize).map { positiveModulo(start + $0, configuration.neuronCount) }
            outputPools.append(pool)
        }

        // Innate food-bearing wiring drives spiking motor pools.
        for bin in 0..<16 {
            let angle = (Double(bin) + 0.5) / 16 * 2 * Double.pi - Double.pi
            for target in outputPools[angle < 0 ? 0 : 1] {
                inputWeights[bin].append(Synapse(target: target, weight: 42 * abs(sin(angle))))
            }
            for target in outputPools[2] {
                inputWeights[bin].append(Synapse(target: target, weight: 24 * max(0, cos(angle))))
            }
        }
        connections = recurrent.enumerated().flatMap { source, synapses in
            synapses.map { NeuralConnection(source: source, target: $0.target, weight: $0.weight) }
        }
        inputConnections = inputWeights.enumerated().flatMap { source, synapses in
            synapses.map { NeuralConnection(source: source, target: $0.target, weight: $0.weight) }
        }
    }

    private func positiveModulo(_ value: Int, _ modulus: Int) -> Int {
        let result = value % modulus
        return result >= 0 ? result : result + modulus
    }
}
