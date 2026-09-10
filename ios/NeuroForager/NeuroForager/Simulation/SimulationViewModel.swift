import Foundation
import SwiftUI
import QuartzCore
import Combine

@MainActor
final class SimulationViewModel: ObservableObject {
    @Published private(set) var environment: WorldEnvironment
    @Published private(set) var brainSnapshot = BrainActivitySnapshot.empty(neuronCount: 500)
    @Published private(set) var isRunning = false
    @Published private(set) var energy: Double = 80
    @Published private(set) var foodEaten = 0

    private var brain = BrainSimulator(configuration: .v01)
    private var sensorEncoder = SensorEncoder()
    private var actionDecoder = ActionDecoder()
    private var displayLink: CADisplayLink?
    private var lastTimestamp: CFTimeInterval?

    init() {
        self.environment = WorldEnvironment(size: CGSize(width: 900, height: 620), seed: 42)
        self.brainSnapshot = brain.snapshot()
    }

    func start() {
        guard displayLink == nil else {
            isRunning = true
            return
        }
        let link = CADisplayLink(target: DisplayLinkProxy { [weak self] timestamp in
            Task { @MainActor in
                self?.tick(timestamp: timestamp)
            }
        }, selector: #selector(DisplayLinkProxy.step(_:)))
        link.preferredFrameRateRange = CAFrameRateRange(minimum: 30, maximum: 60, preferred: 60)
        link.add(to: .main, forMode: .common)
        displayLink = link
        isRunning = true
    }

    func stop() {
        isRunning = false
    }

    func toggleRunning() {
        isRunning.toggle()
    }

    func reset() {
        environment = WorldEnvironment(size: environment.size, seed: UInt64(Date().timeIntervalSince1970))
        brain = BrainSimulator(configuration: .v01)
        energy = 80
        foodEaten = 0
        brainSnapshot = brain.snapshot()
        lastTimestamp = nil
        isRunning = true
    }

    private func tick(timestamp: CFTimeInterval) {
        defer { lastTimestamp = timestamp }
        guard isRunning else { return }

        let rawDelta = lastTimestamp.map { timestamp - $0 } ?? (1.0 / 60.0)
        let dt = min(max(rawDelta, 1.0 / 120.0), 1.0 / 20.0)

        let sensors = sensorEncoder.encode(environment: environment, energy: energy)
        let outputs = brain.step(inputs: sensors, dt: dt)
        let action = actionDecoder.decode(outputs: outputs)
        let result = environment.step(action: action, dt: dt)

        energy = min(100, max(0, energy - dt * 2.0 + (result.consumedFood ? 26 : 0)))
        if result.consumedFood {
            foodEaten += 1
        }
        if energy <= 0 {
            environment.agent.velocity *= 0.94
        }

        brainSnapshot = brain.snapshot()
    }

    deinit {
        displayLink?.invalidate()
    }
}

private final class DisplayLinkProxy: NSObject {
    private let callback: (CFTimeInterval) -> Void

    init(_ callback: @escaping (CFTimeInterval) -> Void) {
        self.callback = callback
    }

    @objc func step(_ link: CADisplayLink) {
        callback(link.timestamp)
    }
}
