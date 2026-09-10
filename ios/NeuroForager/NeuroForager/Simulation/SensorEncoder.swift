import Foundation

struct SensorEncoder {
    static let inputCount = 32

    var range: Double = 240
    var rayCount: Int = 16

    func encode(environment: WorldEnvironment, energy: Double) -> [Double] {
        var inputs = Array(repeating: 0.0, count: Self.inputCount)
        let agent = environment.agent

        for food in environment.foods {
            let dx = Double(food.position.x - agent.position.x)
            let dy = Double(food.position.y - agent.position.y)
            let distance = hypot(dx, dy)
            guard distance <= range else { continue }

            let relativeAngle = normalizeAngle(atan2(dy, dx) - agent.heading)
            let normalized = (relativeAngle + .pi) / (2 * .pi)
            let bin = min(rayCount - 1, max(0, Int(normalized * Double(rayCount))))
            let signal = 1.0 - distance / range
            inputs[bin] = max(inputs[bin], signal)
        }

        let speed = hypot(agent.velocity.dx, agent.velocity.dy)
        inputs[16] = min(1, speed / 320)
        inputs[17] = (energy / 100)
        inputs[18] = sin(agent.heading) * 0.5 + 0.5
        inputs[19] = cos(agent.heading) * 0.5 + 0.5
        inputs[20] = Double(agent.position.x / environment.size.width)
        inputs[21] = Double(agent.position.y / environment.size.height)

        for index in 22..<Self.inputCount {
            let phase = Double(index) * 12.9898 + Double(environment.foods.count) * 78.233
            inputs[index] = (sin(phase) + 1) * 0.05
        }

        return inputs
    }

    private func normalizeAngle(_ angle: Double) -> Double {
        var value = angle
        while value > .pi { value -= 2 * .pi }
        while value < -.pi { value += 2 * .pi }
        return value
    }
}
