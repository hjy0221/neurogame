import CoreGraphics
import Foundation

struct WorldEnvironment {
    struct Agent {
        var position: CGPoint
        var heading: Double
        var velocity: CGVector
        var radius: CGFloat = 12
    }

    struct Food: Identifiable {
        let id: Int
        var position: CGPoint
        var radius: CGFloat = 7
    }

    struct StepResult {
        let consumedFood: Bool
    }

    let size: CGSize
    var agent: Agent
    var foods: [Food]
    private var random: SeededRandom

    init(size: CGSize, seed: UInt64) {
        self.size = size
        self.random = SeededRandom(seed: seed)
        self.agent = Agent(position: CGPoint(x: size.width * 0.5, y: size.height * 0.5), heading: 0, velocity: .zero)
        self.foods = []
        for id in 0..<18 {
            foods.append(Food(id: id, position: WorldEnvironment.randomPoint(in: size, random: &random)))
        }
    }

    mutating func step(action: AgentAction, dt: Double) -> StepResult {
        agent.heading += action.turn * dt * 5.6
        let direction = CGVector(dx: cos(agent.heading), dy: sin(agent.heading))
        agent.velocity.dx += direction.dx * action.thrust * dt * 430
        agent.velocity.dy += direction.dy * action.thrust * dt * 430
        agent.velocity *= 0.94

        let speed = hypot(agent.velocity.dx, agent.velocity.dy)
        if speed > 320 {
            agent.velocity *= 320 / speed
        }

        agent.position.x += agent.velocity.dx * dt
        agent.position.y += agent.velocity.dy * dt
        keepAgentInBounds()

        var consumed = false
        for index in foods.indices {
            let dx = foods[index].position.x - agent.position.x
            let dy = foods[index].position.y - agent.position.y
            let distance = hypot(dx, dy)
            if distance < foods[index].radius + agent.radius {
                foods[index].position = Self.randomPoint(in: size, random: &random)
                consumed = true
            }
        }
        return StepResult(consumedFood: consumed)
    }

    private mutating func keepAgentInBounds() {
        let margin = agent.radius
        if agent.position.x < margin {
            agent.position.x = margin
            agent.velocity.dx = abs(agent.velocity.dx) * 0.35
        } else if agent.position.x > size.width - margin {
            agent.position.x = size.width - margin
            agent.velocity.dx = -abs(agent.velocity.dx) * 0.35
        }

        if agent.position.y < margin {
            agent.position.y = margin
            agent.velocity.dy = abs(agent.velocity.dy) * 0.35
        } else if agent.position.y > size.height - margin {
            agent.position.y = size.height - margin
            agent.velocity.dy = -abs(agent.velocity.dy) * 0.35
        }
    }

    private static func randomPoint(in size: CGSize, random: inout SeededRandom) -> CGPoint {
        CGPoint(
            x: CGFloat(random.nextDouble(in: 32...(Double(size.width) - 32))),
            y: CGFloat(random.nextDouble(in: 32...(Double(size.height) - 32)))
        )
    }
}

extension CGVector {
    static func *= (lhs: inout CGVector, rhs: Double) {
        lhs.dx *= rhs
        lhs.dy *= rhs
    }
}
