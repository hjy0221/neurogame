import SwiftUI

struct SimulationCanvasView: View {
    @ObservedObject var viewModel: SimulationViewModel

    var body: some View {
        Canvas { context, size in
            let world = viewModel.environment
            let scale = min(size.width / world.size.width, size.height / world.size.height)
            let offset = CGPoint(
                x: (size.width - world.size.width * scale) * 0.5,
                y: (size.height - world.size.height * scale) * 0.5
            )

            let worldRect = CGRect(origin: offset, size: CGSize(width: world.size.width * scale, height: world.size.height * scale))
            context.fill(Path(worldRect), with: .color(Color(red: 0.04, green: 0.05, blue: 0.048)))

            drawGrid(context: context, rect: worldRect)

            for food in world.foods {
                let point = map(food.position, scale: scale, offset: offset)
                let rect = CGRect(
                    x: point.x - food.radius * scale,
                    y: point.y - food.radius * scale,
                    width: food.radius * scale * 2,
                    height: food.radius * scale * 2
                )
                context.fill(Path(ellipseIn: rect), with: .color(.mint))
            }

            drawAgent(world.agent, context: context, scale: scale, offset: offset)
        }
        .accessibilityLabel("2D world where the neural agent searches for food")
    }

    private func drawGrid(context: GraphicsContext, rect: CGRect) {
        var path = Path()
        let spacing: CGFloat = 36
        var x = rect.minX
        while x <= rect.maxX {
            path.move(to: CGPoint(x: x, y: rect.minY))
            path.addLine(to: CGPoint(x: x, y: rect.maxY))
            x += spacing
        }
        var y = rect.minY
        while y <= rect.maxY {
            path.move(to: CGPoint(x: rect.minX, y: y))
            path.addLine(to: CGPoint(x: rect.maxX, y: y))
            y += spacing
        }
        context.stroke(path, with: .color(Color.white.opacity(0.05)), lineWidth: 1)
    }

    private func drawAgent(_ agent: WorldEnvironment.Agent, context: GraphicsContext, scale: CGFloat, offset: CGPoint) {
        let center = map(agent.position, scale: scale, offset: offset)
        let radius = max(agent.radius * scale, 8)
        let heading = CGFloat(agent.heading)

        // Six legs make the heading readable even when the agent is moving slowly.
        var legs = Path()
        for segment in 0..<3 {
            let distance = CGFloat(segment - 1) * radius * 0.65
            let bodyPoint = offsetPoint(center, angle: heading, distance: distance)
            for side in [-1.0, 1.0] {
                let knee = offsetPoint(bodyPoint, angle: heading + CGFloat(side) * 1.05, distance: radius * 0.9)
                let foot = offsetPoint(knee, angle: heading + CGFloat(side) * 1.25, distance: radius * 0.55)
                legs.move(to: bodyPoint)
                legs.addLine(to: knee)
                legs.addLine(to: foot)
            }
        }
        context.stroke(legs, with: .color(Color(red: 0.55, green: 0.27, blue: 0.10)), lineWidth: max(1, radius * 0.12))

        // Overlapping warm segments give the agent a recognizable insect body.
        for segment in stride(from: -1, through: 1, by: 1) {
            let distance = CGFloat(segment) * radius * 0.58
            let point = offsetPoint(center, angle: heading, distance: distance)
            let segmentRadius = radius * (segment == 1 ? 0.62 : 0.72)
            let rect = CGRect(x: point.x - segmentRadius, y: point.y - segmentRadius * 0.8, width: segmentRadius * 2, height: segmentRadius * 1.6)
            context.fill(Path(ellipseIn: rect), with: .color(segment == 1 ? Color.orange : Color(red: 0.92, green: 0.50, blue: 0.16)))
            context.stroke(Path(ellipseIn: rect), with: .color(.black.opacity(0.45)), lineWidth: 1)
        }

        let head = offsetPoint(center, angle: heading, distance: radius * 0.92)
        let headRadius = radius * 0.7
        let headRect = CGRect(x: head.x - headRadius, y: head.y - headRadius * 0.82, width: headRadius * 2, height: headRadius * 1.64)
        context.fill(Path(ellipseIn: headRect), with: .color(Color(red: 0.82, green: 0.33, blue: 0.08)))
        context.stroke(Path(ellipseIn: headRect), with: .color(.black.opacity(0.5)), lineWidth: 1)

        // Antennae curve forward from the head and end in small sensory tips.
        var antennae = Path()
        for side in [-1.0, 1.0] {
            let start = offsetPoint(head, angle: heading + CGFloat(side) * 0.72, distance: headRadius * 0.55)
            let tip = offsetPoint(start, angle: heading + CGFloat(side) * 0.48, distance: radius * 1.25)
            antennae.move(to: start)
            antennae.addQuadCurve(to: tip, control: offsetPoint(start, angle: heading + CGFloat(side) * 0.95, distance: radius * 0.9))
            let tipRect = CGRect(x: tip.x - 2, y: tip.y - 2, width: 4, height: 4)
            context.fill(Path(ellipseIn: tipRect), with: .color(.mint))
        }
        context.stroke(antennae, with: .color(Color(red: 0.75, green: 0.42, blue: 0.12)), lineWidth: max(1, radius * 0.1))

        // Bright eyes make the forward direction obvious at a glance.
        for side in [-1.0, 1.0] {
            let eye = offsetPoint(head, angle: heading + CGFloat(side) * 0.58, distance: headRadius * 0.52)
            let eyeRect = CGRect(x: eye.x - radius * 0.16, y: eye.y - radius * 0.16, width: radius * 0.32, height: radius * 0.32)
            context.fill(Path(ellipseIn: eyeRect), with: .color(.black))
            context.fill(Path(ellipseIn: eyeRect.insetBy(dx: radius * 0.07, dy: radius * 0.07)), with: .color(.white))
        }
    }

    private func offsetPoint(_ point: CGPoint, angle: CGFloat, distance: CGFloat) -> CGPoint {
        CGPoint(x: point.x + cos(angle) * distance, y: point.y + sin(angle) * distance)
    }

    private func map(_ point: CGPoint, scale: CGFloat, offset: CGPoint) -> CGPoint {
        CGPoint(x: offset.x + point.x * scale, y: offset.y + point.y * scale)
    }
}
