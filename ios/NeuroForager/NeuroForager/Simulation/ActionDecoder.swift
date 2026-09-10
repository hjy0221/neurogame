import Foundation

struct AgentAction {
    var turn: Double
    var thrust: Double
}

struct ActionDecoder {
    static let outputCount = 4

    func decode(outputs: [Double]) -> AgentAction {
        let left = outputs[safe: 0] ?? 0
        let right = outputs[safe: 1] ?? 0
        let forward = outputs[safe: 2] ?? 0
        let brake = outputs[safe: 3] ?? 0

        let turn = max(-1, min(1, right - left))
        let thrust = max(0, min(1, 0.14 + forward - brake * 0.45))
        return AgentAction(turn: turn, thrust: thrust)
    }
}

private extension Array {
    subscript(safe index: Int) -> Element? {
        indices.contains(index) ? self[index] : nil
    }
}
