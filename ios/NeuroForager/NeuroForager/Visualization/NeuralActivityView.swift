import SwiftUI

struct NeuralActivityView: View {
    let snapshot: BrainActivitySnapshot
    @State private var expanded = false
    @State private var selected: Int?

    var body: some View {
        VStack(spacing: 6) {
            HStack {
                Text("신경망 활동").font(.subheadline.bold())
                Spacer()
                Text("500 뉴런 · 집단별 표시").font(.caption2).foregroundStyle(.secondary)
                Button { expanded = true } label: {
                    Image(systemName: "arrow.up.left.and.arrow.down.right").frame(width: 32, height: 32)
                }.accessibilityLabel("신경망 확대").help("신경망 확대")
            }
            diagram
        }
        .sheet(isPresented: $expanded) {
            VStack {
                HStack {
                    Text("NeuroForager").font(.headline)
                    Spacer()
                    Button("닫기") { expanded = false }
                }
                diagram
            }.padding().preferredColorScheme(.dark)
        }
    }

    private var diagram: some View {
        VStack(spacing: 4) {
            HStack {
                heading("입력층", "32 센서")
                heading("순환 중간층", "440 뉴런")
                heading("출력층", "60 뉴런")
            }
            GeometryReader { geometry in
                let nodes = makeNodes(size: geometry.size)
                let edges = groupedEdges()
                Canvas { context, size in
                    for edge in edges {
                        let a = nodes[edge.source]
                        let b = nodes[edge.target]
                        let focused = selected == nil || selected == edge.source || selected == edge.target
                        var path = Path()
                        let start = CGPoint(x: a.point.x + 13, y: a.point.y)
                        let end = CGPoint(x: b.point.x - 13, y: b.point.y)
                        path.move(to: start)
                        path.addLine(to: end)
                        let opacity = focused ? 0.12 + a.activity * 0.65 : 0.025
                        context.stroke(path, with: .color(a.color.opacity(opacity)), lineWidth: 0.6 + a.activity * 1.5)
                        if a.activity > 0.2 {
                            let p = CGPoint(x: start.x + (end.x - start.x) * 0.65, y: start.y + (end.y - start.y) * 0.65)
                            context.fill(Path(ellipseIn: CGRect(x: p.x - 2, y: p.y - 2, width: 4, height: 4)), with: .color(.yellow.opacity(a.activity)))
                        }
                    }
                    // Same-column and backward synapses are summarized as a feedback arc.
                    let x = size.width / 2
                    var loop = Path()
                    loop.move(to: CGPoint(x: x + 18, y: 16))
                    loop.addCurve(to: CGPoint(x: x + 18, y: size.height - 16), control1: CGPoint(x: x + 58, y: 12), control2: CGPoint(x: x + 58, y: size.height - 12))
                    context.stroke(loop, with: .color(.pink.opacity(0.65)), style: StrokeStyle(lineWidth: 1, dash: [3, 3]))
                    var arrow = Path()
                    arrow.move(to: CGPoint(x: x + 24, y: size.height - 22))
                    arrow.addLine(to: CGPoint(x: x + 18, y: size.height - 16))
                    arrow.addLine(to: CGPoint(x: x + 27, y: size.height - 15))
                    context.stroke(arrow, with: .color(.pink), lineWidth: 1.2)
                    for node in nodes {
                        let rect = CGRect(x: node.point.x - 12, y: node.point.y - 12, width: 24, height: 24)
                        context.fill(Path(ellipseIn: rect), with: .color(Color(white: 0.10)))
                        context.fill(Path(ellipseIn: rect), with: .color(node.color.opacity(0.15 + node.activity * 0.7)))
                        context.stroke(Path(ellipseIn: rect), with: .color(selected == node.id ? .white : node.color), lineWidth: selected == node.id ? 2 : 1)
                        context.draw(Text(node.label).font(.system(size: 9, weight: .semibold)).foregroundColor(.white), at: node.point)
                    }
                }
                .contentShape(Rectangle())
                .onTapGesture { point in
                    let nearest = nodes.min { hypot($0.point.x - point.x, $0.point.y - point.y) < hypot($1.point.x - point.x, $1.point.y - point.y) }
                    selected = nearest?.id == selected ? nil : nearest?.id
                }
                .accessibilityLabel("입력 센서, 순환 중간층, 행동 출력 순서로 표시한 신경망 집단 활동")
            }
            HStack(spacing: 12) {
                Label("활동 강도", systemImage: "circle.fill").foregroundStyle(.cyan)
                Label("순환 연결", systemImage: "arrow.uturn.backward").foregroundStyle(.pink)
                Spacer(minLength: 0)
                Text("발화 \(snapshot.recentSpikeCount)").monospacedDigit()
            }.font(.system(size: 10))
        }
    }

    private func heading(_ title: String, _ subtitle: String) -> some View {
        VStack(spacing: 2) {
            Text(title).font(.system(size: 11, weight: .semibold))
            Text(subtitle).font(.system(size: 9)).foregroundStyle(.secondary)
        }.frame(maxWidth: .infinity)
    }

    private struct Node {
        let id: Int
        let point: CGPoint
        let label: String
        let activity: Double
        let color: Color
    }

    private func makeNodes(size: CGSize) -> [Node] {
        var nodes: [Node] = []
        let labels = [["먹이L", "먹이R", "상태", "기타"], (1...8).map { "H\($0)" }, ["좌회전", "우회전", "전진", "제동"]]
        for column in 0..<3 {
            for row in labels[column].indices {
                let id = column == 0 ? row : (column == 1 ? 4 + row : 12 + row)
                let values: [Double]
                if column == 0 {
                    values = Array(snapshot.inputs.dropFirst(row * 8).prefix(8))
                } else if column == 1 {
                    values = Array(snapshot.activity.dropFirst(row * 55).prefix(55))
                } else {
                    values = snapshot.outputTraces.indices.contains(row) ? [snapshot.outputTraces[row]] : []
                }
                let activity = min(1, values.reduce(0, +) / Double(max(1, values.count)) * (column == 1 ? 3 : 1))
                nodes.append(Node(id: id, point: CGPoint(x: size.width * (Double(column) + 0.5) / 3, y: 16 + (size.height - 32) * (Double(row) + 0.5) / Double(labels[column].count)), label: labels[column][row], activity: activity, color: column == 0 ? .mint : (column == 1 ? .cyan : .yellow)))
            }
        }
        return nodes
    }

    private struct Edge: Hashable { let source: Int; let target: Int }

    private func group(_ neuron: Int) -> Int {
        neuron < 440 ? 4 + neuron / 55 : 12 + (499 - neuron) / 15
    }

    private func groupedEdges() -> [Edge] {
        var edges = Set<Edge>()
        for edge in snapshot.inputConnections {
            edges.insert(Edge(source: edge.source / 8, target: group(edge.target)))
        }
        for edge in snapshot.connections {
            let source = group(edge.source)
            let target = group(edge.target)
            if source < 12 && target >= 12 { edges.insert(Edge(source: source, target: target)) }
        }
        return edges.sorted { $0.source == $1.source ? $0.target < $1.target : $0.source < $1.source }
    }
}
