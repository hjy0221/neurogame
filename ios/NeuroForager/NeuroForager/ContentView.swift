import SwiftUI

struct ContentView: View {
    @StateObject private var viewModel = SimulationViewModel()

    var body: some View {
        VStack(spacing: 0) {
            SimulationCanvasView(viewModel: viewModel)
                .frame(maxWidth: .infinity, maxHeight: .infinity)

            ControlBarView(viewModel: viewModel)

            NeuralActivityView(snapshot: viewModel.brainSnapshot)
                .frame(height: 290)
                .padding(.horizontal, 12)
                .padding(.bottom, 10)
        }
        .background(Color(red: 0.06, green: 0.07, blue: 0.075))
        .foregroundStyle(.white)
        .onAppear { viewModel.start() }
        .onDisappear { viewModel.stop() }
    }
}

private struct ControlBarView: View {
    @ObservedObject var viewModel: SimulationViewModel

    var body: some View {
        HStack(spacing: 14) {
            Button {
                viewModel.toggleRunning()
            } label: {
                Image(systemName: viewModel.isRunning ? "pause.fill" : "play.fill")
                    .frame(width: 32, height: 32)
            }
            .buttonStyle(.borderedProminent)

            Button {
                viewModel.reset()
            } label: {
                Image(systemName: "arrow.clockwise")
                    .frame(width: 32, height: 32)
            }
            .buttonStyle(.bordered)

            VStack(alignment: .leading, spacing: 4) {
                Text("Energy \(Int(viewModel.energy))")
                    .font(.caption.weight(.semibold))
                ProgressView(value: viewModel.energy, total: 100)
                    .tint(.mint)
            }

            VStack(alignment: .trailing, spacing: 4) {
                Text("Food \(viewModel.foodEaten)")
                    .font(.caption.weight(.semibold))
                Text("Spikes \(viewModel.brainSnapshot.recentSpikeCount)")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
            .frame(width: 78, alignment: .trailing)
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 10)
        .background(Color(red: 0.095, green: 0.105, blue: 0.11))
    }
}
