import Foundation

struct SeededRandom {
    private var state: UInt64

    init(seed: UInt64) {
        self.state = seed == 0 ? 0x4d595df4d0f33173 : seed
    }

    mutating func nextUInt64() -> UInt64 {
        state &+= 0x9e3779b97f4a7c15
        var z = state
        z = (z ^ (z >> 30)) &* 0xbf58476d1ce4e5b9
        z = (z ^ (z >> 27)) &* 0x94d049bb133111eb
        return z ^ (z >> 31)
    }

    mutating func nextDouble() -> Double {
        Double(nextUInt64() >> 11) / Double(1 << 53)
    }

    mutating func nextDouble(in range: ClosedRange<Double>) -> Double {
        range.lowerBound + nextDouble() * (range.upperBound - range.lowerBound)
    }

    mutating func nextInt(_ range: Range<Int>) -> Int {
        range.lowerBound + Int(nextUInt64() % UInt64(range.count))
    }

    mutating func nextInt(_ range: ClosedRange<Int>) -> Int {
        range.lowerBound + Int(nextUInt64() % UInt64(range.upperBound - range.lowerBound + 1))
    }
}
