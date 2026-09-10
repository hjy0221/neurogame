# Development

- Keep neural dynamics in pure Swift, independent of SwiftUI and network services.
- Preserve Environment -> SensorEncoder -> BrainSimulator -> ActionDecoder boundaries.
- Render actual topology and activity; never substitute decorative edges for synapses.
- Preserve deterministic seeds in core verification.
- Validate simulator builds after source changes. Test finite values, sensor response and bounded motion when changing dynamics.
- Do not claim biological accuracy or learning without implementing and validating it.
- Keep build products outside the source project. No credentials or external AI APIs.
