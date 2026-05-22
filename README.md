# QSOrbit

**Integrated satellite tracking and downlink reception for amateur radio, with homebrew rotor control.**

> ⚠️ **Status: pre-alpha.** Under active development. Not yet usable.

## What is this?

QSOrbit is an open-source, cross-platform desktop suite for amateur-radio satellite operators with homebrew ground stations. It consolidates the stack of separate programs that operators currently juggle — tracking software, hamlib bridges, SDR receivers, decoders, the Arduino IDE for rotor firmware — into one integrated application.

QSOrbit covers the **receive side** of satellite operations: tracking, SDR reception, decoding, and rotor control. Transmit is not currently supported — see [Out of scope](#out-of-scope-for-now).

The target user is someone running a SatNOGS-style rotor controlled by an Arduino, with one or more RTL-SDR dongles feeding off yagi-or-similar antennas. Designed to extend to other homebrew configurations.

## Planned features

- Dual-SDR support with non-coherent combining for polarization diversity
- Auto-calibration of rotor alignment using known satellite passes
- Pluggable rotor protocols (EasyComm and beyond)
- Modular decoder framework built on gr-satellites
- Per-satellite profiles that remember the settings that worked last time
- IQ recording for offline analysis
- Retro *WarGames*-CAD satellite ID panel — for fun and aesthetic

## Status

Currently in Phase 1 — foundation work. No installable build yet.

## Roadmap

1. **Phase 1** — Foundation: project skeleton, rotor control module, basic satellite tracking
2. **Phase 2** — Single-SDR support: waterfall, demodulation, Doppler correction
3. **Phase 3** — Dual-SDR support: non-coherent combining, polarization estimation
4. **Phase 4** — Decoder framework: gr-satellites integration, per-satellite profiles
5. **Phase 5** — Polish: auto-calibration, AMSAT integration, IQ recording, release prep

## Out of scope (for now)

**Transmit.** QSOrbit is receive-only. It covers the downlink half of satellite operations — tracking, reception, decoding, recording — but does not currently control any transmit hardware. Transmit may be added later, but it's not part of any planned phase above and has no timeline. We won't surface any transmit UI or copy until the feature is real.

## Tech

- Python 3.11+
- PySide6 (Qt) for the UI
- Cross-platform: Windows (primary target), Linux, macOS
- Licensed under [GPL-3.0](LICENSE)

## Getting started

Installation and usage instructions will appear here once Phase 1 lands. For now, this README documents the project's intent and scope.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

[GPL-3.0](LICENSE).
