# IQ test fixtures

Recorded IQ captures used by the DSP and SDR tests. **The capture files
themselves are not committed** — they are several megabytes each, and
`.gitignore` excludes `*.iq` and `*.json` in this directory.

Tests that need a fixture should **skip when it is absent**, the same way the
hardware integration tests skip when no rotor or SDR is attached. A fresh
clone has no fixtures and must still have a green test run.

## Format

Raw 8-bit unsigned interleaved I/Q — `I, Q, I, Q, …`, offset binary, where
127.5 is zero. This is exactly what the RTL-SDR delivers and exactly what
`rtl_sdr.exe` writes, so the files open directly in GNU Radio, `inspectrum`
and similar tools.

The format is deliberately *not* pre-converted to complex floats. The
uint8→complex conversion is part of the code under test; baking it into the
fixture would remove it from test coverage.

Each `.iq` file has a `.json` sidecar recording the capture parameters,
including the **actual** centre frequency and sample rate reported by the
device rather than the requested ones — the tuner PLL and sample clock both
quantise, and spectrum arithmetic must use the actual values.

The sidecar carries a `sidecar_version`. **Version 2** adds two timing keys
to version 1 and changes no existing key, so it is a superset — a v1 file is
still readable, and a reader fills the additions in from `captured_utc` when
they are absent:

- `started_utc` — the capture's **start**, to the millisecond. `captured_utc`
  is the *end* (stamped after the write loop), so anything that needs the
  epoch of the first sample — a replay driving a Doppler curve — reads
  `started_utc`, or derives it as `captured_utc − seconds` for a v1 file.
- `first_block_monotonic_s` — a monotonic-clock reading taken at the first
  block. Two dongles share no wall clock, so a dual capture is aligned by
  pairing each branch's `started_utc` with a monotonic reference at the same
  instant; on a single capture this is informational. `null` only if the
  capture produced no blocks.

## Expected fixtures

| File | Signal | Purpose |
|------|--------|---------|
| `wbfm-99.9.iq` | FM broadcast, 99.9 MHz | Wideband FM demodulation (Chunk E) |
| `nbfm-noaa-162.550.iq` | NOAA weather radio, 162.550 MHz | Narrowband FM demodulation (Chunk G) |
| `nbfm-noaa-162.475.iq` | NOAA weather radio, 162.475 MHz | Second narrowband channel from the same bring-up session |

All captured at 2.048 Msps with the tuner tuned 250 kHz **below** the signal
of interest.

### Why the captures are tuned off-centre

The RTL-SDR has a permanent DC-offset spike at the centre of its passband.
A signal tuned to exactly the centre lands on top of that spike, where its
presence proves nothing — a correctly tuned radio and a DC artifact look
identical. Tuning deliberately off-centre and asserting the signal appears
at the expected offset is a test a mistuned radio fails and cannot fake.

### Why there should be a fixture containing nothing (currently absent)

The set previously included `no-signal-100.05.iq`/`fm-100.3-firstlight.iq`, a
genuine recording of an empty band made by real hardware — real broadband
noise, no signal — against which any "did we receive it?" check must
**fail**. That file was retired (Session 17) and has not yet been
recaptured; recapturing it is a small pending task, not a design change.

The reasoning for wanting a *real* capture rather than synthetic silence
still holds and should guide whoever recaptures it: synthetic silence is too
clean to be a fair test. The retired file had an ADC standard deviation of
12.9, *higher* than a good capture of an actual station — precisely the trap
a real negative fixture exists to catch. Signal level is not signal
presence.

## Regenerating them

The bench capture scripts (`sdr-first-light.py` and friends) currently live
in the maintainer's project-notes folder rather than in this repository, so
these fixtures cannot yet be regenerated from a clone alone. Moving them
into a `tools/` or `scripts/` directory is a candidate for the Phase 2
polish chunk.

The captures are also inherently local: they record whatever stations are
receivable at the maintainer's location. A contributor regenerating them
should substitute a strong local FM station and their own NOAA weather
channel, then update this table.

For reference, the commands that produced the current set:

```
python sdr-first-light.py --freq 99.9   --gain 32.8 --seconds 2 \
    --out tests/fixtures/iq/wbfm-99.9.iq

python sdr-first-light.py --freq 162.550 --gain 49.6 --seconds 2 \
    --smooth 12 --separation 20 --tolerance 15 \
    --out tests/fixtures/iq/nbfm-noaa-162.550.iq

python sdr-first-light.py --freq 162.475 --gain 32.8 --seconds 2 \
    --smooth 12 --tolerance 15 \
    --out tests/fixtures/iq/nbfm-noaa-162.475.iq
```

(The 162.475 sidecar doesn't record a `--separation` value, so it's omitted
above rather than guessed — check the sidecar's `smooth_hz`/`tolerance_hz`
against the script's current flags before relying on this to reproduce it
exactly.)
