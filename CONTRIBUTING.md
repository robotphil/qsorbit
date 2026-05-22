# Contributing to QSOrbit

Thanks for taking an interest in QSOrbit. This document covers how to report bugs, propose features, and submit code.

QSOrbit is in pre-alpha — things change fast, and the conventions below aren't yet enforced by tooling, but they're the direction we're heading.

## Code of conduct

By participating in QSOrbit's spaces — Issues, Pull Requests, Discussions — you agree to abide by our [Code of Conduct](https://github.com/QSOrbit/.github/blob/main/CODE_OF_CONDUCT.md).

## Reporting bugs

Use [GitHub Issues](https://github.com/QSOrbit/qsorbit/issues). The bug-report template prompts for the info we need: what you tried, what you expected, what happened, your environment.

## Proposing features

For substantial ideas — new features, architectural changes — open a [Discussion](https://github.com/QSOrbit/qsorbit/discussions) first so we can talk through it before anyone writes code. For small additions, an Issue is fine.

## Development setup

If you have write access (maintainers), clone the repo directly. Otherwise, fork it first on GitHub and clone your fork.

1. Clone: `git clone https://github.com/QSOrbit/qsorbit.git` (or your fork)
2. `cd qsorbit`
3. Install dependencies: `uv sync` — this creates a virtual environment and installs everything from `pyproject.toml`.
4. Run the app: `uv run python -m qsorbit`
5. Run the tests: `uv run pytest`

If you don't have `uv` installed, see [astral.sh/uv](https://docs.astral.sh/uv/).

## Branch naming

Use a prefix indicating the type of work, followed by a short hyphenated description. All lowercase, hyphens for word breaks.

| Prefix | When to use | Example |
|--------|-------------|---------|
| `feature/` | New functionality | `feature/rotor-serial-comms` |
| `fix/` | Bug fixes | `fix/serial-timeout-handling` |
| `docs/` | Documentation-only changes | `docs/install-guide` |
| `refactor/` | Internal restructuring with no behavior change | `refactor/extract-sdr-interface` |
| `chore/` | Tooling, dependencies, CI tweaks | `chore/update-ruff-config` |

## Commits and pull-request titles

We use a relaxed form of [Conventional Commits](https://www.conventionalcommits.org/).

**Format:** `type: short imperative description`. Example: `feat: add EasyComm serial handshake`.

**Where it applies:**

- **PR titles** — strict. The PR title becomes the squash-merged commit on `main`, so use the format.
- **Commits inside a branch** — loose. They get squashed away on merge. Write whatever keeps you moving (`wip`, `cleanup`, etc.).

**Types:**

| Type | When | Example |
|------|------|---------|
| `feat` | New functionality | `feat: add rotor serial comms` |
| `fix` | Bug fixes | `fix: handle serial timeout` |
| `docs` | Documentation-only changes | `docs: add install guide` |
| `refactor` | Internal restructuring, no behavior change | `refactor: extract sdr interface` |
| `chore` | Tooling, dependencies, CI tweaks | `chore: update ruff config` |

Imperative mood ("add", not "added"). Lowercase. No trailing period.

## Opening a pull request

1. Push your branch: `git push -u origin feature/your-branch`
2. Open a Pull Request against `main`
3. Fill in the PR template
4. Iterate on review feedback
5. A maintainer will squash-merge once approved

We never push directly to `main`, except for truly trivial fixes (a typo in the README, say).

## Style

- **Lint and format**: handled by [ruff](https://docs.astral.sh/ruff/), configured in `pyproject.toml`. Run `uv run ruff check` and `uv run ruff format` before opening a PR.
- **Type hints**: encouraged for public APIs. Not yet enforced.
- **Tests**: new code should have unit tests in `tests/unit/`. Hardware-dependent integration tests live in `tests/integration/` and run separately.

## Architecture notes

QSOrbit's source is organized into `core/` (backend logic) and `ui/` (the PySide6 frontend).

**Important:** `core/` modules must NOT import from `ui/`. UI talks to core; core never talks to UI. This preserves the option of a headless CLI later.

New modules default to `core/` unless they're explicitly Qt-specific.

## Questions

Open a [Discussion](https://github.com/QSOrbit/qsorbit/discussions). The project is small and we're happy to help.
