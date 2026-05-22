"""Entry point for `python -m qsorbit`."""

from qsorbit import __version__


def main() -> None:
    """Print a startup banner and exit. Placeholder until the UI/CLI is built."""
    print(f"QSOrbit v{__version__}")
    print("Integrated satellite tracking and downlink reception for amateur radio.")
    print("(Pre-alpha: nothing functional yet.)")


if __name__ == "__main__":
    main()
