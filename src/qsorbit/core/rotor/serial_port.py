"""Serial port wrapper for rotor communication."""

from __future__ import annotations

import serial

from qsorbit.core.rotor.exceptions import SerialConnectionError, SerialTimeoutError


class SerialPort:
    """A thin wrapper around pyserial.Serial for rotor communication.

    Provides open/close lifecycle management, line-oriented read/write
    operations, and translates pyserial exceptions into QSOrbit-specific
    ones so callers never need to import pyserial directly.

    Supports use as a context manager::

        with SerialPort("COM3", baudrate=9600) as port:
            port.write(b"AZ 180.0\\r\\n")
            response = port.readline()

    Args:
        port: Serial port name (e.g. ``"COM3"`` on Windows,
            ``"/dev/ttyUSB0"`` on Linux).
        baudrate: Baud rate for the connection. Defaults to 9600.
        timeout: Read timeout in seconds. Defaults to 1.0.
        _serial: Optional pre-constructed ``serial.Serial``-compatible
            instance. Intended for unit testing without real hardware —
            pass a mock here so tests never touch the OS serial layer.
    """

    def __init__(
        self,
        port: str,
        baudrate: int = 9600,
        timeout: float = 1.0,
        *,
        _serial: serial.Serial | None = None,
    ) -> None:
        self._port = port
        self._baudrate = baudrate
        self._timeout = timeout
        self._serial = _serial

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_open(self) -> bool:
        """``True`` if the underlying serial port is currently open."""
        return self._serial is not None and self._serial.is_open

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def open(self) -> None:
        """Open the serial port.

        If the port is already open this is a no-op. If no serial instance
        was injected at construction time, one is created here.

        Raises:
            SerialConnectionError: If the port cannot be opened.
        """
        if self.is_open:
            return
        try:
            if self._serial is None:
                self._serial = serial.Serial(
                    port=self._port,
                    baudrate=self._baudrate,
                    timeout=self._timeout,
                )
            else:
                self._serial.open()
        except serial.SerialException as exc:
            raise SerialConnectionError(
                f"Could not open serial port {self._port!r}: {exc}"
            ) from exc

    def close(self) -> None:
        """Close the serial port if it is open.

        Safe to call on an already-closed port or before ``open()``.
        """
        if self._serial is not None and self._serial.is_open:
            self._serial.close()

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def write(self, data: bytes) -> None:
        """Write raw bytes to the serial port.

        Args:
            data: Bytes to transmit.

        Raises:
            SerialConnectionError: If the port is not open, or if the
                write fails (e.g. device disconnected mid-transfer).
        """
        if not self.is_open:
            raise SerialConnectionError("Serial port is not open.")
        try:
            self._serial.write(data)
        except serial.SerialException as exc:
            raise SerialConnectionError(f"Write failed on {self._port!r}: {exc}") from exc

    def readline(self) -> bytes:
        """Read one line (up to and including ``\\n``) from the serial port.

        Returns:
            The bytes read, including the newline terminator.

        Raises:
            SerialConnectionError: If the port is not open, or if the
                read fails (e.g. device disconnected).
            SerialTimeoutError: If the read times out before a full line
                is received (pyserial returns ``b""`` on timeout).
        """
        if not self.is_open:
            raise SerialConnectionError("Serial port is not open.")
        try:
            data = self._serial.readline()
        except serial.SerialException as exc:
            raise SerialConnectionError(f"Read failed on {self._port!r}: {exc}") from exc
        if not data:
            raise SerialTimeoutError(f"Read timed out on {self._port!r} after {self._timeout}s.")
        return data

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> SerialPort:
        """Open the port on entering a ``with`` block."""
        self.open()
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        """Close the port on exiting a ``with`` block, even if an exception occurred."""
        self.close()
