"""Unit tests for the SerialPort wrapper.

All tests use a mock serial instance so no hardware is required.
The mock is injected via the private ``_serial`` parameter — see the
``_make_serial`` helper below.
"""

from unittest.mock import MagicMock, PropertyMock, patch

import pytest
import serial

from qsorbit.core.rotor.exceptions import SerialConnectionError, SerialTimeoutError
from qsorbit.core.rotor.serial_port import SerialPort

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_serial(*, is_open: bool = True) -> MagicMock:
    """Return a MagicMock configured to behave like a ``serial.Serial`` instance.

    Args:
        is_open: The value that ``mock.is_open`` will return.
    """
    mock = MagicMock(spec=serial.Serial)
    type(mock).is_open = PropertyMock(return_value=is_open)
    return mock


# ---------------------------------------------------------------------------
# Connection lifecycle — open()
# ---------------------------------------------------------------------------


class TestOpen:
    def test_open_calls_serial_open_on_injected_instance(self):
        """When a serial instance is injected, open() calls its .open() method."""
        mock_serial = _make_serial(is_open=False)
        port = SerialPort("COM3", _serial=mock_serial)
        port.open()
        mock_serial.open.assert_called_once()

    def test_open_constructs_serial_when_none_injected(self):
        """When no serial instance is provided, open() constructs one via serial.Serial()."""
        # Create the mock instance before entering the patch context — inside the patch,
        # serial.Serial is already a MagicMock, so spec=serial.Serial would fail.
        mock_instance = _make_serial(is_open=True)
        with patch("qsorbit.core.rotor.serial_port.serial.Serial") as MockSerial:
            MockSerial.return_value = mock_instance
            port = SerialPort("COM3", baudrate=9600, timeout=1.0)
            port.open()
            MockSerial.assert_called_once_with(port="COM3", baudrate=9600, timeout=1.0)

    def test_open_is_idempotent_when_already_open(self):
        """Calling open() on an already-open port is a no-op."""
        mock_serial = _make_serial(is_open=True)
        port = SerialPort("COM3", _serial=mock_serial)
        port.open()
        port.open()
        mock_serial.open.assert_not_called()

    def test_open_raises_serial_connection_error_on_failure(self):
        """serial.SerialException is wrapped in SerialConnectionError."""
        mock_serial = _make_serial(is_open=False)
        mock_serial.open.side_effect = serial.SerialException("no such port")
        port = SerialPort("COM3", _serial=mock_serial)
        with pytest.raises(SerialConnectionError, match="COM3"):
            port.open()


# ---------------------------------------------------------------------------
# Connection lifecycle — close()
# ---------------------------------------------------------------------------


class TestClose:
    def test_close_calls_serial_close(self):
        """close() on an open port closes the underlying serial instance."""
        mock_serial = _make_serial(is_open=True)
        port = SerialPort("COM3", _serial=mock_serial)
        port.close()
        mock_serial.close.assert_called_once()

    def test_close_on_already_closed_port_is_safe(self):
        """close() on a closed port does not raise and does not call serial.close()."""
        mock_serial = _make_serial(is_open=False)
        port = SerialPort("COM3", _serial=mock_serial)
        port.close()
        mock_serial.close.assert_not_called()

    def test_close_before_open_is_safe(self):
        """close() before any open() call does not raise."""
        port = SerialPort("COM3")
        port.close()  # should not raise


# ---------------------------------------------------------------------------
# is_open property
# ---------------------------------------------------------------------------


class TestIsOpen:
    def test_is_open_true_when_serial_is_open(self):
        port = SerialPort("COM3", _serial=_make_serial(is_open=True))
        assert port.is_open is True

    def test_is_open_false_when_serial_is_closed(self):
        port = SerialPort("COM3", _serial=_make_serial(is_open=False))
        assert port.is_open is False

    def test_is_open_false_when_no_serial_instance(self):
        port = SerialPort("COM3")
        assert port.is_open is False


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------


class TestContextManager:
    def test_context_manager_opens_port_on_enter(self):
        """__enter__ triggers open(), which calls .open() on the injected serial."""
        mock_serial = _make_serial(is_open=False)
        port = SerialPort("COM3", _serial=mock_serial)
        port.__enter__()
        mock_serial.open.assert_called_once()

    def test_context_manager_closes_port_on_exit(self):
        """__exit__ triggers close() on the underlying serial instance."""
        mock_serial = _make_serial(is_open=True)
        port = SerialPort("COM3", _serial=mock_serial)
        port.__exit__(None, None, None)
        mock_serial.close.assert_called_once()

    def test_context_manager_closes_port_when_exception_raised(self):
        """Port is closed even when an exception propagates out of the with block."""
        mock_serial = _make_serial(is_open=True)
        port = SerialPort("COM3", _serial=mock_serial)
        with pytest.raises(RuntimeError):
            with port:
                raise RuntimeError("something went wrong")
        mock_serial.close.assert_called_once()

    def test_context_manager_returns_self(self):
        """__enter__ returns the SerialPort instance itself."""
        mock_serial = _make_serial(is_open=False)
        port = SerialPort("COM3", _serial=mock_serial)
        result = port.__enter__()
        assert result is port


# ---------------------------------------------------------------------------
# Write path
# ---------------------------------------------------------------------------


class TestWrite:
    def test_write_sends_bytes_to_serial(self):
        """write() passes the byte string through to the underlying serial instance."""
        mock_serial = _make_serial(is_open=True)
        port = SerialPort("COM3", _serial=mock_serial)
        port.write(b"AZ 180.0\r\n")
        mock_serial.write.assert_called_once_with(b"AZ 180.0\r\n")

    def test_write_raises_when_port_not_open(self):
        """write() on a closed port raises SerialConnectionError."""
        mock_serial = _make_serial(is_open=False)
        port = SerialPort("COM3", _serial=mock_serial)
        with pytest.raises(SerialConnectionError, match="not open"):
            port.write(b"AZ 180.0\r\n")

    def test_write_raises_on_serial_exception(self):
        """serial.SerialException during write is wrapped in SerialConnectionError."""
        mock_serial = _make_serial(is_open=True)
        mock_serial.write.side_effect = serial.SerialException("device disconnected")
        port = SerialPort("COM3", _serial=mock_serial)
        with pytest.raises(SerialConnectionError):
            port.write(b"AZ 180.0\r\n")


# ---------------------------------------------------------------------------
# Read path
# ---------------------------------------------------------------------------


class TestReadline:
    def test_readline_returns_received_bytes(self):
        """readline() returns whatever the underlying serial instance reads."""
        mock_serial = _make_serial(is_open=True)
        mock_serial.readline.return_value = b"AZ180.0\r\n"
        port = SerialPort("COM3", _serial=mock_serial)
        assert port.readline() == b"AZ180.0\r\n"

    def test_readline_raises_timeout_on_empty_response(self):
        """An empty response (pyserial's timeout indicator) raises SerialTimeoutError."""
        mock_serial = _make_serial(is_open=True)
        mock_serial.readline.return_value = b""
        port = SerialPort("COM3", _serial=mock_serial)
        with pytest.raises(SerialTimeoutError):
            port.readline()

    def test_readline_raises_when_port_not_open(self):
        """readline() on a closed port raises SerialConnectionError."""
        mock_serial = _make_serial(is_open=False)
        port = SerialPort("COM3", _serial=mock_serial)
        with pytest.raises(SerialConnectionError, match="not open"):
            port.readline()

    def test_readline_raises_on_serial_exception(self):
        """serial.SerialException during read is wrapped in SerialConnectionError."""
        mock_serial = _make_serial(is_open=True)
        mock_serial.readline.side_effect = serial.SerialException("device disconnected")
        port = SerialPort("COM3", _serial=mock_serial)
        with pytest.raises(SerialConnectionError):
            port.readline()
