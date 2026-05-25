"""Rotor control module.

Provides serial communication and EasyComm II protocol support for
Arduino-controlled antenna rotors.
"""

from qsorbit.core.rotor.exceptions import RotorError, SerialConnectionError, SerialTimeoutError
from qsorbit.core.rotor.serial_port import SerialPort

__all__ = [
    "RotorError",
    "SerialConnectionError",
    "SerialTimeoutError",
    "SerialPort",
]
