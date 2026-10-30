"""Serial protocol of the Teensy controller (firmware v4.1, see docs/serial_protocol.md)."""

import select
import struct
import time

import numpy as np
import serial
import serial.tools.list_ports

FW_MIN = (4, 1)  # oldest Teensy firmware this driver works with (v4.0 misses ~10% of the samples)

INIT_OK, INIT_STUCK, INIT_TOO_MANY = 0, 1, 2

FRAME_SYNC = 0xA5
FRAME_DATA = ord("D")
FRAME_END = ord("E")
FRAME_SIZE = 22

TEENSY_USB_VID = 0x16C0


def _crc8_table():
    table = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        crc = i
        for _ in range(8):
            crc = ((crc << 1) ^ 0x07) & 0xFF if crc & 0x80 else (crc << 1) & 0xFF
        table[i] = crc
    return table


CRC8_TABLE = _crc8_table()


def crc8_rows(rows):
    """CRC-8 (polynomial 0x07) of each row of a (N, k) uint8 array."""
    crc = np.zeros(len(rows), dtype=np.uint8)
    for col in range(rows.shape[1]):
        crc = CRC8_TABLE[crc ^ rows[:, col]]
    return crc


def find_port():
    """Serial port of the first Teensy found, or None."""
    for p in serial.tools.list_ports.comports():
        if p.vid == TEENSY_USB_VID:
            return p.device
    return None


class ControllerError(RuntimeError):
    pass


class Controller:
    """Request/reply commands of the Teensy, plus the raw stream.

    Module ``i`` is the ``i``-th module from the Teensy (bus address ``i + 1``);
    masks have bit ``i`` set for module ``i``.
    """

    def __init__(self, port, timeout=2.0):
        # the Teensy's USB serial ignores the baud rate: use a standard one
        self.ser = serial.Serial(port, 115200, timeout=timeout)
        self.timeout = timeout
        self.streaming = False
        # a previous session may have left the Teensy streaming: stop it and
        # wait until the frames still in flight have arrived
        self._send(b"x")
        self._wait_quiet(0.2, 1.5)

    def _wait_quiet(self, quiet, limit):
        """Drop incoming bytes until none arrive for ``quiet`` s (at most ``limit`` s)."""
        self.ser.timeout = quiet
        t_end = time.monotonic() + limit
        while time.monotonic() < t_end and self.ser.read(max(1, self.ser.in_waiting)):
            pass
        self.ser.timeout = self.timeout
        self.ser.reset_input_buffer()

    def close(self):
        if self.streaming:
            self.stop_stream()
        self.ser.close()

    # ------------------------------------------------------------------ basics

    def _send(self, data):
        # while streaming the Teensy only takes 'x' (stop) and 'i' (LED); any
        # other request would get no reply and its bytes could read as 'x'
        if self.streaming and data[:1] not in (b"x", b"i"):
            raise ControllerError("Only LED commands work while streaming: stop the stream first")
        self.ser.write(bytes(data))

    def _reply(self):
        """Read a reply: timestamp (8), length (1), data."""
        head = self.ser.read(9)
        if len(head) != 9:
            raise ControllerError("No reply from the Teensy")
        n = head[8]
        data = self.ser.read(n)
        if len(data) != n:
            raise ControllerError("Incomplete reply from the Teensy")
        return data

    def version(self):
        self._send(b"v")
        data = self._reply()
        return data[0], data[1]

    def check_version(self):
        try:
            major, minor = self.version()
        except ControllerError:
            raise ControllerError("No reply from the Teensy (is the right port open?)") from None
        if (major, minor) < FW_MIN:
            raise ControllerError(
                f"Teensy firmware v{major}.{minor} is too old: flash "
                f"firmware/uctrl_teensy (v{FW_MIN[0]}.{FW_MIN[1]} or newer)")
        return major, minor

    def mirror(self, b=(1, 2, 3)):
        self._send(b"m" + bytes(b))
        return tuple(self._reply()) == tuple(b)

    def address_modules(self):
        """Run the module addressing; return the number of modules."""
        self._send(b"s")
        data = self._reply()
        nb, status = data[0], data[1]
        if status == INIT_STUCK:
            raise ControllerError(
                "Module addressing stuck: a module never received its address. "
                "Check the ribbon cables (IO1 line), then power-cycle the "
                "wristband with the Teensy")
        return nb, status

    def set_spi_clock(self, hz):
        self._send(b"c" + struct.pack("<I", int(hz)))
        return struct.unpack("<I", self._reply())[0]

    def set_led(self, module, led, rgb):
        """Set LED ``led`` (0 or 1) of ``module`` (0-based) to ``rgb``.

        Also works while streaming, at the cost of about one sample per module.
        """
        r, g, b = (int(v) & 0xFF for v in rgb)
        self._send(bytes([ord("i"), module + 1, led + 1, r, g, b]))

    # -------------------------------------------------------- module registers

    def read_regs(self, mask, reg, n):
        """Read ``n`` registers from ``reg`` on every masked module.

        Returns a (modules, n) uint8 array, in module order.
        """
        cmd = bytes([reg | 0x80])
        self._send(b"r" + struct.pack("<IBB", mask, n, len(cmd)) + cmd)
        data = self._reply()
        nb = bin(mask).count("1")
        if len(data) != n * nb:
            raise ControllerError(f"Expected {n * nb} bytes, got {len(data)}")
        return np.frombuffer(data, dtype=np.uint8).reshape(nb, n)

    def write_regs(self, mask, reg, values):
        """Write ``values`` to the registers from ``reg`` on every masked module."""
        values = bytes(values)
        cmd = bytes([reg & 0x7F])
        self._send(b"w" + struct.pack("<IBB", mask, len(values), len(cmd)) + cmd + values)

    # ----------------------------------------------------------------- stream

    def start_stream(self, mask, drdy_mask):
        """Start the conversions of the masked modules and stream their samples."""
        self._send(b"A" + struct.pack("<IB", mask, drdy_mask))
        self.streaming = True

    def stop_stream(self):
        """Stop streaming and drop the frames still in flight."""
        self._send(b"x")
        tail = b""
        t_end = time.monotonic() + 0.5
        self.ser.timeout = 0.05
        while time.monotonic() < t_end:
            chunk = self.ser.read(max(1, self.ser.in_waiting))
            tail = (tail + chunk)[-4 * FRAME_SIZE:]
            if bytes([FRAME_SYNC, FRAME_END]) in tail:
                break
        self.ser.timeout = self.timeout
        self.ser.reset_input_buffer()
        self.streaming = False

    def read_stream(self, timeout):
        """Bytes received so far, waiting at most ``timeout`` s for the first one."""
        n = self.ser.in_waiting
        if n == 0 and timeout > 0:
            try:
                select.select([self.ser.fileno()], [], [], timeout)
            except (AttributeError, OSError, ValueError):  # port without a file descriptor
                time.sleep(min(timeout, 0.005))
            n = self.ser.in_waiting
        return self.ser.read(n) if n else b""
