## Serial protocol (Teensy firmware v4.1)

Each request starts with a one-byte command; the rest of the bytes depend on the command. Replies (when there is one) are `ts` (8 bytes, `micros()`), `len` (1 byte) and `len` bytes of data. Multi-byte values are little-endian.

Module `i` of a mask (bit `i`) is the module at bus address `i + 1`, the `i`-th from the Teensy. The address bus idles at 16, which no module uses, and the bus holds at most 15 modules.

| Cmd | Request | Reply data |
|---|---|---|
| `r` | mask(4) n(1) nc(1) cmd(nc) | n bytes per masked module (at most 255 in total) |
| `w` | mask(4) n(1) nc(1) cmd(nc) val(n) | none |
| `s` | — | nb(1) status(1): 0 ok, 1 addressing stuck (a module never got its address), 2 more than 15 modules |
| `n` | — | nb(1) |
| `m` | v1 v2 v3 | v1 v2 v3 |
| `v` | — | major(1) minor(1) |
| `i` | addr(1) led(1) r g b | none. I2C write of `led r g b` to the COM_MOD at `addr` (= module index + 1), `led` 1 or 2 |
| `c` | hz(4) | hz(4): SPI clock set (100 kHz to 20 MHz, 4 MHz at power-up) |
| `A` | mask(4) drdy(1) | streamed frames (below) until `x` |
| `x` | — | `E` frame when streaming, nothing otherwise |

### Streaming `A` / `x`
`A` starts the conversions of the masked modules (back to back on the bus) and the Teensy then polls their `DATA_STATUS` register and ECG data. A new sample is seen when all the `drdy` bits are set (0xE0: the three ECG channels) or when the ECG data changes: polled continuously, `DATA_STATUS` alone misses about 10% of the samples. The channels of a module update up to ~90 µs apart, so 100 µs after it sees the sample the Teensy reads the 9 ECG bytes again and sends a 22-byte frame:

| Bytes | Content |
|---|---|
| 0 | sync 0xA5 |
| 1 | type: `D` data, `E` end of stream |
| 2 | module index (bit of the mask) |
| 3-6 | per-module sample counter |
| 7-10 | `micros()` when the sample was seen (wraps every 71.6 min) |
| 11 | `DATA_STATUS` when the sample was seen |
| 12-20 | ECG channel 1, 2, 3: 3 bytes each, MSB first |
| 21 | CRC-8 (polynomial 0x07) of bytes 1-20 |

While streaming, the Teensy only accepts `x`, which stops the stream, puts the masked modules in standby and sends an `E` frame, and `i`, whose ~0.5 ms I2C transfer can cost about one sample per module. Other bytes are discarded. `clvhd.Stream` parses these frames, drops the ones with a bad CRC, and aligns the modules on module 0's sample times.
