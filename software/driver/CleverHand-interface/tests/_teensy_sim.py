#!/usr/bin/env python3
"""Fake Teensy (firmware protocol v4.1) on a pseudo-terminal, for testing the host driver.

13 ADS1293 modules with different clock rates, timestamp jitter of up to
350 us (the Teensy's poll round at 4 MHz with every module ready),
a micros() wraparound a few seconds in, and injected faults:
  module 5 drops one sample in 2000, module 7 sends one duplicate in 1000,
  one frame in 5000 gets a corrupted byte.
All modules sample the same signal, so after alignment their values should match.

    FAULTS=0 python teensy_sim.py PATH_FILE   # writes the pty path to PATH_FILE

Set FAULTS=0 to disable the injected faults. When a stream stops, the counts
of injected faults are printed as JSON. BOOT_S=1 holds the commands for 1 s
after the port opens, like the Teensy's setup() delay after a power-up.
"""
import json
import math
import os
import random
import select
import struct
import sys
import time
import tty

NB = 13
ODR = 200000 / (4 * 4 * 6)
ADC_MAX = 0xF30000
random.seed(1)
FAULTS = os.environ.get('FAULTS', '1') == '1'
PPM = [random.uniform(-40, 40) for _ in range(NB)]
PHASE = [random.uniform(0, 1 / ODR) for _ in range(NB)]
MICROS_BASE = 2**32 - 3_000_000  # wraps 3 s after start
JITTER_S = 350e-6
BOOT_S = float(os.environ.get('BOOT_S', '0'))

regs = [bytearray(0x50) for _ in range(NB)]
for r in regs:
    r[0x40] = 0x01  # REVID


def crc8(data):
    crc = 0
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = ((crc << 1) ^ 0x07) & 0xFF if crc & 0x80 else (crc << 1) & 0xFF
    return crc


def signal(t, ch):
    return (1e-3 * math.sin(2 * math.pi * 37 * t) + 0.5e-3 * math.sin(2 * math.pi * 110 * t)) * (ch + 1) / 3


def code(v):
    c = int(round((v * 3.5 / 4.8 + 0.5) * ADC_MAX))
    return c.to_bytes(3, 'big')


t_start = time.monotonic()


def micros(t):
    return (MICROS_BASE + int((t - t_start) * 1e6)) & 0xFFFFFFFF


def reply(fd, data):
    os.write(fd, struct.pack('<QB', 0, len(data)) + bytes(data))


def frame(typ, module, counter, t_us, status, ecg):
    f = bytes([0xA5, typ, module]) + struct.pack('<II', counter, t_us) + bytes([status]) + ecg
    return f + bytes([crc8(f[1:])])


def main():
    master, slave = os.openpty()
    tty.setraw(master)
    path = os.ttyname(slave)
    with open(sys.argv[1], 'w') as f:
        f.write(path)
    print('sim on', path, 'ppm', json.dumps([round(p, 3) for p in PPM]), flush=True)
    time.sleep(BOOT_S)  # the host's bytes wait in the pty meanwhile
    rx = bytearray()
    streaming = False
    mask = 0
    next_n = [0] * NB
    counters = [0] * NB
    stream_t0 = 0.0
    stats = {'drop': 0, 'dup': 0, 'crc': 0}

    def need(n):
        while len(rx) < n:
            r, _, _ = select.select([master], [], [], 1.0)
            if r:
                rx.extend(os.read(master, 4096))

    while True:
        r, _, _ = select.select([master], [], [], 0.001)
        if r:
            try:
                rx.extend(os.read(master, 4096))
            except OSError:
                return
        while rx:
            c = chr(rx[0])
            if streaming:
                if c == 'i':  # LEDs are also accepted while streaming
                    need(6); print('led', json.dumps(list(rx[1:6])), flush=True); del rx[:6]
                    continue
                del rx[0]
                if c == 'x':
                    streaming = False
                    for m in range(NB):
                        regs[m][0] = 0x02
                    os.write(master, frame(ord('E'), 0, 0, micros(time.monotonic()), 0, bytes(9)))
                    print('stream stopped', json.dumps(stats), flush=True)
                continue
            if c == 'v':
                del rx[0]; reply(master, [4, 1])
            elif c == 's':
                del rx[0]; reply(master, [NB, 0])
            elif c == 'n':
                del rx[0]; reply(master, [NB])
            elif c == 'm':
                need(4); reply(master, rx[1:4]); del rx[:4]
            elif c == 'c':
                need(5); reply(master, rx[1:5]); del rx[:5]
            elif c == 'i':
                need(6); print('led', json.dumps(list(rx[1:6])), flush=True); del rx[:6]
            elif c == 'x':
                del rx[0]  # not streaming: nothing to stop, no reply
            elif c in 'rw':
                need(7)
                m_mask, n, n_cmd = struct.unpack('<IBB', rx[1:7])
                total = 7 + n_cmd + (n if c == 'w' else 0)
                need(total)
                reg = rx[7] & 0x7F
                vals = rx[7 + n_cmd:total]
                del rx[:total]
                out = bytearray()
                for m in range(NB):
                    if m_mask & (1 << m):
                        if c == 'r':
                            out += regs[m][reg:reg + n]
                        else:
                            regs[m][reg:reg + n] = vals
                if c == 'r':
                    reply(master, out)
            elif c == 'A':
                need(6)
                mask = struct.unpack('<I', rx[1:5])[0]
                del rx[:6]
                streaming = True
                stream_t0 = time.monotonic()
                next_n = [0] * NB
                counters = [0] * NB
                print('stream started', flush=True)
            else:
                del rx[0]
        if streaming:
            now = time.monotonic()
            due = []  # (detection time, frame): sent in time order, as the Teensy polls round-robin
            for m in range(NB):
                if not mask & (1 << m):
                    continue
                rate = ODR * (1 + PPM[m] * 1e-6)
                while True:
                    t_true = stream_t0 + PHASE[m] + next_n[m] / rate
                    t_det = t_true + random.uniform(0, JITTER_S)
                    if t_det > now:
                        break
                    next_n[m] += 1
                    if FAULTS and m == 5 and random.random() < 1 / 2000:
                        stats['drop'] += 1
                        continue
                    ecg = b''.join(code(signal(t_true, ch)) for ch in range(3))
                    fr = bytearray(frame(ord('D'), m, counters[m], micros(t_det), 0xE0, ecg))
                    counters[m] += 1
                    if FAULTS and random.random() < 1 / 5000:
                        fr[random.randrange(1, 22)] ^= 0x10
                        stats['crc'] += 1
                    due.append((t_det, bytes(fr)))
                    if FAULTS and m == 7 and random.random() < 1 / 1000:
                        stats['dup'] += 1
                        due.append((t_det + 80e-6, frame(ord('D'), m, counters[m], micros(t_det + 80e-6), 0xE0, ecg)))
                        counters[m] += 1
            out_bytes = b''.join(fr for _, fr in sorted(due, key=lambda x: x[0]))
            if out_bytes:
                os.write(master, out_bytes)


if __name__ == '__main__':
    main()
