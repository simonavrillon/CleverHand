"""Continuous acquisition: frame parsing, per-module clock fit and alignment.

The Teensy streams every ECG sample of every module, stamped with the time
(``micros()``) at which it first saw the sample. Each module runs on its own
crystal, so this module fits each module's sample period and offset, detects
dropped or duplicated samples, and resamples all modules onto the sample
times of module 0. The first SETTLE_SAMPLES samples of each module are
dropped while the ADS1293 decimation filters settle after START_CONV.
"""

import threading
import time

import numpy as np

from . import ads1293
from .protocol import FRAME_DATA, FRAME_SIZE, FRAME_SYNC, crc8_rows

MAX_WAIT_US = 200_000      # wait this long for a late module, then output NaN
FIRST_FRAMES = 256         # frames indexed from the nominal period, before the fit takes over
ALIGN_MIN_SAMPLES = 256    # samples before a module's fit is used to align
CLOCK_TAU_S = 20           # time constant of the clock fit
HISTORY_S = 1.0            # samples kept per module for the alignment
# first samples of each module dropped while the decimation filters settle
# after START_CONV (measured: from -550 mV back to the input within 5 samples)
SETTLE_SAMPLES = 8


class LinearClock:
    """Fit y = a + slope * x with exponential forgetting.

    Models a module's sample times (y, us) against its sample index (x). The
    sums are kept relative to the last point so they stay precise over long
    recordings.
    """

    def __init__(self, forgetting):
        self.lam = forgetting
        self.n = 0
        self.S = self.Sx = self.Sy = self.Sxx = self.Sxy = 0.0
        self.x0 = self.y0 = 0.0

    def update(self, x, y):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        m = len(x)
        if m == 0:
            return
        if self.n == 0:
            self.x0, self.y0 = x[0], y[0]
        dx, dy = x - self.x0, y - self.y0
        w = self.lam ** np.arange(m - 1, -1, -1)  # weight of each new point
        decay = self.lam ** m
        self.S = decay * self.S + w.sum()
        self.Sx = decay * self.Sx + (w * dx).sum()
        self.Sy = decay * self.Sy + (w * dy).sum()
        self.Sxx = decay * self.Sxx + (w * dx * dx).sum()
        self.Sxy = decay * self.Sxy + (w * dx * dy).sum()
        self.n += m
        self._recenter(x[-1], y[-1])

    def _recenter(self, x0, y0):
        ax, ay = x0 - self.x0, y0 - self.y0
        self.Sxx += -2 * ax * self.Sx + ax * ax * self.S
        self.Sxy += -ax * self.Sy - ay * self.Sx + ax * ay * self.S
        self.Sx -= ax * self.S
        self.Sy -= ay * self.S
        self.x0, self.y0 = x0, y0

    def slope(self):
        den = self.S * self.Sxx - self.Sx * self.Sx
        return (self.S * self.Sxy - self.Sx * self.Sy) / den if den > 0 else 0.0

    def _intercept(self, b):
        return (self.Sy - b * self.Sx) / self.S

    def predict(self, x):
        b = self.slope()
        return self.y0 + self._intercept(b) + b * (np.asarray(x, dtype=np.float64) - self.x0)

    def inverse(self, y):
        b = self.slope()
        return self.x0 + (np.asarray(y, dtype=np.float64) - self.y0 - self._intercept(b)) / b


class _Module:
    def __init__(self, nominal_period_us, forgetting, keep):
        self.T = nominal_period_us
        self.clock = LinearClock(forgetting)
        self.keep = keep
        self.to_settle = SETTLE_SAMPLES
        self.started = False
        self.first = []  # (t, volts) batches until FIRST_FRAMES have arrived
        self.indexed = False
        self.last_ts32 = 0
        self.epoch = 0
        self.k_last = -1
        self.t_last = 0.0
        self.k = np.zeros(0, dtype=np.int64)  # received samples, oldest first
        self.v = np.zeros((0, 3))
        self.samples = self.dropped = self.duplicates = 0

    def add(self, ts32, volts):
        """Add the frames of one batch (arrival order)."""
        if self.to_settle:
            n = min(self.to_settle, len(ts32))
            self.to_settle -= n
            ts32, volts = ts32[n:], volts[n:]
            if len(ts32) == 0:
                return
        # unwrap micros() (wraps every 71.6 min)
        ts = ts32.astype(np.int64)
        prev = np.concatenate([[self.last_ts32 if self.started else ts[0]], ts[:-1]])
        wraps = np.cumsum((ts < prev) & (prev - ts > 2**31))
        t = (self.epoch + wraps).astype(np.float64) * 2**32 + ts
        self.epoch += int(wraps[-1])
        self.last_ts32 = int(ts[-1])

        self.started = True
        if not self.indexed:
            self.first.append((t, volts))
            if sum(len(b[0]) for b in self.first) < FIRST_FRAMES:
                return
            t, volts = (np.concatenate(c) for c in zip(*self.first, strict=True))
            self.first, self.indexed = [], True
            n = FIRST_FRAMES
            self._accept(self._first_indices(t[:n]), t[:n], volts[:n])
            t, volts = t[n:], volts[n:]
        i = 0
        while i < len(t):
            # from the fitted clock, which is centred on the timestamp jitter
            # (so tolerates nearly a period of it), in slices short enough
            # that extrapolating a young fit stays accurate
            m = max(16, self.clock.n // 4)
            k = np.rint(self.clock.inverse(t[i:i + m])).astype(np.int64)
            self._accept(k, t[i:i + m], volts[i:i + m])
            i += m

    def _first_indices(self, t):
        """Sample indices of the first frames, from the nominal period.

        A timestamp lags its sample by the poll latency (0 to one poll round),
        so the timestamps modulo the period gather on an arc narrower than the
        period. Indexing from the centre of that arc tolerates drops,
        duplicates and almost a period of jitter; comparing consecutive
        timestamps would need the jitter under half a period. The crystal
        tolerance (< 100 ppm) moves the arc by only ~12 us over these frames.
        """
        phase = np.sort(np.mod(t - t[0], self.T))
        gaps = np.diff(np.concatenate([phase, [phase[0] + self.T]]))
        g = int(np.argmax(gaps))  # the widest empty stretch is outside the arc
        start = phase[(g + 1) % len(phase)]
        centre = t[0] + start + (self.T - gaps[g]) / 2
        k = np.rint((t - centre) / self.T).astype(np.int64)
        return k - k[0]

    def _accept(self, k, t, volts):
        # a frame that does not advance the index is a duplicate
        before = np.maximum.accumulate(np.concatenate([[self.k_last], k]))[:-1]
        new = k > before
        self.duplicates += int((~new).sum())
        k, t, volts = k[new], t[new], volts[new]
        if len(k) == 0:
            return
        base = self.k_last if self.k_last >= 0 else k[0] - 1
        self.dropped += int((np.diff(np.concatenate([[base], k])) - 1).sum())
        self.samples += len(k)
        self.k_last, self.t_last = int(k[-1]), float(t[-1])
        self.clock.update(k, t)

        self.k = np.concatenate([self.k, k])[-self.keep:]
        self.v = np.concatenate([self.v, volts])[-self.keep:]

    def values_at(self, ks):
        """Values at integer indices (NaN where the sample is missing)."""
        out = np.full((len(ks), 3), np.nan)
        if len(self.k) == 0:
            return out
        idx = np.searchsorted(self.k, ks).clip(0, len(self.k) - 1)
        hit = self.k[idx] == ks
        out[hit] = self.v[idx[hit]]
        return out

    def interpolate(self, t):
        """Values at times t (us), linear between the two nearest samples."""
        x = self.clock.inverse(t)
        i0 = np.floor(x).astype(np.int64)
        frac = (x - i0)[:, None]
        return self.values_at(i0) * (1 - frac) + self.values_at(i0 + 1) * frac

    def rate(self):
        b = self.clock.slope() if self.indexed else 0.0
        return 1e6 / b if b > 0 else 0.0


class Stream:
    """Aligned stream of all modules.

    Use as a context manager (starts and stops the acquisition) or call
    ``start()``/``stop()``. ``read()`` returns ``(t_host, t_device, data)``:
    sample times in s on the host clock (``time.monotonic()``) and on the controller clock, and the
    data, (N, 3 * modules) in V with NaN where a sample is missing.

    A background thread drains the serial port continuously, so the Teensy
    never waits on a full buffer however often ``read()`` is called.
    """

    def __init__(self, ctrl, nb_modules, cfg):
        self.ctrl = ctrl
        self.nb = nb_modules
        self.cfg = cfg
        self.nominal_rate = cfg.odr()
        self.adc_max = cfg.adc_max()
        self.channel_labels = [f"m{m}_ch{ch + 1}" for m in range(nb_modules) for ch in range(3)]
        self.running = False

    # -------------------------------------------------------------- control

    def start(self):
        T = 1e6 / self.nominal_rate
        lam = 1 - 1 / (CLOCK_TAU_S * self.nominal_rate)
        keep = int(HISTORY_S * self.nominal_rate) + 64
        self.modules = [_Module(T, lam, keep) for _ in range(self.nb)]
        self.rx = bytearray()
        self.crc_errors = 0
        self._resyncing = False
        self.next_k = 0
        self.latest_t = 0.0
        self.block_start = None
        self.block_min = 0.0
        self.block_mins = []
        self._pending = bytearray()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._error = None
        self.ctrl.start_stream((1 << self.nb) - 1, self.cfg.drdy_mask())
        self._thread = threading.Thread(target=self._drain, name="clvhd-reader", daemon=True)
        self._thread.start()
        self.running = True

    def stop(self):
        if self.running:
            self._stop.set()
            self._thread.join()
            self.ctrl.stop_stream()
        self.running = False

    def _drain(self):
        try:
            while not self._stop.is_set():
                data = self.ctrl.read_stream(0.01)
                if data:
                    with self._lock:
                        self._pending += data
        except Exception as e:  # reported by read()
            self._error = e

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()

    # ----------------------------------------------------------------- data

    def read(self, timeout=0.02):
        """Aligned samples received during the next ``timeout`` s (0: only what already arrived).

        Processing the samples in batches keeps the CPU load low, so call it
        with a timeout of 10-50 ms rather than in a tight loop.
        """
        if timeout > 0:
            time.sleep(timeout)
        if self._error is not None:
            raise self._error
        with self._lock:
            self.rx += self._pending
            self._pending = bytearray()
        frames = self._parse()
        if len(frames):
            module = frames[:, 2]
            ts32 = np.ascontiguousarray(frames[:, 7:11]).view("<u4")[:, 0]
            volts = ads1293.ecg_volts(frames[:, 12:21].reshape(-1, 3, 3), self.adc_max, self.cfg.enable)
            for m in np.unique(module):
                if m < self.nb:
                    sel = module == m
                    self.modules[m].add(ts32[sel], volts[sel])
            self.latest_t = max([self.latest_t, *(mod.t_last for mod in self.modules)])
        return self._emit()

    def _parse(self):
        """Valid data frames of the receive buffer (resynchronising after a bad CRC)."""
        buf = np.frombuffer(bytes(self.rx), dtype=np.uint8)
        pos, out = 0, []
        while len(buf) - pos >= FRAME_SIZE:
            k = (len(buf) - pos) // FRAME_SIZE
            rows = buf[pos:pos + k * FRAME_SIZE].reshape(k, FRAME_SIZE)
            ok = (rows[:, 0] == FRAME_SYNC) & (crc8_rows(rows[:, 1:-1]) == rows[:, -1])
            n_ok = k if ok.all() else int(np.argmin(ok))
            out.append(rows[:n_ok])
            pos += n_ok * FRAME_SIZE
            if n_ok > 0:
                self._resyncing = False
            if n_ok < k:  # bad frame: skip one byte and look for the next sync
                if not self._resyncing:
                    self.crc_errors += 1  # once per corruption
                    self._resyncing = True
                pos += 1
        del self.rx[:pos]
        frames = np.concatenate(out) if out else np.zeros((0, FRAME_SIZE), dtype=np.uint8)
        return frames[frames[:, 1] == FRAME_DATA]

    def _host_offset(self):
        # the smallest (host - device) difference is the one with the least
        # transfer latency; keep the minimum over the last ten 1 s blocks
        now = time.monotonic()
        diff = now - self.latest_t * 1e-6
        if self.block_start is None or now - self.block_start > 1.0:
            if self.block_start is not None:
                self.block_mins = [*self.block_mins, self.block_min][-10:]
            self.block_start, self.block_min = now, diff
        else:
            self.block_min = min(self.block_min, diff)
        return min([self.block_min, *self.block_mins])

    def _emit(self):
        empty = (np.zeros(0), np.zeros(0), np.zeros((0, 3 * self.nb)))
        if self.latest_t <= 0:
            return empty
        offset = self._host_offset()
        m0 = self.modules[0]
        if m0.samples < ALIGN_MIN_SAMPLES:
            return empty
        if len(m0.k) and self.next_k < m0.k[0]:
            self.next_k = int(m0.k[0])  # history lost
        ks = np.arange(self.next_k, m0.k_last + 1)
        t = m0.clock.predict(ks)

        # output a sample once every module has data past it, or once it is
        # MAX_WAIT_US older than the newest data
        ok = np.ones(len(ks), dtype=bool)
        for mod in self.modules[1:]:
            if mod.samples >= ALIGN_MIN_SAMPLES:
                reached = t < mod.clock.predict(mod.k_last)
            else:
                reached = np.zeros(len(ks), dtype=bool)
            ok &= reached | (self.latest_t - t >= MAX_WAIT_US)
        n = len(ks) if ok.all() else int(np.argmin(ok))
        if n == 0:
            return empty
        ks, t = ks[:n], t[:n]

        data = np.full((n, 3 * self.nb), np.nan)
        data[:, 0:3] = m0.values_at(ks)
        for b, mod in enumerate(self.modules[1:], start=1):
            if mod.samples >= ALIGN_MIN_SAMPLES:
                data[:, 3 * b:3 * b + 3] = mod.interpolate(t)
        self.next_k += n
        t_device = t * 1e-6
        return t_device + offset, t_device, data

    def stats(self):
        """Per module: samples, dropped, duplicates, rate_hz (controller clock), ppm vs module 0."""
        out = []
        rate0 = self.modules[0].rate()
        for mod in self.modules:
            r = mod.rate()
            out.append(dict(samples=mod.samples, dropped=mod.dropped, duplicates=mod.duplicates,
                            rate_hz=r, ppm=(r / rate0 - 1) * 1e6 if rate0 > 0 and r > 0 else 0.0))
        return out
