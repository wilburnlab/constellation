"""A stand-in Thermo raw file for the reader's tests (no DLLs, no CLR).

``_FakeRaw`` answers the slice of CommonCore's ``IRawDataPlus`` the range
reader calls; ``scan`` builds one stand-in scan; ``mixed_raw`` is a small
run with every way a scan's peaks are made. The ``fake_centroider``
fixture (in ``conftest.py``) stands in for ``Scan.FromFile`` + Thermo's
centroider.
"""

from __future__ import annotations

import numpy as np


class Obj:
    """Attribute bag standing in for a CommonCore object."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


class FakeEvent:
    def __init__(self, mass=None, width=None, offset=0.0):
        self._mass, self._width, self._offset = mass, width, offset

    def GetMass(self, idx):  # noqa: ARG002
        if self._mass is None:
            raise RuntimeError("no precursor")
        return self._mass

    def GetIsolationWidth(self, idx):  # noqa: ARG002
        if self._width is None:
            raise RuntimeError("no width")
        return self._width

    def GetIsolationWidthOffset(self, idx):  # noqa: ARG002
        return self._offset


def scan(
    scan_type: str,
    *,
    rt_min: float,
    acquired: str = "centroid",
    mz=(),
    intensity=None,
    stream=None,
    trailer=(),
    event=None,
):
    """One stand-in scan. ``acquired`` is ``centroid`` or ``profile``;
    ``stream`` is ``None`` (no centroid stream), ``()`` (an empty one) or
    ``(mz, intensity, resolution, noise, baseline)``. ``trailer`` entries
    are ``(label, display string)`` or ``(label, display string, typed
    value)``; without a typed value the display string is its value."""
    mz = np.asarray(mz, dtype=np.float64)
    intensity = (
        np.asarray(intensity, dtype=np.float64)
        if intensity is not None
        else np.arange(1, mz.size + 1, dtype=np.float64)
    )
    return {
        "stats": Obj(
            TIC=float(intensity.sum()),
            BasePeakMass=float(mz[np.argmax(intensity)]) if mz.size else 0.0,
            BasePeakIntensity=float(intensity.max()) if mz.size else 0.0,
            StartTime=rt_min,
            ScanType=scan_type,
            IsCentroidScan=acquired == "centroid",
        ),
        "mz": mz,
        "intensity": intensity,
        "stream": stream,
        "trailer": Obj(Labels=[t[0] for t in trailer], Values=[t[1] for t in trailer]),
        "typed": {t[0]: (t[2] if len(t) > 2 else _typed(t[1])) for t in trailer},
        "event": event or FakeEvent(),
    }


def _typed(display: str):
    try:
        return float(display) if "." in display else int(display)
    except ValueError:
        return display


class FakeRaw:
    """The slice of IRawDataPlus the range reader calls.

    The trailer header is every label any scan has, in first-seen order;
    ``header=False`` stands in for a file that gives no header (or no
    typed values), so only the display strings are there to read.
    """

    def __init__(self, scans: dict[int, dict], *, header: bool = True):
        self.scans = scans
        self.RunHeaderEx = Obj(FirstSpectrum=min(scans), LastSpectrum=max(scans))
        labels: dict[str, None] = {}
        for s in scans.values():
            labels.update(dict.fromkeys(s["trailer"].Labels))
        self.header = list(labels) if header else None

    def GetScanStatsForScanNumber(self, n):
        return self.scans[n]["stats"]

    def GetScanEvents(self, lo, hi):
        return [self.scans[n]["event"] for n in range(lo, hi + 1)]

    def GetScanEventForScanNumber(self, n):
        return self.scans[n]["event"]

    def GetSegmentedScanFromScanNumber(self, n, stats):  # noqa: ARG002
        s = self.scans[n]
        return Obj(Positions=s["mz"], Intensities=s["intensity"])

    def GetCentroidStream(self, n, include_ref):  # noqa: ARG002
        stream = self.scans[n]["stream"]
        if stream is None or stream == ():
            return Obj(Length=0, Masses=None, Intensities=None)
        mz, inten, res, noise, base = (np.asarray(a, dtype=np.float64) for a in stream)
        return Obj(
            Length=mz.size, Masses=mz, Intensities=inten,
            Resolutions=res, Noises=noise, Baselines=base,
        )

    def GetTrailerExtraInformation(self, n):
        return self.scans[n]["trailer"]

    def GetTrailerExtraHeaderInformation(self):
        if self.header is None:
            raise RuntimeError("no trailer header")
        return [Obj(Label=label, DataType="FLOAT", StringLengthOrPrecision=3) for label in self.header]

    def GetTrailerExtraValues(self, n):
        typed = self.scans[n]["typed"]
        # Every header entry, the ones this scan does not have included.
        return [typed.get(label, 0) for label in self.header]

    def Dispose(self):
        pass


MS1 = "FTMS + p NSI Full ms [400.0000-1600.0000]"
MS2 = "ITMS + c NSI r Full ms2 500.2500@hcd30.00 [200.0000-1500.0000]"
IT_MS1 = "ITMS + p NSI r Full ms [400.0000-1200.0000]"


def mixed_raw() -> "FakeRaw":
    """Four scans, one per way a scan's peaks are made in centroid mode."""
    return FakeRaw({
        # Orbitrap profile MS1 with the instrument's centroid stream.
        1: scan(MS1, rt_min=0.5, acquired="profile", mz=[400.0, 400.1, 400.2],
                intensity=[1.0, 9.0, 2.0],
                stream=([400.1, 500.0], [9.0, 4.0], [6e4, 5e4], [1.0, 1.5], [0.1, 0.2]),
                trailer=[("Ion Injection Time (ms):", "15.000"), ("AGC:", "On")]),
        # Ion-trap centroid MS2.
        2: scan(MS2, rt_min=0.51, mz=[201.0, 305.5, 410.2],
                trailer=[("Monoisotopic M/Z:", "500.2512"), ("Charge State:", "2"),
                         ("MS2 Isolation Width:", "2.00"), ("Master Scan Number:", "1")],
                event=FakeEvent(mass=500.25, width=2.0009)),
        # Ion-trap profile MS1: no centroid stream at all.
        3: scan(IT_MS1, rt_min=1.2, acquired="profile",
                mz=[450.0, 450.1, 450.2, 450.3], intensity=[1.0, 8.0, 10.0, 2.0]),
        # Orbitrap profile MS1 whose stream exists but is empty.
        4: scan(MS1, rt_min=1.3, acquired="profile", mz=[400.0, 400.1],
                intensity=[3.0, 4.0], stream=()),
    })


def long_raw(n: int, *, first: int = 1) -> "FakeRaw":
    """``n`` ion-trap centroid scans, an MS1 then MS2s, a few peaks each."""
    scans = {}
    for i in range(first, first + n):
        ms1 = (i - first) % 10 == 0
        scans[i] = scan(
            IT_MS1.replace(" p ", " c ") if ms1 else MS2, rt_min=i / 600.0,
            mz=[300.0 + i % 7, 400.0 + i % 11, 500.0 + i % 13],
            intensity=[float(i), 2.0 * i, 3.0 * i],
            event=None if ms1 else FakeEvent(mass=500.25, width=2.0),
        )
    return FakeRaw(scans)
