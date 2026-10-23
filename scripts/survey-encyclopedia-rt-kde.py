"""Record what EncyclopeDIA's retention-time fit returns for synthetic points.

``massspec.rt.encyclopedia`` replays the jar's fit
(``RetentionTimeFilter.getFilter(points, 0.95f)``) bit for bit. The lab's
reference searches check that on real data, but they are not in the
repository; this script runs the pinned jar's own classes on small
synthetic point sets chosen to reach every branch of the fit (the linear
fallbacks at 1, 2 and 20 points, the first KDE at 21, a zero-width axis,
values landing exactly on half cells, tied neighbours, horizontal and
vertical runs, a walk that stops on the top edge, outliers) and writes
what the jar returned, so the replay is tested without lab data::

    tests/data/encyclopedia_rt_kde/<version>/<case>.json

Each file holds the points (``x``, ``y``: library and observed time, in
the order given); per point the jar's warp, delta, 95 % flag and inverse
warp of ``y``; and the warp at 201 query times running a tenth of the
range past each end (the fit's shape, clamped ends included, without
its thousands of knots). Every value is as Java printed it
(``Float.toString`` / ``Double.toString`` parse back to the same
number).

Needs the installed jar (``constellation doctor``) and a ``javac`` (the
conda environment's JDK); the harness runs under the jar's bundled JRE,
the JVM a search uses. Only synthetic files in a temporary directory are
given to Java. Run from project root::

    python3 scripts/survey-encyclopedia-rt-kde.py
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

from constellation.thirdparty import encyclopedia as _adapter  # noqa: F401  (registers the tool)
from constellation.thirdparty.encyclopedia import MINIMUM_VERSION
from constellation.thirdparty.jvm import _resolve_java
from constellation.thirdparty.registry import find

_HARNESS = r"""
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import edu.washington.gs.maccoss.encyclopedia.algorithms.alignment.RetentionTimeFilter;
import edu.washington.gs.maccoss.encyclopedia.utils.graphing.XYPoint;

public class RtKdeSurvey {
    public static void main(String[] args) throws Exception {
        for (int c = 0; c + 1 < args.length; c += 2) {
            ArrayList<XYPoint> points = new ArrayList<>();
            ArrayList<Double> queries = new ArrayList<>();
            for (String line : Files.readAllLines(Path.of(args[c]))) {
                String[] f = line.split("\t");
                if (f[0].equals("Q")) {
                    queries.add(Double.parseDouble(f[1]));
                } else {
                    points.add(new XYPoint(Double.parseDouble(f[1]), Double.parseDouble(f[2])));
                }
            }
            RetentionTimeFilter filter = RetentionTimeFilter.getFilter(points, 0.95f);
            try (PrintWriter out = new PrintWriter(args[c + 1])) {
                for (Double q : queries) {
                    out.println("Q\t" + filter.getYValue((float) q.doubleValue()));
                }
                for (XYPoint p : points) {
                    float x = (float) p.getX();
                    float y = (float) p.getY();
                    out.println("P\t" + filter.getYValue(x) + "\t" + filter.getDelta(y, x) + "\t"
                        + filter.isWithinMiddle95(y, x) + "\t" + filter.getXValue(y));
                }
            }
        }
    }
}
"""


def _cases() -> dict[str, list[tuple[float, float]]]:
    """Point sets, deterministic, values rounded so they print short."""
    rng = random.Random(20261022)

    def r(v: float) -> float:
        return round(v, 6)

    def curve(t: float) -> float:
        return 2.0 + 100.0 * (t / 100.0) ** 0.75

    cases: dict[str, list[tuple[float, float]]] = {}
    cases["n1"] = [(12.5, 14.0)]
    cases["n2"] = [(10.0, 12.0), (50.0, 61.0)]
    cases["n20"] = [(r(x), r(1.1 * x + 3 + rng.gauss(0, 0.5))) for x in (rng.uniform(5, 95) for _ in range(20))]
    cases["n21"] = [(r(x), r(curve(x) + rng.gauss(0, 0.5))) for x in (rng.uniform(5, 95) for _ in range(21))]
    cases["zero_width_x"] = [(5.0, r(rng.uniform(10, 90))) for _ in range(40)]
    # x and y span exactly 0..2999, so a value k + 0.5 sits on a half cell.
    half = [(0.0, 0.0), (2999.0, 2999.0)]
    half += [(k + 0.5, min(2999.0, k + 0.5 + (k % 7))) for k in range(10, 2980, 37)]
    cases["half_cells"] = half
    # A lattice of single points: zero density between them, exact ties
    # between neighbours.
    cases["ties"] = [(float(10 * i), float(10 * i + (5 if i % 3 == 0 else 0))) for i in range(40)]
    # Plateaus and jumps: horizontal and vertical runs in the walk.
    steps = []
    for k in range(6):
        steps += [(r(20 * k + rng.uniform(0, 15)), r(30.0 * k + rng.gauss(0, 0.2))) for _ in range(25)]
        steps += [(r(20 * k + 15 + rng.gauss(0, 0.2)), r(30.0 * k + rng.uniform(0, 30))) for _ in range(25)]
    cases["steps"] = steps
    # The data reach the top of y before the right of x, as in the lab's
    # gas-phase-fraction first pass.
    top = [(r(x), r(x + rng.gauss(0, 0.6))) for x in (rng.uniform(0, 80) for _ in range(400))]
    top += [(r(rng.uniform(80, 100)), r(80 + rng.gauss(0, 0.3))) for _ in range(60)]
    cases["edge_y"] = top
    outliers = []
    for _ in range(1500):
        x = rng.betavariate(1.6, 1.4) * 100
        y = curve(x) + rng.gauss(0, 0.4) if rng.random() > 0.3 else rng.uniform(0, 105)
        outliers.append((r(x), r(y)))
    cases["outliers"] = outliers
    return cases


def _queries(points: list[tuple[float, float]]) -> list[float]:
    lo = min(p[0] for p in points)
    hi = max(p[0] for p in points)
    pad = 0.1 * (hi - lo) if hi > lo else 1.0
    return [round(lo - pad + (hi - lo + 2 * pad) * k / 200, 6) for k in range(201)]


def _run(java: Path, javac: str, jar: Path, cases: dict[str, list[tuple[float, float]]], timeout: float) -> dict:
    lib = sorted((jar.parent / "lib").glob("*.jar"))
    classpath = os.pathsep.join(str(p) for p in [jar, *lib])
    work = Path(tempfile.mkdtemp(prefix="rt-kde-survey-"))
    try:
        (work / "RtKdeSurvey.java").write_text(_HARNESS)
        subprocess.run(
            [javac, "--release", "21", "-cp", classpath, "-d", str(work), str(work / "RtKdeSurvey.java")],
            check=True,
        )
        argv = [str(java), f"-Djava.io.tmpdir={work / 'tmp'}", "-cp", os.pathsep.join([str(work), classpath]), "RtKdeSurvey"]
        (work / "tmp").mkdir()
        for name, points in cases.items():
            lines = [f"P\t{x!r}\t{y!r}\n" for x, y in points]
            lines += [f"Q\t{q!r}\n" for q in _queries(points)]
            (work / f"{name}.in").write_text("".join(lines))
            argv += [str(work / f"{name}.in"), str(work / f"{name}.out")]
        proc = subprocess.Popen(argv, cwd=work, start_new_session=True)
        try:
            proc.wait(timeout=timeout)
        finally:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        if proc.returncode != 0:
            raise RuntimeError(f"the harness exited with status {proc.returncode}")
        results = {}
        for name, points in cases.items():
            queried, rows = [], []
            for line in (work / f"{name}.out").read_text().splitlines():
                f = line.split("\t")
                if f[0] == "Q":
                    queried.append(float(f[1]))
                else:
                    rows.append((float(f[1]), float(f[2]), f[3] == "true", float(f[4])))
            results[name] = {
                "case": name,
                "n": len(points),
                "x": [p[0] for p in points],
                "y": [p[1] for p in points],
                "query_x": _queries(points),
                "query_warp": queried,
                "warp": [r[0] for r in rows],
                "delta": [r[1] for r in rows],
                "within95": [r[2] for r in rows],
                "inverse_y": [r[3] for r in rows],
            }
        return results
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="default: tests/data/encyclopedia_rt_kde/<version>/")
    parser.add_argument("--javac", default=shutil.which("javac") or "javac")
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args(argv)

    handle = find("encyclopedia")
    version = handle.version or MINIMUM_VERSION
    java, java_source, _ = _resolve_java(handle)
    print(f"jar:  {handle.path} ({version})\njava: {java} ({java_source})", file=sys.stderr)
    out_dir = args.output_dir or Path("tests/data/encyclopedia_rt_kde") / version
    out_dir.mkdir(parents=True, exist_ok=True)
    results = _run(Path(java), args.javac, Path(handle.path), _cases(), args.timeout)
    for name, record in results.items():
        if any(isinstance(v, float) and not math.isfinite(v) for v in record["warp"]):
            print(f"{name}: non-finite warp", file=sys.stderr)
        (out_dir / f"{name}.json").write_text(json.dumps(record, separators=(",", ":")) + "\n")
        print(f"{name}: {record['n']} points", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
