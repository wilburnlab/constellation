"""Phase 2+ cluster subpackage — fingerprint computation, genome-guided
clustering, weighted-PWM consensus building, de novo clustering stubs.

The genome-guided clusterer is the production workhorse
(:mod:`.cluster_genome`); the de novo Phase 3 stub
(:mod:`.cluster`) and consensus re-export (:mod:`.consensus`) are
included here so all cluster-stage modules share a directory.
"""

from __future__ import annotations
