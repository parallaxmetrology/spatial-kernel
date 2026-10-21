"""Parallax success detection: agreement of N independent reads, with Ω as uncertainty.

The contrarian thesis of this package is that the *judge* of a robot outcome should
not itself be a learned model that the policy can learn to fool. But a single
deterministic check is also brittle — one camera centroid is viewpoint-dependent
and easy to satisfy accidentally. The parallax move is to make success the
*agreement of several genuinely independent reads* of the same outcome, and to
report their disagreement (Ω) as the honest confidence of the verdict.

This is the Parallax-Ω construction from the seismology work, repointed:

    Ω = mean pairwise distance between N independent structural reads of one scene.
        low Ω  → the observables concur → a coherent, trustworthy "success".
        high Ω → the observables disagree → the outcome is contested; do not trust
                 any single read.

The independence requirement is load-bearing and easy to violate. Five reads that
are all transforms of the same RGB pixel surface measure *transform diversity*, not
physical agreement — exactly the failure documented in the seismology Guardrail 3
("five critics in different rooms, comparing notes", not one critic in five mirrors).
Each read therefore declares the observable *surface* it derives from, and the audit
counts distinct surfaces — not reads — as the independence dimension.

Candidate independent surfaces for a manipulation outcome:
    rgb        — appearance / gradient structure
    depth      — geometric occupancy above the support plane
    proprio    — gripper/end-effector pose and force trace
    time       — scene-settling / motion-cessation signature

A verdict that rests on one surface is reported as such, with its independence
flagged low — the number does not pretend to a robustness it has not earned.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Read:
    """One independent read of an outcome.

    centroid: normalized object position, kernel convention (delta_x, delta_y),
              each roughly in [-0.5, 0.5]; 0 = frame centre.
    in_region: did this read place the object inside the task's target region?
    surface:  the physical observable this read derives from (independence key).
    """

    name: str
    surface: str
    centroid: tuple[float, float]
    in_region: bool
    weight: float = 1.0
    note: str = ""


@dataclass(frozen=True)
class ParallaxVerdict:
    success: bool
    confidence: float            # [0,1], derived from Ω and region agreement
    omega: float                 # mean pairwise centroid disagreement
    region_agreement: float      # fraction of reads voting in-region
    unanimous: bool
    independence: int            # number of distinct observable surfaces
    independence_note: str
    n_reads: int
    contributing: list[str]
    detail: dict[str, Any] = field(default_factory=dict)


def _pairwise_omega(reads: list[Read]) -> float:
    """Mean weighted pairwise Euclidean distance between read centroids."""
    n = len(reads)
    if n < 2:
        return 0.0
    total = 0.0
    wsum = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            (xi, yi), (xj, yj) = reads[i].centroid, reads[j].centroid
            d = math.hypot(xi - xj, yi - yj)
            w = reads[i].weight * reads[j].weight
            total += w * d
            wsum += w
    return float(total / wsum) if wsum > 0 else 0.0


def _confidence_from_omega(omega: float, region_agreement: float, omega_scale: float) -> float:
    """Confidence falls off as reads disagree, and is gated by region agreement.

    omega_scale is the disagreement (in normalized-centroid units) at which spatial
    confidence has decayed to 1/e. region_agreement multiplies in so that even
    tightly-clustered reads that all agree the object is in the *wrong* place cannot
    report high success confidence.
    """
    spatial = math.exp(-omega / max(omega_scale, 1e-6))
    return float(max(0.0, min(1.0, spatial * region_agreement)))


def fuse_reads(
    reads: list[Read],
    *,
    omega_scale: float = 0.18,
    omega_success_max: float = 0.20,
    min_region_agreement: float = 1.0,
) -> ParallaxVerdict:
    """Fuse N independent reads into a success verdict with Ω-derived confidence.

    success requires: unanimous-enough region agreement AND coherent observables
    (Ω below omega_success_max). The two conditions guard different failures —
    region agreement guards "all reads agree it's in the wrong place"; the Ω gate
    guards "the reads disagree about where it even is".

    omega_scale       — Ω at which spatial confidence decays to 1/e.
    omega_success_max  — max disagreement still allowed to call success.
    min_region_agreement — required fraction voting in-region (1.0 = unanimous).
    """
    if not reads:
        return ParallaxVerdict(
            success=False, confidence=0.0, omega=0.0, region_agreement=0.0,
            unanimous=False, independence=0,
            independence_note="no reads supplied.", n_reads=0, contributing=[],
        )

    omega = _pairwise_omega(reads)
    wsum = sum(r.weight for r in reads) or 1.0
    region_agreement = sum(r.weight for r in reads if r.in_region) / wsum
    unanimous = all(r.in_region for r in reads) or all(not r.in_region for r in reads)

    surfaces = sorted({r.surface for r in reads})
    independence = len(surfaces)
    if independence >= 3:
        ind_note = f"{independence} independent surfaces ({', '.join(surfaces)}); robust."
    elif independence == 2:
        ind_note = f"2 independent surfaces ({', '.join(surfaces)}); minimally robust."
    else:
        ind_note = (
            f"single surface ('{surfaces[0]}'); Ω measures transform diversity, "
            "not physical agreement — treat confidence as provisional."
        )

    confidence = _confidence_from_omega(omega, region_agreement, omega_scale)
    # A single-surface verdict cannot earn full confidence — cap it (Guardrail 3).
    if independence < 2:
        confidence = min(confidence, 0.5)

    success = bool(
        region_agreement >= min_region_agreement
        and omega <= omega_success_max
        and any(r.in_region for r in reads)
    )

    return ParallaxVerdict(
        success=success,
        confidence=round(confidence, 4),
        omega=round(omega, 4),
        region_agreement=round(float(region_agreement), 4),
        unanimous=unanimous,
        independence=independence,
        independence_note=ind_note,
        n_reads=len(reads),
        contributing=[r.name for r in reads],
        detail={
            "surfaces": surfaces,
            "omega_scale": omega_scale,
            "omega_success_max": omega_success_max,
            "min_region_agreement": min_region_agreement,
            "per_read": [
                {
                    "name": r.name,
                    "surface": r.surface,
                    "centroid": [round(r.centroid[0], 4), round(r.centroid[1], 4)],
                    "in_region": r.in_region,
                    "weight": r.weight,
                    "note": r.note,
                }
                for r in reads
            ],
        },
    )
