# -*- coding: utf-8 -*-
from __future__ import annotations

"""
ace_protocol.py
===============
Description of a multi-flip-angle / ACE-MRI acquisition schedule.

This module answers exactly one question: *what did the scanner play, and
when?*  It holds no physics (see `signal_models.py`), no kinetic models
(see `kinetic_models.py`) and no fitting code (see `fitting*.py`).

A single-flip-angle acquisition is the one-segment special case, not a
separate code path -- see `ACEProtocol.single_fa`.

IMPORTANT: this module must not import anything from `dce_mri`.
`config.py` imports it, so any back-reference would risk an import cycle.
Nothing here needs one: a schedule is just times, angles and TRs.

Units
-----
    t_edges : s
    alphas  : degrees  (nominal, i.e. before any B1 scaling)
    TRs     : ms       (matching config.AcquisitionConfig.TR)

Public API
----------
    ACEProtocol   -- piecewise FA/TR schedule
    FrameArrays   -- per-frame view of a schedule, returned by frame_arrays()
"""


from dataclasses import dataclass
from typing import NamedTuple

import numpy as np


class FrameArrays(NamedTuple):
    """Per-frame expansion of an ACEProtocol over a given time vector."""
    alpha_deg: np.ndarray   # (T,) nominal flip angle per frame, deg
    TR_ms:     np.ndarray   # (T,) repetition time per frame, ms
    seg_id:    np.ndarray   # (T,) int segment index per frame
    keep:      np.ndarray   # (T,) bool; False for settling frames after a switch


@dataclass(frozen=True)
class ACEProtocol:
    """
    Piecewise flip-angle / TR encoding schedule.

    Segment i covers t_edges[i] <= t < t_edges[i+1], during which the
    scanner plays nominal flip angle alphas[i] at repetition time TRs[i].
    A time falling exactly on an edge belongs to the later segment; times
    past the final edge fall in the last segment.

    Fields
    ------
    t_edges  : (N+1,) segment boundary times (s), must start at 0 and increase
    alphas   : (N,)   nominal flip angles (deg)
    TRs      : (N,)   repetition times (ms)
    n_settle : frames to drop immediately after each flip-angle switch, to
               exclude data acquired before the new steady state is reached.
               Counted in FRAMES, not seconds, so the appropriate value
               depends on frame duration; the presets copy each paper's choice.
    """

    t_edges:  np.ndarray
    alphas:   np.ndarray
    TRs:      np.ndarray
    n_settle: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "t_edges", np.asarray(self.t_edges, dtype=np.float64))
        object.__setattr__(self, "alphas", np.asarray(self.alphas,  dtype=np.float64))
        object.__setattr__(self, "TRs",  np.asarray(self.TRs,     dtype=np.float64))

        if len(self.t_edges) != len(self.alphas) + 1:
            raise ValueError("t_edges must have len(alphas)+1 entries")
        if len(self.alphas) != len(self.TRs):
            raise ValueError("alphas and TRs must have the same length")
        if self.t_edges[0] != 0.0:
            raise ValueError("t_edges must start at 0")
        if np.any(np.diff(self.t_edges) <= 0):
            raise ValueError("t_edges must be strictly increasing")
        if np.any(self.alphas <= 0) or np.any(self.TRs <= 0):
            raise ValueError("alphas and TRs must be positive")
        if self.n_settle < 0:
            raise ValueError("n_settle must be >= 0")

    # -- queries ----------------------------------------------------------

    @property
    def n_segments(self) -> int:
        return len(self.alphas)

    @property
    def is_single_fa(self) -> bool:
        """True if one flip angle / TR is played for the whole acquisition."""
        return self.n_segments == 1

    def segment_of(self, t: np.ndarray) -> np.ndarray:
        """Segment index for each time in `t`."""
        t = np.atleast_1d(np.asarray(t, dtype=np.float64))
        return np.clip(np.searchsorted(self.t_edges, t, side="right") - 1,
                       0, self.n_segments - 1)

    def frame_arrays(self, time_var: np.ndarray) -> FrameArrays:
        """Expand the schedule onto the frame times in `time_var`."""
        seg = self.segment_of(time_var)

        keep = np.ones(seg.shape, dtype=bool)
        if self.n_settle > 0:
            for i in np.flatnonzero(np.diff(seg) != 0) + 1:
                keep[i:i + self.n_settle] = False

        return FrameArrays(alpha_deg=self.alphas[seg],
                           TR_ms=self.TRs[seg],
                           seg_id=seg,
                           keep=keep)

    # -- presets ----------------------------------------------------------

    @classmethod
    def single_fa(cls, alpha: float = 10.0, TR_ms: float = 3.64,
                  t_max: float = 150.0) -> "ACEProtocol":
        """Conventional single-flip-angle DCE acquisition (one segment)."""
        return cls(t_edges=np.array([0.0, t_max]),
                   alphas=np.array([alpha]),
                   TRs=np.array([TR_ms]))

    @classmethod
    def zhang2017_ace(cls, t_max: float = 529.0, n_settle: int = 1,
                      rescale: bool = False) -> "ACEProtocol":
        """
        ACE-MRI schedule of Zhang et al., NMR Biomed 2017.

        The published segment durations are tied to the number of
        excitations needed to reach steady state at each TR, so by default
        they are kept at their real second-values and only the final
        return-to-baseline segment is stretched to reach `t_max`.

        Pass rescale=True to scale every edge proportionally instead. That
        produces a self-similar but time-compressed schedule whose segments
        no longer satisfy the steady-state assumption, so use it only for
        exploring protocol design, not for simulating the published method.
        """
        edges_pub = np.array([0., 267., 294., 321., 348., 383., 496., 529.])
        alphas    = np.array([10., 12., 8., 5., 2., 90., 10.])
        TRs       = np.array([12., 12., 12., 12., 12., 100., 12.])

        if rescale:
            edges = edges_pub * (t_max / edges_pub[-1])
        else:
            if t_max <= edges_pub[-2]:
                raise ValueError(
                    f"t_max={t_max} is too short for the unscaled Zhang2017 "
                    f"schedule (needs > {edges_pub[-2]} s); pass rescale=True "
                    "for a self-similar shorter version instead."
                )
            edges = edges_pub.copy()
            edges[-1] = t_max

        return cls(edges, alphas, TRs, n_settle=n_settle)

    @classmethod
    def zhang2019_dfa(cls, t_max: float, a1: float = 8.0, a2: float = 25.0,
                      TR_ms: float = 3.64, n_settle: int = 2) -> "ACEProtocol":
        """
        Dual-flip-angle schedule of Zhang et al., NMR Biomed 2019:
        three equal segments, a1 - a2 - a1.
        """
        edges = np.array([0.0, t_max / 3.0, 2.0 * t_max / 3.0, t_max])
        return cls(edges, np.array([a1, a2, a1]), np.array([TR_ms] * 3),
                   n_settle=n_settle)