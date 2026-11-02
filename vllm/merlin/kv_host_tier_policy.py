# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""Host-RAM KV tier: the probation/protected admission policy (flag).

An SLRU-like split of published
host units, byte-weighted (host rows), with no step clock:

* a newly published unit enters PROBATION, unless it is the successor of a
  protected unit (the session came back: it inherits the segment) or its
  keys hit the GHOST list (a recently evicted unit's keys: the session came
  back after losing its host copy);
* a host use (hit / restore) promotes a unit to PROTECTED;
* PROTECTED is capped at ``PROTECTED_SHARE`` of published rows; the least
  recently promoted protected units are demoted to probation past the cap.
  The cap is re-applied whenever a unit is protected, dropped or evicted, or
  its row count changes. The single most recently promoted unit is never
  demoted, so one unit larger than the cap may hold the segment alone;
* within the LRU victim class, probation units are evicted before protected
  ones, and a copy that is not for a returning session (no ghost hit, no
  protected predecessor) never displaces a protected unit: it is bypassed
  (``probation_bypassed``) instead.

The flag ``MERLIN_KV_HOST_TIER__ADMISSION`` (``slru``; unset / ``off`` /
``lru`` = off) defaults OFF. Off, every hook is a no-op, every unit ranks 0
and ``UnitTable.plan`` orders the LRU class by last use alone (the converged
default). Duplicate classes (running twins, imminent twins) and idle rows
are never affected by this policy.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass

ADMISSION_ENV = "MERLIN_KV_HOST_TIER__ADMISSION"
MODES_OFF = ("", "0", "off", "lru")
MODES_ON = ("slru",)

PROBATION = 0
PROTECTED = 1

# Share of published host rows the protected segment may hold.
PROTECTED_SHARE = 0.8
# Plain block hashes remembered from evicted units (FIFO).
GHOST_KEYS_MAX = 1 << 16


@dataclass
class AdmissionCounters:
    probation_admitted: int = 0  # units published into probation
    protected_admitted: int = 0  # units published straight into protected
    probation_bypassed: int = 0  # short plans that refused to take protected
    promoted: int = 0  # probation -> protected on a host use
    demoted: int = 0  # protected -> probation past the cap
    ghost_hits: int = 0  # publications whose keys hit the ghost list


class LruAdmission:
    """Flag off: no segments; the LRU class is ordered by last use."""

    enabled = False

    def __init__(self) -> None:
        self.counters = AdmissionCounters()

    def rank(self, unit_id: int) -> int:
        return PROBATION

    def admits_protected(
        self, key_sets: Iterable[frozenset], preds: Iterable[int]
    ) -> bool:
        return True

    def on_publish(
        self, unit_id: int, rows: int, *, inherit: Iterable[int] = (),
        keys: frozenset = frozenset(),
    ) -> None:
        return None

    def on_host_use(self, unit_id: int, rows: int) -> None:
        return None

    def on_evict(self, unit_id: int, keys: frozenset) -> None:
        return None

    def on_drop(self, unit_id: int) -> None:
        return None

    def clear(self) -> None:
        return None


class SlruAdmission(LruAdmission):
    """Probation / protected segments over published units (rows-weighted)."""

    enabled = True

    def __init__(
        self, protected_share: float = PROTECTED_SHARE,
        ghost_max: int = GHOST_KEYS_MAX,
    ) -> None:
        super().__init__()
        self.protected_share = protected_share
        self.ghost_max = ghost_max
        # unit id -> rows, in promotion order (oldest first).
        self._protected: OrderedDict[int, int] = OrderedDict()
        self._probation: dict[int, int] = {}
        # Running row totals of the two segments (kept in step with the maps).
        self._prob_rows = 0
        self._prot_rows = 0
        self._ghost: OrderedDict[object, None] = OrderedDict()

    # -- queries -------------------------------------------------------------

    def rank(self, unit_id: int) -> int:
        return PROTECTED if unit_id in self._protected else PROBATION

    def ghost_hit(self, keys: Iterable[object]) -> bool:
        return any(k in self._ghost for k in keys)

    def admits_protected(
        self, key_sets: Iterable[frozenset], preds: Iterable[int]
    ) -> bool:
        """A copy may displace protected units only for a returning session:
        a protected predecessor, or keys that hit the ghost list."""
        if any(p in self._protected for p in preds):
            return True
        return any(self.ghost_hit(keys) for keys in key_sets)

    # -- events --------------------------------------------------------------

    def on_publish(
        self, unit_id: int, rows: int, *, inherit: Iterable[int] = (),
        keys: frozenset = frozenset(),
    ) -> None:
        if unit_id in self._protected:
            # Re-publication keeps an earned segment.
            self._set_protected(unit_id, rows)
            self._rebalance()
            return
        if unit_id in self._probation:
            self._set_probation(unit_id, rows)
            self._rebalance()
            return
        inherit = list(inherit)
        if any(p in self._protected for p in inherit):
            self._protect(unit_id, rows)
            self.counters.protected_admitted += 1
        elif keys and self.ghost_hit(keys):
            self.counters.ghost_hits += 1
            self._protect(unit_id, rows)
            self.counters.protected_admitted += 1
        else:
            self._set_probation(unit_id, rows)
            self.counters.probation_admitted += 1
        for k in keys:
            self._ghost.pop(k, None)

    def on_host_use(self, unit_id: int, rows: int) -> None:
        if unit_id in self._protected:
            self._set_protected(unit_id, rows)
            self._protected.move_to_end(unit_id)
            self._rebalance()
            return
        if unit_id in self._probation:
            self._prob_rows -= self._probation.pop(unit_id)
            self.counters.promoted += 1
            self._protect(unit_id, rows)

    def on_evict(self, unit_id: int, keys: frozenset) -> None:
        # Sorted: the ghost list is FIFO-bounded, so insertion order
        # decides which ghosts survive; set order of bytes is per process.
        for k in sorted(keys):
            self._ghost[k] = None
            self._ghost.move_to_end(k)
        while len(self._ghost) > self.ghost_max:
            self._ghost.popitem(last=False)
        self.on_drop(unit_id)

    def on_drop(self, unit_id: int) -> None:
        self._prob_rows -= self._probation.pop(unit_id, 0)
        prot = self._protected.pop(unit_id, None)
        if prot is not None:
            self._prot_rows -= prot
        # A removal shrinks the published total, so the cap is re-applied.
        self._rebalance()

    def clear(self) -> None:
        self._probation.clear()
        self._protected.clear()
        self._prob_rows = self._prot_rows = 0
        self._ghost.clear()

    # -- internals -----------------------------------------------------------

    def _set_protected(self, unit_id: int, rows: int) -> None:
        self._prot_rows += rows - self._protected.get(unit_id, 0)
        self._protected[unit_id] = rows

    def _set_probation(self, unit_id: int, rows: int) -> None:
        self._prob_rows += rows - self._probation.get(unit_id, 0)
        self._probation[unit_id] = rows

    def _protect(self, unit_id: int, rows: int) -> None:
        self._set_protected(unit_id, rows)
        self._protected.move_to_end(unit_id)
        self._rebalance()

    def _rebalance(self) -> None:
        """Demote the least recently promoted protected units to probation
        until the protected segment is within its share of published rows.
        The most recently promoted unit always stays protected."""
        cap = self.protected_share * (self._prob_rows + self._prot_rows)
        while self._prot_rows > cap and len(self._protected) > 1:
            old, n = self._protected.popitem(last=False)
            self._prot_rows -= n
            self._set_probation(old, n)
            self.counters.demoted += 1


def admission_mode(raw: object) -> str:
    """Normalise the flag value: ``"off"`` or ``"slru"``; anything else is
    an error (raised at boot by ``UnitTable``)."""
    mode = ("" if raw is None else str(raw)).strip().lower()
    if mode in MODES_OFF:
        return "off"
    if mode in MODES_ON:
        return mode
    raise ValueError(
        f"{ADMISSION_ENV} must be one of {MODES_OFF + MODES_ON} (got {raw!r})"
    )


def make_admission_policy(raw: object = None) -> LruAdmission:
    return SlruAdmission() if admission_mode(raw) == "slru" else LruAdmission()


__all__ = [
    "ADMISSION_ENV",
    "AdmissionCounters",
    "LruAdmission",
    "SlruAdmission",
    "admission_mode",
    "make_admission_policy",
]
