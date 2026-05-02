"""Domain mixer.

Interleaves multiple (named) iterators with weights from
`PerplexityCorrelationCurriculum`. Reentrant — can be re-iterated freely —
and deterministic given a seed.

Usage:

    mixer = DomainMixer(
        sources={"fineweb_edu": iter1, "starcoder": iter2, ...},
        curriculum=curriculum,         # optional; uses .weights dict
        seed=0,
    )
    for item in mixer:
        ...

If `curriculum` is None, weights default to uniform.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Dict, Iterator, Optional

from sprl.training.curriculum import PerplexityCorrelationCurriculum

# A "source factory" returns a fresh iterator (so the mixer is reentrant).
SourceFactory = Callable[[], Iterator]


@dataclass
class _Source:
    name: str
    factory: SourceFactory
    iterator: Optional[Iterator] = None
    exhausted: bool = False


class DomainMixer:
    """Weighted-random interleaver over named sources.

    Each source is a *factory* (zero-arg callable producing an iterator). The
    mixer can therefore be iterated multiple times without callers worrying
    about iterator exhaustion.

    Weights are read live from `curriculum.weights` (if provided) so the mixer
    reflects curriculum updates without rebuild.
    """

    def __init__(
        self,
        sources: Dict[str, SourceFactory],
        *,
        curriculum: Optional[PerplexityCorrelationCurriculum] = None,
        weights: Optional[Dict[str, float]] = None,
        seed: int = 0,
        restart_exhausted: bool = True,
    ):
        if not sources:
            raise ValueError("DomainMixer requires at least one source.")
        self.sources: Dict[str, _Source] = {
            n: _Source(n, f) for n, f in sources.items()
        }
        self.curriculum = curriculum
        self._explicit_weights = weights
        self.seed = seed
        self.restart_exhausted = restart_exhausted

    # ------------------------------------------------------------------
    def _current_weights(self) -> Dict[str, float]:
        if self._explicit_weights is not None:
            w = self._explicit_weights
        elif self.curriculum is not None:
            w = self.curriculum.weights
        else:
            n = len(self.sources)
            w = {k: 1.0 / n for k in self.sources}
        w = {k: float(w.get(k, 0.0)) for k in self.sources}
        s = sum(w.values())
        if s <= 0:
            n = len(self.sources)
            return {k: 1.0 / n for k in self.sources}
        return {k: v / s for k, v in w.items()}

    # ------------------------------------------------------------------
    def __iter__(self) -> Iterator:
        rng = random.Random(self.seed)
        # Fresh iterators at every __iter__ — that is the reentrancy guarantee.
        for s in self.sources.values():
            s.iterator = s.factory()
            s.exhausted = False
        names = list(self.sources.keys())

        while True:
            w = self._current_weights()
            weights = [w[n] for n in names]
            if all(self.sources[n].exhausted for n in names):
                return
            masked = [
                weights[i] if not self.sources[n].exhausted else 0.0
                for i, n in enumerate(names)
            ]
            total = sum(masked)
            if total <= 0:
                return
            masked = [m / total for m in masked]
            pick = rng.choices(names, weights=masked, k=1)[0]
            src = self.sources[pick]
            try:
                yield next(src.iterator)
            except StopIteration:
                if self.restart_exhausted:
                    src.iterator = src.factory()
                    try:
                        yield next(src.iterator)
                    except StopIteration:
                        src.exhausted = True
                else:
                    src.exhausted = True
