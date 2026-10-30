# Copyright 2026 ROBI Contributors.
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations
from typing import Callable, Generator
from escape_ood.suites.spec import Condition, Suite, enumerate_conditions

class Task:
    pass

    def __init__(self, condition: Condition) -> None:
        self.condition = condition

    @property
    def instruction(self) -> str:
        """Language for this episode. Comes from the condition, so it matches the axis."""
        return self.condition.instruction

    def run(self, ctx) -> Generator:
        raise NotImplementedError

class ConditionQueue:
    """Deterministic queue over (condition x repeat), advancing only on confirmed success.

    Reproducible from the suite alone: no RNG decides *which* conditions get collected, only
    per-episode nuisance variation (spawn yaw) inside an episode. Two runs of the same axis
    therefore cover the same conditions in the same order.

    ``partition_idx`` / ``partition_total`` split the work across parallel collectors by
    striding, so each gets an interleaved slice rather than a contiguous block — a contiguous
    split would make one worker collect only far-from-support conditions, and a crash would
    then bias the dataset along the very axis being measured.
    """

    def __init__(self, suite: Suite, object_keys: list[str], *, demos_per_condition: int=5, splits: tuple[str, ...]=('train',), partition_idx: int=0, partition_total: int=1, predicate: Callable[[Condition], bool] | None=None) -> None:
        if partition_total < 1 or not 0 <= partition_idx < partition_total:
            raise ValueError(f'bad partition {partition_idx}/{partition_total}')
        self.suite = suite
        self.demos_per_condition = max(1, int(demos_per_condition))
        conds = enumerate_conditions(suite, object_keys, splits=splits)
        if predicate is not None:
            conds = [c for c in conds if predicate(c)]
            if not conds:
                raise ValueError(f"predicate excluded every condition of suite '{suite.name}'")
        conds.sort(key=lambda c: c.key())
        self._all = conds
        self._pending: list[Condition] = [c for (i, c) in enumerate(conds) if i % partition_total == partition_idx]
        self._remaining: dict[str, int] = {c.key(): self.demos_per_condition for c in self._pending}
        self._order: list[Condition] = list(self._pending)
        self._cursor = 0

    @property
    def total_episodes(self) -> int:
        return len(self._order) * self.demos_per_condition

    @property
    def collected(self) -> int:
        return self.total_episodes - sum(self._remaining.values())

    @property
    def done(self) -> bool:
        return all((v == 0 for v in self._remaining.values()))

    def summary(self) -> str:
        return f'suite={self.suite.name} conditions={len(self._order)} demos/condition={self.demos_per_condition} collected={self.collected}/{self.total_episodes}'

    def next_condition(self) -> Condition | None:
        """The condition to attempt next, or None when the queue is exhausted.

        Round-robins rather than draining one condition at a time, so an interrupted run still
        yields a dataset spread across the whole suite instead of a prefix of it.
        """
        if self.done:
            return None
        n = len(self._order)
        for offset in range(n):
            c = self._order[(self._cursor + offset) % n]
            if self._remaining[c.key()] > 0:
                self._cursor = (self._cursor + offset) % n
                return c
        return None

    def confirm_success(self, condition: Condition) -> None:
        """Record one successful demonstration. Only this advances the queue."""
        k = condition.key()
        if k not in self._remaining:
            raise KeyError(f'condition not in this queue partition: {k}')
        if self._remaining[k] > 0:
            self._remaining[k] -= 1
        self._cursor = (self._cursor + 1) % max(len(self._order), 1)

    def record_failure(self, condition: Condition) -> None:
        """A failed attempt. Deliberately does not decrement — it advances the cursor only,
        so a persistently-failing condition does not block the rest of the suite."""
        if condition.key() not in self._remaining:
            raise KeyError(f'condition not in this queue partition: {condition.key()}')
        self._cursor = (self._cursor + 1) % max(len(self._order), 1)
