"""Bounded, GUI-thread scheduling for interactive physical slice requests.

Only one native worker runs at a time. Input replaces a single pending request,
so wheel bursts do not accumulate arrays or spawn threads. A completed frame
may be shown while input continues, avoiding starvation when rendering takes
longer than the input interval. Once input settles, only the latest request may
be presented. Source/workflow changes always invalidate earlier frames.
"""
from __future__ import annotations

from dataclasses import dataclass
from threading import get_ident
from typing import Any, Hashable


@dataclass(frozen=True)
class SliceTicket:
    epoch: int
    revision: int


@dataclass(frozen=True)
class SliceRequest:
    ticket: SliceTicket
    payload: Any


_UNSET = object()


class LatestSliceQueue:
    """GUI-owned bookkeeping; worker execution and cancellation stay external.

    ``context`` should contain immutable source and workflow identity, e.g.
    ``(id(volume), tab_index, layout_epoch)``. Every requested geometry change
    calls :meth:`request`; no in-epoch worker cancellation is needed. Pass its
    ticket through the worker result and call :meth:`accepts` before painting,
    then :meth:`complete` when the worker actually finishes. ``take`` must be
    called only when it is safe to launch that next native worker.

    ``invalidate`` retains ownership of a running old worker until completion,
    preventing overlapping workers during source replacement. Its result can
    never be presented. Call ``invalidate`` on restore, close, deleted panes,
    or when an empty request means the desired images are already displayed.
    """

    def __init__(self) -> None:
        self._owner = get_ident()
        self._epoch = 0
        self._revision = 0
        self._context: Hashable | object = _UNSET
        self._latest: SliceTicket | None = None
        self._active: SliceRequest | None = None
        self._pending: SliceRequest | None = None

    def _check_thread(self) -> None:
        if get_ident() != self._owner:
            raise RuntimeError("Slice scheduling must stay on its owning GUI thread")

    @property
    def active(self) -> SliceRequest | None:
        return self._active

    @property
    def pending(self) -> SliceRequest | None:
        return self._pending

    @property
    def latest_ticket(self) -> SliceTicket | None:
        return self._latest

    @property
    def active_is_current(self) -> bool:
        """False permits cancellation of an obsolete source/workflow worker."""
        return self._active is not None and self._active.ticket.epoch == self._epoch

    def invalidate(self, context: Hashable | object = _UNSET) -> None:
        self._check_thread()
        self._epoch += 1
        self._context = context
        self._latest = None
        self._pending = None

    def request(self, payload: Any, *, context: Hashable) -> SliceTicket:
        self._check_thread()
        # Requiring a hashable context prevents an accidental comparison of
        # volume arrays and makes this constant-time for ordinary tuple keys.
        hash(context)
        if self._context is _UNSET or context != self._context:
            self.invalidate(context)
        self._revision += 1
        ticket = SliceTicket(self._epoch, self._revision)
        self._latest = ticket
        self._pending = SliceRequest(ticket, payload)
        return ticket

    def take(self) -> SliceRequest | None:
        self._check_thread()
        if self._active is not None or self._pending is None:
            return None
        self._active, self._pending = self._pending, None
        return self._active

    def accepts(self, ticket: SliceTicket, *, interacting: bool) -> bool:
        self._check_thread()
        if (self._active is None or self._active.ticket != ticket
                or ticket.epoch != self._epoch or self._latest is None):
            return False
        return bool(interacting or ticket == self._latest)

    def complete(self, ticket: SliceTicket) -> bool:
        """Release exactly the worker that finished, including obsolete ones."""
        self._check_thread()
        if self._active is None or self._active.ticket != ticket:
            return False
        self._active = None
        return True
