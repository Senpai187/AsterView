"""No Qt or GPU: bounded work, source isolation, and navigation starvation."""
from concurrent.futures import ThreadPoolExecutor
import gc
import weakref

import pytest

from asterview.slice_scheduler import LatestSliceQueue


def test_continuous_input_presents_finished_frame_and_runs_only_latest_pending():
    queue = LatestSliceQueue()
    first = queue.request("plane 1", context=("source A", "Explore"))
    assert queue.take().ticket == first
    for index in range(2, 101):
        latest = queue.request(f"plane {index}", context=("source A", "Explore"))
        assert queue.take() is None
    assert queue.active.payload == "plane 1"
    assert queue.pending.payload == "plane 100"
    assert queue.accepts(first, interacting=True)
    assert queue.complete(first)
    last = queue.take()
    assert last.ticket == latest and last.payload == "plane 100"
    assert queue.pending is None
    assert queue.accepts(last.ticket, interacting=True)
    assert queue.complete(last.ticket)
    assert queue.take() is None


def test_settle_rejects_intermediate_frame_and_requires_exact_latest():
    queue = LatestSliceQueue()
    first = queue.request(1, context=("source A", "Explore"))
    queue.take()
    last = queue.request(2, context=("source A", "Explore"))
    assert queue.accepts(first, interacting=True)
    assert not queue.accepts(first, interacting=False)
    queue.complete(first)
    queue.take()
    assert queue.accepts(last, interacting=False)


@pytest.mark.parametrize("next_context", [
    ("source B", "Explore", 1),
    ("source A", "TMJ", 1),
    ("source A", "Explore", 2),
])
def test_source_workflow_and_layout_epochs_never_present_stale_frames(next_context):
    queue = LatestSliceQueue()
    old = queue.request("old source pixels", context=("source A", "Explore", 1))
    queue.take()
    current = queue.request("new source pixels", context=next_context)
    assert current.epoch > old.epoch
    assert not queue.active_is_current
    assert not queue.accepts(old, interacting=True)
    assert not queue.accepts(old, interacting=False)
    assert queue.take() is None
    queue.complete(old)
    assert queue.take().ticket == current
    assert queue.active_is_current
    assert queue.accepts(current, interacting=False)


def test_explicit_restore_or_empty_current_request_invalidates_all_pending_work():
    queue = LatestSliceQueue()
    context = ("source A", "Explore")
    old = queue.request("old", context=context)
    queue.take()
    queue.request("pending", context=context)
    queue.invalidate(context)
    assert queue.latest_ticket is None and queue.pending is None
    assert not queue.accepts(old, interacting=True)
    assert queue.take() is None
    current = queue.request("restored", context=context)
    assert current.epoch > old.epoch
    assert queue.complete(old)
    assert queue.take().ticket == current


def test_duplicate_completion_cannot_release_another_running_worker():
    queue = LatestSliceQueue()
    old = queue.request("old", context=1)
    queue.take()
    current = queue.request("current", context=1)
    queue.complete(old)
    queue.take()
    assert not queue.complete(old)
    assert queue.active.ticket == current
    assert not queue.accepts(old, interacting=True)


def test_coalescing_releases_replaced_payloads():
    class Payload:
        pass

    queue = LatestSliceQueue()
    first, pending = Payload(), Payload()
    old_ref, pending_ref = weakref.ref(first), weakref.ref(pending)
    first_ticket = queue.request(first, context=1)
    queue.take()
    queue.request(pending, context=1)
    del first, pending
    queue.request(Payload(), context=1)
    gc.collect()
    assert old_ref() is not None
    assert pending_ref() is None
    queue.complete(first_ticket)
    gc.collect()
    assert old_ref() is None


def test_scheduler_cannot_be_mutated_from_worker_thread():
    queue = LatestSliceQueue()
    with ThreadPoolExecutor(max_workers=1) as worker:
        result = worker.submit(lambda: queue.request("pixels", context=1))
        with pytest.raises(RuntimeError, match="owning GUI thread"):
            result.result()
