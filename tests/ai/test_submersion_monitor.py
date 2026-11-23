"""Rule-based submersion monitor (Blueprint M6) on the stub pool (25 m x 12.5 m, exits (0,2) and (25,10.5))."""
from __future__ import annotations

from lifeguard.ai.features.submersion import PoolTrackView, SubmersionMonitor
from lifeguard.system.geometry.stub import StubPoolGeometry

DT = 0.1


def _view(
    track_id: int,
    xy: tuple[float, float] | None,
    visible: bool = True,
    in_water: bool = True,
    first_seen_t: float = 0.0,
) -> PoolTrackView:
    return PoolTrackView(track_id, visible, xy, in_water, first_seen_t)


def _monitor_with_disappearance(xy: tuple[float, float] = (10.0, 6.0), in_water: bool = True):
    """Track 1 visible at xy for t = 0..1.0, then gone at t = 1.1."""
    monitor = SubmersionMonitor(StubPoolGeometry())
    for i in range(11):
        monitor.update(round(i * DT, 3), [_view(1, xy, in_water=in_water)])
    timers = monitor.update(1.1, [])
    return monitor, timers


def test_disappearance_in_interior_starts_timer_at_last_visible_time() -> None:
    _, timers = _monitor_with_disappearance()
    assert len(timers) == 1
    assert timers[0].track_id == 1
    assert timers[0].start_t == 1.0
    assert timers[0].xy == (10.0, 6.0)


def test_lost_state_counts_as_not_visible() -> None:
    monitor = SubmersionMonitor(StubPoolGeometry())
    monitor.update(0.0, [_view(1, (10.0, 6.0))])
    timers = monitor.update(0.1, [_view(1, (10.0, 6.0), visible=False)])
    assert [t.track_id for t in timers] == [1]


def test_timer_persists_until_reappearance() -> None:
    monitor, _ = _monitor_with_disappearance()
    for i in range(12, 200):
        timers = monitor.update(round(i * DT, 3), [])
    assert [t.track_id for t in timers] == [1]


def test_new_track_nearby_ends_timer_and_is_merged() -> None:
    monitor, _ = _monitor_with_disappearance()
    timers = monitor.update(5.0, [_view(7, (11.2, 6.3), first_seen_t=4.9)])
    assert timers == []
    assert monitor.merged_from(7) == (1,)


def test_same_id_nearby_ends_timer() -> None:
    monitor, _ = _monitor_with_disappearance()
    assert monitor.update(3.0, [_view(1, (10.5, 6.0))]) == []
    assert monitor.merged_from(1) == ()


def test_same_id_far_away_does_not_end_timer() -> None:
    """A likely ID switch must not cancel the timer."""
    monitor, _ = _monitor_with_disappearance()
    timers = monitor.update(3.0, [_view(1, (16.0, 6.0))])
    assert [t.track_id for t in timers] == [1]


def test_existing_swimmer_passing_over_the_spot_does_not_end_timer() -> None:
    monitor, _ = _monitor_with_disappearance()
    timers = monitor.update(3.0, [_view(2, (10.0, 6.0), first_seen_t=0.0)])
    assert [t.track_id for t in timers] == [1]


def test_new_track_far_away_does_not_end_timer() -> None:
    monitor, _ = _monitor_with_disappearance()
    timers = monitor.update(3.0, [_view(8, (20.0, 3.0), first_seen_t=2.9)])
    assert [t.track_id for t in timers] == [1]


def test_no_timer_near_exit_edge_or_out_of_water() -> None:
    assert _monitor_with_disappearance(xy=(0.5, 2.0))[1] == []    # at the ladder (exit)
    assert _monitor_with_disappearance(xy=(12.0, 0.5))[1] == []   # within 1 m of the wall
    assert _monitor_with_disappearance(in_water=False)[1] == []   # on the deck


def test_merge_chain_keeps_all_old_ids() -> None:
    monitor, _ = _monitor_with_disappearance()
    monitor.update(5.0, [_view(7, (10.5, 6.0), first_seen_t=4.9)])
    monitor.update(5.1, [_view(7, (10.5, 6.0), first_seen_t=4.9)])
    monitor.update(5.2, [])  # 7 goes under again
    monitor.update(8.0, [_view(9, (10.6, 6.1), first_seen_t=7.9)])
    assert monitor.merged_from(9) == (7, 1)


def test_reset_clears_timers() -> None:
    monitor, _ = _monitor_with_disappearance()
    monitor.reset()
    assert monitor.timers == {}
    assert monitor.update(2.0, []) == []
