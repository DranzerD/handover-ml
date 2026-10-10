"""Simulator behaviour and fault-injection semantics."""

from __future__ import annotations

import numpy as np
import pytest

from cellselect.sim.confparser import ConfigError, parse_config
from cellselect.sim.context import Environment, SimContext, stable_seed
from cellselect.sim.faults import COMPONENTS, FAULTS
from cellselect.sim.radio import harq_outcome, select_cqi, simulate_link
from cellselect.sim.rrc import connection_setup, radio_link_monitor
from cellselect.sim.scheduler import n_prb
from cellselect.sim.session import simulate_outages


def ctx(*faults: str, seed: int = 1) -> SimContext:
    return SimContext(frozenset(faults), Environment(), seed)


def test_stable_seed_is_deterministic_and_distinct() -> None:
    assert stable_seed(1, "A") == stable_seed(1, "A")
    assert stable_seed(1, "A") != stable_seed(1, "B")


def test_every_fault_targets_a_known_component() -> None:
    assert all(f.component in COMPONENTS for f in FAULTS.values())


def test_link_simulation_is_deterministic() -> None:
    a = simulate_link(ctx(), 15.0, 200, "pedestrian", "NR", 106, 30)
    b = simulate_link(ctx(), 15.0, 200, "pedestrian", "NR", 106, 30)
    np.testing.assert_allclose(a.tput_mbps, b.tput_mbps)


def test_cqi_offset_fault_only_manifests_at_high_sinr() -> None:
    sinr = np.array([5.0, 10.0, 19.0, 20.0])
    clean, faulty = select_cqi(sinr, ctx()), select_cqi(sinr, ctx("F_CQI_OFFSET"))
    assert (faulty[:2] == clean[:2]).all()
    assert (faulty[2:] > clean[2:]).all()


def test_harq_fault_increases_residual_loss() -> None:
    sinr, cqi = np.full(10, 3.0), np.full(10, 6)
    clean, _ = harq_outcome(sinr, cqi, ctx(), "NR")
    faulty, _ = harq_outcome(sinr, cqi, ctx("F_HARQ_MAXRETX"), "NR")
    assert faulty.mean() > 100 * clean.mean()


def test_prb_fault_only_affects_100mhz() -> None:
    assert n_prb(40, 30, "NR", ctx("F_SCHED_PRB")) == n_prb(40, 30, "NR", ctx())
    assert n_prb(100, 30, "NR", ctx("F_SCHED_PRB")) < n_prb(100, 30, "NR", ctx())


def test_t310_fault_declares_rlf_for_short_fade() -> None:
    trace = np.r_[np.full(50, 10.0), np.full(30, -15.0), np.full(100, 10.0)]  # 300 ms fade @10 ms
    assert radio_link_monitor(ctx(), trace, 10.0) == []
    assert len(radio_link_monitor(ctx("F_T310_SCALE"), trace, 10.0)) >= 1


def test_t300_fault_breaks_cell_edge_setup_only() -> None:
    assert connection_setup(ctx(), -3.5, 400).success_rate > 0.97
    assert connection_setup(ctx("F_RRC_T300"), -3.5, 400).success_rate < 0.9
    assert connection_setup(ctx("F_RRC_T300"), 15.0, 400).success_rate == 1.0


def test_config_range_fault_rejects_band_maximum() -> None:
    raw = {"band": "n78", "bandwidth_mhz": 100}
    assert parse_config(raw, ctx()).bandwidth_mhz == 100
    with pytest.raises(ConfigError):
        parse_config(raw, ctx("F_CONFIG_RANGE"))


def test_backoff_fault_only_hurts_repeated_outages() -> None:
    assert simulate_outages(ctx(), 0.3, 1).max_recovery_s == pytest.approx(
        simulate_outages(ctx("F_BACKOFF"), 0.3, 1).max_recovery_s
    )
    assert (
        simulate_outages(ctx("F_BACKOFF"), 0.3, 4).max_recovery_s
        > simulate_outages(ctx(), 0.3, 4).max_recovery_s + 0.3
    )
