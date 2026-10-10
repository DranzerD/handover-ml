"""Throughput and scheduling tests. (Simulated - see cellselect.sim; not real RF/modem testing.)"""

from __future__ import annotations

import pytest

from cellselect.suite.catalog import specs_for_template
from cellselect.suite.checks import run_check

pytestmark = pytest.mark.cellular


@pytest.mark.parametrize("spec", specs_for_template("dl_throughput"), ids=lambda s: s.test_id)
def test_downlink_throughput(spec, cs_ctx, cs_lab_time, request):
    """DL throughput vs. 3GPP-style reference at the measured SINR."""
    request.node.cs_kpis = run_check(spec, cs_ctx)


@pytest.mark.parametrize("spec", specs_for_template("ul_throughput"), ids=lambda s: s.test_id)
def test_uplink_throughput_power_control(spec, cs_ctx, cs_lab_time, request):
    """UL throughput with open/closed-loop power control."""
    request.node.cs_kpis = run_check(spec, cs_ctx)


@pytest.mark.parametrize("spec", specs_for_template("multi_user"), ids=lambda s: s.test_id)
def test_multi_user_fairness(spec, cs_ctx, cs_lab_time, request):
    """Proportional-fair scheduling fairness across UEs."""
    request.node.cs_kpis = run_check(spec, cs_ctx)
