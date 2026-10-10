"""Latency, packet-loss and buffering tests. (Simulated - see cellselect.sim; not real RF/modem testing.)"""

from __future__ import annotations

import pytest

from cellselect.suite.catalog import specs_for_template
from cellselect.suite.checks import run_check

pytestmark = pytest.mark.cellular


@pytest.mark.parametrize("spec", specs_for_template("latency_load"), ids=lambda s: s.test_id)
def test_latency_under_load(spec, cs_ctx, cs_lab_time, request):
    """p95 user-plane latency under cell load."""
    request.node.cs_kpis = run_check(spec, cs_ctx)


@pytest.mark.parametrize("spec", specs_for_template("packet_loss"), ids=lambda s: s.test_id)
def test_packet_loss_under_interference(spec, cs_ctx, cs_lab_time, request):
    """Residual loss after HARQ under interference/fading."""
    request.node.cs_kpis = run_check(spec, cs_ctx)


@pytest.mark.parametrize("spec", specs_for_template("burst_buffer"), ids=lambda s: s.test_id)
def test_burst_buffering(spec, cs_ctx, cs_lab_time, request):
    """Bursty traffic is buffered without drops."""
    request.node.cs_kpis = run_check(spec, cs_ctx)
