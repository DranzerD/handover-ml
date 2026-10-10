"""Smoke tests (critical, always selected). (Simulated - see cellselect.sim; not real RF/modem testing.)"""

from __future__ import annotations

import pytest

from cellselect.suite.catalog import specs_for_template
from cellselect.suite.checks import run_check

pytestmark = pytest.mark.cellular


@pytest.mark.parametrize("spec", specs_for_template("smoke_attach"), ids=lambda s: s.test_id)
def test_attach(spec, cs_ctx, cs_lab_time, request):
    """UE attaches at good SINR."""
    request.node.cs_kpis = run_check(spec, cs_ctx)


@pytest.mark.parametrize("spec", specs_for_template("smoke_dl"), ids=lambda s: s.test_id)
def test_basic_downlink(spec, cs_ctx, cs_lab_time, request):
    """Basic DL throughput sanity."""
    request.node.cs_kpis = run_check(spec, cs_ctx)


@pytest.mark.parametrize("spec", specs_for_template("smoke_latency"), ids=lambda s: s.test_id)
def test_basic_latency(spec, cs_ctx, cs_lab_time, request):
    """Ping latency at light load."""
    request.node.cs_kpis = run_check(spec, cs_ctx)


@pytest.mark.parametrize("spec", specs_for_template("smoke_handover"), ids=lambda s: s.test_id)
def test_basic_handover(spec, cs_ctx, cs_lab_time, request):
    """One intra-frequency handover."""
    request.node.cs_kpis = run_check(spec, cs_ctx)


@pytest.mark.parametrize("spec", specs_for_template("smoke_config"), ids=lambda s: s.test_id)
def test_default_config(spec, cs_ctx, cs_lab_time, request):
    """Default configuration parses."""
    request.node.cs_kpis = run_check(spec, cs_ctx)
