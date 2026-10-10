"""Mobility / handover drive tests. (Simulated - see cellselect.sim; not real RF/modem testing.)"""

from __future__ import annotations

import pytest

from cellselect.suite.catalog import specs_for_template
from cellselect.suite.checks import run_check

pytestmark = pytest.mark.cellular


@pytest.mark.parametrize("spec", specs_for_template("intra_handover"), ids=lambda s: s.test_id)
def test_intra_frequency_handover(spec, cs_ctx, cs_lab_time, request):
    """Intra-frequency handover: no RLF, no ping-pong, short interruption."""
    request.node.cs_kpis = run_check(spec, cs_ctx)


@pytest.mark.parametrize("spec", specs_for_template("inter_handover"), ids=lambda s: s.test_id)
def test_inter_frequency_handover(spec, cs_ctx, cs_lab_time, request):
    """Inter-frequency handover interruption time."""
    request.node.cs_kpis = run_check(spec, cs_ctx)


@pytest.mark.parametrize("spec", specs_for_template("ping_pong"), ids=lambda s: s.test_id)
def test_border_ping_pong(spec, cs_ctx, cs_lab_time, request):
    """UE loitering at the cell border does not ping-pong."""
    request.node.cs_kpis = run_check(spec, cs_ctx)
