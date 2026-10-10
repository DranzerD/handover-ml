"""Connection establishment and reconnection tests. (Simulated - see cellselect.sim; not real RF/modem testing.)"""

from __future__ import annotations

import pytest

from cellselect.suite.catalog import specs_for_template
from cellselect.suite.checks import run_check

pytestmark = pytest.mark.cellular


@pytest.mark.parametrize("spec", specs_for_template("rrc_setup"), ids=lambda s: s.test_id)
def test_rrc_connection_setup(spec, cs_ctx, cs_lab_time, request):
    """RRC setup success rate and latency."""
    request.node.cs_kpis = run_check(spec, cs_ctx)


@pytest.mark.parametrize("spec", specs_for_template("reconnection"), ids=lambda s: s.test_id)
def test_reconnection(spec, cs_ctx, cs_lab_time, request):
    """Re-establishment and session recovery after link loss."""
    request.node.cs_kpis = run_check(spec, cs_ctx)
