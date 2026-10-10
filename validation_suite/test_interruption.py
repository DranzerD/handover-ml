"""Network interruption tests. (Simulated - see cellselect.sim; not real RF/modem testing.)"""

from __future__ import annotations

import pytest

from cellselect.suite.catalog import specs_for_template
from cellselect.suite.checks import run_check

pytestmark = pytest.mark.cellular


@pytest.mark.parametrize("spec", specs_for_template("outage_recovery"), ids=lambda s: s.test_id)
def test_outage_recovery(spec, cs_ctx, cs_lab_time, request):
    """Outage detection (RLF iff > T310) and back-off recovery."""
    request.node.cs_kpis = run_check(spec, cs_ctx)


@pytest.mark.parametrize("spec", specs_for_template("rlm_fade"), ids=lambda s: s.test_id)
def test_radio_link_monitoring(spec, cs_ctx, cs_lab_time, request):
    """RLF declared only for fades longer than T310."""
    request.node.cs_kpis = run_check(spec, cs_ctx)
