"""Signal strength / link-quality tests. (Simulated - see cellselect.sim; not real RF/modem testing.)"""

from __future__ import annotations

import pytest

from cellselect.suite.catalog import specs_for_template
from cellselect.suite.checks import run_check

pytestmark = pytest.mark.cellular


@pytest.mark.parametrize("spec", specs_for_template("rsrp_measurement"), ids=lambda s: s.test_id)
def test_rsrp_measurement_accuracy(spec, cs_ctx, cs_lab_time, request):
    """RSRP reporting accuracy vs. link budget."""
    request.node.cs_kpis = run_check(spec, cs_ctx)


@pytest.mark.parametrize("spec", specs_for_template("cqi_link_adaptation"), ids=lambda s: s.test_id)
def test_cqi_link_adaptation(spec, cs_ctx, cs_lab_time, request):
    """CQI selection keeps BLER inside the target window."""
    request.node.cs_kpis = run_check(spec, cs_ctx)


@pytest.mark.parametrize("spec", specs_for_template("link_degradation"), ids=lambda s: s.test_id)
def test_link_degradation_recovery(spec, cs_ctx, cs_lab_time, request):
    """Throughput recovers after link degradation without RLF."""
    request.node.cs_kpis = run_check(spec, cs_ctx)
