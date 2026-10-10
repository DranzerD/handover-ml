"""Regression (tight-tolerance KPI) and long-duration stability tests. (Simulated - see cellselect.sim; not real RF/modem testing.)"""

from __future__ import annotations

import pytest

from cellselect.suite.catalog import specs_for_template
from cellselect.suite.checks import run_check

pytestmark = pytest.mark.cellular


@pytest.mark.parametrize("spec", specs_for_template("kpi_regression"), ids=lambda s: s.test_id)
def test_kpi_regression(spec, cs_ctx, cs_lab_time, request):
    """Composite KPI regression with tight tolerances."""
    request.node.cs_kpis = run_check(spec, cs_ctx)


@pytest.mark.parametrize("spec", specs_for_template("stability"), ids=lambda s: s.test_id)
def test_long_duration_stability(spec, cs_ctx, cs_lab_time, request):
    """Soak test: memory growth, RLF from short fades, recovery."""
    request.node.cs_kpis = run_check(spec, cs_ctx)
