"""Configuration-change tests. (Simulated - see cellselect.sim; not real RF/modem testing.)"""

from __future__ import annotations

import pytest

from cellselect.suite.catalog import specs_for_template
from cellselect.suite.checks import run_check

pytestmark = pytest.mark.cellular


@pytest.mark.parametrize("spec", specs_for_template("config_apply"), ids=lambda s: s.test_id)
def test_apply_config_change(spec, cs_ctx, cs_lab_time, request):
    """Configuration changes are validated and take effect."""
    request.node.cs_kpis = run_check(spec, cs_ctx)


@pytest.mark.parametrize("spec", specs_for_template("config_validation"), ids=lambda s: s.test_id)
def test_config_validation(spec, cs_ctx, cs_lab_time, request):
    """Configuration schema validation accepts/rejects correctly."""
    request.node.cs_kpis = run_check(spec, cs_ctx)
