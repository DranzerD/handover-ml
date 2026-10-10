"""Deterministic, lightweight cellular network simulator with controlled fault injection.

This is a *behavioural* model (link budget, link adaptation, HARQ, scheduling, RRC timers,
handover, reconnection, configuration parsing). It is not an RF/modem simulator and makes no
claim of 3GPP conformance; it exists so the validation suite has realistic, deterministic
cause-and-effect between injected faults and KPI-based test failures.
"""

from cellselect.sim.context import Environment, SimContext
from cellselect.sim.faults import COMPONENTS, FAULTS, Fault

__all__ = ["Environment", "SimContext", "FAULTS", "COMPONENTS", "Fault"]
