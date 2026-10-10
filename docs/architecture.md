# Architecture

```mermaid
flowchart LR
    subgraph SIM["Simulated system under test (src/cellselect/sim)"]
        direction TB
        R[radio.py<br/>path loss · CQI/BLER · HARQ · L3 filter]
        S[scheduler.py<br/>PRB tables · proportional fair]
        T[transport.py<br/>latency · buffers · memory]
        C[rrc.py<br/>RRC setup · RLM T310/N310/N311]
        M[mobility.py<br/>A3 handover · TTT · ping-pong]
        X[session.py · power.py · confparser.py · bands.py]
        F[(faults.py<br/>22 injectable bugs)]
        F -. activates .-> R & S & T & C & M & X
    end

    subgraph SUITE["Validation suite"]
        CAT[catalog.py<br/>160 tests · IDs · tags · lab durations]
        CHK[checks.py + reference.py<br/>KPI assertions vs. independent oracles]
        PYT[validation_suite/*.py<br/>pytest harness + conftest plugin]
        RUN[runner.py<br/>in-process runner + fault attribution]
    end

    subgraph DATA["Dataset (src/cellselect/data)"]
        SCN[scenarios.py<br/>change = visible context + hidden faults]
        GEN[generate.py<br/>chronological history generation]
        FEAT[features.py<br/>HistoryState: leakage-safe features]
    end

    subgraph ML["Models (src/cellselect/ml)"]
        TR[train.py<br/>LR · RF · HistGB, temporal split,<br/>forward-chaining CV]
        PRED[predictor.py<br/>p(fail) + tree uncertainty]
    end

    subgraph SEL["Selection engine (src/cellselect/selection)"]
        ENG[engine.py<br/>full · random · history · coverage · ML<br/>critical tests · fallbacks · explanations]
    end

    subgraph OUT["Benchmark & reports"]
        BEN[benchmark/run.py<br/>multi-seed replay of held-out changes]
        REP[reporting/*<br/>HTML + JSON + matplotlib]
    end

    CLI{{cli.py / orchestrate.py}}

    SIM --> CHK
    CAT --> CHK --> PYT & RUN
    SCN --> GEN
    RUN --> GEN --> FEAT
    FEAT --> TR --> PRED --> ENG
    FEAT --> ENG
    ENG --> PYT
    ENG --> BEN --> REP
    PYT --> REP
    CLI --- GEN & TR & ENG & PYT & BEN & REP
```

## Data flow for one change

1. **Change arrives** (`ChangeScenario`). The selector sees only `ChangeContext`: change type, area,
   touched components, size, lab environment. Injected faults stay hidden in `ground_truth`.
2. **Features** are computed by `HistoryState.features_for_change` from static test metadata, the
   visible change context and outcomes of *earlier* changes only.
3. **Prediction**: the persisted pipeline (`artifacts/models/<model>.joblib`) returns `p(fail)` per
   test; for random forests the per-tree standard deviation is used as an uncertainty signal.
4. **Selection** (`MLRiskCoverage`):
   1. critical tests (always, even over budget, with a warning);
   2. *safe fallback* — tests with fewer than `min_history` executions run unconditionally;
   3. *coverage guard* — every changed component gets at least one test (up to 10 % overrun);
   4. cost-aware greedy on `(p_i · residual_risk + λ · coverage_gain) / duration_i`, where residual
      risk discounts components already covered by likely-failing selected tests;
   5. priority ordering by risk density `p_i / duration_i` (fast, likely failures first).
   Scenario-level fallbacks to coverage-greedy: missing/incompatible model, inference error, change
   type/area never seen in training.
5. **Execution**: `cellselect run/compare` invokes pytest on `validation_suite/` with
   `--cs-scenario` (activates the hidden faults in the simulator) and `--cs-selection` (deselects and
   reorders). Results are written to JSON by the conftest plugin.
6. **Attribution**: a test *detects* fault *f* if it fails with *f* alone and passes with no faults
   under the identical seed/environment. Failures that also occur without faults are noise.

## Why an in-process runner *and* pytest?

History generation executes ~62 000 tests and the benchmark replays 360 held-out changes; spawning
pytest for each would add minutes of interpreter start-up. Both paths call the same
`checks.run_check` with the same `SimContext` seeds; `tests/test_integration.py` asserts that
pytest and the in-process runner produce identical outcomes on a faulty scenario, and every
`compare` run re-checks that the selected tests' pytest outcomes match the full pytest run.

## Cost model

Each test has a *nominal lab duration* (e.g. a 3 km/h handover drive ≈ 342 s, soak tests
900–3600 s). Budgets and runtime reduction use these durations because they model what a real lab
spends; the simulated checks themselves take milliseconds. Selection overhead is measured
wall-clock and added to the selected time for end-to-end efficiency. `--time-scale` optionally
sleeps `duration × scale` per test so pytest wall-clock also reflects lab time.
