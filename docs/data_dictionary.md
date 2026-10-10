# Data dictionary

All data is **synthetic**, produced by `cellselect generate` from the behavioural simulator.
Nothing here comes from real devices, modems or RF equipment.

## `executions.csv` — one row per (change, test) execution

| Column | Type | Role | Description |
|---|---|---|---|
| `change_id` | int | key | Chronological change index (0 = oldest). Used for grouping/splitting, never as a feature. |
| `test_id` | str | key | Unique test ID, e.g. `HO-INTRA-007`. Not a feature (would not generalise to new tests). |
| `test_category` | cat | feature (static) | smoke, signal_quality, throughput, latency_loss, connection, handover, interruption, configuration, regression, stability |
| `template` | cat | feature (static) | Test template (e.g. `intra_handover`) |
| `band` | cat | feature (static) | Band under test (`b13`, `n3`, `n41`, `n78`, `na`) |
| `duration_log` | float | feature (static) | log10 of nominal lab duration (s) |
| `n_tags` | int | feature (static) | Number of component coverage tags |
| `is_critical` | 0/1 | feature (static) | Mandatory test flag |
| `change_type` | cat | feature (change) | feature, bugfix, refactor, config_update |
| `change_area` | cat | feature (change) | phy, mac, transport, rrc, mobility, session, power, config |
| `overlap_count` | int | feature (change) | Test tags ∩ changed components |
| `overlap_frac` | float | feature (change) | `overlap_count / n_tags` |
| `primary_overlap` | 0/1 | feature (change) | Test's primary component was changed |
| `neighbor_overlap` | int | feature (change) | Tags that are dependency-graph neighbours of changed components |
| `lines_changed_log` | float | feature (change) | log10 lines changed |
| `files_changed` | int | feature (change) | Files changed |
| `n_changed_components` | int | feature (change) | Number of declared changed components |
| `device` | cat | feature (env) | Device under test (devA/devB/devC: different noise figures) |
| `firmware_branch` | cat | feature (env) | main / release |
| `lab_interference_db` | float | feature (env) | Uncontrolled lab interference |
| `temperature_c` | float | feature (env) | Lab temperature (affects noise figure) |
| `hist_n_runs` | int | feature (history) | Prior executions of this test |
| `hist_fail_rate` | float | feature (history) | Beta(0.5, 9.5)-smoothed historical failure rate |
| `hist_fail_rate_last10` / `_last25` | float | feature (history) | Failure rate over the test's last 10 / 25 executions |
| `last_failed` | 0/1 | feature (history) | Previous execution failed |
| `runs_since_fail` | int | feature (history) | Executions since last failure (capped at 100) |
| `flip_rate_last20` | float | feature (history) | Pass↔fail transitions over the last 20 runs (flakiness) |
| `hist_overlap_runs` | int | feature (history) | Prior runs where the change touched this test's components |
| `hist_fail_rate_when_overlap` | float | feature (history) | Smoothed failure rate in those runs |
| `comp_recent_fail_rate` | float | feature (history) | Mean over the test's components of "any test on this component failed" in the last 10 changes |
| `category_prev_fail_frac` | float | feature (history) | Fraction of same-category tests that failed in the previous change |
| `duration_s` | float | metadata | Nominal lab duration (cost) |
| `failed` | 0/1 | **target** | Outcome of *this* execution |
| `failure_kind` | str | analysis only | `fault` (attributed to an injected fault), `noise` (fails without faults), `interaction` (only fails with a fault combination), empty if passed |
| `message` | str | analysis only | Assertion message |
| `wall_s` | float | analysis only | Measured in-process execution time |

**Leakage rule:** every history feature for change *t* is computed from changes `< t` before the
outcomes of *t* are recorded (`HistoryState.features_for_change` → run → `HistoryState.update`).
`failed`, `failure_kind`, `message`, `wall_s` are never model inputs (`FEATURE_COLUMNS` is explicit
and unit-tested).

## `changes.jsonl` — one JSON object per change

* `visible`: `change_id`, `change_type`, `area`, `changed_components`, `lines_changed`,
  `files_changed`, `env`, `seed` — what selection may use.
* `ground_truth` (hidden from selection): `new_faults`, `carried_faults` (unfixed from earlier
  changes), `active_faults`, `fault_details`, `detected_by_full_suite`, `noise_failures`.

## `fault_detection.csv`

`change_id, fault_id, test_id` — test detects the fault (fails with the fault alone, passes with no
faults, same seed and environment).

## `catalog.json` / `data/catalog/test_catalog.csv`

Test ID, template, category, coverage tags (components), coverage items (tags + `cat:` + `band:`),
nominal duration, critical flag, parameters, description.

## Model artifacts (`artifacts/models/`)

* `<model>.joblib` — fitted sklearn `Pipeline` (one-hot + scaler/passthrough + classifier), decision
  threshold, feature list.
* `model_card.json` — train/test change ranges, CV and held-out metrics for every model and the
  heuristic baselines, selected best model, seen change types/areas (used for OOD fallback).
* `test_predictions.csv` — held-out predictions for plots.
