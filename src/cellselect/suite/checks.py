"""KPI checks for every test template.

Each check runs the simulator under a SimContext (faults + environment + seed), computes KPIs,
compares them against reference oracles and raises ``AssertionError`` on violation. The
returned dict holds the measured KPIs (recorded in execution results and reports).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from cellselect.sim.bands import band_info
from cellselect.sim.confparser import ConfigError, apply_config_change, dl_fraction, parse_config
from cellselect.sim.context import SimContext
from cellselect.sim.mobility import linear_path, oscillating_path, simulate_drive
from cellselect.sim.power import simulate_ul_sinr
from cellselect.sim.radio import (
    ar1_trace,
    bler,
    l3_filter,
    noise_per_re_dbm,
    pathloss_db,
    rsrp_dbm,
    select_cqi,
    simulate_link,
    spectral_efficiency,
)
from cellselect.sim.rrc import connection_setup, radio_link_monitor, reestablishment_time_ms
from cellselect.sim.scheduler import n_prb, proportional_fair
from cellselect.sim.session import simulate_outages
from cellselect.sim.transport import latency_samples_ms, memory_growth_mb, simulate_bursty_queue
from cellselect.suite import reference as ref
from cellselect.suite.catalog import TestSpec

KPIs = dict[str, float]
CheckFn = Callable[[TestSpec, SimContext], KPIs]
CHECKS: dict[str, CheckFn] = {}


def check(template: str) -> Callable[[CheckFn], CheckFn]:
    def deco(fn: CheckFn) -> CheckFn:
        CHECKS[template] = fn
        return fn

    return deco


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


# ------------------------------------------------------------------------------- helpers
def _dl_link(
    ctx: SimContext,
    band: str,
    bw: int,
    sinr: float | np.ndarray,
    n_steps: int,
    fading: str = "static",
    layers: int = 2,
    fraction: float | None = None,
):
    info = band_info(band, ctx)
    prb = n_prb(bw, info.scs_khz, info.rat, ctx)
    frac = info.dl_fraction if fraction is None else fraction
    return simulate_link(ctx, sinr, n_steps, fading, info.rat, prb, info.scs_khz, layers, frac)


def _latency_p95(ctx: SimContext, band: str, load: float, n_packets: int = 3000) -> float:
    info = band_info(band, ctx)
    link = simulate_link(ctx, 12.0, 200, "static", info.rat, 106, info.scs_khz)
    slot = 15.0 / info.scs_khz
    rtt = 8.0 if info.scs_khz == 15 else 4.0
    lat = latency_samples_ms(ctx, load, link.expected_tx, slot, rtt, n_packets)
    return float(np.percentile(lat, 95))


def _fade_trace(
    ctx: SimContext,
    total_ms: float,
    fades: list[tuple[float, float]],
    dt_ms: float = 10.0,
    base_db: float = 10.0,
    floor_db: float = -15.0,
    ramp_ms: float = 30.0,
) -> np.ndarray:
    t = np.arange(0.0, total_ms, dt_ms)
    trace = np.full_like(t, base_db)
    for start, length in fades:
        down = np.clip((t - start) / ramp_ms, 0, 1)
        up = np.clip((t - (start + length)) / ramp_ms, 0, 1)
        depth = down * (1 - up)
        trace = trace - depth * (base_db - floor_db)
    return trace + ctx.rng.normal(0.0, 1.0, size=t.shape)


# ------------------------------------------------------------------------------- smoke
@check("smoke_attach")
def check_smoke_attach(spec: TestSpec, ctx: SimContext) -> KPIs:
    res = connection_setup(ctx, spec.params["sinr_db"], 200)
    expect(res.success_rate >= 0.99, f"attach success {res.success_rate:.3f} < 0.99")
    return {"success_rate": res.success_rate, "setup_ms": res.mean_time_ms}


@check("smoke_dl")
def check_smoke_dl(spec: TestSpec, ctx: SimContext) -> KPIs:
    p = spec.params
    link = _dl_link(ctx, p["band"], p["bw"], p["sinr_db"], 300)
    ratio = link.mean_tput / ref.ref_dl_tput_mbps(p["band"], p["bw"], float(link.sinr_db.mean()))
    expect(ratio >= 0.7, f"DL throughput ratio {ratio:.3f} < 0.70")
    return {"tput_mbps": link.mean_tput, "ratio": ratio}


@check("smoke_latency")
def check_smoke_latency(spec: TestSpec, ctx: SimContext) -> KPIs:
    p95 = _latency_p95(ctx, spec.params["band"], spec.params["load"], 1000)
    expect(p95 <= 30.0, f"p95 latency {p95:.1f} ms > 30 ms")
    return {"p95_ms": p95}


@check("smoke_handover")
def check_smoke_handover(spec: TestSpec, ctx: SimContext) -> KPIs:
    p = spec.params
    res = simulate_drive(
        ctx, linear_path(120, 380, p["speed_kmh"], 0.04), 0.04, p["speed_kmh"], ttt_ms=p["ttt_ms"]
    )
    expect(len(res.handovers) >= 1, "no handover executed")
    expect(res.rlf_count == 0, f"{res.rlf_count} radio link failures")
    return {"handovers": len(res.handovers), "rlf": res.rlf_count}


@check("smoke_config")
def check_smoke_config(spec: TestSpec, ctx: SimContext) -> KPIs:
    try:
        cfg = parse_config({"band": spec.params["band"], "bandwidth_mhz": spec.params["bw"]}, ctx)
    except ConfigError as exc:
        raise AssertionError(f"default config rejected: {exc}") from exc
    return {"bandwidth_mhz": float(cfg.bandwidth_mhz)}


# ------------------------------------------------------------------------------- signal quality
@check("rsrp_measurement")
def check_rsrp(spec: TestSpec, ctx: SimContext) -> KPIs:
    band, indoor = spec.params["band"], spec.params["location"] == "indoor"
    freq, scs, rat, _, _, bw = ref.REF_BANDS[band]
    info = band_info(band, ctx)
    true = float(rsrp_dbm(46.0, ref.ref_prb(band, bw), pathloss_db(info.dl_freq_mhz, 150.0, ctx, indoor)))
    true += float(ctx.rng.normal(0.0, 1.0))  # static shadowing in the lab
    n = int(spec.duration_s / 0.04)
    std, rho = (2.5, 0.5) if indoor else (1.0, 0.5)
    samples = true + ar1_trace(ctx.rng, n, std, rho)
    reported = np.round(l3_filter(samples, ctx))
    err_mean = float(reported.mean() - ref.ref_rsrp_dbm(band, 150.0, indoor))
    err_std = float((reported - true).std())
    expect(abs(err_mean) <= 3.0, f"RSRP bias {err_mean:+.2f} dB exceeds 3 dB vs link budget")
    expect(err_std <= 1.8, f"RSRP report jitter {err_std:.2f} dB > 1.8 dB")
    return {"rsrp_dbm": float(reported.mean()), "bias_db": err_mean, "jitter_db": err_std}


@check("cqi_link_adaptation")
def check_cqi(spec: TestSpec, ctx: SimContext) -> KPIs:
    p = spec.params
    link = _dl_link(ctx, p["band"], ref.REF_BANDS[p["band"]][5], p["sinr_db"], 300)
    mean_bler = float(link.bler.mean())
    cqi_err = float(link.cqi.mean() - ref.ref_cqi(float(link.sinr_db.mean())))
    expect(mean_bler <= 0.2, f"BLER {mean_bler:.3f} above 20 % target window")
    expect(abs(cqi_err) <= 1.2, f"mean CQI deviates {cqi_err:+.2f} from reference at measured SINR")
    return {"bler": mean_bler, "cqi_error": cqi_err}


@check("link_degradation")
def check_degradation(spec: TestSpec, ctx: SimContext) -> KPIs:
    p = spec.params
    dt = 0.02
    t = np.arange(0.0, spec.duration_s, dt)
    if p["profile"] == "gradual":
        mean = np.interp(t, [0, 20, 40, 50, 70, 90], [15, 15, -3, -3, 15, 15])
    elif p["profile"] == "step":
        mean = np.where((t >= 20) & (t < 40), 2.0, 15.0)
    else:
        mean = np.full_like(t, 15.0)
        for start in (20.0, 45.0, 70.0):
            mean = np.where((t >= start) & (t < start + 0.6), -14.0, mean)
    link = _dl_link(ctx, p["band"], ref.REF_BANDS[p["band"]][5], mean, len(t), fading="pedestrian")
    rlf = radio_link_monitor(ctx, link.sinr_db, dt * 1000.0)
    tail = link.tput_mbps[t >= spec.duration_s - 15]
    ratio = float(tail.mean() / ref.ref_dl_tput_mbps(p["band"], ref.REF_BANDS[p["band"]][5], 15.0))
    loss = float(link.residual_loss.mean())
    expect(len(rlf) == 0, f"{len(rlf)} RLF during recoverable degradation")
    expect(ratio >= 0.75, f"post-recovery throughput ratio {ratio:.3f} < 0.75")
    expect(loss <= 0.01, f"residual loss {loss:.4f} > 1 %")
    return {"rlf": float(len(rlf)), "recovery_ratio": ratio, "residual_loss": loss}


# ------------------------------------------------------------------------------- throughput
@check("dl_throughput")
def check_dl_tput(spec: TestSpec, ctx: SimContext) -> KPIs:
    p = spec.params
    link = _dl_link(ctx, p["band"], p["bw"], p["sinr_db"], int(spec.duration_s * 10))
    expected = ref.ref_dl_tput_mbps(p["band"], p["bw"], float(link.sinr_db.mean()))
    ratio = link.mean_tput / expected
    expect(
        ratio >= 0.85, f"DL throughput {link.mean_tput:.1f} Mbps is {ratio:.0%} of reference {expected:.1f}"
    )
    return {"tput_mbps": link.mean_tput, "ratio": ratio}


@check("ul_throughput")
def check_ul_tput(spec: TestSpec, ctx: SimContext) -> KPIs:
    p = spec.params
    info = band_info(p["band"], ctx)
    pl = float(pathloss_db(info.dl_freq_mhz, p["distance_m"], ctx))
    m = 50
    sinr = simulate_ul_sinr(ctx, pl, m, info.scs_khz, p["pl_error_db"], int(spec.duration_s * 10))
    est = np.concatenate(([sinr[0]], sinr[:-1]))
    cqi = select_cqi(est, ctx)
    b = bler(sinr, cqi, ctx, info.rat)
    tput = (
        m
        * 12
        * 14
        * 1000.0
        * (info.scs_khz / 15)
        * 0.75
        * spectral_efficiency(cqi, info.rat)
        * info.ul_fraction
        * (1 - b)
        / 1e6
    )
    _, scs, rat, _, ul, _ = ref.REF_BANDS[p["band"]]
    ref_sinr = ref.ref_ul_sinr_db(ref.ref_pathloss_db(info.dl_freq_mhz, p["distance_m"]), m, scs)
    expected = ref.ref_rate_mbps(m, scs, ref.ref_se(ref_sinr, rat), 1, ul)
    ratio = float(tput.mean() / expected)
    expect(ratio >= 0.8, f"UL throughput ratio {ratio:.3f} < 0.80 (ref SINR {ref_sinr:.1f} dB)")
    return {"ul_tput_mbps": float(tput.mean()), "ratio": ratio, "ul_sinr_db": float(sinr[50:].mean())}


@check("multi_user")
def check_multi_user(spec: TestSpec, ctx: SimContext) -> KPIs:
    p = spec.params
    users = p["users"]
    sinrs = np.full(users, 15.0) if p["traffic"] == "equal" else np.linspace(5.0, 25.0, users)
    info = band_info(p["band"], ctx)
    prb = n_prb(40, info.scs_khz, info.rat, ctx)
    served = proportional_fair(ctx, sinrs, 600, prb, info.scs_khz, info.rat)
    ref_rates = np.array([ref.ref_rate_mbps(106, 30, ref.ref_se(s, "NR"), 1, 1.0) for s in sinrs])
    norm = served / ref_rates * users
    fairness = ref.jain(norm)
    expect(fairness >= 0.8, f"normalised Jain fairness {fairness:.3f} < 0.80")
    expect(norm.min() >= 0.3, f"starved UE: min normalised share {norm.min():.3f} < 0.30")
    return {"jain": fairness, "min_share": float(norm.min())}


# ------------------------------------------------------------------------------- latency / loss
@check("latency_load")
def check_latency(spec: TestSpec, ctx: SimContext) -> KPIs:
    p = spec.params
    p95 = _latency_p95(ctx, p["band"], p["load"])
    limit = ref.ref_latency_p95_ms(p["load"], ref.REF_BANDS[p["band"]][1])
    expect(p95 <= limit, f"p95 latency {p95:.1f} ms exceeds {limit:.1f} ms at load {p['load']}")
    return {"p95_ms": p95, "limit_ms": limit}


@check("packet_loss")
def check_packet_loss(spec: TestSpec, ctx: SimContext) -> KPIs:
    p = spec.params
    fading = "vehicular" if p["mobility"] == "vehicular" else "static"
    link = _dl_link(ctx, p["band"], 40, 15.0 - p["interference_db"], int(spec.duration_s * 10), fading=fading)
    loss = float(link.residual_loss.mean())
    expect(loss <= 5e-3, f"residual packet loss {loss:.2e} > 0.5 %")
    return {"residual_loss": loss}


@check("burst_buffer")
def check_burst(spec: TestSpec, ctx: SimContext) -> KPIs:
    p = spec.params
    arrived, dropped = simulate_bursty_queue(
        ctx, int(spec.duration_s * 10), p["burst_packets"], p["period_steps"], 50.0
    )
    rate = dropped / max(arrived, 1)
    expect(rate <= 0.005, f"buffer drop rate {rate:.3%} > 0.5 %")
    return {"drop_rate": rate}


# ------------------------------------------------------------------------------- connection
@check("rrc_setup")
def check_rrc_setup(spec: TestSpec, ctx: SimContext) -> KPIs:
    res = connection_setup(ctx, spec.params["sinr_db"], 400)
    expect(res.success_rate >= 0.97, f"setup success {res.success_rate:.3f} < 0.97")
    expect(res.mean_time_ms <= 120.0, f"mean setup time {res.mean_time_ms:.1f} ms > 120 ms")
    return {"success_rate": res.success_rate, "setup_ms": res.mean_time_ms}


@check("reconnection")
def check_reconnection(spec: TestSpec, ctx: SimContext) -> KPIs:
    p = spec.params
    times = [reestablishment_time_ms(ctx, p["cause"] == "rlf", 10.0) for _ in range(p["cycles"])]
    out = simulate_outages(ctx, 0.3, p["cycles"])
    limit = ref.ref_backoff_recovery_s(0.3) + 0.5
    expect(max(times) <= 400.0, f"re-establishment took {max(times):.0f} ms > 400 ms")
    expect(out.max_recovery_s <= limit, f"session recovery {out.max_recovery_s:.2f} s > {limit:.2f} s")
    return {"max_reest_ms": max(times), "max_recovery_s": out.max_recovery_s}


# ------------------------------------------------------------------------------- handover
@check("intra_handover")
def check_intra_ho(spec: TestSpec, ctx: SimContext) -> KPIs:
    p = spec.params
    info = band_info(p["band"], ctx)
    prb = ref.ref_prb(p["band"], ref.REF_BANDS[p["band"]][5])
    res = simulate_drive(
        ctx,
        linear_path(120, 380, p["speed_kmh"], 0.04),
        0.04,
        p["speed_kmh"],
        freq_mhz=(info.dl_freq_mhz, info.dl_freq_mhz),
        n_prb=(prb, prb),
        scs_khz=info.scs_khz,
        ttt_ms=p["ttt_ms"],
    )
    expect(len(res.handovers) >= 1, "no handover executed")
    expect(res.rlf_count == 0, f"{res.rlf_count} radio link failure(s) during handover drive")
    expect(res.ping_pongs <= 1, f"{res.ping_pongs} ping-pong handovers")
    expect(res.max_interruption_ms <= 60.0, f"interruption {res.max_interruption_ms:.0f} ms > 60 ms")
    return {
        "handovers": float(len(res.handovers)),
        "rlf": float(res.rlf_count),
        "ping_pong": float(res.ping_pongs),
        "max_interruption_ms": res.max_interruption_ms,
    }


@check("inter_handover")
def check_inter_ho(spec: TestSpec, ctx: SimContext) -> KPIs:
    p = spec.params
    a, b = band_info(p["band"], ctx), band_info(p["target_band"], ctx)
    prb = (ref.ref_prb(a.name, ref.REF_BANDS[a.name][5]), ref.ref_prb(b.name, ref.REF_BANDS[b.name][5]))
    res = simulate_drive(
        ctx,
        linear_path(50, 450, p["speed_kmh"], 0.04),
        0.04,
        p["speed_kmh"],
        freq_mhz=(a.dl_freq_mhz, b.dl_freq_mhz),
        n_prb=prb,
        scs_khz=b.scs_khz,
        ttt_ms=160.0,
    )
    expect(len(res.handovers) >= 1, "no inter-frequency handover executed")
    expect(res.rlf_count == 0, f"{res.rlf_count} radio link failure(s)")
    expect(res.max_interruption_ms <= 60.0, f"interruption {res.max_interruption_ms:.0f} ms > 60 ms")
    return {"handovers": float(len(res.handovers)), "max_interruption_ms": res.max_interruption_ms}


@check("ping_pong")
def check_ping_pong(spec: TestSpec, ctx: SimContext) -> KPIs:
    p = spec.params
    std = 3.0 if p["fading"] == "high" else 1.0
    path = oscillating_path(250.0, 25.0, 3.0, spec.duration_s, 0.04)
    res = simulate_drive(
        ctx,
        path,
        0.04,
        3.0,
        fading_std_db=std,
        a3_offset_db=2.0,
        hysteresis_db=p["hysteresis_db"],
        ttt_ms=160.0,
    )
    expect(res.ping_pongs <= 2, f"{res.ping_pongs} ping-pong handovers (limit 2)")
    expect(res.rlf_count == 0, f"{res.rlf_count} radio link failure(s)")
    return {"handovers": float(len(res.handovers)), "ping_pong": float(res.ping_pongs)}


# ------------------------------------------------------------------------------- interruptions
@check("outage_recovery")
def check_outage(spec: TestSpec, ctx: SimContext) -> KPIs:
    p = spec.params
    outage_ms = p["outage_s"] * 1000.0
    trace = _fade_trace(ctx, outage_ms + 5000.0, [(2000.0, outage_ms)], floor_db=-20.0)
    rlf = radio_link_monitor(ctx, trace, 10.0)
    expected_rlf = outage_ms > 1100.0
    out = simulate_outages(ctx, p["outage_s"], p["count"])
    limit = ref.ref_backoff_recovery_s(p["outage_s"]) + 0.6
    expect(
        bool(rlf) == expected_rlf,
        f"RLF declared={bool(rlf)} but expected={expected_rlf} for {outage_ms:.0f} ms outage",
    )
    expect(out.max_recovery_s <= limit, f"recovery {out.max_recovery_s:.2f} s > {limit:.2f} s")
    return {"rlf": float(len(rlf)), "max_recovery_s": out.max_recovery_s}


@check("rlm_fade")
def check_rlm(spec: TestSpec, ctx: SimContext) -> KPIs:
    fade = float(spec.params["fade_ms"])
    trace = _fade_trace(ctx, fade + 4000.0, [(1000.0, fade)])
    rlf = radio_link_monitor(ctx, trace, 10.0)
    if fade > 1100.0:
        expect(len(rlf) >= 1, f"no RLF for {fade:.0f} ms fade")
        first_ms = rlf[0] * 10.0 - 1000.0
        expect(
            950.0 <= first_ms <= 1200.0, f"RLF declared {first_ms:.0f} ms after fade start (expected ~T310)"
        )
    else:
        expect(len(rlf) == 0, f"spurious RLF for {fade:.0f} ms fade (< T310)")
    return {"rlf": float(len(rlf))}


# ------------------------------------------------------------------------------- configuration
def _cfg_tput(ctx: SimContext, cfg) -> float:
    info = band_info(cfg.band, ctx)
    prb = n_prb(cfg.bandwidth_mhz, info.scs_khz, info.rat, ctx)
    link = simulate_link(ctx, 20.0, 200, "static", info.rat, prb, info.scs_khz, 2, dl_fraction(cfg, ctx))
    return link.mean_tput, float(link.sinr_db.mean())


@check("config_apply")
def check_config_apply(spec: TestSpec, ctx: SimContext) -> KPIs:
    p = spec.params
    try:
        cur = parse_config({"band": p["band"], "bandwidth_mhz": p["from"]["bw"]}, ctx)
    except ConfigError as exc:
        raise AssertionError(f"baseline config rejected: {exc}") from exc
    try:
        new = apply_config_change(cur, p["change"], ctx)
        accepted = True
    except ConfigError:
        new, accepted = None, False
    expect(
        accepted == (p["expect"] == "accept"),
        f"change {p['change']} accepted={accepted}, expected {p['expect']}",
    )
    kpis: KPIs = {"accepted": float(accepted)}
    if not accepted or new is None:
        return kpis
    ch = p["change"]
    if "bandwidth_mhz" in ch or "tdd_pattern" in ch or "band" in ch:
        (before, s_before), (after, s_after) = _cfg_tput(ctx, cur), _cfg_tput(ctx, new)
        exp_after = ref.ref_dl_tput_mbps(new.band, new.bandwidth_mhz, s_after, tdd_pattern=new.tdd_pattern)
        exp_before = ref.ref_dl_tput_mbps(cur.band, cur.bandwidth_mhz, s_before, tdd_pattern=cur.tdd_pattern)
        ratio_err = (after / exp_after) / (before / exp_before) - 1.0
        kpis.update(tput_after_mbps=after, ratio_error=ratio_err)
        expect(abs(ratio_err) <= 0.15, f"throughput scaling off by {ratio_err:+.1%} after change")
        expect(after >= 0.8 * exp_after, f"throughput {after:.0f} Mbps < 80 % of reference {exp_after:.0f}")
    if "band" in ch:
        info = band_info(new.band, ctx)
        rsrp = float(
            rsrp_dbm(
                46.0, ref.ref_prb(new.band, new.bandwidth_mhz), pathloss_db(info.dl_freq_mhz, 200.0, ctx)
            )
        )
        exp = (
            46.0
            - 10 * np.log10(12 * ref.ref_prb(new.band, new.bandwidth_mhz))
            + 17.0
            - ref.ref_pathloss_db(ref.REF_BANDS[new.band][0], 200.0)
        )
        expect(abs(rsrp - exp) <= 2.0, f"RSRP after band change {rsrp:.1f} dBm vs expected {exp:.1f}")
    if "mtu" in ch:
        packet = ch["mtu"] - 100
        delivered = packet <= new.effective_mtu
        kpis["effective_mtu"] = float(new.effective_mtu)
        expect(delivered, f"{packet}-byte packets dropped: effective MTU {new.effective_mtu}")
    return kpis


@check("config_validation")
def check_config_validation(spec: TestSpec, ctx: SimContext) -> KPIs:
    try:
        parse_config(dict(spec.params["raw"]), ctx)
        accepted = True
    except ConfigError:
        accepted = False
    expect(
        accepted == (spec.params["expect"] == "accept"),
        f"config {spec.params['raw']} accepted={accepted}, expected {spec.params['expect']}",
    )
    return {"accepted": float(accepted)}


# ------------------------------------------------------------------------------- regression
_PROFILES = {
    # sinr, fading, load, speed (None = no drive), setup sinr
    "urban": (13.0, "pedestrian", 0.75, 30.0, 3.0),
    "suburban": (18.0, "pedestrian", 0.6, 60.0, 6.0),
    "highway": (10.0, "vehicular", 0.5, 120.0, 5.0),
    "indoor": (None, "pedestrian", 0.75, None, 3.0),
}


@check("kpi_regression")
def check_regression(spec: TestSpec, ctx: SimContext) -> KPIs:
    p = spec.params
    sinr, fading, load, speed, setup_sinr = _PROFILES[p["profile"]]
    info = band_info(p["band"], ctx)
    kpis: KPIs = {}
    if sinr is None:  # indoor: SINR derived from the indoor link budget at 60 m
        noise = noise_per_re_dbm(info.scs_khz, 7.0)
        rsrp = float(rsrp_dbm(46.0, 273, pathloss_db(info.dl_freq_mhz, 60.0, ctx, indoor=True)))
        sinr_used = min(rsrp - noise, 30.0) - 12.0  # 12 dB indoor interference rise
        ref_sinr = min(ref.ref_rsrp_dbm(p["band"], 60.0, indoor=True) - noise, 30.0) - 12.0
    else:
        sinr_used = ref_sinr = sinr
    link = _dl_link(ctx, p["band"], 100, sinr_used, 1200, fading=fading)
    ratio = link.mean_tput / ref.ref_dl_tput_mbps(
        p["band"], 100, float(link.sinr_db.mean()) + (ref_sinr - sinr_used)
    )
    kpis["tput_ratio"] = ratio
    expect(ratio >= 0.75, f"[{p['profile']}] DL throughput ratio {ratio:.3f} < 0.75")
    p95 = _latency_p95(ctx, p["band"], load)
    limit = 0.9 * ref.ref_latency_p95_ms(load, info.scs_khz)
    kpis["p95_ms"] = p95
    expect(p95 <= limit, f"[{p['profile']}] p95 latency {p95:.1f} > {limit:.1f} ms")
    setup = connection_setup(ctx, setup_sinr, 300)
    kpis["setup_success"] = setup.success_rate
    expect(setup.success_rate >= 0.99, f"[{p['profile']}] setup success {setup.success_rate:.3f} < 0.99")
    if speed is not None:
        res = simulate_drive(ctx, linear_path(180, 320, speed, 0.04), 0.04, speed, fading_std_db=2.5)
        kpis.update(rlf=float(res.rlf_count), ping_pong=float(res.ping_pongs))
        expect(
            res.rlf_count == 0 and res.ping_pongs <= 1,
            f"[{p['profile']}] mobility KPIs: rlf={res.rlf_count} ping_pong={res.ping_pongs}",
        )
    return kpis


# ------------------------------------------------------------------------------- stability
@check("stability")
def check_stability(spec: TestSpec, ctx: SimContext) -> KPIs:
    p = spec.params
    soak = float(p["soak_s"])
    bursty = p["traffic"] == "bursty"
    growth = memory_growth_mb(ctx, soak, 600.0 if bursty else 120.0)
    expect(growth <= 2.5, f"memory grew {growth:.1f} MB over {soak / 60:.0f} min soak (limit 2.5 MB)")
    n_fades = int(soak // 300)
    rlf_total = 0
    for _ in range(n_fades):
        length = float(ctx.rng.uniform(150.0, 400.0))
        rlf_total += len(
            radio_link_monitor(
                ctx, _fade_trace(ctx, length + 1500.0, [(500.0, length)], floor_db=-14.0), 10.0
            )
        )
    expect(rlf_total == 0, f"{rlf_total} RLF(s) from short fades during soak")
    count = int(soak // 600) if bursty else 1
    out = simulate_outages(ctx, 0.5, count)
    limit = ref.ref_backoff_recovery_s(0.5) + 0.6
    expect(out.max_recovery_s <= limit, f"recovery after outage {out.max_recovery_s:.2f} s > {limit:.2f} s")
    setup = connection_setup(ctx, 8.0, 200)
    expect(setup.success_rate >= 0.99, f"re-attach success {setup.success_rate:.3f}")
    return {"memory_growth_mb": growth, "rlf": float(rlf_total), "max_recovery_s": out.max_recovery_s}


def run_check(spec: TestSpec, ctx: SimContext) -> KPIs:
    """Execute the KPI check for ``spec`` (raises AssertionError on failure)."""
    try:
        fn = CHECKS[spec.template]
    except KeyError as exc:
        raise KeyError(f"no check registered for template {spec.template!r}") from exc
    return fn(spec, ctx)
