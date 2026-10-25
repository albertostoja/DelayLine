import pickle
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.optimize import least_squares

import Simulation as S

try:
    from hardware_ops import HardwareOps
except Exception:
    HardwareOps = None


DEFAULT_LINEAR_STAGE_SERIALS = {
    "M1": "27266900",
    "M2": "27266901",
}
DEFAULT_MOVING_LINEAR_STAGES = ("M1", "M2")
DEFAULT_LINEAR_STAGE_GUARD_MM = getattr(S, "DEFAULT_LINEAR_STAGE_GUARD_MM", 0.10)
DEFAULT_LINEAR_STAGE_REPLAN_TOLERANCE_MM = 1e-3

DEFAULT_ROTATION_CONTROLLER = "newport"
DEFAULT_ROTATION_DEGREES_PER_SUBSTEP = 0.0023
# Approximate physical response for N_R dry runs; hardware command estimates
# remain separate. Real staging takes about three times as many steps.
DEFAULT_REFLECTION_DRY_RUN_DEGREES_PER_SUBSTEP = 0.0007

DEFAULT_ACTUATOR_MAP = {
    "M1.dx": {
        "kind": "linear",
        "mirror": "M1",
        "serial": DEFAULT_LINEAR_STAGE_SERIALS["M1"],
        "direction": 1.0,
    },
    "M2.dx": {
        "kind": "linear",
        "mirror": "M2",
        "serial": DEFAULT_LINEAR_STAGE_SERIALS["M2"],
        "direction": -1.0,
    },
    "M1.dangle": {
        "kind": "rotation",
        "mirror": "M1",
        "controller": DEFAULT_ROTATION_CONTROLLER,
        "actuator": 1,
        "direction": 1,
    },
    "M2.dangle": {
        "kind": "rotation",
        "mirror": "M2",
        "controller": DEFAULT_ROTATION_CONTROLLER,
        "actuator": 3,
        "direction": 1,
    },
    "M3.dangle": {
        "kind": "rotation",
        "mirror": "M3",
        "controller": DEFAULT_ROTATION_CONTROLLER,
        "actuator": 5,
        "direction": 1,
    },
    "M4.dangle": {
        "kind": "rotation",
        "mirror": "M4",
        "controller": DEFAULT_ROTATION_CONTROLLER,
        "actuator": 7,
        "direction": 1,
    },
}

DEFAULT_ROTATION_CALIBRATION = {
    "M1.dangle": DEFAULT_ROTATION_DEGREES_PER_SUBSTEP,
    "M2.dangle": DEFAULT_ROTATION_DEGREES_PER_SUBSTEP,
    "M3.dangle": DEFAULT_ROTATION_DEGREES_PER_SUBSTEP,
    "M4.dangle": DEFAULT_ROTATION_DEGREES_PER_SUBSTEP,
}


def _merged_actuator_map(actuator_map):
    merged = {
        label: dict(config)
        for label, config in DEFAULT_ACTUATOR_MAP.items()
    }
    if actuator_map:
        for label, config in actuator_map.items():
            base = dict(merged.get(label, {}))
            base.update(config)
            merged[label] = base
    return merged


def _linear_stage_order_from_actuator_map(actuator_map):
    ordered = []
    for mirror_name in ("M1", "M2", "M3"):
        mapping = actuator_map.get(f"{mirror_name}.dx")
        if mapping is None or mapping.get("kind") != "linear":
            continue
        mapped_mirror = mapping.get("mirror", mirror_name)
        if mapped_mirror not in ordered:
            ordered.append(mapped_mirror)
    return tuple(ordered) if ordered else DEFAULT_MOVING_LINEAR_STAGES


def _normalized_rotation_calibration(rotation_calibration):
    normalized = dict(DEFAULT_ROTATION_CALIBRATION)
    if rotation_calibration:
        for label, value in rotation_calibration.items():
            if isinstance(value, dict):
                value = value.get("degrees_per_substep", DEFAULT_ROTATION_DEGREES_PER_SUBSTEP)
            normalized[label] = float(value)
    return normalized


def _x_to_mirrors(x, base_mirrors):
    return S.unpack_variables(x, *base_mirrors)


def _mirrors_to_lists(mirrors):
    return [np.array(mirror, dtype=float).tolist() for mirror in mirrors]


def _quadcell_readout_from_sim_qc(sim_qc, qc_readout_sign):
    if qc_readout_sign == 0:
        raise ValueError("qc_readout_sign must be nonzero.")
    return np.asarray(sim_qc, dtype=float) / float(qc_readout_sign)


def _sim_qc_from_quadcell_readout(qc_readout, qc_readout_sign):
    return float(qc_readout_sign) * np.asarray(qc_readout, dtype=float)


def _planned_step_qc_readout(step, qc_readout_sign):
    sim_qc = np.array([step["qc1_error"], step["qc2_error"]], dtype=float)
    return _quadcell_readout_from_sim_qc(sim_qc, qc_readout_sign)


def _read_quadcell_y(
        hardware,
        *,
        times,
        delay,
        dry_run,
        dry_run_x,
        base_mirrors,
        qc_readout_sign):
    """Read the hardware QC coordinate that corresponds to simulated QC error.

    HardwareOps.quads.get_xy_position() returns [QC1_x, QC1_y, QC2_x, QC2_y].
    The OPD actuator feedback should use QC x, i.e. indices 0 and 2.  The
    returned "y" key is kept as a backwards-compatible alias for older executor
    code/logging that used that name.
    """
    if dry_run:
        sim_qc = np.array(
            S.quadcell_errors_from_variables(dry_run_x, *base_mirrors),
            dtype=float
        )
        x = _quadcell_readout_from_sim_qc(sim_qc, qc_readout_sign)
        return {
            "raw": [float(x[0]), np.nan, float(x[1]), np.nan],
            "x": x,
            "y": x,
            "axis": "x",
        }

    if hardware is None or not hasattr(hardware, "quads"):
        raise ValueError("A HardwareOps-like object with .quads is required unless dry_run=True.")

    # For live actuator feedback, use the freshest quadcell readout. The
    # times/delay arguments are kept for API compatibility and dry-run parity.
    raw = hardware.quads.get_xy_position()
    if len(raw) < 4:
        raise ValueError(f"Expected four quadcell readout values, got {raw}.")

    x = np.array([raw[0], raw[2]], dtype=float)
    return {
        "raw": list(raw),
        "x": x,
        "y": x,
        "axis": "x",
    }


def _normalize_qc_min_signals(qc_min_signal=0.04, qc_min_signals=None):
    if qc_min_signals is None:
        return np.array([float(qc_min_signal), float(qc_min_signal)], dtype=float)
    values = np.array(qc_min_signals, dtype=float)
    if values.size != 2:
        raise ValueError("qc_min_signals must contain exactly two thresholds.")
    return values


def _read_quadcell_x_with_signal(
        hardware,
        *,
        dry_run,
        dry_run_x,
        base_mirrors,
        qc_readout_sign,
        qc_min_signal=0.04,
        qc_min_signals=None,
        dry_run_signal_strengths=(1.0, 1.0)):
    thresholds = _normalize_qc_min_signals(qc_min_signal, qc_min_signals)

    if dry_run:
        sim_qc = np.array(
            S.quadcell_errors_from_variables(dry_run_x, *base_mirrors),
            dtype=float
        )
        x = _quadcell_readout_from_sim_qc(sim_qc, qc_readout_sign)
        signals = np.array(dry_run_signal_strengths, dtype=float)
        if signals.size == 1:
            signals = np.repeat(signals, 2)
        valid_mask = signals >= thresholds
        return {
            "raw": [float(x[0]), np.nan, float(x[1]), np.nan],
            "x": x,
            "y": x,
            "axis": "x",
            "signal_strengths": signals.tolist(),
            "signal_thresholds": thresholds.tolist(),
            "valid_mask": valid_mask.tolist(),
            "valid": bool(np.all(valid_mask) and np.all(np.isfinite(x))),
            "error": None,
        }

    if hardware is None or not hasattr(hardware, "quads"):
        raise ValueError("A HardwareOps-like object with .quads is required unless dry_run=True.")

    serial_numbers = list(hardware.quads.serial_numbers)
    if len(serial_numbers) != 2:
        raise ValueError(f"Expected two quadcells, got {serial_numbers}.")

    signals = []
    normalized_x = []
    values = []
    raw = []
    read_error = None
    for sn in serial_numbers:
        try:
            if hasattr(hardware.quads, "get_status"):
                status_full = hardware.quads.get_status(sn)
            else:
                status_full = hardware.quads.controllers[sn].Status
            status = status_full.PositionDifference
            signal_sum = float(status_full.Sum)
            x_mm, y_mm = hardware.quads.raw_to_mm(status.X, status.Y, signal_sum)
            signals.append(signal_sum)
            normalized_x.append(float(status.X) / signal_sum if signal_sum > 0 else np.nan)
            values.extend([float(x_mm), float(y_mm)])
            raw.extend([status.X, status.Y])
        except Exception as exc:
            read_error = str(exc)
            signals.append(np.nan)
            normalized_x.append(np.nan)
            values.extend([np.nan, np.nan])
            raw.extend([np.nan, np.nan])

    x = np.array([values[0], values[2]], dtype=float)
    signals = np.array(signals, dtype=float)
    valid_mask = np.isfinite(signals) & (signals >= thresholds)
    # A ratio above one alone does not establish controller saturation. Keep
    # the legacy metadata fields, but do not impose the removed software clip.
    saturated = np.zeros(2, dtype=bool)
    return {
        "raw": raw,
        "x": x,
        "y": x,
        "axis": "x",
        "signal_strengths": signals.tolist(),
        "signal_thresholds": thresholds.tolist(),
        "valid_mask": valid_mask.tolist(),
        "signal_valid": bool(np.all(valid_mask)),
        "normalized_x": list(normalized_x),
        "position_saturated_mask": saturated.tolist(),
        "position_saturated": bool(np.any(saturated)),
        "valid": bool(np.all(valid_mask) and np.all(np.isfinite(x)) and not np.any(saturated)),
        "error": read_error,
    }


def _qc_model_comparison(x_estimate, base_mirrors, qc, qc_readout_sign):
    predicted = _quadcell_readout_from_sim_qc(
        S.quadcell_errors_from_variables(x_estimate, *base_mirrors), qc_readout_sign
    )
    residual = np.asarray(qc["x"], dtype=float) - predicted
    return {
        "measured_qc_x": np.asarray(qc["x"], dtype=float).tolist(),
        "predicted_qc_x": predicted.tolist(),
        "qc_residual_x": residual.tolist(),
        "qc_residual_norm": float(np.linalg.norm(residual)),
        "qc_position_valid": bool(qc["valid"]),
        "qc_position_saturated": bool(qc.get("position_saturated", False)),
    }


def _qc_invalid_details(qc):
    issues = []
    signals = qc.get("signal_strengths", [])
    thresholds = qc.get("signal_thresholds", [np.nan] * len(signals))
    for index, (signal, threshold) in enumerate(zip(signals, thresholds), start=1):
        if not np.isfinite(signal):
            issues.append(f"QC{index} signal is nonfinite")
        elif np.isfinite(threshold) and signal < threshold:
            issues.append(f"QC{index} signal={signal:.6g} < minimum={threshold:.6g}")
    for index, position in enumerate(qc.get("x", []), start=1):
        if not np.isfinite(position):
            issues.append(f"QC{index} position is nonfinite")
    if qc.get("error"):
        issues.append(f"read error: {qc['error']}")
    return "; ".join(issues) or "QC signal/position is invalid"


def _staging_qc_failure(qc, qc_limit):
    if qc.get("position_saturated", False):
        return "stage_qc_saturated: QC position is clipped; the spot position is unknown."
    if not qc["valid"]:
        return "stage_qc_invalid: " + _qc_invalid_details(qc) + " during same-N_R staging."
    max_abs = float(np.max(np.abs(qc["x"])))
    if max_abs > float(qc_limit):
        return f"stage_qc_limit: QC max abs {max_abs:.6g} exceeds {qc_limit:g} during staging."
    return None


def _initial_linear_stage_locs(
        hardware,
        actuator_map,
        *,
        M1_linear_loc=None,
        M2_linear_loc=None,
        M3_linear_loc=None,
        dry_run=False,
        prefer_hardware_positions=False):
    provided = {
        "M1": M1_linear_loc,
        "M2": M2_linear_loc,
        "M3": M3_linear_loc,
    }
    locs = {}
    midpoint = getattr(S, "LINEAR_STAGE_TRAVEL_MM", 24.0) / 2.0
    active_linear_stages = list(_linear_stage_order_from_actuator_map(actuator_map))
    for mirror_name, provided_loc in provided.items():
        if provided_loc is not None and mirror_name not in active_linear_stages:
            active_linear_stages.append(mirror_name)

    for mirror_name in active_linear_stages:
        provided_loc = provided.get(mirror_name)
        label = f"{mirror_name}.dx"
        mapping = actuator_map.get(label)
        if (
            prefer_hardware_positions and
            not dry_run and
            hardware is not None and
            getattr(hardware, "stages", None) is not None and
            mapping is not None
        ):
            locs[mirror_name] = float(hardware.stages.get_position(mapping["serial"]))
        elif provided_loc is not None:
            locs[mirror_name] = float(provided_loc)
            continue
        elif (
            not dry_run and
            hardware is not None and
            getattr(hardware, "stages", None) is not None and
            mapping is not None
        ):
            locs[mirror_name] = float(hardware.stages.get_position(mapping["serial"]))
        else:
            locs[mirror_name] = float(midpoint)

    return locs


def _linear_stage_allowed_position_range(position_mm, linear_stage_guard_mm=DEFAULT_LINEAR_STAGE_GUARD_MM):
    if hasattr(S, "_linear_stage_allowed_loc_range"):
        return S._linear_stage_allowed_loc_range(
            position_mm,
            linear_stage_guard_mm=linear_stage_guard_mm
        )

    travel_mm = float(getattr(S, "LINEAR_STAGE_TRAVEL_MM", 24.0))
    guard = float(linear_stage_guard_mm)
    if guard < 0:
        raise ValueError("linear_stage_guard_mm must be non-negative.")
    if guard * 2 >= travel_mm:
        raise ValueError(
            f"linear_stage_guard_mm={guard} leaves no usable travel for {travel_mm} mm stages."
        )
    position_mm = float(position_mm)
    if position_mm < -1e-9 or position_mm > travel_mm + 1e-9:
        raise ValueError(f"linear stage position must be between 0 and {travel_mm} mm.")
    position_mm = float(np.clip(position_mm, 0.0, travel_mm))
    return min(guard, position_mm), max(travel_mm - guard, position_mm)


def _clip_linear_hardware_delta_to_range(delta_mm, before_position_mm, loc_min, loc_max):
    delta_mm = float(delta_mm)
    before_position_mm = float(before_position_mm)
    target_position_mm = before_position_mm + delta_mm
    clipped_target_position_mm = float(np.clip(target_position_mm, loc_min, loc_max))
    clipped_delta_mm = clipped_target_position_mm - before_position_mm
    return clipped_delta_mm, target_position_mm, clipped_target_position_mm


def assimilate_rotation_angle_from_qc(
        x_current,
        axis_index,
        measured_qc_y,
        M1,
        M2,
        M3,
        M4,
        *,
        qc_readout_sign=-1.0,
        angle_prior=None,
        angle_window=None,
        qc_fit_scale=0.1,
        prior_sigma=None):
    """Fit one simulated rotation angle to measured quadcell Y readouts."""
    x_current = np.array(x_current, dtype=float)
    measured_qc_y = np.array(measured_qc_y, dtype=float)
    target_sim_qc = _sim_qc_from_quadcell_readout(measured_qc_y, qc_readout_sign)

    if axis_index is None:
        raise ValueError("axis_index is required for rotation assimilation.")

    if angle_prior is None:
        angle_prior = x_current[axis_index]
    angle_prior = float(angle_prior)

    if angle_window is None:
        angle_window = max(0.05, abs(angle_prior - x_current[axis_index]) * 2.0 + 0.02)
    angle_window = float(abs(angle_window))

    if prior_sigma is None:
        prior_sigma = max(0.02, angle_window / 2.0)

    lower = angle_prior - angle_window
    upper = angle_prior + angle_window

    def residuals(theta):
        x_trial = x_current.copy()
        x_trial[axis_index] = theta[0]
        sim_qc = np.array(
            S.quadcell_errors_from_variables(x_trial, M1, M2, M3, M4),
            dtype=float
        )
        res = list((sim_qc - target_sim_qc) / float(qc_fit_scale))
        if prior_sigma is not None and prior_sigma > 0:
            res.append((theta[0] - angle_prior) / float(prior_sigma))
        return np.array(res, dtype=float)

    res = least_squares(
        residuals,
        x0=np.array([angle_prior], dtype=float),
        bounds=(np.array([lower]), np.array([upper])),
    )

    x_fit = x_current.copy()
    x_fit[axis_index] = res.x[0]
    sim_qc_fit = np.array(
        S.quadcell_errors_from_variables(x_fit, M1, M2, M3, M4),
        dtype=float
    )
    fit_error = sim_qc_fit - target_sim_qc

    return x_fit, {
        "success": bool(res.success),
        "message": res.message,
        "angle": float(res.x[0]),
        "angle_prior": angle_prior,
        "angle_window": angle_window,
        "target_sim_qc": target_sim_qc.tolist(),
        "fit_sim_qc": sim_qc_fit.tolist(),
        "fit_error": fit_error.tolist(),
        "fit_error_norm": float(np.linalg.norm(fit_error)),
        "cost": float(res.cost),
    }


def _assimilate_staging_rotation_from_qc(
        x_current, x_before_step, step, measured_qc, base_mirrors, *,
        expected_reflections, qc_readout_sign, max_fit_error):
    """Fit staging on its source branch without trusting optimizer status alone."""
    axis = int(step["axis_index"])
    planned_angle = float(S.pack_variables(*step["mirrors"])[axis])
    target = _sim_qc_from_quadcell_readout(measured_qc, qc_readout_sign)
    priors = (
        ("pulse_prior", float(x_current[axis])),
        ("relative_plan_prior", float(x_before_step[axis] + step["command_value"])),
        ("planned_endpoint_prior", planned_angle),
        ("previous_angle_prior", float(x_before_step[axis])),
    )
    candidates = []
    attempts = []
    seen = set()
    for source, prior in priors:
        if prior in seen or not np.isfinite(prior):
            continue
        seen.add(prior)
        trial = np.array(x_current, dtype=float).copy()
        trial[axis] = prior
        count = int(S.get_reflection_count(*S.unpack_variables(trial, *base_mirrors)))
        if count != expected_reflections:
            attempts.append({
                "prior_source": source, "angle_prior": prior, "seed_reflections": count,
                "accepted": False, "reason": "seed is outside the expected reflection branch",
            })
            continue
        predicted = np.asarray(S.quadcell_errors_from_variables(trial, *base_mirrors), dtype=float)
        if not np.all(np.isfinite(predicted)):
            attempts.append({
                "prior_source": source, "angle_prior": prior, "seed_reflections": count,
                "accepted": False, "reason": "seed is outside the expected finite-QC branch",
            })
            continue
        candidates.append((float(np.linalg.norm(predicted - target)), source, prior))
    candidates.sort(key=lambda candidate: candidate[0])

    for _, source, prior in candidates:
        try:
            fitted, info = assimilate_rotation_angle_from_qc(
                x_current, axis, measured_qc, *base_mirrors,
                qc_readout_sign=qc_readout_sign, angle_prior=prior,
                angle_window=max(0.05, abs(x_current[axis] - prior) + 0.02),
            )
            count = int(S.get_reflection_count(*S.unpack_variables(fitted, *base_mirrors)))
            error = float(info["fit_error_norm"])
            accepted = bool(
                info["success"] and np.all(np.isfinite(fitted)) and np.isfinite(error) and
                error <= max_fit_error and count == expected_reflections
            )
            attempts.append({
                "prior_source": source, "angle_prior": prior, "angle": float(fitted[axis]),
                "optimizer_success": bool(info["success"]), "fitted_reflections": count,
                "fit_error_norm": error, "accepted": accepted,
            })
            if accepted:
                return fitted, {
                    **info, "accepted": True, "expected_reflections": int(expected_reflections),
                    "fitted_reflections": count, "prior_source": source, "fit_attempts": attempts,
                }
        except Exception as exc:
            attempts.append({"prior_source": source, "angle_prior": prior, "accepted": False, "error": str(exc)})
    return np.array(x_current, dtype=float).copy(), {
        "success": False, "accepted": False, "expected_reflections": int(expected_reflections),
        "message": f"No same-N_R={expected_reflections} angle fit matched the measured QCs within {max_fit_error:g} mm.",
        "fit_attempts": attempts,
    }


def _update_rotation_calibration(
        calibration,
        actuator_label,
        actual_angle_delta,
        commanded_substeps,
        *,
        update_rate=0.2,
        clip_fraction=0.2):
    if commanded_substeps == 0:
        return calibration.get(actuator_label, DEFAULT_ROTATION_DEGREES_PER_SUBSTEP)

    observed = abs(float(actual_angle_delta)) / abs(int(commanded_substeps))
    if not np.isfinite(observed) or observed <= 0:
        return calibration.get(actuator_label, DEFAULT_ROTATION_DEGREES_PER_SUBSTEP)

    old = calibration.get(actuator_label, DEFAULT_ROTATION_DEGREES_PER_SUBSTEP)
    lower = old * (1.0 - clip_fraction)
    upper = old * (1.0 + clip_fraction)
    observed = float(np.clip(observed, lower, upper))
    new_value = (1.0 - update_rate) * old + update_rate * observed
    calibration[actuator_label] = new_value
    return new_value


def _dry_run_rotation_delta_for_steps(
        actuator_label,
        sim_steps,
        rotation_calibration,
        rng,
        *,
        dry_run_rotation_error=0.10,
        dry_run_pulse_response=None,
        pulse_substeps=None):
    sim_steps = int(sim_steps)
    if sim_steps == 0:
        return 0.0

    sign_key = "+" if sim_steps > 0 else "-"
    if dry_run_pulse_response is not None:
        entry = dry_run_pulse_response.get(actuator_label)
        if isinstance(entry, dict):
            if sign_key in entry:
                value = float(entry[sign_key])
                reference_steps = abs(int(entry.get("pulse_substeps", pulse_substeps or abs(sim_steps))))
                reference_steps = max(reference_steps, 1)
                return value * (abs(sim_steps) / reference_steps)
            if "degrees_per_substep" in entry:
                base = float(entry["degrees_per_substep"]) * sim_steps
                return base
        elif entry is not None:
            return float(entry) * sim_steps

    degrees_per_substep = rotation_calibration.get(
        actuator_label,
        DEFAULT_ROTATION_DEGREES_PER_SUBSTEP
    )
    error_factor = 1.0 + rng.uniform(-dry_run_rotation_error, dry_run_rotation_error)
    return sim_steps * degrees_per_substep * error_factor


def _pulse_response_from_table(
        pulse_table,
        actuator_label,
        sign,
        pulse_substeps,
        fallback_degrees_per_substep):
    sign_key = "+" if sign >= 0 else "-"
    fallback = sign * int(pulse_substeps) * float(fallback_degrees_per_substep)
    if not pulse_table:
        return fallback, "fallback_degrees_per_substep"

    responses = pulse_table.get("responses", pulse_table)
    entry = responses.get(actuator_label, {}).get(sign_key)
    if not entry:
        return fallback, "fallback_degrees_per_substep"

    value = float(entry.get("mean_angle_delta", fallback))
    if not np.isfinite(value) or abs(value) <= 1e-12:
        return fallback, "fallback_degrees_per_substep"
    return value, "pulse_calibration"


def calibrate_rotation_pulse_response(
        M1,
        M2,
        M3,
        M4,
        hardware=None,
        *,
        actuator_map=None,
        rotation_calibration=None,
        actuators=("M1.dangle", "M2.dangle", "M3.dangle", "M4.dangle"),
        directions=(1, -1),
        pulse_substeps=2,
        repeats=3,
        qc_min_signal=0.04,
        qc_min_signals=None,
        qc_readout_sign=-1.0,
        rotation_settle_delay=0.04,
        recenter_after_calibration=True,
        recenter_after_each_actuator=True,
        recenter_tolerance=0.2,
        recenter_max_pulses=20,
        recenter_min_improvement=0.01,
        dry_run=False,
        dry_run_rotation_error=0.10,
        dry_run_signal_strengths=(1.0, 1.0),
        dry_run_pulse_response=None,
        rng_seed=None,
        profile=True,
        profile_sink=None):
    """Measure local angle response for small signed rotation pulses."""
    if profile and profile_sink is None:
        profile_sink = print

    t0 = time.perf_counter()

    def log(message):
        if not profile:
            return
        profile_sink(f"[calibrate_rotation {time.perf_counter() - t0:.3f}s] {message}")

    actuator_map = _merged_actuator_map(actuator_map)
    rotation_calibration = _normalized_rotation_calibration(rotation_calibration)
    rng = np.random.default_rng(rng_seed)
    base_mirrors = (
        np.array(M1, dtype=float),
        np.array(M2, dtype=float),
        np.array(M3, dtype=float),
        np.array(M4, dtype=float),
    )
    x_estimate = S.pack_variables(*base_mirrors)
    x_initial = x_estimate.copy()
    x_physical = x_estimate.copy()
    pulse_substeps = int(abs(pulse_substeps))
    repeats = int(repeats)
    if pulse_substeps <= 0:
        raise ValueError("pulse_substeps must be positive.")
    if repeats <= 0:
        raise ValueError("repeats must be positive.")

    responses = {}
    samples = []
    recenter_log = []
    failure_reason = None

    def command_pulse(label, mapping, axis_index, sim_steps):
        nonlocal failure_reason
        hardware_direction = int(np.sign(mapping.get("direction", 1)) or 1)
        hardware_steps = hardware_direction * int(sim_steps)
        x_before = x_estimate.copy()

        if dry_run:
            actual_delta = _dry_run_rotation_delta_for_steps(
                label,
                sim_steps,
                rotation_calibration,
                rng,
                dry_run_rotation_error=dry_run_rotation_error,
                dry_run_pulse_response=dry_run_pulse_response,
                pulse_substeps=pulse_substeps,
            )
            x_physical[axis_index] += actual_delta
        else:
            if hardware is None or getattr(hardware, "rotation_stages", None) is None:
                raise ValueError("hardware.rotation_stages is required unless dry_run=True.")
            controller = mapping.get("controller", DEFAULT_ROTATION_CONTROLLER)
            actuator = int(mapping["actuator"])
            hardware.rotation_stages.move_relative_steps(controller, actuator, hardware_steps)

        x_estimate[axis_index] += int(sim_steps) * rotation_calibration.get(
            label,
            DEFAULT_ROTATION_DEGREES_PER_SUBSTEP
        )
        if rotation_settle_delay and rotation_settle_delay > 0:
            time.sleep(rotation_settle_delay)

        qc = _read_quadcell_x_with_signal(
            hardware,
            dry_run=dry_run,
            dry_run_x=x_physical,
            base_mirrors=base_mirrors,
            qc_readout_sign=qc_readout_sign,
            qc_min_signal=qc_min_signal,
            qc_min_signals=qc_min_signals,
            dry_run_signal_strengths=dry_run_signal_strengths,
        )
        if not qc["valid"]:
            failure_reason = f"QC signal invalid during calibration for {label}."
            return x_before, qc, None, hardware_steps

        x_fit, assimilation = assimilate_rotation_angle_from_qc(
            x_estimate,
            axis_index,
            qc["y"],
            *base_mirrors,
            qc_readout_sign=qc_readout_sign,
            angle_prior=x_estimate[axis_index],
        )
        x_estimate[:] = x_fit
        return x_before, qc, assimilation, hardware_steps

    def calibration_qc_sensitivities(allowed_actuators=None):
        allowed = None if allowed_actuators is None else set(allowed_actuators)
        grouped = {}
        for sample in samples:
            if allowed is not None and sample.get("actuator") not in allowed:
                continue
            if not sample.get("qc_after_signal_valid", False):
                continue
            try:
                before = np.array(sample["qc_before"], dtype=float)
                after = np.array(sample["qc_after"], dtype=float)
            except Exception:
                continue
            if not np.all(np.isfinite(before)) or not np.all(np.isfinite(after)):
                continue
            key = (sample["actuator"], sample["sign"])
            grouped.setdefault(key, []).append(after - before)

        sensitivities = {}
        for key, values in grouped.items():
            arr = np.array(values, dtype=float)
            if arr.size:
                sensitivities[key] = np.mean(arr, axis=0)
        return sensitivities

    def run_post_calibration_recenter(label, allowed_actuators):
        nonlocal failure_reason
        sensitivities = calibration_qc_sensitivities(allowed_actuators)
        current_qc = _read_quadcell_x_with_signal(
            hardware,
            dry_run=dry_run,
            dry_run_x=x_physical,
            base_mirrors=base_mirrors,
            qc_readout_sign=qc_readout_sign,
            qc_min_signal=qc_min_signal,
            qc_min_signals=qc_min_signals,
            dry_run_signal_strengths=dry_run_signal_strengths,
        )
        if not current_qc["valid"]:
            failure_reason = f"QC signal invalid before {label} calibration recenter."
            return

        for pulse_index in range(1, int(recenter_max_pulses) + 1):
            current_x = np.array(current_qc["x"], dtype=float)
            current_norm = float(np.linalg.norm(current_x))
            current_max_abs = float(np.max(np.abs(current_x)))
            if current_max_abs <= float(recenter_tolerance):
                break

            candidates = []
            for actuator_label in allowed_actuators:
                mapping = actuator_map.get(actuator_label)
                if mapping is None or mapping.get("kind") != "rotation":
                    continue
                axis_index = int({"M1.dangle": 1, "M2.dangle": 3, "M3.dangle": 5, "M4.dangle": 7}[actuator_label])
                for sign_key, sign in (("+", 1), ("-", -1)):
                    qc_delta = sensitivities.get((actuator_label, sign_key))
                    if qc_delta is None:
                        continue
                    predicted = current_x + qc_delta
                    candidates.append((
                        float(np.linalg.norm(predicted)),
                        float(np.max(np.abs(predicted))),
                        actuator_label,
                        mapping,
                        axis_index,
                        sign,
                        sign_key,
                        qc_delta,
                    ))

            if not candidates:
                recenter_log.append({
                    "phase": label,
                    "pulse": int(pulse_index),
                    "success": False,
                    "failure_reason": "No measured calibration QC sensitivities available.",
                    "qc_x": current_x.tolist(),
                    "allowed_actuators": list(allowed_actuators),
                })
                break

            candidates.sort(key=lambda row: (row[0], row[1]))
            _, _, actuator_label, mapping, axis_index, sign, sign_key, qc_delta = candidates[0]
            hardware_direction = int(np.sign(mapping.get("direction", 1)) or 1)
            sim_steps = int(sign * pulse_substeps)
            hardware_steps = hardware_direction * sim_steps
            x_before = x_estimate.copy()
            x_physical_before = x_physical.copy()

            if dry_run:
                actual_delta = _dry_run_rotation_delta_for_steps(
                    actuator_label,
                    sim_steps,
                    rotation_calibration,
                    rng,
                    dry_run_rotation_error=dry_run_rotation_error,
                    dry_run_pulse_response=dry_run_pulse_response,
                    pulse_substeps=pulse_substeps,
                )
                x_physical[axis_index] += actual_delta
            else:
                controller = mapping.get("controller", DEFAULT_ROTATION_CONTROLLER)
                actuator = int(mapping["actuator"])
                hardware.rotation_stages.move_relative_steps(controller, actuator, hardware_steps)

            x_estimate[axis_index] += sim_steps * rotation_calibration.get(
                actuator_label,
                DEFAULT_ROTATION_DEGREES_PER_SUBSTEP
            )
            if rotation_settle_delay and rotation_settle_delay > 0:
                time.sleep(rotation_settle_delay)

            after_qc = _read_quadcell_x_with_signal(
                hardware,
                dry_run=dry_run,
                dry_run_x=x_physical,
                base_mirrors=base_mirrors,
                qc_readout_sign=qc_readout_sign,
                qc_min_signal=qc_min_signal,
                qc_min_signals=qc_min_signals,
                dry_run_signal_strengths=dry_run_signal_strengths,
            )
            after_norm = (
                float(np.linalg.norm(after_qc["x"]))
                if after_qc["valid"] else float("inf")
            )
            improved = (
                after_qc["valid"] and
                after_norm < current_norm - float(recenter_min_improvement)
            )

            entry = {
                "phase": label,
                "pulse": int(pulse_index),
                "actuator": actuator_label,
                "sign": sign_key,
                "hardware_steps": int(hardware_steps),
                "sim_steps": int(sim_steps),
                "predicted_qc_delta": np.array(qc_delta, dtype=float).tolist(),
                "before_qc_x": current_x.tolist(),
                "after_qc_x": after_qc["x"].tolist(),
                "before_norm": float(current_norm),
                "after_norm": float(after_norm),
                "accepted": bool(improved),
            }

            if improved:
                try:
                    x_fit, assimilation = assimilate_rotation_angle_from_qc(
                        x_estimate,
                        axis_index,
                        after_qc["y"],
                        *base_mirrors,
                        qc_readout_sign=qc_readout_sign,
                        angle_prior=x_estimate[axis_index],
                    )
                    x_estimate[:] = x_fit
                    if dry_run:
                        x_physical[:] = x_estimate
                    entry["assimilation"] = assimilation
                except Exception as exc:
                    entry["assimilation_error"] = str(exc)
                recenter_log.append(entry)
                current_qc = after_qc
                log(
                    f"{label} recenter pulse={pulse_index} {actuator_label} {sign_key} "
                    f"qc=({after_qc['x'][0]:.3f},{after_qc['x'][1]:.3f})"
                )
                continue

            if dry_run:
                x_physical[:] = x_physical_before
            else:
                controller = mapping.get("controller", DEFAULT_ROTATION_CONTROLLER)
                actuator = int(mapping["actuator"])
                hardware.rotation_stages.move_relative_steps(controller, actuator, -hardware_steps)
                if rotation_settle_delay and rotation_settle_delay > 0:
                    time.sleep(rotation_settle_delay)
            x_estimate[:] = x_before
            entry["rolled_back"] = True
            recenter_log.append(entry)
            break

    start_qc = _read_quadcell_x_with_signal(
        hardware,
        dry_run=dry_run,
        dry_run_x=x_physical,
        base_mirrors=base_mirrors,
        qc_readout_sign=qc_readout_sign,
        qc_min_signal=qc_min_signal,
        qc_min_signals=qc_min_signals,
        dry_run_signal_strengths=dry_run_signal_strengths,
    )
    if not start_qc["valid"]:
        failure_reason = "QC signal invalid before rotation pulse calibration."

    if failure_reason is None:
        for actuator_label in actuators:
            mapping = actuator_map.get(actuator_label)
            if mapping is None:
                failure_reason = f"No hardware mapping for actuator {actuator_label}."
                break
            if mapping.get("kind") != "rotation":
                failure_reason = f"Calibration only supports rotation actuators, got {actuator_label}."
                break
            axis_index = int({"M1.dangle": 1, "M2.dangle": 3, "M3.dangle": 5, "M4.dangle": 7}[actuator_label])
            responses.setdefault(actuator_label, {})

            for direction in directions:
                sign = int(np.sign(direction) or 1)
                sign_key = "+" if sign > 0 else "-"
                signed_samples = []
                for repeat_index in range(1, repeats + 1):
                    if failure_reason is not None:
                        break

                    before_qc = _read_quadcell_x_with_signal(
                        hardware,
                        dry_run=dry_run,
                        dry_run_x=x_physical,
                        base_mirrors=base_mirrors,
                        qc_readout_sign=qc_readout_sign,
                        qc_min_signal=qc_min_signal,
                        qc_min_signals=qc_min_signals,
                        dry_run_signal_strengths=dry_run_signal_strengths,
                    )
                    if not before_qc["valid"]:
                        failure_reason = f"QC signal invalid before calibration pulse for {actuator_label} {sign_key}."
                        break

                    x_before, after_qc, assimilation, hardware_steps = command_pulse(
                        actuator_label,
                        mapping,
                        axis_index,
                        sign * pulse_substeps,
                    )
                    if failure_reason is not None:
                        break

                    measured_delta = float(x_estimate[axis_index] - x_before[axis_index])
                    signed_samples.append(measured_delta)

                    x_return_before, return_qc, return_assimilation, return_hardware_steps = command_pulse(
                        actuator_label,
                        mapping,
                        axis_index,
                        -sign * pulse_substeps,
                    )
                    sample = {
                        "actuator": actuator_label,
                        "sign": sign_key,
                        "repeat": repeat_index,
                        "pulse_substeps": int(pulse_substeps),
                        "hardware_steps": int(hardware_steps),
                        "angle_delta": measured_delta,
                        "qc_before": before_qc["x"].tolist(),
                        "qc_after": after_qc["x"].tolist(),
                        "qc_after_signal_valid": bool(after_qc["valid"]),
                        "fit_error_norm": None if assimilation is None else assimilation.get("fit_error_norm"),
                        "return_hardware_steps": int(return_hardware_steps),
                        "return_qc": return_qc["x"].tolist(),
                        "return_qc_signal_valid": bool(return_qc["valid"]),
                        "return_fit_error_norm": (
                            None if return_assimilation is None
                            else return_assimilation.get("fit_error_norm")
                        ),
                    }
                    samples.append(sample)

                    if failure_reason is not None:
                        break

                values = np.array(signed_samples, dtype=float)
                responses[actuator_label][sign_key] = {
                    "success": bool(values.size > 0 and failure_reason is None),
                    "pulse_substeps": int(pulse_substeps),
                    "mean_angle_delta": float(np.mean(values)) if values.size else np.nan,
                    "std_angle_delta": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
                    "mean_degrees_per_substep": float(np.mean(values) / pulse_substeps) if values.size else np.nan,
                    "n_samples": int(values.size),
                    "samples": values.tolist(),
                }
                if values.size:
                    log(
                        f"{actuator_label} {sign_key}{pulse_substeps} substeps: "
                        f"mean={np.mean(values):.6g} deg std={responses[actuator_label][sign_key]['std_angle_delta']:.3g}"
                    )
                if failure_reason is not None:
                    break
            if (
                failure_reason is None and
                recenter_after_calibration and
                recenter_after_each_actuator
            ):
                run_post_calibration_recenter(
                    f"post-{actuator_label}",
                    (actuator_label,),
                )
            if failure_reason is not None:
                break

    def calibration_qc_sensitivities():
        grouped = {}
        for sample in samples:
            if not sample.get("qc_after_signal_valid", False):
                continue
            try:
                before = np.array(sample["qc_before"], dtype=float)
                after = np.array(sample["qc_after"], dtype=float)
            except Exception:
                continue
            if not np.all(np.isfinite(before)) or not np.all(np.isfinite(after)):
                continue
            key = (sample["actuator"], sample["sign"])
            grouped.setdefault(key, []).append(after - before)

        sensitivities = {}
        for key, values in grouped.items():
            arr = np.array(values, dtype=float)
            if arr.size:
                sensitivities[key] = np.mean(arr, axis=0)
        return sensitivities

    if (
        failure_reason is None and
        recenter_after_calibration and
        not recenter_after_each_actuator
    ):
        sensitivities = calibration_qc_sensitivities()
        current_qc = _read_quadcell_x_with_signal(
            hardware,
            dry_run=dry_run,
            dry_run_x=x_physical,
            base_mirrors=base_mirrors,
            qc_readout_sign=qc_readout_sign,
            qc_min_signal=qc_min_signal,
            qc_min_signals=qc_min_signals,
            dry_run_signal_strengths=dry_run_signal_strengths,
        )
        if not current_qc["valid"]:
            failure_reason = "QC signal invalid before post-calibration recenter."
        else:
            for pulse_index in range(1, int(recenter_max_pulses) + 1):
                current_x = np.array(current_qc["x"], dtype=float)
                current_norm = float(np.linalg.norm(current_x))
                current_max_abs = float(np.max(np.abs(current_x)))
                if current_max_abs <= float(recenter_tolerance):
                    break

                candidates = []
                for actuator_label in actuators:
                    mapping = actuator_map.get(actuator_label)
                    if mapping is None or mapping.get("kind") != "rotation":
                        continue
                    axis_index = int({"M1.dangle": 1, "M2.dangle": 3, "M3.dangle": 5, "M4.dangle": 7}[actuator_label])
                    for sign_key, sign in (("+", 1), ("-", -1)):
                        qc_delta = sensitivities.get((actuator_label, sign_key))
                        if qc_delta is None:
                            continue
                        predicted = current_x + qc_delta
                        candidates.append((
                            float(np.linalg.norm(predicted)),
                            float(np.max(np.abs(predicted))),
                            actuator_label,
                            mapping,
                            axis_index,
                            sign,
                            sign_key,
                            qc_delta,
                        ))

                if not candidates:
                    recenter_log.append({
                        "pulse": int(pulse_index),
                        "success": False,
                        "failure_reason": "No measured calibration QC sensitivities available.",
                        "qc_x": current_x.tolist(),
                    })
                    break

                candidates.sort(key=lambda row: (row[0], row[1]))
                _, _, actuator_label, mapping, axis_index, sign, sign_key, qc_delta = candidates[0]
                hardware_direction = int(np.sign(mapping.get("direction", 1)) or 1)
                sim_steps = int(sign * pulse_substeps)
                hardware_steps = hardware_direction * sim_steps
                x_before = x_estimate.copy()
                x_physical_before = x_physical.copy()

                if dry_run:
                    actual_delta = _dry_run_rotation_delta_for_steps(
                        actuator_label,
                        sim_steps,
                        rotation_calibration,
                        rng,
                        dry_run_rotation_error=dry_run_rotation_error,
                        dry_run_pulse_response=dry_run_pulse_response,
                        pulse_substeps=pulse_substeps,
                    )
                    x_physical[axis_index] += actual_delta
                else:
                    controller = mapping.get("controller", DEFAULT_ROTATION_CONTROLLER)
                    actuator = int(mapping["actuator"])
                    hardware.rotation_stages.move_relative_steps(controller, actuator, hardware_steps)

                x_estimate[axis_index] += sim_steps * rotation_calibration.get(
                    actuator_label,
                    DEFAULT_ROTATION_DEGREES_PER_SUBSTEP
                )
                if rotation_settle_delay and rotation_settle_delay > 0:
                    time.sleep(rotation_settle_delay)

                after_qc = _read_quadcell_x_with_signal(
                    hardware,
                    dry_run=dry_run,
                    dry_run_x=x_physical,
                    base_mirrors=base_mirrors,
                    qc_readout_sign=qc_readout_sign,
                    qc_min_signal=qc_min_signal,
                    qc_min_signals=qc_min_signals,
                    dry_run_signal_strengths=dry_run_signal_strengths,
                )
                after_norm = (
                    float(np.linalg.norm(after_qc["x"]))
                    if after_qc["valid"] else float("inf")
                )
                improved = (
                    after_qc["valid"] and
                    after_norm < current_norm - float(recenter_min_improvement)
                )

                entry = {
                    "pulse": int(pulse_index),
                    "actuator": actuator_label,
                    "sign": sign_key,
                    "hardware_steps": int(hardware_steps),
                    "sim_steps": int(sim_steps),
                    "predicted_qc_delta": np.array(qc_delta, dtype=float).tolist(),
                    "before_qc_x": current_x.tolist(),
                    "after_qc_x": after_qc["x"].tolist(),
                    "before_norm": float(current_norm),
                    "after_norm": float(after_norm),
                    "accepted": bool(improved),
                }

                if improved:
                    try:
                        x_fit, assimilation = assimilate_rotation_angle_from_qc(
                            x_estimate,
                            axis_index,
                            after_qc["y"],
                            *base_mirrors,
                            qc_readout_sign=qc_readout_sign,
                            angle_prior=x_estimate[axis_index],
                        )
                        x_estimate[:] = x_fit
                        if dry_run:
                            x_physical[:] = x_estimate
                        entry["assimilation"] = assimilation
                    except Exception as exc:
                        entry["assimilation_error"] = str(exc)
                    recenter_log.append(entry)
                    current_qc = after_qc
                    log(
                        f"post-cal recenter pulse={pulse_index} {actuator_label} {sign_key} "
                        f"qc=({after_qc['x'][0]:.3f},{after_qc['x'][1]:.3f})"
                    )
                    continue

                if dry_run:
                    x_physical[:] = x_physical_before
                else:
                    controller = mapping.get("controller", DEFAULT_ROTATION_CONTROLLER)
                    actuator = int(mapping["actuator"])
                    hardware.rotation_stages.move_relative_steps(controller, actuator, -hardware_steps)
                    if rotation_settle_delay and rotation_settle_delay > 0:
                        time.sleep(rotation_settle_delay)
                x_estimate[:] = x_before
                entry["rolled_back"] = True
                recenter_log.append(entry)
                break

    final_qc = _read_quadcell_x_with_signal(
        hardware,
        dry_run=dry_run,
        dry_run_x=x_physical,
        base_mirrors=base_mirrors,
        qc_readout_sign=qc_readout_sign,
        qc_min_signal=qc_min_signal,
        qc_min_signals=qc_min_signals,
        dry_run_signal_strengths=dry_run_signal_strengths,
    )

    calibrated_mirrors = S.unpack_variables(x_estimate, *base_mirrors)
    model_comparison = _qc_model_comparison(x_estimate, base_mirrors, final_qc, qc_readout_sign)
    result = {
        "success": failure_reason is None,
        "failure_reason": failure_reason,
        "pulse_substeps": int(pulse_substeps),
        "repeats": int(repeats),
        "responses": responses,
        "samples": samples,
        "start_qc_x": start_qc["x"].tolist(),
        "start_qc_signal_strengths": start_qc["signal_strengths"],
        "start_qc_signal_valid": bool(start_qc["valid"]),
        "final_qc_x": final_qc["x"].tolist(),
        "final_qc_signal_strengths": final_qc["signal_strengths"],
        "final_qc_signal_valid": bool(final_qc["valid"]),
        "recenter_after_calibration": bool(recenter_after_calibration),
        "recenter_after_each_actuator": bool(recenter_after_each_actuator),
        "recenter_tolerance": float(recenter_tolerance),
        "recenter_log": recenter_log,
        "x_estimate": x_estimate.copy(),
        "initial_x": x_initial,
        "angle_delta_from_initial": (x_estimate - x_initial)[[1, 3, 5, 7]].tolist(),
        "qc_shift_from_initial": (final_qc["x"] - start_qc["x"]).tolist(),
        "initial_model_comparison": _qc_model_comparison(x_initial, base_mirrors, start_qc, qc_readout_sign),
        "final_model_comparison": model_comparison,
        "mirror_state_source": "qc_fitted_angle_estimate",
        "x_physical": x_physical.copy(),
        "calibrated_mirrors": calibrated_mirrors,
        "calibrated_mirrors_list": _mirrors_to_lists(calibrated_mirrors),
        "qc_min_signal": float(qc_min_signal),
        "qc_min_signals": (
            None if qc_min_signals is None
            else np.array(qc_min_signals, dtype=float).tolist()
        ),
    }
    log(
        f"done success={result['success']} failure={failure_reason} "
        f"qc_shift={np.round(result['qc_shift_from_initial'], 4).tolist()} "
        f"model_qc_residual={model_comparison['qc_residual_norm']:.4g} "
        "state=qc_fitted_angle_estimate"
    )
    return result


def _execute_linear_step(
        step,
        mapping,
        hardware,
        x_model,
        x_physical,
        linear_stage_locs,
        *,
        dry_run,
        linear_settle_delay,
        linear_stage_guard_mm=DEFAULT_LINEAR_STAGE_GUARD_MM,
        linear_stage_replan_tolerance_mm=DEFAULT_LINEAR_STAGE_REPLAN_TOLERANCE_MM):
    axis_index = step["axis_index"]
    command_value = float(step["command_value"])
    direction = float(mapping.get("direction", 1.0))
    if direction == 0:
        raise ValueError(f"Linear actuator {step.get('actuator')} has zero hardware direction.")
    planned_hardware_delta = direction * command_value
    hardware_delta = planned_hardware_delta
    serial = mapping["serial"]
    mirror_name = mapping.get("mirror", step["actuator"].split(".")[0])

    before_stage_position = linear_stage_locs.get(mirror_name)
    after_stage_position = None
    before_hardware_position = None
    after_hardware_position = None
    actual_sim_delta = command_value

    if dry_run:
        if before_stage_position is None:
            before_stage_position = 0.0
        before_stage_position = float(before_stage_position)
        loc_min, loc_max = _linear_stage_allowed_position_range(
            before_stage_position,
            linear_stage_guard_mm=linear_stage_guard_mm
        )
        hardware_delta, planned_after_stage_position, after_stage_position = (
            _clip_linear_hardware_delta_to_range(
                planned_hardware_delta,
                before_stage_position,
                loc_min,
                loc_max
            )
        )
        clipped_after_stage_position = after_stage_position
        actual_sim_delta = hardware_delta / direction
        x_physical[axis_index] += actual_sim_delta
    else:
        if hardware is None or getattr(hardware, "stages", None) is None:
            raise ValueError("hardware.stages is required to execute linear moves.")
        before_hardware_position = float(hardware.stages.get_position(serial))
        if before_stage_position is None:
            before_stage_position = before_hardware_position
        else:
            before_stage_position = float(before_stage_position)
        loc_min, loc_max = _linear_stage_allowed_position_range(
            before_hardware_position,
            linear_stage_guard_mm=linear_stage_guard_mm
        )
        hardware_delta, planned_after_stage_position, clipped_after_stage_position = (
            _clip_linear_hardware_delta_to_range(
                planned_hardware_delta,
                before_hardware_position,
                loc_min,
                loc_max
            )
        )
        hardware.stages.move_relative(serial, hardware_delta)
        if linear_settle_delay and linear_settle_delay > 0:
            time.sleep(linear_settle_delay)
        after_hardware_position = float(hardware.stages.get_position(serial))
        actual_hardware_delta = after_hardware_position - before_hardware_position
        actual_sim_delta = actual_hardware_delta / direction
        after_stage_position = after_hardware_position

    x_model[axis_index] += actual_sim_delta
    linear_stage_locs[mirror_name] = after_stage_position
    linear_stage_replan_tolerance_mm = float(linear_stage_replan_tolerance_mm)
    hardware_delta_error = float(abs(hardware_delta - planned_hardware_delta))
    sim_delta_error = float(abs(actual_sim_delta - command_value))
    command_was_clipped = hardware_delta_error > linear_stage_replan_tolerance_mm
    actual_motion_mismatch = sim_delta_error > linear_stage_replan_tolerance_mm
    replan_recommended = command_was_clipped or actual_motion_mismatch

    return {
        "kind": "linear",
        "serial": serial,
        "planned_sim_delta": command_value,
        "planned_hardware_delta": planned_hardware_delta,
        "hardware_delta": hardware_delta,
        "actual_sim_delta": actual_sim_delta,
        "hardware_delta_error": hardware_delta_error,
        "sim_delta_error": sim_delta_error,
        "linear_stage_replan_tolerance_mm": linear_stage_replan_tolerance_mm,
        "linear_stage_clipped": bool(command_was_clipped),
        "linear_stage_actual_mismatch": bool(actual_motion_mismatch),
        "replan_recommended": bool(replan_recommended),
        "replan_reason": (
            "linear stage command clipped to guarded travel range"
            if command_was_clipped else (
                "linear stage actual motion differed from planned motion"
                if actual_motion_mismatch else None
            )
        ),
        "before_position": before_hardware_position if not dry_run else before_stage_position,
        "after_position": after_hardware_position if not dry_run else after_stage_position,
        "planned_after_stage_position": planned_after_stage_position,
        "clipped_after_stage_position": clipped_after_stage_position,
        "commanded_after_stage_position": after_stage_position,
        "guarded_position_range": [float(loc_min), float(loc_max)],
        "before_hardware_position": before_hardware_position,
        "after_hardware_position": after_hardware_position,
        "before_stage_position": before_stage_position,
        "after_stage_position": after_stage_position,
        "before_sim_position": before_stage_position,
        "after_sim_position": after_stage_position,
    }


def _execute_rotation_step(
        step,
        mapping,
        hardware,
        x_model,
        x_physical,
        base_mirrors,
        rotation_calibration,
        rng,
        *,
        dry_run,
        dry_run_rotation_error,
        qc_readout_sign,
        qc_step_tolerance,
        qc_replan_tolerance,
        max_qc_error,
        qc_plan_limit,
        qc_safety_margin,
        min_qc_step_tolerance,
        clip_qc_target_to_safety,
        fast_qc_avg,
        fast_qc_delay,
        max_rotation_chunks_per_step,
        max_rotation_chunk_substeps,
        min_rotation_chunk_substeps,
        calibration_update_rate,
        calibration_clip_fraction):
    actuator_label = step["actuator"]
    axis_index = step["axis_index"]
    planned_angle_delta = float(step["command_value"])
    degrees_per_substep = rotation_calibration.get(
        actuator_label,
        DEFAULT_ROTATION_DEGREES_PER_SUBSTEP
    )
    if degrees_per_substep <= 0:
        raise ValueError(f"degrees_per_substep must be positive for {actuator_label}.")

    controller = mapping.get("controller", DEFAULT_ROTATION_CONTROLLER)
    actuator = int(mapping["actuator"])
    hardware_direction = int(np.sign(mapping.get("direction", 1)) or 1)
    angle_direction = int(np.sign(planned_angle_delta) or 1)
    target_qc_y = _planned_step_qc_readout(step, qc_readout_sign)
    target_sim_qc = _sim_qc_from_quadcell_readout(target_qc_y, qc_readout_sign)
    planned_target_sim_qc = target_sim_qc.copy()
    if clip_qc_target_to_safety:
        safe_qc_limit = max(0.0, max_qc_error - qc_safety_margin)
        target_sim_qc = np.clip(target_sim_qc, -safe_qc_limit, safe_qc_limit)
        target_qc_y = _quadcell_readout_from_sim_qc(target_sim_qc, qc_readout_sign)
    target_qc_margin = max_qc_error - float(np.max(np.abs(target_sim_qc)))
    target_plan_margin = qc_plan_limit - float(np.max(np.abs(target_sim_qc)))
    target_is_recovery = target_plan_margin < 0.0
    effective_qc_step_tolerance = min(
        qc_step_tolerance,
        max(float(min_qc_step_tolerance), 0.5 * max(0.0, target_qc_margin))
    )

    before_qc = _read_quadcell_y(
        hardware,
        times=fast_qc_avg,
        delay=fast_qc_delay,
        dry_run=dry_run,
        dry_run_x=x_physical,
        base_mirrors=base_mirrors,
        qc_readout_sign=qc_readout_sign,
    )
    current_y = before_qc["y"]
    start_y = current_y.copy()
    target_delta = target_qc_y - start_y
    target_norm = float(np.linalg.norm(target_delta))
    best_distance = float(np.linalg.norm(target_qc_y - current_y))
    best_y = current_y.copy()
    chunks_without_improvement = 0
    total_commanded_substeps = 0
    total_sim_substeps = 0
    chunk_logs = []

    predicted_total_substeps = max(
        abs(planned_angle_delta) / degrees_per_substep,
        float(min_rotation_chunk_substeps)
    )
    base_chunk = int(np.ceil(predicted_total_substeps / 5.0))
    base_chunk = int(np.clip(
        base_chunk,
        int(min_rotation_chunk_substeps),
        int(max_rotation_chunk_substeps)
    ))

    stop_reason = None
    if best_distance <= effective_qc_step_tolerance:
        stop_reason = "already_within_qc_step_tolerance"

    for chunk_index in range(1, max_rotation_chunks_per_step + 1):
        if stop_reason is not None:
            break

        remaining_distance = float(np.linalg.norm(target_qc_y - current_y))
        if remaining_distance <= effective_qc_step_tolerance:
            stop_reason = "reached_qc_step_tolerance"
            break

        if target_norm > 1e-12:
            progress = float(np.dot(current_y - start_y, target_delta) / np.dot(target_delta, target_delta))
        else:
            progress = 1.0

        scale = 1.0
        current_sim_qc = _sim_qc_from_quadcell_readout(current_y, qc_readout_sign)
        current_qc_margin = max_qc_error - float(np.max(np.abs(current_sim_qc)))
        current_plan_margin = qc_plan_limit - float(np.max(np.abs(current_sim_qc)))

        if (
            progress > 0.75 or
            remaining_distance < 2.0 * effective_qc_step_tolerance or
            current_qc_margin < qc_safety_margin or
            (not target_is_recovery and current_plan_margin < qc_safety_margin)
        ):
            scale = 0.5
        if (
            progress > 0.9 or
            remaining_distance < effective_qc_step_tolerance or
            current_qc_margin < 0.5 * qc_safety_margin or
            (not target_is_recovery and current_plan_margin < 0.5 * qc_safety_margin)
        ):
            scale = 0.25

        chunk_substeps = max(
            int(min_rotation_chunk_substeps),
            int(np.ceil(base_chunk * scale))
        )
        if current_qc_margin < 2.0 * qc_safety_margin or (
            not target_is_recovery and current_plan_margin < 2.0 * qc_safety_margin
        ):
            chunk_substeps = int(min_rotation_chunk_substeps)
        sim_steps = angle_direction * chunk_substeps
        hardware_steps = hardware_direction * sim_steps
        x_physical_before_chunk = x_physical.copy()

        if dry_run:
            error_factor = 1.0 + rng.uniform(-dry_run_rotation_error, dry_run_rotation_error)
            x_physical[axis_index] += sim_steps * degrees_per_substep * error_factor
        else:
            if hardware is None or getattr(hardware, "rotation_stages", None) is None:
                raise ValueError("hardware.rotation_stages is required to execute rotation moves.")
            hardware.rotation_stages.move_relative_steps(controller, actuator, hardware_steps)

        total_commanded_substeps += hardware_steps
        total_sim_substeps += sim_steps
        after_qc = _read_quadcell_y(
            hardware,
            times=fast_qc_avg,
            delay=fast_qc_delay,
            dry_run=dry_run,
            dry_run_x=x_physical,
            base_mirrors=base_mirrors,
            qc_readout_sign=qc_readout_sign,
        )
        current_y = after_qc["y"]
        current_sim_qc = _sim_qc_from_quadcell_readout(current_y, qc_readout_sign)
        current_qc_margin = max_qc_error - float(np.max(np.abs(current_sim_qc)))
        current_plan_margin = qc_plan_limit - float(np.max(np.abs(current_sim_qc)))
        distance = float(np.linalg.norm(target_qc_y - current_y))
        if target_norm > 1e-12:
            progress = float(np.dot(current_y - start_y, target_delta) / np.dot(target_delta, target_delta))
        else:
            progress = 1.0

        improved = distance < best_distance
        if improved:
            best_distance = distance
            best_y = current_y.copy()
            chunks_without_improvement = 0
        else:
            chunks_without_improvement += 1

        chunk_logs.append({
            "chunk": chunk_index,
            "hardware_steps": hardware_steps,
            "qc_x": current_y.tolist(),
            "qc_y": current_y.tolist(),
            "distance_to_target": distance,
            "progress": progress,
            "improved": bool(improved),
        })

        if current_qc_margin <= 0.0:
            if dry_run:
                x_physical[:] = x_physical_before_chunk
            else:
                hardware.rotation_stages.move_relative_steps(controller, actuator, -hardware_steps)
            total_commanded_substeps -= hardware_steps
            total_sim_substeps -= sim_steps
            rollback_qc = _read_quadcell_y(
                hardware,
                times=fast_qc_avg,
                delay=fast_qc_delay,
                dry_run=dry_run,
                dry_run_x=x_physical,
                base_mirrors=base_mirrors,
                qc_readout_sign=qc_readout_sign,
            )
            current_y = rollback_qc["y"]
            current_sim_qc = _sim_qc_from_quadcell_readout(current_y, qc_readout_sign)
            current_qc_margin = max_qc_error - float(np.max(np.abs(current_sim_qc)))
            distance = float(np.linalg.norm(target_qc_y - current_y))
            if target_norm > 1e-12:
                progress = float(np.dot(current_y - start_y, target_delta) / np.dot(target_delta, target_delta))
            else:
                progress = 1.0
            if len(chunk_logs) > 0:
                chunk_logs[-1]["rolled_back"] = True
                chunk_logs[-1]["rollback_qc_x"] = current_y.tolist()
                chunk_logs[-1]["rollback_qc_y"] = current_y.tolist()
                chunk_logs[-1]["rollback_distance_to_target"] = distance
                chunk_logs[-1]["rollback_qc_margin"] = current_qc_margin
            best_distance = distance
            best_y = current_y.copy()
            stop_reason = "qc_bound_rollback"
            break
        if distance <= effective_qc_step_tolerance:
            stop_reason = "reached_qc_step_tolerance"
            break
        if current_qc_margin < qc_safety_margin:
            stop_reason = "near_qc_bound"
            break
        if not target_is_recovery and current_plan_margin < 0.0:
            stop_reason = "outside_qc_plan_limit"
            break
        if progress >= 1.0 and not improved:
            stop_reason = "overshot_or_past_target"
            break
        if chunks_without_improvement >= 3:
            stop_reason = "stalled_without_improvement"
            break

    if stop_reason is None:
        stop_reason = "max_rotation_chunks_reached"

    expected_angle_delta = total_sim_substeps * degrees_per_substep
    angle_prior = x_model[axis_index] + expected_angle_delta
    angle_window = max(0.05, abs(expected_angle_delta) * 2.0 + 0.02)

    x_before_assimilation = x_model.copy()
    x_fit, assimilation = assimilate_rotation_angle_from_qc(
        x_model,
        axis_index,
        best_y,
        *base_mirrors,
        qc_readout_sign=qc_readout_sign,
        angle_prior=angle_prior,
        angle_window=angle_window,
    )
    x_model[:] = x_fit

    actual_angle_delta = x_model[axis_index] - x_before_assimilation[axis_index]
    updated_calibration = _update_rotation_calibration(
        rotation_calibration,
        actuator_label,
        actual_angle_delta,
        total_sim_substeps,
        update_rate=calibration_update_rate,
        clip_fraction=calibration_clip_fraction,
    )

    after_distance = float(np.linalg.norm(target_qc_y - best_y))
    final_sim_qc = _sim_qc_from_quadcell_readout(best_y, qc_readout_sign)
    final_qc_margin = max_qc_error - float(np.max(np.abs(final_sim_qc)))
    replan_recommended = (
        after_distance > qc_replan_tolerance or
        final_qc_margin < qc_safety_margin or
        (
            not target_is_recovery and
            float(np.max(np.abs(final_sim_qc))) > qc_plan_limit
        ) or
        stop_reason in {
            "near_qc_bound",
            "qc_bound_exceeded",
            "qc_bound_rollback",
            "outside_qc_plan_limit",
        }
    )

    return {
        "kind": "rotation",
        "controller": controller,
        "actuator": actuator,
        "planned_angle_delta": planned_angle_delta,
        "degrees_per_substep_start": degrees_per_substep,
        "degrees_per_substep_end": updated_calibration,
        "target_qc_x": target_qc_y.tolist(),
        "target_qc_y": target_qc_y.tolist(),
        "planned_target_sim_qc": planned_target_sim_qc.tolist(),
        "target_sim_qc": target_sim_qc.tolist(),
        "qc_target_clipped": bool(np.max(np.abs(planned_target_sim_qc - target_sim_qc)) > 1e-12),
        "effective_qc_step_tolerance": float(effective_qc_step_tolerance),
        "final_qc_margin": float(final_qc_margin),
        "qc_plan_limit": float(qc_plan_limit),
        "start_qc_x": start_y.tolist(),
        "start_qc_y": start_y.tolist(),
        "final_qc_x": best_y.tolist(),
        "final_qc_y": best_y.tolist(),
        "distance_to_target": after_distance,
        "total_commanded_substeps": int(total_commanded_substeps),
        "total_sim_substeps": int(total_sim_substeps),
        "stop_reason": stop_reason,
        "replan_recommended": bool(replan_recommended),
        "assimilation": assimilation,
        "chunks": chunk_logs,
    }


def _staging_angle_feedback_step(
        step, x_estimate, measured_qc, base_mirrors, *, qc_readout_sign,
        degrees_per_substep, qc_step_tolerance, qc_safety_limit, expected_reflections):
    """Express an absolute axis-angle goal as a locally observable QC change."""
    axis = int(step["axis_index"])
    goal_x = np.array(x_estimate, dtype=float).copy()
    goal_angle = float(S.pack_variables(*step["mirrors"])[axis])
    angle_delta = goal_angle - float(goal_x[axis])
    goal_x[axis] = goal_angle
    if int(S.get_reflection_count(*S.unpack_variables(goal_x, *base_mirrors))) != expected_reflections:
        raise ValueError("updated absolute-angle goal leaves the staging reflection branch")
    start_prediction = _quadcell_readout_from_sim_qc(
        S.quadcell_errors_from_variables(x_estimate, *base_mirrors), qc_readout_sign)
    goal_prediction = _quadcell_readout_from_sim_qc(
        S.quadcell_errors_from_variables(goal_x, *base_mirrors), qc_readout_sign)
    target_qc = np.asarray(measured_qc, dtype=float) + goal_prediction - start_prediction
    if not np.all(np.isfinite(target_qc)) or np.max(np.abs(target_qc)) > qc_safety_limit:
        raise ValueError("updated absolute-angle QC goal is outside the staging safety window")

    tolerance = float(qc_step_tolerance)
    if abs(angle_delta) > 1e-12:
        sensitivity = float(np.linalg.norm(goal_prediction - start_prediction) / abs(angle_delta))
        if sensitivity > 1e-12:
            tolerance = min(tolerance, max(1e-8, 0.5 * degrees_per_substep * sensitivity))
    feedback_step = dict(step)
    feedback_step["command_value"] = angle_delta
    sim_goal = _sim_qc_from_quadcell_readout(target_qc, qc_readout_sign)
    feedback_step.update(qc1_error=float(sim_goal[0]), qc2_error=float(sim_goal[1]))
    return feedback_step, tolerance, {
        "goal_mode": "absolute_angle", "target_angle_deg": goal_angle,
        "axis_angle_error_before_deg": float(x_estimate[axis] - goal_angle),
        "qc_step_tolerance_used": tolerance, "model_goal_x": goal_x.tolist(),
        "original_plan_qc_x": _planned_step_qc_readout(step, qc_readout_sign).tolist(),
        "updated_goal_qc_x": target_qc.tolist(),
    }


def _execute_rotation_step_fixed(
        step,
        mapping,
        hardware,
        x_estimate,
        x_physical,
        base_mirrors,
        rotation_calibration,
        rng,
        *,
        dry_run,
        dry_run_rotation_error,
        qc_readout_sign,
        qc_step_tolerance,
        qc_safety_limit,
        fast_qc_avg,
        fast_qc_delay,
        max_rotation_chunks_per_step,
        max_rotation_chunk_substeps,
        min_rotation_chunk_substeps,
        rotation_settle_delay,
        qc_reader=None,
        dry_run_delta=None,
        sample_callback=None,
        stop_on_target_crossing=False):
    actuator_label = step["actuator"]
    axis_index = step["axis_index"]
    planned_angle_delta = float(step["command_value"])
    degrees_per_substep = rotation_calibration.get(
        actuator_label,
        DEFAULT_ROTATION_DEGREES_PER_SUBSTEP
    )
    if degrees_per_substep <= 0:
        raise ValueError(f"degrees_per_substep must be positive for {actuator_label}.")

    controller = mapping.get("controller", DEFAULT_ROTATION_CONTROLLER)
    actuator = int(mapping["actuator"])
    hardware_direction = int(np.sign(mapping.get("direction", 1)) or 1)
    angle_direction = int(np.sign(planned_angle_delta) or 1)
    target_qc_y = _planned_step_qc_readout(step, qc_readout_sign)

    def read_qc():
        if qc_reader is not None:
            return qc_reader()
        return _read_quadcell_y(
            hardware,
            times=fast_qc_avg,
            delay=fast_qc_delay,
            dry_run=dry_run,
            dry_run_x=x_physical,
            base_mirrors=base_mirrors,
            qc_readout_sign=qc_readout_sign,
        )

    before_qc = read_qc()
    current_y = before_qc["y"]
    start_y = current_y.copy()
    target_delta = target_qc_y - start_y
    target_norm = float(np.dot(target_delta, target_delta))
    best_distance = float(np.linalg.norm(target_qc_y - current_y))
    best_y = current_y.copy()
    chunks_without_improvement = 0
    total_commanded_substeps = 0
    total_sim_substeps = 0
    chunk_logs = []
    rollback_count = 0

    predicted_total_substeps = max(
        abs(planned_angle_delta) / degrees_per_substep,
        float(min_rotation_chunk_substeps)
    )
    base_chunk = int(np.ceil(predicted_total_substeps / 5.0))
    base_chunk = int(np.clip(
        base_chunk,
        int(min_rotation_chunk_substeps),
        int(max_rotation_chunk_substeps)
    ))

    stop_reason = None
    failure_reason = None
    if not before_qc.get("valid", True):
        stop_reason = "qc_signal_invalid"
        failure_reason = "QC signal invalid before staging move: " + _qc_invalid_details(before_qc) + "."
    elif best_distance <= qc_step_tolerance:
        stop_reason = "already_within_qc_step_tolerance"

    for chunk_index in range(1, max_rotation_chunks_per_step + 1):
        if stop_reason is not None:
            break

        remaining_distance = float(np.linalg.norm(target_qc_y - current_y))
        if remaining_distance <= qc_step_tolerance:
            stop_reason = "reached_qc_step_tolerance"
            break

        progress = 1.0
        if target_norm > 1e-12:
            progress = float(np.dot(current_y - start_y, target_delta) / target_norm)

        scale = 1.0
        if progress > 0.75 or remaining_distance < 2.0 * qc_step_tolerance:
            scale = 0.5
        if progress > 0.9 or remaining_distance < qc_step_tolerance:
            scale = 0.25

        chunk_substeps = max(
            int(min_rotation_chunk_substeps),
            int(np.ceil(base_chunk * scale))
        )
        sim_steps = angle_direction * chunk_substeps
        hardware_steps = hardware_direction * sim_steps
        x_physical_before_chunk = x_physical.copy()
        x_estimate_before_chunk = x_estimate.copy()

        if dry_run:
            if dry_run_delta is None:
                error_factor = 1.0 + rng.uniform(-dry_run_rotation_error, dry_run_rotation_error)
                actual_delta = sim_steps * degrees_per_substep * error_factor
            else:
                actual_delta = dry_run_delta(sim_steps)
            x_physical[axis_index] += actual_delta
        else:
            if hardware is None or getattr(hardware, "rotation_stages", None) is None:
                raise ValueError("hardware.rotation_stages is required to execute rotation moves.")
            hardware.rotation_stages.move_relative_steps(controller, actuator, hardware_steps)

        x_estimate[axis_index] += sim_steps * degrees_per_substep
        total_commanded_substeps += hardware_steps
        total_sim_substeps += sim_steps
        if rotation_settle_delay and rotation_settle_delay > 0:
            time.sleep(rotation_settle_delay)

        after_qc = read_qc()
        current_y = after_qc["y"]
        distance = float(np.linalg.norm(target_qc_y - current_y))
        progress = 1.0
        if target_norm > 1e-12:
            progress = float(np.dot(current_y - start_y, target_delta) / target_norm)

        improved = distance < best_distance
        if improved:
            best_distance = distance
            best_y = current_y.copy()
            chunks_without_improvement = 0
        else:
            chunks_without_improvement += 1

        chunk_log = {
            "chunk": chunk_index,
            "hardware_steps": int(hardware_steps),
            "sim_steps": int(sim_steps),
            "qc_x": current_y.tolist(),
            "qc_y": current_y.tolist(),
            "distance_to_target": distance,
            "progress": progress,
            "improved": bool(improved),
        }
        chunk_logs.append(chunk_log)
        if sample_callback is not None:
            sample_callback(chunk_log, after_qc)
        if not after_qc.get("valid", True):
            stop_reason = "qc_signal_invalid"
            failure_reason = "QC signal invalid during staging move: " + _qc_invalid_details(after_qc) + "."
            break

        if float(np.max(np.abs(current_y))) > qc_safety_limit:
            if dry_run:
                x_physical[:] = x_physical_before_chunk
            else:
                hardware.rotation_stages.move_relative_steps(controller, actuator, -hardware_steps)
            x_estimate[:] = x_estimate_before_chunk
            total_commanded_substeps -= hardware_steps
            total_sim_substeps -= sim_steps
            rollback_count += 1
            if rotation_settle_delay and rotation_settle_delay > 0:
                time.sleep(rotation_settle_delay)
            rollback_qc = read_qc()
            current_y = rollback_qc["y"]
            distance = float(np.linalg.norm(target_qc_y - current_y))
            chunk_log["rolled_back"] = True
            chunk_log["rollback_qc_x"] = current_y.tolist()
            chunk_log["rollback_qc_y"] = current_y.tolist()
            chunk_log["rollback_distance_to_target"] = distance
            best_distance = distance
            best_y = current_y.copy()
            stop_reason = "qc_safety_rollback"
            failure_reason = (
                f"Measured QC exceeded +/-{qc_safety_limit} mm during {actuator_label}."
            )
            break

        if distance <= qc_step_tolerance:
            stop_reason = "reached_qc_step_tolerance"
            break
        if stop_on_target_crossing and progress >= 1.0:
            stop_reason = "overshot_or_past_target"
            break
        if progress >= 1.0 and not improved:
            stop_reason = "overshot_or_past_target"
            break
        if chunks_without_improvement >= 3:
            stop_reason = "stalled_without_improvement"
            break

    if stop_reason is None:
        stop_reason = "max_rotation_chunks_reached"

    return {
        "kind": "rotation_fixed_plan",
        "controller": controller,
        "actuator": actuator,
        "planned_angle_delta": planned_angle_delta,
        "degrees_per_substep": degrees_per_substep,
        "target_qc_x": target_qc_y.tolist(),
        "target_qc_y": target_qc_y.tolist(),
        "start_qc_x": start_y.tolist(),
        "start_qc_y": start_y.tolist(),
        "final_qc_x": current_y.tolist(),
        "final_qc_y": current_y.tolist(),
        "best_qc_x": best_y.tolist(),
        "best_qc_y": best_y.tolist(),
        "distance_to_target": float(np.linalg.norm(target_qc_y - current_y)),
        "best_distance_to_target": float(best_distance),
        "total_commanded_substeps": int(total_commanded_substeps),
        "total_sim_substeps": int(total_sim_substeps),
        "rollback_count": int(rollback_count),
        "stop_reason": stop_reason,
        "failure_reason": failure_reason,
        "chunks": chunk_logs,
    }


def _path_metrics_from_x(x, base_mirrors, include_edge_ends=False):
    qc1_error, qc2_error = S.quadcell_errors_from_variables(x, *base_mirrors)
    mirrors = S.unpack_variables(x, *base_mirrors)
    edge_summary = S.reflection_edge_summary(
        x, *base_mirrors,
        include_ends=include_edge_ends
    )
    return {
        "OPD": float(S.OPD_from_variables(x, *base_mirrors)),
        "reflection_count": int(S.get_reflection_count(*mirrors)),
        "qc1_error": float(qc1_error),
        "qc2_error": float(qc2_error),
        "qc_difference": float(qc1_error - qc2_error),
        "min_reflection_u": float(edge_summary["min_u"]),
        "max_reflection_u": float(edge_summary["max_u"]),
        "closest_edge_margin": float(edge_summary["closest_edge_margin"]),
    }


def _planner_failure_is_final_qc_only(failure_reason):
    if failure_reason is None:
        return False
    text = str(failure_reason)
    return (
        text.startswith("Final QC offset ") and
        "exceeds final tolerance" in text
    )


def execute_OPD_closed_loop(
        target_OPD,
        M1,
        M2,
        M3,
        M4,
        hardware=None,
        *,
        actuator_map=None,
        rotation_calibration=None,
        M1_linear_loc=None,
        M2_linear_loc=None,
        M3_linear_loc=None,
        linear_stage_guard_mm=DEFAULT_LINEAR_STAGE_GUARD_MM,
        linear_stage_replan_tolerance_mm=DEFAULT_LINEAR_STAGE_REPLAN_TOLERANCE_MM,
        replan_every=5,
        qc_step_tolerance=0.15,
        qc_replan_tolerance=0.35,
        qc_safety_margin=0.2,
        min_qc_step_tolerance=0.03,
        clip_qc_target_to_safety=False,
        final_qc_tolerance=0.5,
        final_OPD_relaxed_tolerance=0.5,
        qc_detector_limit=3.9,
        qc_plan_limit=1.5,
        qc_hardware_stop=3.5,
        fast_qc_avg=3,
        fast_qc_delay=0.3,
        final_qc_avg=5,
        final_qc_delay=0.3,
        linear_settle_delay=1.0,
        qc_readout_sign=-1.0,
        max_replans=50,
        max_total_steps=300,
        target_OPD_tolerance=0.05,
        max_rotation_chunks_per_step=50,
        max_rotation_chunk_substeps=50,
        min_rotation_chunk_substeps=1,
        calibration_update_rate=0.2,
        calibration_clip_fraction=0.2,
        dry_run=False,
        dry_run_rotation_error=0.10,
        rng_seed=None,
        profile=True,
        profile_sink=None,
        choose_OPD_kwargs=None):
    """Execute an OPD change with real quadcell feedback around choose_OPD.

    Returns (mirrors, result, execution), mirroring choose_OPD's calling style.
    The execution dict contains hardware logs, planner logs, final state, and
    failure information.
    """
    if profile and profile_sink is None:
        profile_sink = print

    t0 = time.perf_counter()

    def log(message):
        if not profile:
            return
        line = f"[execute_OPD {time.perf_counter() - t0:.3f}s] {message}"
        profile_sink(line)

    actuator_map = _merged_actuator_map(actuator_map)
    active_linear_stages = _linear_stage_order_from_actuator_map(actuator_map)
    rotation_calibration = _normalized_rotation_calibration(rotation_calibration)
    rng = np.random.default_rng(rng_seed)
    choose_OPD_kwargs = dict(choose_OPD_kwargs or {})
    qc_hardware_stop = float(qc_hardware_stop)
    qc_detector_limit = float(qc_detector_limit)
    qc_plan_limit = float(qc_plan_limit)
    linear_stage_guard_mm = float(linear_stage_guard_mm)
    linear_stage_replan_tolerance_mm = float(linear_stage_replan_tolerance_mm)
    final_OPD_acceptance_tolerance = max(float(target_OPD_tolerance), float(final_OPD_relaxed_tolerance))

    base_mirrors = (
        np.array(M1, dtype=float),
        np.array(M2, dtype=float),
        np.array(M3, dtype=float),
        np.array(M4, dtype=float),
    )
    x_model = S.pack_variables(*base_mirrors)
    x_physical = x_model.copy()

    linear_stage_locs = _initial_linear_stage_locs(
        hardware,
        actuator_map,
        M1_linear_loc=M1_linear_loc,
        M2_linear_loc=M2_linear_loc,
        M3_linear_loc=M3_linear_loc,
        dry_run=dry_run,
        prefer_hardware_positions=True,
    )

    execution_log = []
    planner_runs = []
    failure_reason = None
    final_res = SimpleNamespace(success=False, message="Closed-loop execution did not finish.")
    latest_plan = None
    total_accepted_steps = 0
    replan_reason = "initial"

    log(
        f"start target_OPD={target_OPD:.3f} dry_run={dry_run} "
        f"linear_locs={linear_stage_locs} "
        f"linear_stage_guard={linear_stage_guard_mm:.3f}"
    )

    for replan_index in range(1, max_replans + 1):
        current_mirrors = _x_to_mirrors(x_model, base_mirrors)
        current_OPD = S.OPD_from_variables(x_model, *base_mirrors)
        log(
            f"plan replan={replan_index} reason={replan_reason} "
            f"OPD={current_OPD:.3f}"
        )

        planner_profile = []
        planner_kwargs = dict(choose_OPD_kwargs)
        planner_kwargs.setdefault("qc_detector_limit", qc_detector_limit)
        planner_kwargs.setdefault("qc_plan_limit", qc_plan_limit)
        planner_kwargs.setdefault("qc_hardware_stop", qc_hardware_stop)
        planner_kwargs.setdefault("final_OPD_relaxed_tolerance", final_OPD_relaxed_tolerance)
        planner_kwargs.setdefault("final_center_qc_priority", True)
        planner_kwargs.setdefault("moving_linear_stages", active_linear_stages)
        planner_kwargs.setdefault("linear_stage_guard_mm", linear_stage_guard_mm)
        mirrors_opt, final_res, latest_plan = S.choose_OPD(
            target_OPD,
            *current_mirrors,
            return_actuation_plan=True,
            final_center_qc_threshold=final_qc_tolerance,
            final_qc_tolerance=final_qc_tolerance,
            target_OPD_tolerance=target_OPD_tolerance,
            M1_linear_loc=linear_stage_locs["M1"],
            M2_linear_loc=linear_stage_locs["M2"],
            M3_linear_loc=linear_stage_locs.get("M3"),
            profile=bool(profile),
            profile_sink=planner_profile.append,
            **planner_kwargs,
        )
        planner_runs.append({
            "replan": replan_index,
            "reason": replan_reason,
            "plan": latest_plan,
            "profile": planner_profile,
        })

        if latest_plan.get("failure_reason") is not None:
            failure_reason = "Planner failed: " + latest_plan["failure_reason"]
            log(failure_reason)
            break

        steps = latest_plan.get("steps", [])
        if len(steps) == 0:
            replan_reason = "planner_returned_no_steps"
            break

        accepted_this_plan = 0
        replan_reason = None

        for step in steps:
            if total_accepted_steps >= max_total_steps:
                failure_reason = f"Reached max_total_steps={max_total_steps}."
                break

            actuator_label = step.get("actuator")
            if actuator_label not in actuator_map:
                failure_reason = f"No hardware mapping for actuator {actuator_label}."
                break

            mapping = actuator_map[actuator_label]
            step_t0 = time.perf_counter()
            before_qc = _read_quadcell_y(
                hardware,
                times=fast_qc_avg,
                delay=fast_qc_delay,
                dry_run=dry_run,
                dry_run_x=x_physical,
                base_mirrors=base_mirrors,
                qc_readout_sign=qc_readout_sign,
            )

            try:
                if mapping["kind"] == "linear":
                    detail = _execute_linear_step(
                        step,
                        mapping,
                        hardware,
                        x_model,
                        x_physical,
                        linear_stage_locs,
                        dry_run=dry_run,
                        linear_settle_delay=linear_settle_delay,
                        linear_stage_guard_mm=linear_stage_guard_mm,
                        linear_stage_replan_tolerance_mm=linear_stage_replan_tolerance_mm,
                    )
                elif mapping["kind"] == "rotation":
                    detail = _execute_rotation_step(
                        step,
                        mapping,
                        hardware,
                        x_model,
                        x_physical,
                        base_mirrors,
                        rotation_calibration,
                        rng,
                        dry_run=dry_run,
                        dry_run_rotation_error=dry_run_rotation_error,
                        qc_readout_sign=qc_readout_sign,
                        qc_step_tolerance=qc_step_tolerance,
                        qc_replan_tolerance=qc_replan_tolerance,
                        max_qc_error=qc_hardware_stop,
                        qc_plan_limit=qc_plan_limit,
                        qc_safety_margin=qc_safety_margin,
                        min_qc_step_tolerance=min_qc_step_tolerance,
                        clip_qc_target_to_safety=clip_qc_target_to_safety,
                        fast_qc_avg=fast_qc_avg,
                        fast_qc_delay=fast_qc_delay,
                        max_rotation_chunks_per_step=max_rotation_chunks_per_step,
                        max_rotation_chunk_substeps=max_rotation_chunk_substeps,
                        min_rotation_chunk_substeps=min_rotation_chunk_substeps,
                        calibration_update_rate=calibration_update_rate,
                        calibration_clip_fraction=calibration_clip_fraction,
                    )
                else:
                    failure_reason = f"Unknown actuator kind {mapping['kind']} for {actuator_label}."
                    break
            except Exception as exc:
                failure_reason = f"Hardware execution failed for {actuator_label}: {exc}"
                break

            after_qc = _read_quadcell_y(
                hardware,
                times=fast_qc_avg,
                delay=fast_qc_delay,
                dry_run=dry_run,
                dry_run_x=x_physical,
                base_mirrors=base_mirrors,
                qc_readout_sign=qc_readout_sign,
            )
            sim_qc = np.array(
                S.quadcell_errors_from_variables(x_model, *base_mirrors),
                dtype=float
            )
            current_OPD = S.OPD_from_variables(x_model, *base_mirrors)

            total_accepted_steps += 1
            accepted_this_plan += 1

            entry = {
                "execution_step": total_accepted_steps,
                "planner_replan": replan_index,
                "planner_step": step.get("step"),
                "actuator": actuator_label,
                "command_value": step.get("command_value"),
                "planned_OPD": step.get("OPD"),
                "actual_model_OPD": current_OPD,
                "planned_qc_x": _planned_step_qc_readout(step, qc_readout_sign).tolist(),
                "planned_qc_y": _planned_step_qc_readout(step, qc_readout_sign).tolist(),
                "before_qc_raw": before_qc["raw"],
                "before_qc_x": before_qc["x"].tolist(),
                "before_qc_y": before_qc["y"].tolist(),
                "after_qc_raw": after_qc["raw"],
                "after_qc_x": after_qc["x"].tolist(),
                "after_qc_y": after_qc["y"].tolist(),
                "model_sim_qc": sim_qc.tolist(),
                "linear_stage_locs": dict(linear_stage_locs),
                "detail": detail,
                "dt": time.perf_counter() - step_t0,
            }
            execution_log.append(entry)
            log(
                f"step={total_accepted_steps} actuator={actuator_label} "
                f"OPD={current_OPD:.3f} qc_x=({after_qc['x'][0]:.3f},{after_qc['x'][1]:.3f})"
            )

            if detail.get("replan_recommended"):
                distance_to_target = detail.get("distance_to_target")
                detail_reason = detail.get("replan_reason")
                if distance_to_target is not None:
                    replan_reason = (
                        f"{actuator_label} QC target miss "
                        f"{float(distance_to_target):.3f} mm"
                    )
                elif detail_reason:
                    replan_reason = f"{actuator_label} {detail_reason}"
                else:
                    replan_reason = f"{actuator_label} requested replanning"
                break

            if accepted_this_plan >= replan_every:
                replan_reason = f"accepted {accepted_this_plan} steps from current plan"
                break

            if (
                abs(current_OPD - target_OPD) <= final_OPD_acceptance_tolerance and
                max(abs(after_qc["y"][0]), abs(after_qc["y"][1])) <= final_qc_tolerance
            ):
                replan_reason = "target_reached"
                break

        if failure_reason is not None:
            log(failure_reason)
            break

        final_qc = _read_quadcell_y(
            hardware,
            times=final_qc_avg if replan_reason == "target_reached" else fast_qc_avg,
            delay=final_qc_delay if replan_reason == "target_reached" else fast_qc_delay,
            dry_run=dry_run,
            dry_run_x=x_physical,
            base_mirrors=base_mirrors,
            qc_readout_sign=qc_readout_sign,
        )
        current_OPD = S.OPD_from_variables(x_model, *base_mirrors)

        if (
            abs(current_OPD - target_OPD) <= final_OPD_acceptance_tolerance and
            max(abs(final_qc["y"][0]), abs(final_qc["y"][1])) <= final_qc_tolerance
        ):
            replan_reason = "target_reached"
            final_res = SimpleNamespace(
                success=True,
                message="Closed-loop OPD execution reached target tolerances."
            )
            log(
                f"done OPD={current_OPD:.3f} target={target_OPD:.3f} "
                f"qc_x=({final_qc['x'][0]:.3f},{final_qc['x'][1]:.3f})"
            )
            break

        if replan_reason is None:
            replan_reason = "plan_steps_exhausted_before_tolerance"

    else:
        failure_reason = f"Reached max_replans={max_replans}."

    final_mirrors = _x_to_mirrors(x_model, base_mirrors)
    final_OPD = S.OPD_from_variables(x_model, *base_mirrors)
    final_sim_qc = S.quadcell_errors_from_variables(x_model, *base_mirrors)
    final_res = S.set_OPD_result_full_x(final_res, *final_mirrors)
    final_success = failure_reason is None and getattr(final_res, "success", False)
    measured_qc_values = []
    rollback_count = 0
    for entry in execution_log:
        for key in ("before_qc_y", "after_qc_y"):
            if key in entry:
                measured_qc_values.extend(abs(float(v)) for v in entry[key])
        detail = entry.get("detail", {})
        rollback_count += sum(1 for chunk in detail.get("chunks", []) if chunk.get("rolled_back"))
    max_abs_measured_qc = max(measured_qc_values) if measured_qc_values else 0.0

    execution = {
        "success": bool(final_success),
        "failure_reason": failure_reason,
        "target_OPD": float(target_OPD),
        "final_OPD": float(final_OPD),
        "final_OPD_error": float(final_OPD - target_OPD),
        "final_sim_qc": [float(final_sim_qc[0]), float(final_sim_qc[1])],
        "qc_readout_axis": "x",
        "max_abs_measured_qc": float(max_abs_measured_qc),
        "rollback_count": int(rollback_count),
        "linear_stage_locs": dict(linear_stage_locs),
        "moving_linear_stages": list(active_linear_stages),
        "rotation_calibration": dict(rotation_calibration),
        "execution_log": execution_log,
        "planner_runs": planner_runs,
        "actuation_plan": latest_plan,
        "dry_run": bool(dry_run),
        "replan_every": int(replan_every),
        "qc_step_tolerance": float(qc_step_tolerance),
        "qc_replan_tolerance": float(qc_replan_tolerance),
        "qc_safety_margin": float(qc_safety_margin),
        "qc_detector_limit": float(qc_detector_limit),
        "qc_plan_limit": float(qc_plan_limit),
        "qc_hardware_stop": float(qc_hardware_stop),
        "linear_stage_guard_mm": float(linear_stage_guard_mm),
        "linear_stage_replan_tolerance_mm": float(linear_stage_replan_tolerance_mm),
        "min_qc_step_tolerance": float(min_qc_step_tolerance),
        "clip_qc_target_to_safety": bool(clip_qc_target_to_safety),
        "final_qc_tolerance": float(final_qc_tolerance),
        "final_OPD_relaxed_tolerance": float(final_OPD_relaxed_tolerance),
        "final_OPD_acceptance_tolerance": float(final_OPD_acceptance_tolerance),
        "fast_qc_avg": int(fast_qc_avg),
        "fast_qc_delay": float(fast_qc_delay),
        "final_qc_avg": int(final_qc_avg),
        "final_qc_delay": float(final_qc_delay),
        "linear_settle_delay": float(linear_settle_delay),
    }

    if not final_success and failure_reason is None:
        execution["failure_reason"] = "Closed-loop execution stopped before success criteria were met."

    return final_mirrors, final_res, execution


def _coerce_profile_step_interval(profile_step_interval):
    if profile_step_interval is None:
        return None
    interval = int(profile_step_interval)
    if interval < 0:
        raise ValueError("profile_step_interval must be non-negative or None.")
    return interval


def _should_log_fixed_plan_step(step_index, n_steps, actuator_label, mapping_kind, profile_step_interval, detail):
    if profile_step_interval is None:
        return True
    if step_index == 1 or step_index == n_steps:
        return True
    if mapping_kind == "linear":
        return True
    if detail.get("failure_reason") is not None or detail.get("replan_recommended"):
        return True
    if profile_step_interval <= 0:
        return False
    return (step_index % profile_step_interval) == 0


def _prepare_OPD_fixed_plan(
        target_OPD,
        M1,
        M2,
        M3,
        M4,
        hardware,
        *,
        actuator_map=None,
        M1_linear_loc=None,
        M2_linear_loc=None,
        M3_linear_loc=None,
        linear_stage_guard_mm=DEFAULT_LINEAR_STAGE_GUARD_MM,
        qc_plan_limit=1.5,
        qc_detector_limit=3.9,
        qc_hardware_stop=3.5,
        final_qc_tolerance=0.5,
        final_OPD_tolerance=0.5,
        dry_run=False,
        profile=True,
        choose_OPD_kwargs=None):
    actuator_map = _merged_actuator_map(actuator_map)
    active_linear_stages = _linear_stage_order_from_actuator_map(actuator_map)
    choose_OPD_kwargs = dict(choose_OPD_kwargs or {})
    qc_plan_limit = float(qc_plan_limit)
    qc_detector_limit = float(qc_detector_limit)
    qc_hardware_stop = float(qc_hardware_stop)
    linear_stage_guard_mm = float(linear_stage_guard_mm)

    base_mirrors = (
        np.array(M1, dtype=float),
        np.array(M2, dtype=float),
        np.array(M3, dtype=float),
        np.array(M4, dtype=float),
    )
    x_estimate = S.pack_variables(*base_mirrors)
    x_physical = x_estimate.copy()

    linear_stage_locs = _initial_linear_stage_locs(
        hardware,
        actuator_map,
        M1_linear_loc=M1_linear_loc,
        M2_linear_loc=M2_linear_loc,
        M3_linear_loc=M3_linear_loc,
        dry_run=dry_run,
        prefer_hardware_positions=True,
    )

    planner_kwargs = dict(choose_OPD_kwargs)
    planner_kwargs.setdefault("qc_plan_limit", qc_plan_limit)
    planner_kwargs.setdefault("qc_detector_limit", qc_detector_limit)
    planner_kwargs.setdefault("qc_hardware_stop", qc_hardware_stop)
    planner_kwargs.setdefault("final_qc_tolerance", final_qc_tolerance)
    planner_kwargs.setdefault("final_center_qc_threshold", final_qc_tolerance)
    planner_kwargs.setdefault("final_OPD_relaxed_tolerance", final_OPD_tolerance)
    planner_kwargs.setdefault("final_center_qc_priority", True)
    planner_kwargs.setdefault("moving_linear_stages", active_linear_stages)
    planner_kwargs.setdefault("linear_stage_guard_mm", linear_stage_guard_mm)

    planner_profile = []
    mirrors_opt, planner_res, actuation_plan = S.choose_OPD(
        target_OPD,
        *base_mirrors,
        return_actuation_plan=True,
        M1_linear_loc=linear_stage_locs["M1"],
        M2_linear_loc=linear_stage_locs["M2"],
        M3_linear_loc=linear_stage_locs.get("M3"),
        profile=bool(profile),
        profile_sink=planner_profile.append,
        **planner_kwargs,
    )

    return {
        "actuator_map": actuator_map,
        "active_linear_stages": active_linear_stages,
        "base_mirrors": base_mirrors,
        "x_estimate": x_estimate,
        "x_physical": x_physical,
        "linear_stage_locs": linear_stage_locs,
        "planner_kwargs": planner_kwargs,
        "planner_profile": planner_profile,
        "mirrors_opt": mirrors_opt,
        "planner_res": planner_res,
        "actuation_plan": actuation_plan,
    }


def plan_OPD_fixed_plan(
        target_OPD,
        M1,
        M2,
        M3,
        M4,
        hardware=None,
        *,
        actuator_map=None,
        rotation_calibration=None,
        M1_linear_loc=None,
        M2_linear_loc=None,
        M3_linear_loc=None,
        linear_stage_guard_mm=DEFAULT_LINEAR_STAGE_GUARD_MM,
        linear_stage_replan_tolerance_mm=DEFAULT_LINEAR_STAGE_REPLAN_TOLERANCE_MM,
        qc_plan_limit=1.5,
        qc_detector_limit=3.9,
        qc_hardware_stop=3.5,
        qc_step_tolerance=0.15,
        final_qc_tolerance=0.5,
        final_OPD_tolerance=0.5,
        require_final_qc=False,
        allow_final_qc_planner_failure=True,
        fast_qc_avg=3,
        fast_qc_delay=0.04,
        final_qc_avg=5,
        final_qc_delay=0.04,
        linear_settle_delay=2.0,
        rotation_settle_delay=0.04,
        step_settle_delay=0.04,
        qc_readout_sign=-1.0,
        max_total_steps=300,
        max_rotation_chunks_per_step=50,
        max_rotation_chunk_substeps=50,
        min_rotation_chunk_substeps=1,
        dry_run=False,
        dry_run_rotation_error=0.10,
        rng_seed=None,
        profile=True,
        profile_sink=None,
        profile_step_interval=1,
        choose_OPD_kwargs=None,
        **legacy_hardware_kwargs):
    """Preview the exact plan setup used by execute_OPD_fixed_plan."""
    del (
        rotation_calibration,
        linear_stage_replan_tolerance_mm,
        qc_step_tolerance,
        require_final_qc,
        fast_qc_avg,
        fast_qc_delay,
        final_qc_avg,
        final_qc_delay,
        linear_settle_delay,
        rotation_settle_delay,
        step_settle_delay,
        qc_readout_sign,
        max_rotation_chunks_per_step,
        max_rotation_chunk_substeps,
        min_rotation_chunk_substeps,
        dry_run_rotation_error,
        rng_seed,
        profile_step_interval,
    )
    if profile and profile_sink is None:
        profile_sink = print

    t0 = time.perf_counter()

    def log(message):
        if not profile:
            return
        profile_sink(f"[plan_fixed_OPD {time.perf_counter() - t0:.3f}s] {message}")

    choose_OPD_kwargs = dict(choose_OPD_kwargs or {})
    if "qc_safety_limit" in legacy_hardware_kwargs:
        qc_hardware_stop = legacy_hardware_kwargs.pop("qc_safety_limit")
    if len(legacy_hardware_kwargs) > 0:
        unknown = ", ".join(sorted(legacy_hardware_kwargs))
        raise TypeError(f"plan_OPD_fixed_plan() got unexpected keyword argument(s): {unknown}")

    setup = _prepare_OPD_fixed_plan(
        target_OPD,
        M1,
        M2,
        M3,
        M4,
        hardware,
        actuator_map=actuator_map,
        M1_linear_loc=M1_linear_loc,
        M2_linear_loc=M2_linear_loc,
        M3_linear_loc=M3_linear_loc,
        linear_stage_guard_mm=linear_stage_guard_mm,
        qc_plan_limit=qc_plan_limit,
        qc_detector_limit=qc_detector_limit,
        qc_hardware_stop=qc_hardware_stop,
        final_qc_tolerance=final_qc_tolerance,
        final_OPD_tolerance=final_OPD_tolerance,
        dry_run=dry_run,
        profile=profile,
        choose_OPD_kwargs=choose_OPD_kwargs,
    )
    actuation_plan = setup["actuation_plan"]
    steps = actuation_plan.get("steps", [])
    planner_failure_reason = actuation_plan.get("failure_reason")
    planner_failure_ignored = bool(
        planner_failure_reason is not None and
        allow_final_qc_planner_failure and
        _planner_failure_is_final_qc_only(planner_failure_reason) and
        len(steps) > 0
    )
    failure_reason = None
    if planner_failure_reason is not None and not planner_failure_ignored:
        failure_reason = "Planner failed: " + planner_failure_reason
    elif len(steps) == 0:
        failure_reason = "Planner returned no actuation steps."
    max_total_steps_would_stop = len(steps) > int(max_total_steps)

    actuation_plan["fixed_plan_preview"] = True
    actuation_plan["execution_preview"] = {
        "planned_steps": int(len(steps)),
        "max_total_steps": int(max_total_steps),
        "max_total_steps_would_stop": bool(max_total_steps_would_stop),
        "failure_reason": failure_reason,
        "planner_failure_ignored": bool(planner_failure_ignored),
        "planner_failure_reason": planner_failure_reason,
        "linear_stage_locs": dict(setup["linear_stage_locs"]),
        "moving_linear_stages": list(setup["active_linear_stages"]),
        "linear_stage_guard_mm": float(linear_stage_guard_mm),
        "planner_kwargs": dict(setup["planner_kwargs"]),
        "planner_profile": setup["planner_profile"],
    }
    log(
        f"target_OPD={target_OPD:.3f} planned_steps={len(steps)} "
        f"max_total_steps={int(max_total_steps)} "
        f"would_stop_at_max_total_steps={max_total_steps_would_stop} "
        f"linear_locs={setup['linear_stage_locs']} "
        f"linear_stage_guard={float(linear_stage_guard_mm):.3f} "
        f"failure={failure_reason}"
    )
    return setup["mirrors_opt"], setup["planner_res"], actuation_plan


def execute_OPD_fixed_plan(
        target_OPD,
        M1,
        M2,
        M3,
        M4,
        hardware=None,
        *,
        actuator_map=None,
        rotation_calibration=None,
        M1_linear_loc=None,
        M2_linear_loc=None,
        M3_linear_loc=None,
        linear_stage_guard_mm=DEFAULT_LINEAR_STAGE_GUARD_MM,
        linear_stage_replan_tolerance_mm=DEFAULT_LINEAR_STAGE_REPLAN_TOLERANCE_MM,
        qc_plan_limit=1.5,
        qc_detector_limit=3.9,
        qc_hardware_stop=3.5,
        qc_step_tolerance=0.15,
        final_qc_tolerance=0.5,
        final_OPD_tolerance=0.5,
        require_final_qc=False,
        allow_final_qc_planner_failure=True,
        fast_qc_avg=3,
        fast_qc_delay=0.04,
        final_qc_avg=5,
        final_qc_delay=0.04,
        linear_settle_delay=2.0,
        rotation_settle_delay=0.04,
        step_settle_delay=0.04,
        qc_readout_sign=-1.0,
        max_total_steps=300,
        max_rotation_chunks_per_step=50,
        max_rotation_chunk_substeps=50,
        min_rotation_chunk_substeps=1,
        dry_run=False,
        dry_run_rotation_error=0.10,
        rng_seed=None,
        profile=True,
        profile_sink=None,
        profile_step_interval=1,
        choose_OPD_kwargs=None,
        **legacy_hardware_kwargs):
    """Execute one precomputed choose_OPD plan without intermediate replanning.

    Returned execution records, including failed runs, can be passed to
    plot_fixed_plan_quadcell_overlay and plot_fixed_plan_reflection_u_overlay.
    Rotation chunks retain measured QCs and estimated/dry-run angle snapshots.
    """
    if profile and profile_sink is None:
        profile_sink = print

    t0 = time.perf_counter()

    def log(message):
        if not profile:
            return
        profile_sink(f"[execute_fixed_OPD {time.perf_counter() - t0:.3f}s] {message}")

    rotation_calibration = _normalized_rotation_calibration(rotation_calibration)
    rng = np.random.default_rng(rng_seed)
    choose_OPD_kwargs = dict(choose_OPD_kwargs or {})
    if "qc_safety_limit" in legacy_hardware_kwargs:
        qc_hardware_stop = legacy_hardware_kwargs.pop("qc_safety_limit")
    if len(legacy_hardware_kwargs) > 0:
        unknown = ", ".join(sorted(legacy_hardware_kwargs))
        raise TypeError(f"execute_OPD_fixed_plan() got unexpected keyword argument(s): {unknown}")
    qc_plan_limit = float(qc_plan_limit)
    qc_detector_limit = float(qc_detector_limit)
    qc_hardware_stop = float(qc_hardware_stop)
    linear_stage_guard_mm = float(linear_stage_guard_mm)
    linear_stage_replan_tolerance_mm = float(linear_stage_replan_tolerance_mm)
    profile_step_interval = _coerce_profile_step_interval(profile_step_interval)

    setup = _prepare_OPD_fixed_plan(
        target_OPD,
        M1,
        M2,
        M3,
        M4,
        hardware,
        actuator_map=actuator_map,
        M1_linear_loc=M1_linear_loc,
        M2_linear_loc=M2_linear_loc,
        M3_linear_loc=M3_linear_loc,
        linear_stage_guard_mm=linear_stage_guard_mm,
        qc_plan_limit=qc_plan_limit,
        qc_detector_limit=qc_detector_limit,
        qc_hardware_stop=qc_hardware_stop,
        final_qc_tolerance=final_qc_tolerance,
        final_OPD_tolerance=final_OPD_tolerance,
        dry_run=dry_run,
        profile=profile,
        choose_OPD_kwargs=choose_OPD_kwargs,
    )
    actuator_map = setup["actuator_map"]
    active_linear_stages = setup["active_linear_stages"]
    base_mirrors = setup["base_mirrors"]
    x_estimate = setup["x_estimate"]
    x_physical = setup["x_physical"]
    linear_stage_locs = setup["linear_stage_locs"]
    planner_profile = setup["planner_profile"]
    mirrors_opt = setup["mirrors_opt"]
    planner_res = setup["planner_res"]
    actuation_plan = setup["actuation_plan"]

    steps = actuation_plan.get("steps", [])
    failure_reason = None
    planner_failure_reason = actuation_plan.get("failure_reason")
    planner_failure_ignored = False
    if actuation_plan.get("failure_reason") is not None:
        if (
            allow_final_qc_planner_failure and
            _planner_failure_is_final_qc_only(planner_failure_reason) and
            len(steps) > 0
        ):
            planner_failure_ignored = True
        else:
            failure_reason = "Planner failed: " + actuation_plan["failure_reason"]
    elif len(steps) == 0:
        failure_reason = "Planner returned no actuation steps."

    log(
        f"start target_OPD={target_OPD:.3f} dry_run={dry_run} "
        f"planned_steps={len(steps)} failure={failure_reason} "
        f"ignored_planner_failure={planner_failure_ignored} "
        f"linear_locs={linear_stage_locs} "
        f"linear_stage_guard={linear_stage_guard_mm:.3f}"
    )

    execution_log = []

    for step_index, step in enumerate(steps, start=1):
        if failure_reason is not None:
            break
        if step_index > max_total_steps:
            failure_reason = f"Reached max_total_steps={max_total_steps}."
            break

        if step_index > 1 and step_settle_delay and step_settle_delay > 0:
            time.sleep(step_settle_delay)

        actuator_label = step.get("actuator")
        if actuator_label not in actuator_map:
            failure_reason = f"No hardware mapping for actuator {actuator_label}."
            break

        mapping = actuator_map[actuator_label]
        step_t0 = time.perf_counter()
        before_qc = _read_quadcell_y(
            hardware,
            times=fast_qc_avg,
            delay=fast_qc_delay,
            dry_run=dry_run,
            dry_run_x=x_physical,
            base_mirrors=base_mirrors,
            qc_readout_sign=qc_readout_sign,
        )

        estimate_x_before_step = x_estimate.copy()
        physical_x_before_step = x_physical.copy() if dry_run else None
        rotation_samples = []

        def record_rotation_sample(sample, qc):
            sample.update(
                estimate_x_after_pulse=x_estimate.tolist(),
                physical_x_after_pulse=x_physical.tolist() if dry_run else None,
                qc_signal_valid=bool(qc.get("valid", True)),
            )
            rotation_samples.append(sample)

        try:
            if mapping["kind"] == "linear":
                detail = _execute_linear_step(
                    step,
                    mapping,
                    hardware,
                    x_estimate,
                    x_physical,
                    linear_stage_locs,
                    dry_run=dry_run,
                    linear_settle_delay=linear_settle_delay,
                    linear_stage_guard_mm=linear_stage_guard_mm,
                    linear_stage_replan_tolerance_mm=linear_stage_replan_tolerance_mm,
                )
            elif mapping["kind"] == "rotation":
                detail = _execute_rotation_step_fixed(
                    step,
                    mapping,
                    hardware,
                    x_estimate,
                    x_physical,
                    base_mirrors,
                    rotation_calibration,
                    rng,
                    dry_run=dry_run,
                    dry_run_rotation_error=dry_run_rotation_error,
                    qc_readout_sign=qc_readout_sign,
                    qc_step_tolerance=qc_step_tolerance,
                    qc_safety_limit=qc_hardware_stop,
                    fast_qc_avg=fast_qc_avg,
                    fast_qc_delay=fast_qc_delay,
                    max_rotation_chunks_per_step=max_rotation_chunks_per_step,
                    max_rotation_chunk_substeps=max_rotation_chunk_substeps,
                    min_rotation_chunk_substeps=min_rotation_chunk_substeps,
                    rotation_settle_delay=rotation_settle_delay,
                    sample_callback=record_rotation_sample,
                )
            else:
                failure_reason = f"Unknown actuator kind {mapping['kind']} for {actuator_label}."
                break
        except Exception as exc:
            failure_reason = f"Hardware execution failed for {actuator_label}: {exc}"
            execution_log.append({
                "execution_step": step_index,
                "planner_step": step.get("step"),
                "actuator": actuator_label,
                "before_qc_raw": before_qc["raw"],
                "before_qc_x": before_qc["x"].tolist(),
                "before_qc_y": before_qc["y"].tolist(),
                "estimate_x_before_step": estimate_x_before_step.tolist(),
                "physical_x_before_step": (
                    None if physical_x_before_step is None else physical_x_before_step.tolist()
                ),
                "estimate_x_after_step": x_estimate.tolist(),
                "physical_x_after_step": x_physical.tolist() if dry_run else None,
                "failure_reason": failure_reason,
                "detail": {"failure_reason": failure_reason, "chunks": rotation_samples},
                "dt": time.perf_counter() - step_t0,
            })
            break

        after_qc = _read_quadcell_y(
            hardware,
            times=fast_qc_avg,
            delay=fast_qc_delay,
            dry_run=dry_run,
            dry_run_x=x_physical,
            base_mirrors=base_mirrors,
            qc_readout_sign=qc_readout_sign,
        )
        measured_sim_qc = _sim_qc_from_quadcell_readout(after_qc["y"], qc_readout_sign)
        pre_assimilation_x = x_estimate.copy()
        pre_assimilation_sim_qc = np.array(
            S.quadcell_errors_from_variables(pre_assimilation_x, *base_mirrors),
            dtype=float
        )
        assimilation = None
        if mapping["kind"] == "rotation":
            axis_index = step.get("axis_index")
            try:
                x_fit, assimilation = assimilate_rotation_angle_from_qc(
                    x_estimate,
                    axis_index,
                    after_qc["y"],
                    *base_mirrors,
                    qc_readout_sign=qc_readout_sign,
                    angle_prior=x_estimate[axis_index],
                )
                x_estimate[:] = x_fit
                assimilation["applied"] = True
                assimilation["angle_delta_from_dead_reckoned"] = float(
                    x_estimate[axis_index] - pre_assimilation_x[axis_index]
                )
            except Exception as exc:
                assimilation = {
                    "success": False,
                    "applied": False,
                    "message": str(exc),
                    "angle_delta_from_dead_reckoned": 0.0,
                }
        post_assimilation_sim_qc = np.array(
            S.quadcell_errors_from_variables(x_estimate, *base_mirrors),
            dtype=float
        )
        planned_sim_qc = np.array([step["qc1_error"], step["qc2_error"]], dtype=float)
        planned_qc_y = _planned_step_qc_readout(step, qc_readout_sign)
        target_miss = float(np.linalg.norm(planned_qc_y - after_qc["y"]))
        estimate_sim_qc = post_assimilation_sim_qc.copy()
        physical_OPD = S.OPD_from_variables(x_physical, *base_mirrors) if dry_run else None
        estimate_OPD = S.OPD_from_variables(x_estimate, *base_mirrors)
        planned_OPD = float(step.get("OPD", np.nan))
        estimate_metrics = _path_metrics_from_x(
            x_estimate,
            base_mirrors,
            include_edge_ends=actuation_plan.get("include_edge_ends", False),
        )
        physical_metrics = None
        if dry_run:
            physical_metrics = _path_metrics_from_x(
                x_physical,
                base_mirrors,
                include_edge_ends=actuation_plan.get("include_edge_ends", False),
            )

        entry = {
            "execution_step": step_index,
            "planner_step": step.get("step"),
            "actuator": actuator_label,
            "command_value": step.get("command_value"),
            "planned_OPD": planned_OPD,
            "estimate_OPD": float(estimate_OPD),
            "physical_OPD": None if physical_OPD is None else float(physical_OPD),
            "OPD_divergence": None if physical_OPD is None else float(physical_OPD - planned_OPD),
            "planned_qc_x": planned_qc_y.tolist(),
            "planned_qc_y": planned_qc_y.tolist(),
            "before_qc_raw": before_qc["raw"],
            "before_qc_x": before_qc["x"].tolist(),
            "before_qc_y": before_qc["y"].tolist(),
            "estimate_x_before_step": estimate_x_before_step.tolist(),
            "physical_x_before_step": (
                None if physical_x_before_step is None else physical_x_before_step.tolist()
            ),
            "estimate_x_after_step": x_estimate.tolist(),
            "physical_x_after_step": x_physical.tolist() if dry_run else None,
            "after_qc_raw": after_qc["raw"],
            "after_qc_x": after_qc["x"].tolist(),
            "after_qc_y": after_qc["y"].tolist(),
            "planned_sim_qc": planned_sim_qc.tolist(),
            "measured_sim_qc": measured_sim_qc.tolist(),
            "estimate_sim_qc": estimate_sim_qc.tolist(),
            "pre_assimilation_sim_qc": pre_assimilation_sim_qc.tolist(),
            "post_assimilation_sim_qc": post_assimilation_sim_qc.tolist(),
            "assimilation": assimilation,
            "estimate_path_metrics": estimate_metrics,
            "physical_path_metrics": physical_metrics,
            "qc_target_miss": target_miss,
            "qc_divergence": (measured_sim_qc - planned_sim_qc).tolist(),
            "linear_stage_locs": dict(linear_stage_locs),
            "detail": detail,
            "dt": time.perf_counter() - step_t0,
        }
        execution_log.append(entry)
        if _should_log_fixed_plan_step(
            step_index,
            len(steps),
            actuator_label,
            mapping.get("kind"),
            profile_step_interval,
            detail,
        ):
            log(
                f"step={step_index}/{len(steps)} actuator={actuator_label} "
                f"miss={target_miss:.3f} qc_x=({after_qc['x'][0]:.3f},{after_qc['x'][1]:.3f})"
            )

        if detail.get("failure_reason") is not None:
            failure_reason = detail["failure_reason"]
            break
        if detail.get("replan_recommended"):
            detail_reason = detail.get("replan_reason") or "requested replanning"
            failure_reason = (
                f"{actuator_label} {detail_reason}; fixed plan stopped after the "
                "clipped move so a new plan can be computed from the current hardware state."
            )
            break
        if float(np.max(np.abs(after_qc["y"]))) > qc_hardware_stop:
            failure_reason = (
                f"Measured QC exceeded +/-{qc_hardware_stop} mm after {actuator_label}."
            )
            break

    final_qc = _read_quadcell_y(
        hardware,
        times=final_qc_avg,
        delay=final_qc_delay,
        dry_run=dry_run,
        dry_run_x=x_physical,
        base_mirrors=base_mirrors,
        qc_readout_sign=qc_readout_sign,
    )
    if execution_log and "after_qc_x" not in execution_log[-1]:
        execution_log[-1].update(
            after_qc_raw=final_qc["raw"],
            after_qc_x=final_qc["x"].tolist(),
            after_qc_y=final_qc["y"].tolist(),
        )
    final_x_for_report = x_estimate
    final_mirrors = _x_to_mirrors(final_x_for_report, base_mirrors)
    final_OPD = S.OPD_from_variables(final_x_for_report, *base_mirrors)
    final_sim_qc = S.quadcell_errors_from_variables(final_x_for_report, *base_mirrors)
    final_OPD_error = float(final_OPD - target_OPD)
    final_physical_mirrors = _x_to_mirrors(x_physical, base_mirrors) if dry_run else None

    measured_qc_values = []
    rollback_count = 0
    max_step_target_miss = 0.0
    max_abs_OPD_divergence = 0.0
    for entry in execution_log:
        for key in ("before_qc_y", "after_qc_y"):
            measured_qc_values.extend(abs(float(v)) for v in entry.get(key, []))
        max_step_target_miss = max(max_step_target_miss, float(entry.get("qc_target_miss", 0.0)))
        if entry.get("OPD_divergence") is not None:
            max_abs_OPD_divergence = max(max_abs_OPD_divergence, abs(float(entry["OPD_divergence"])))
        detail = entry.get("detail", {})
        rollback_count += int(detail.get("rollback_count", 0))
    max_abs_measured_qc = max(measured_qc_values) if measured_qc_values else float(np.max(np.abs(final_qc["y"])))

    final_success = (
        failure_reason is None and
        abs(final_OPD_error) <= final_OPD_tolerance and
        (
            not require_final_qc or
            max(abs(float(final_qc["y"][0])), abs(float(final_qc["y"][1]))) <= final_qc_tolerance
        )
    )
    final_qc_max_abs = max(abs(float(final_qc["y"][0])), abs(float(final_qc["y"][1])))
    final_qc_within_tolerance = final_qc_max_abs <= final_qc_tolerance
    success_checks = {
        "no_failure_reason": failure_reason is None,
        "final_OPD_within_tolerance": abs(final_OPD_error) <= final_OPD_tolerance,
        "final_qc_required": bool(require_final_qc),
        "final_qc_within_tolerance": bool(final_qc_within_tolerance),
        "needs_final_recenter": bool(not final_qc_within_tolerance),
        "final_OPD_error": final_OPD_error,
        "final_OPD_tolerance": float(final_OPD_tolerance),
        "final_qc_max_abs": float(final_qc_max_abs),
        "final_qc_tolerance": float(final_qc_tolerance),
        "planner_failure_ignored": bool(planner_failure_ignored),
        "planner_failure_reason": planner_failure_reason,
    }
    if failure_reason is None and not final_success:
        failed_checks = [
            name for name, ok in success_checks.items()
            if (
                name.endswith("_within_tolerance") and
                not ok and
                (name != "final_qc_within_tolerance" or require_final_qc)
            )
        ]
        failure_reason = "Fixed-plan final checks failed: " + ", ".join(failed_checks)
    final_res = SimpleNamespace(
        success=bool(final_success),
        message=(
            (
                "Fixed-plan OPD execution reached OPD tolerance; final QC still needs recentering."
                if not final_qc_within_tolerance
                else "Fixed-plan OPD execution reached target tolerances."
            )
            if final_success
            else "Fixed-plan OPD execution stopped before success criteria were met."
        )
    )
    final_res = S.set_OPD_result_full_x(final_res, *final_mirrors)

    execution = {
        "success": bool(final_success),
        "failure_reason": failure_reason if failure_reason is not None else (None if final_success else final_res.message),
        "success_checks": success_checks,
        "fixed_plan": True,
        "fixed_plan_substep_log": True,
        "completion_message": final_res.message,
        "initial_qc_x": (
            execution_log[0]["before_qc_x"] if execution_log else final_qc["x"].tolist()
        ),
        "initial_estimate_x": S.pack_variables(*base_mirrors).tolist(),
        "initial_physical_x": S.pack_variables(*base_mirrors).tolist() if dry_run else None,
        "planner_failure_ignored": bool(planner_failure_ignored),
        "planner_failure_reason": planner_failure_reason,
        "needs_final_recenter": bool(not final_qc_within_tolerance),
        "target_OPD": float(target_OPD),
        "final_OPD": float(final_OPD),
        "final_OPD_error": final_OPD_error,
        "final_sim_qc": [float(final_sim_qc[0]), float(final_sim_qc[1])],
        "final_assimilated_mirrors": _mirrors_to_lists(final_mirrors),
        "final_physical_mirrors": (
            None if final_physical_mirrors is None
            else _mirrors_to_lists(final_physical_mirrors)
        ),
        "final_qc_x": final_qc["x"].tolist(),
        "final_qc_y": final_qc["y"].tolist(),
        "max_abs_measured_qc": float(max_abs_measured_qc),
        "max_step_target_miss": float(max_step_target_miss),
        "max_abs_OPD_divergence": float(max_abs_OPD_divergence),
        "rollback_count": int(rollback_count),
        "linear_stage_locs": dict(linear_stage_locs),
        "moving_linear_stages": list(active_linear_stages),
        "rotation_calibration": dict(rotation_calibration),
        "execution_log": execution_log,
        "planner_runs": [{
            "replan": 1,
            "reason": "initial_fixed_plan",
            "plan": actuation_plan,
            "profile": planner_profile,
        }],
        "actuation_plan": actuation_plan,
        "dry_run": bool(dry_run),
        "qc_plan_limit": float(qc_plan_limit),
        "qc_detector_limit": float(qc_detector_limit),
        "qc_hardware_stop": float(qc_hardware_stop),
        "qc_step_tolerance": float(qc_step_tolerance),
        "linear_stage_guard_mm": float(linear_stage_guard_mm),
        "linear_stage_replan_tolerance_mm": float(linear_stage_replan_tolerance_mm),
        "final_qc_tolerance": float(final_qc_tolerance),
        "final_OPD_tolerance": float(final_OPD_tolerance),
        "require_final_qc": bool(require_final_qc),
        "allow_final_qc_planner_failure": bool(allow_final_qc_planner_failure),
        "qc_readout_axis": "x",
        "qc_readout_sign": float(qc_readout_sign),
        "fast_qc_avg": int(fast_qc_avg),
        "fast_qc_delay": float(fast_qc_delay),
        "final_qc_avg": int(final_qc_avg),
        "final_qc_delay": float(final_qc_delay),
        "linear_settle_delay": float(linear_settle_delay),
        "rotation_settle_delay": float(rotation_settle_delay),
        "step_settle_delay": float(step_settle_delay),
        "profile_step_interval": None if profile_step_interval is None else int(profile_step_interval),
    }

    log(
        f"done success={final_success} OPD_error={final_OPD_error:.3f} "
        f"final_qc_x=({final_qc['x'][0]:.3f},{final_qc['x'][1]:.3f}) "
        f"failure={execution['failure_reason']}"
    )

    return final_mirrors, final_res, execution


def execute_reflection_count_fixed_plan(
        target_N_R,
        M1,
        M2,
        M3,
        M4,
        hardware=None,
        *,
        actuator_map=None,
        rotation_calibration=None,
        center_n_tries=4000,
        angle_perturb=1.5,
        center_retry_angle_scales=S.DEFAULT_CENTER_RETRY_ANGLE_SCALES,
        seed=0,
        u_min=0.1,
        u_max=0.9,
        sigma_edge=0.1,
        final_qc_tolerance=0.5,
        qc_reacquire_limit=3.0,
        reacquire_stop_mode="window",
        reacquire_center_tolerance=0.0,
        stage_qc_limit=3.0,
        final_center_qc_safety_limit=3.5,
        final_center_max_axis_splits=80,
        final_center_waypoint_depth=4,
        final_center_feedback_max_pulses=80,
        final_center_min_improvement=0.01,
        require_target_N_R_estimate_for_reacquire_stop=False,
        require_beam_departure_before_reacquire=True,
        min_reacquire_pulse_fraction=0.0,
        target_reacquire_extend_until_reacquired=True,
        reacquire_angle_scan_limit=1.0,
        reacquire_scan_samples=1001,
        reacquire_max_first_leg_candidates=50,
        reacquire_stage_max_candidates=24,
        reacquisition_strategy="forced_stage_one_axis",
        preflight_before_calibration=None,
        forced_stage_samples=41,
        forced_stage_free_angle_regularization=0.02,
        forced_stage_max_nfev=160,
        surface_stage_active_actuators=("M1.dangle", "M2.dangle", "M3.dangle"),
        surface_stage_sweep_actuator="M4.dangle",
        surface_stage_distance_metric="active_3d",
        surface_stage_match_target_3d=True,
        surface_stage_qc_tolerance=0.05,
        surface_stage_path_qc_limit=2.0,
        surface_stage_qc_step_tolerance=0.15,
        surface_stage_angle_fit_qc_tolerance=0.15,
        surface_stage_qc_overshoot_allowance=0.1,
        surface_stage_max_chunks_per_step=50,
        surface_stage_sweep_half_span_deg=None,
        surface_stage_sweep_step_deg=0.05,
        surface_stage_sweep_refinement_levels=8,
        surface_stage_sweep_expand_to_bounds=True,
        surface_stage_sweep_max_half_span_deg=None,
        surface_stage_sweep_min_step_deg=1e-4,
        surface_stage_preferred_axis=None,
        surface_stage_line_max_steps_per_side=400,
        surface_stage_line_step_deg=0.1,
        surface_stage_projection_refinement_samples=3,
        surface_stage_max_path_candidates=2,
        surface_stage_repair_source_bounds=True,
        surface_stage_source_repair_n_tries=None,
        surface_stage_source_repair_angle_perturb=None,
        surface_stage_stop_after_stage=False,
        surface_stage_target_jump_include_active_residuals=True,
        surface_stage_validate_target_sweep=True,
        surface_stage_learn_rotation_response=True,
        target_center_after_jump=True,
        target_center_u_min=None,
        target_center_u_max=None,
        max_target_jump_center_candidates=12,
        final_center_after_reacquire=True,
        qc_min_signal=0.04,
        qc_min_signals=None,
        qc_readout_sign=-1.0,
        max_total_steps=300,
        max_rotation_chunk_substeps=20,
        min_rotation_chunk_substeps=1,
        rotation_settle_delay=0.04,
        step_settle_delay=0.04,
        calibrate_before=True,
        calibration_pulse_substeps=2,
        calibration_repeats=3,
        calibration_recenter_after=True,
        calibration_recenter_after_each_actuator=True,
        calibration_recenter_tolerance=0.2,
        calibration_recenter_max_pulses=20,
        calibration_recenter_min_improvement=0.01,
        rotation_pulse_calibration=None,
        dry_run=False,
        dry_run_use_settle_delays=False,
        dry_run_rotation_error=0.10,
        dry_run_signal_strengths=(1.0, 1.0),
        dry_run_pulse_response=None,
        rng_seed=None,
        profile=True,
        profile_sink=None):
    """Execute a rotation-only reflection-count change with beam reacquisition.

    Surface staging uses the OPD fixed-plan controller to reach each simulated
    QC endpoint, with a separate overshoot allowance above the planned bound.
    step_settle_delay is the extra wait between actuator steps; it defaults to
    0.04 seconds, matching execute_OPD_fixed_plan.
    The staging rollback guard is surface_stage_path_qc_limit plus
    surface_stage_qc_overshoot_allowance, not a calibrated detector edge.
    Planned pulses use calibration_pulse_substeps (or the calibration table's
    pulse size); feedback chunks may command fewer steps than that pulse size.
    Without explicit calibration, dry-run physical motion uses an approximate
    0.0007 degrees per motor step. Explicit dry_run_pulse_response takes
    precedence, followed by rotation_pulse_calibration and rotation_calibration.
    The target jump may leave the QCs and stops on valid reacquisition. With
    reacquire_stop_mode="center_crossing", at least one measured QC x position
    must hit zero or cross it between consecutive valid samples after departure.
    Both QCs must still have valid signals within the reacquisition window.
    reacquire_center_tolerance optionally accepts positions near zero (mm).
    Lost signal resets the crossing history. If the primary jump reaches its
    planned angle before reacquiring, it can keep
    sweeping in the same direction until reacquisition or the pulse budget.
    max_total_steps limits commanded chunks/pulses, not planner steps.
    max_rotation_chunk_substeps also caps target-jump commands. A calibrated
    pulse can be split into smaller commands with QC feedback after each;
    min_reacquire_pulse_fraction still measures completed calibrated motion.
    Hardware angles are estimates, not encoder measurements.
    Surface target jumps validate the first one-axis QC return in both model
    directions. Reliable staging QC fits refine uncalibrated step response.
    Staging fits must preserve the source N_R and match both QCs within
    surface_stage_angle_fit_qc_tolerance (norm in mm). Failed fits stop staging
    before another actuator move and require an inverse refresh.
    Dry-run reflection counts use simulated physical angles, not pulse priors.
    On hardware, QC reacquisition can succeed while the pulse-model N_R is
    unverified; an optional model mismatch becomes a warning pending inverse
    refresh, not a failed measured stop. Dry-run count mismatches still fail.
    target_N_R_reached is a legacy model flag; its source is recorded explicitly.
    With profile=True, surface planning reports target/staging distances and
    route-search progress. Routing skips strictly worse-distance candidates
    only after a fully validated route succeeds; equal-distance ties remain.
    Surface target centering uses the same narrower random-search retries as
    trace_reflection_count_surface_family, controlled by center_retry_angle_scales.
    Surface strategies trace source and target four-corner patches, select the
    target point nearest the starting angles, then the source stage nearest that
    target. Surface tracing knobs apply to both sheets; no dense family is built.
    With surface_stage_match_target_3d, staging first tries the target's exact
    active angles and solves only the source sweep angle, then refines the
    nearest QC-feasible stage if necessary. This uses surface_stage_path_qc_limit
    at the endpoint; surface_stage_qc_tolerance still defines the traced sheets.
    New surface plans track absolute single-axis angle goals through local QC
    changes computed from the current fitted state. Their effective QC tolerance
    is capped at half an estimated motor step to prevent cumulative drift along
    a centered sheet. This still uses QC feedback, not encoder measurements.
    """
    if profile and profile_sink is None:
        profile_sink = print

    t0 = time.perf_counter()

    def log(message):
        if not profile:
            return
        profile_sink(f"[execute_N_R {time.perf_counter() - t0:.3f}s] {message}")

    center_retry_angle_scales = S._normalize_center_retry_angle_scales(center_retry_angle_scales)
    if (surface_stage_sweep_max_half_span_deg is not None and
            (not np.isfinite(surface_stage_sweep_max_half_span_deg) or surface_stage_sweep_max_half_span_deg <= 0)):
        raise ValueError("surface_stage_sweep_max_half_span_deg must be finite and positive.")
    if not np.isfinite(surface_stage_sweep_min_step_deg) or surface_stage_sweep_min_step_deg < 0:
        raise ValueError("surface_stage_sweep_min_step_deg must be finite and nonnegative.")
    actuator_map = _merged_actuator_map(actuator_map)
    explicit_rotation_calibration = set(rotation_calibration or ())
    rotation_calibration = _normalized_rotation_calibration(rotation_calibration)
    if dry_run and dry_run_pulse_response is None:
        dry_run_pulse_response = {}
        for label, estimated_response in rotation_calibration.items():
            response = {
                "degrees_per_substep": (
                    estimated_response if label in explicit_rotation_calibration
                    else DEFAULT_REFLECTION_DRY_RUN_DEGREES_PER_SUBSTEP
                ),
            }
            if rotation_pulse_calibration:
                reference_steps = int(abs(rotation_pulse_calibration.get(
                    "pulse_substeps", calibration_pulse_substeps,
                )))
                if reference_steps <= 0:
                    raise ValueError("execution pulse_substeps must be positive.")
                for sign, sign_key in ((1, "+"), (-1, "-")):
                    delta, source = _pulse_response_from_table(
                        rotation_pulse_calibration, label, sign,
                        reference_steps, estimated_response,
                    )
                    if source == "pulse_calibration":
                        response[sign_key] = delta
                        response["pulse_substeps"] = reference_steps
            dry_run_pulse_response[label] = response
    requested_rotation_settle_delay = float(rotation_settle_delay)
    requested_step_settle_delay = float(step_settle_delay)
    if dry_run and not bool(dry_run_use_settle_delays):
        rotation_settle_delay = 0.0
        step_settle_delay = 0.0
    rng = np.random.default_rng(rng_seed)
    if dry_run:
        log(f"dry_run_motion_response={dry_run_pulse_response}")
    base_mirrors = (
        np.array(M1, dtype=float),
        np.array(M2, dtype=float),
        np.array(M3, dtype=float),
        np.array(M4, dtype=float),
    )
    x_estimate = S.pack_variables(*base_mirrors)
    x_physical = x_estimate.copy()
    calibration_result = None
    pulse_table = rotation_pulse_calibration
    reacquisition_strategy = str(reacquisition_strategy)
    valid_reacquisition_strategies = {
        "forced_stage_one_axis",
        "surface_stage_then_jump",
        "surface_stage_then_refresh",
        "staged_one_axis",
        "direct_scan",
    }
    if reacquisition_strategy not in valid_reacquisition_strategies:
        raise ValueError(
            "reacquisition_strategy must be one of "
            f"{sorted(valid_reacquisition_strategies)}, got {reacquisition_strategy!r}."
        )
    reacquire_stop_mode = str(reacquire_stop_mode)
    if reacquire_stop_mode not in {"window", "center_crossing"}:
        raise ValueError("reacquire_stop_mode must be 'window' or 'center_crossing'.")
    reacquire_center_tolerance = float(reacquire_center_tolerance)
    if not np.isfinite(reacquire_center_tolerance) or reacquire_center_tolerance < 0:
        raise ValueError("reacquire_center_tolerance must be finite and nonnegative.")
    if not dry_run and (
            hardware is None or getattr(hardware, "rotation_stages", None) is None or
            getattr(hardware, "quads", None) is None):
        raise ValueError("dry_run=False requires hardware=H with .rotation_stages and .quads.")
    if not np.isfinite(rotation_settle_delay) or rotation_settle_delay < 0:
        raise ValueError("rotation_settle_delay must be finite and nonnegative.")
    max_rotation_chunk_substeps = int(max_rotation_chunk_substeps)
    min_rotation_chunk_substeps = int(min_rotation_chunk_substeps)
    if not 1 <= min_rotation_chunk_substeps <= max_rotation_chunk_substeps:
        raise ValueError("Rotation chunk limits must satisfy 1 <= min <= max.")
    surface_stage_angle_fit_qc_tolerance = float(surface_stage_angle_fit_qc_tolerance)
    if not np.isfinite(surface_stage_angle_fit_qc_tolerance) or surface_stage_angle_fit_qc_tolerance < 0:
        raise ValueError("surface_stage_angle_fit_qc_tolerance must be finite and nonnegative.")

    def planner_timing_text(plan):
        timing = plan.get("planner_timing") if isinstance(plan, dict) else None
        if not timing:
            return "planner_timing=unavailable"
        preferred = (
            "total",
            "direct_scan",
            "stage_generation",
            "stage_recenter_solve",
            "stage_path_validation",
            "stage_jump_scan",
            "target_center_solve",
            "target_center_path",
            "source_repair",
            "target_solve",
            "source_line_trace",
            "source_sweep_trace",
            "source_corner_trace",
            "source_boundary_refinement",
            "target_line_trace",
            "target_sweep_trace",
            "target_corner_trace",
            "target_boundary_refinement",
            "target_surface_projection",
            "stage_target_match",
            "projection",
            "target_sweep_validation",
            "candidate_path_validation",
            "candidate_validation",
        )
        parts = []
        for key in preferred:
            if key in timing:
                parts.append(f"{key}={float(timing[key]):.3f}s")
        return "planner_timing " + " ".join(parts)

    def plan_reacquisition_from(mirrors):
        if reacquisition_strategy == "direct_scan":
            plan_t0 = time.perf_counter()
            mirrors_target_plan, planner_res_plan, actuation_plan_plan = S.plan_reflection_count_reacquisition(
                *mirrors,
                target_N_R=target_N_R,
                qc_reacquire_limit=qc_reacquire_limit,
                angle_scan_limit=reacquire_angle_scan_limit,
                scan_samples=reacquire_scan_samples,
                max_first_leg_candidates=reacquire_max_first_leg_candidates,
            )
            actuation_plan_plan["planner_timing"] = {
                "total": float(time.perf_counter() - plan_t0)
            }
            actuation_plan_plan["reacquisition_strategy"] = "direct_scan"
            actuation_plan_plan.setdefault("requires_inverse_refresh", False)
            actuation_plan_plan.setdefault("qc_only_reacquire_stop", False)
            actuation_plan_plan.setdefault("stage_plan", None)
            actuation_plan_plan.setdefault("target_jump_step", None)
            return mirrors_target_plan, planner_res_plan, actuation_plan_plan

        if reacquisition_strategy in {"surface_stage_then_refresh", "surface_stage_then_jump"}:
            if surface_stage_qc_overshoot_allowance < 0 or surface_stage_qc_step_tolerance < 0:
                raise ValueError("Stage QC tolerance and overshoot allowance must be nonnegative.")
            return S.plan_reflection_count_surface_stage(
                *mirrors,
                target_N_R=target_N_R,
                active_actuators=surface_stage_active_actuators,
                sweep_actuator=surface_stage_sweep_actuator,
                distance_metric=surface_stage_distance_metric,
                match_target_active_angles=surface_stage_match_target_3d,
                stage_qc_tolerance=surface_stage_qc_tolerance,
                stage_path_qc_limit=surface_stage_path_qc_limit,
                qc_reacquire_limit=qc_reacquire_limit,
                sweep_half_span_deg=(
                    reacquire_angle_scan_limit
                    if surface_stage_sweep_half_span_deg is None
                    else surface_stage_sweep_half_span_deg
                ),
                sweep_step_deg=surface_stage_sweep_step_deg,
                sweep_refinement_levels=surface_stage_sweep_refinement_levels,
                sweep_expand_to_bounds=surface_stage_sweep_expand_to_bounds,
                sweep_max_half_span_deg=surface_stage_sweep_max_half_span_deg,
                sweep_min_step_deg=surface_stage_sweep_min_step_deg,
                preferred_axis=surface_stage_preferred_axis,
                line_max_steps_per_side=surface_stage_line_max_steps_per_side,
                line_step_deg=surface_stage_line_step_deg,
                u_min=u_min,
                u_max=u_max,
                center_n_tries=center_n_tries,
                center_angle_perturb=angle_perturb,
                center_retry_angle_scales=center_retry_angle_scales,
                center_seed=seed,
                center_u_min=target_center_u_min,
                center_u_max=target_center_u_max,
                center_sigma_edge=sigma_edge,
                center_final_qc_tolerance=final_qc_tolerance,
                center_enforce_final_u_bounds=True,
                center_final_u_tolerance=1e-9,
                sigma_edge=sigma_edge,
                projection_refinement_samples=surface_stage_projection_refinement_samples,
                max_stage_path_candidates=surface_stage_max_path_candidates,
                repair_source_out_of_bounds=surface_stage_repair_source_bounds,
                source_repair_n_tries=surface_stage_source_repair_n_tries,
                source_repair_angle_perturb=surface_stage_source_repair_angle_perturb,
                execute_target_jump=not bool(surface_stage_stop_after_stage),
                target_jump_include_active_residuals=surface_stage_target_jump_include_active_residuals,
                validate_target_sweep=surface_stage_validate_target_sweep,
                target_sweep_angle_scan_limit=reacquire_angle_scan_limit,
                target_sweep_scan_samples=reacquire_scan_samples,
                target_sweep_stop_mode=reacquire_stop_mode,
                target_sweep_center_tolerance=reacquire_center_tolerance,
                stage_max_axis_splits=final_center_max_axis_splits,
                stage_waypoint_depth=final_center_waypoint_depth,
                stage_motion_samples_per_step=15,
                stage_fast_motion_samples_per_step=5,
                profile_callback=(lambda msg: log("planner " + msg)) if profile else None,
            )

        stage_search_mode = (
            "forced"
            if reacquisition_strategy == "forced_stage_one_axis"
            else "random"
        )
        return S.plan_reflection_count_staged_reacquisition(
            *mirrors,
            target_N_R=target_N_R,
            qc_reacquire_limit=qc_reacquire_limit,
            stage_qc_limit=stage_qc_limit,
            angle_scan_limit=reacquire_angle_scan_limit,
            scan_samples=reacquire_scan_samples,
            stage_n_tries=center_n_tries,
            stage_angle_perturb=angle_perturb,
            stage_seed=seed,
            stage_search_mode=stage_search_mode,
            forced_stage_samples=forced_stage_samples,
            forced_stage_free_angle_regularization=forced_stage_free_angle_regularization,
            forced_stage_max_nfev=forced_stage_max_nfev,
            stage_sigma_edge=sigma_edge,
            target_center_after_jump=target_center_after_jump,
            target_qc_tolerance=final_qc_tolerance,
            target_center_u_min=target_center_u_min,
            target_center_u_max=target_center_u_max,
            max_target_jump_center_candidates=max_target_jump_center_candidates,
            max_stage_candidates=reacquire_stage_max_candidates,
            stage_max_axis_splits=final_center_max_axis_splits,
            stage_waypoint_depth=final_center_waypoint_depth,
            u_min=u_min,
            u_max=u_max,
            profile_callback=(lambda msg: log("planner " + msg)),
        )

    staged_qc_only_strategy = reacquisition_strategy in {
        "forced_stage_one_axis",
        "staged_one_axis",
    }
    surface_stage_strategy = reacquisition_strategy in {
        "surface_stage_then_refresh",
        "surface_stage_then_jump",
    }
    surface_stage_stage_only = bool(surface_stage_strategy and surface_stage_stop_after_stage)
    preflight_strategy = staged_qc_only_strategy or surface_stage_strategy
    if preflight_before_calibration is None:
        preflight_before_calibration = not surface_stage_strategy
    preflight_before_calibration = bool(preflight_before_calibration)

    if calibrate_before and preflight_strategy and preflight_before_calibration:
        try:
            _, preflight_res, preflight_plan = plan_reacquisition_from(base_mirrors)
            preflight_failure_reason = preflight_plan.get("failure_reason")
        except Exception as exc:
            preflight_res = SimpleNamespace(success=False, message=str(exc))
            preflight_plan = {
                "steps": [],
                "n_steps": 0,
                "reflection_count_change": True,
                "reflection_count_reacquisition": True,
                "target_N_R": int(target_N_R),
                "reacquisition_strategy": reacquisition_strategy,
                "requires_inverse_refresh": False,
                "qc_only_reacquire_stop": staged_qc_only_strategy,
                "stage_only_refresh": surface_stage_stage_only,
                "stage_plan": None,
                "target_jump_step": None,
                "failure_reason": str(exc),
            }
            preflight_failure_reason = str(exc)

        if preflight_failure_reason is not None:
            final_res = S.set_OPD_result_full_x(
                SimpleNamespace(
                    success=False,
                    message="Reflection-count execution stopped before calibration."
                ),
                *base_mirrors
            )
            execution = {
                "success": False,
                "failure_reason": "Planner failed before calibration: " + str(preflight_failure_reason),
                "completion_status": "failed",
                "completion_message": str(final_res.message),
                "warnings": [],
                "reflection_count_change": True,
                "reflection_count_reacquisition": True,
                "rotation_only": True,
                "target_N_R": int(target_N_R),
                "reacquisition_strategy": reacquisition_strategy,
                "requires_inverse_refresh": False,
                "suggested_next_step": None,
                "stage_only_refresh": surface_stage_stage_only,
                "stage_reached": False,
                "target_N_R_reached": False,
                "calibration_result": None,
                "actuation_plan": preflight_plan,
                "planner_result": preflight_res,
                "execution_log": [],
                "dry_run": bool(dry_run),
            }
            return base_mirrors, final_res, execution

        log(
            f"preflight plan found search_mode={preflight_plan.get('search_mode')} "
            f"before calibration {planner_timing_text(preflight_plan)}"
        )

    if calibrate_before:
        calibration_result = calibrate_rotation_pulse_response(
            *base_mirrors,
            hardware=hardware,
            actuator_map=actuator_map,
            rotation_calibration=rotation_calibration,
            pulse_substeps=calibration_pulse_substeps,
            repeats=calibration_repeats,
            qc_min_signal=qc_min_signal,
            qc_min_signals=qc_min_signals,
            qc_readout_sign=qc_readout_sign,
            rotation_settle_delay=rotation_settle_delay,
            recenter_after_calibration=calibration_recenter_after,
            recenter_after_each_actuator=calibration_recenter_after_each_actuator,
            recenter_tolerance=calibration_recenter_tolerance,
            recenter_max_pulses=calibration_recenter_max_pulses,
            recenter_min_improvement=calibration_recenter_min_improvement,
            dry_run=dry_run,
            dry_run_rotation_error=dry_run_rotation_error,
            dry_run_signal_strengths=dry_run_signal_strengths,
            dry_run_pulse_response=dry_run_pulse_response,
            rng_seed=rng_seed,
            profile=profile,
            profile_sink=profile_sink,
        )
        pulse_table = calibration_result
        if not calibration_result["success"]:
            final_mirrors = calibration_result["calibrated_mirrors"]
            final_res = S.set_OPD_result_full_x(
                SimpleNamespace(
                    success=False,
                    message="Reflection-count execution stopped during calibration."
                ),
                *final_mirrors
            )
            execution = {
                "success": False,
                "failure_reason": "Calibration failed: " + str(calibration_result["failure_reason"]),
                "completion_status": "failed",
                "completion_message": str(final_res.message),
                "warnings": [],
                "reflection_count_change": True,
                "rotation_only": True,
                "target_N_R": int(target_N_R),
                "reacquisition_strategy": reacquisition_strategy,
                "requires_inverse_refresh": False,
                "suggested_next_step": None,
                "calibration_result": calibration_result,
                "actuation_plan": None,
                "execution_log": [],
                "dry_run": bool(dry_run),
            }
            return final_mirrors, final_res, execution

        base_mirrors = tuple(np.array(m, dtype=float) for m in calibration_result["calibrated_mirrors"])
        x_estimate = S.pack_variables(*base_mirrors)
        x_physical = np.array(calibration_result["x_physical"], dtype=float)
        log("calibration complete; planning from QC-fitted post-calibration angle estimate (no image refit)")

    try:
        mirrors_target, planner_res, actuation_plan = plan_reacquisition_from(base_mirrors)
        planner_failure_reason = actuation_plan.get("failure_reason")
    except Exception as exc:
        mirrors_target = base_mirrors
        planner_res = SimpleNamespace(success=False, message=str(exc))
        actuation_plan = {
            "steps": [],
            "n_steps": 0,
            "reflection_count_change": True,
            "reflection_count_reacquisition": True,
            "target_N_R": int(target_N_R),
            "reacquisition_strategy": reacquisition_strategy,
            "requires_inverse_refresh": False,
            "qc_only_reacquire_stop": bool(
                staged_qc_only_strategy or
                (surface_stage_strategy and not surface_stage_stop_after_stage)
            ),
            "stage_only_refresh": surface_stage_stage_only,
            "stage_plan": None,
            "target_jump_step": None,
            "failure_reason": str(exc),
        }
        planner_failure_reason = str(exc)

    steps = actuation_plan.get("steps", [])
    center_crossing_required = bool(
        reacquire_stop_mode == "center_crossing" and any(
            step.get("target_jump_step", True) and not step.get("disable_reacquire_stop", False)
            for step in steps
        )
    )
    failure_reason = None
    if planner_failure_reason is not None:
        failure_reason = "Planner failed: " + planner_failure_reason
    elif len(steps) == 0:
        start_reflections = S.get_reflection_count(*base_mirrors)
        if surface_stage_stage_only:
            failure_reason = None
        elif start_reflections != int(target_N_R):
            failure_reason = "Planner returned no rotation steps."

    pulse_budget = {"max_total_steps": int(max_total_steps), "nominal_plan_pulses": 0,
                    "nominal_stage_pulses": 0, "steps": []}
    execution_pulse_substeps = int(abs(
        pulse_table.get("pulse_substeps", calibration_pulse_substeps)
        if isinstance(pulse_table, dict) else calibration_pulse_substeps
    ))
    if execution_pulse_substeps <= 0:
        raise ValueError("execution pulse_substeps must be positive.")
    for planned_step in steps:
        label = planned_step.get("actuator")
        delta = float(planned_step["command_value"])
        pulse_delta, _ = _pulse_response_from_table(
            pulse_table, label, int(np.sign(delta) or 1), execution_pulse_substeps,
            rotation_calibration.get(label, DEFAULT_ROTATION_DEGREES_PER_SUBSTEP),
        )
        count = max(1, int(round(abs(delta / pulse_delta)))) if delta else 0
        pulse_budget["steps"].append(count)
        pulse_budget["nominal_plan_pulses"] += count
        if not planned_step.get("target_jump_step", True):
            pulse_budget["nominal_stage_pulses"] += count
    pulse_budget["nominal_rotation_settle_seconds"] = (
        pulse_budget["nominal_plan_pulses"] * float(rotation_settle_delay)
    )
    if failure_reason is None and not surface_stage_strategy and pulse_budget["nominal_stage_pulses"] > max_total_steps:
        failure_reason = (
            f"Insufficient pulse budget before staging: stage needs "
            f"{pulse_budget['nominal_stage_pulses']} pulses, max_total_steps={max_total_steps}. "
            "No staging moves executed; calibration may already have moved the mirrors."
        )
    log(
        f"pulse_budget stage={pulse_budget['nominal_stage_pulses']} "
        f"full_plan={pulse_budget['nominal_plan_pulses']} limit={max_total_steps} "
        f"rotation_settle_delay={rotation_settle_delay:g}s per pulse"
    )
    execution_start_qc = _read_quadcell_x_with_signal(
        hardware, dry_run=dry_run, dry_run_x=x_physical, base_mirrors=base_mirrors,
        qc_readout_sign=qc_readout_sign, qc_min_signal=qc_min_signal,
        qc_min_signals=qc_min_signals, dry_run_signal_strengths=dry_run_signal_strengths,
    )
    initial_model_comparison = _qc_model_comparison(
        x_estimate, base_mirrors, execution_start_qc, qc_readout_sign
    )
    log(
        f"start QC measured={np.round(execution_start_qc['x'], 4).tolist()} "
        f"predicted={np.round(initial_model_comparison['predicted_qc_x'], 4).tolist()} "
        f"residual_norm={initial_model_comparison['qc_residual_norm']:.4g} "
        f"position_valid={execution_start_qc['valid']}"
    )

    log(
        f"start target_N_R={int(target_N_R)} dry_run={dry_run} "
        f"strategy={reacquisition_strategy} "
        f"stage_goal_mode={'absolute_angle' if actuation_plan.get('staging_angle_targets') else 'fixed_qc'} "
        f"planned_steps={len(steps)} search_mode={actuation_plan.get('search_mode')} "
        f"qc_reacquire_limit={qc_reacquire_limit} "
        f"reacquire_stop_mode={reacquire_stop_mode} "
        f"{planner_timing_text(actuation_plan)} failure={failure_reason}"
    )

    execution_log = []
    center_execution_log = []
    execution_warnings = []
    total_commanded_substeps = 0
    total_execution_pulses = 0
    beam_reacquired_during_sweep = False
    beam_departed_reacquire_window = not bool(require_beam_departure_before_reacquire)
    beam_departed_at = None
    qc_center_crossed_mask = np.zeros(2, dtype=bool)
    qc_center_crossed_at = [None, None]
    previous_center_qc_x = None
    reacquired_qc = None
    reacquired_at = None
    last_reacquisition_axis = None
    last_reacquisition_actuator = None
    last_reacquisition_assimilation = None
    last_reacquired_by = None
    stage_arrival = None
    track_stage_angles = bool(surface_stage_strategy and actuation_plan.get("staging_angle_targets", False))
    requires_inverse_refresh = False
    staging_rotation_observations = {}
    stage_only_refresh = bool(actuation_plan.get("stage_only_refresh", False))
    execution_t0 = time.perf_counter()
    final_center_path_failure = None
    stage_step_indices = [
        index for index, step in enumerate(steps, start=1)
        if (
            step.get("reflection_count_surface_stage_move") or
            step.get("reflection_count_stage_move")
        )
    ]
    last_stage_step_index = max(stage_step_indices) if stage_step_indices else 0
    stage_reached_during_execution = bool(
        failure_reason is None and actuation_plan.get("stage_x") is not None and
        last_stage_step_index == 0 and
        np.allclose(x_estimate, actuation_plan["stage_x"], rtol=0, atol=1e-9)
    )

    for step_index, step in enumerate(steps, start=1):
        if failure_reason is not None:
            break
        if step_index > 1 and step_settle_delay and step_settle_delay > 0:
            time.sleep(step_settle_delay)

        actuator_label = step.get("actuator")
        mapping = actuator_map.get(actuator_label)
        if mapping is None:
            failure_reason = f"No hardware mapping for actuator {actuator_label}."
            break
        if mapping.get("kind") != "rotation":
            failure_reason = f"Reflection-count executor only supports rotation actuators, got {actuator_label}."
            break

        axis_index = int(step["axis_index"])
        planned_angle_delta = float(step["command_value"])
        target_jump_step = bool(step.get("target_jump_step", True))
        if surface_stage_strategy and (target_jump_step or track_stage_angles) and step.get("mirrors") is not None:
            planned_angle_delta = float(S.pack_variables(*step["mirrors"])[axis_index] - x_estimate[axis_index])
        target_center_step = bool(step.get("target_center_step", False))
        reacquire_stop_disabled = bool(step.get("disable_reacquire_stop", False))
        qc_only_reacquire_stop = bool(
            actuation_plan.get("qc_only_reacquire_stop", False) and
            target_jump_step and
            not reacquire_stop_disabled
        )
        reacquire_stop_enabled = bool(target_jump_step and not reacquire_stop_disabled)
        target_jump_sequence_part = step.get("target_jump_sequence_part")
        extend_target_reacquire_sweep = bool(
            surface_stage_strategy and
            target_reacquire_extend_until_reacquired and
            reacquire_stop_enabled and
            not target_center_step and
            target_jump_sequence_part == "sweep"
        )
        runtime_sweep_validation = None
        if surface_stage_strategy and surface_stage_validate_target_sweep and target_jump_sequence_part == "sweep":
            planned_stage = actuation_plan.get("stage_x")
            if planned_stage is not None:
                angle_axes = [1, 3, 5, 7]
                estimated_error = x_estimate[angle_axes] - np.asarray(planned_stage)[angle_axes]
                stage_arrival = {"estimated_angle_error_deg": estimated_error.tolist()}
                if dry_run:
                    stage_arrival["physical_angle_error_deg"] = (
                        x_physical[angle_axes] - np.asarray(planned_stage)[angle_axes]).tolist()
                log(f"stage arrival estimated_angle_error_deg={np.round(estimated_error, 7).tolist()}")
            validation_x = x_physical if dry_run else x_estimate
            runtime_sweep_validation = S.scan_reflection_count_sweep_reacquisition(
                *S.unpack_variables(validation_x, *base_mirrors), target_N_R,
                sweep_actuator=actuator_label, qc_reacquire_limit=qc_reacquire_limit,
                angle_scan_limit=reacquire_angle_scan_limit, scan_samples=reacquire_scan_samples,
                stop_mode=reacquire_stop_mode, center_tolerance=reacquire_center_tolerance,
                u_min=u_min, u_max=u_max,
            )
            expected_source = int(actuation_plan.get("stage_reflections", actuation_plan.get("start_reflections")))
            if runtime_sweep_validation["start_N_R"] != expected_source or not runtime_sweep_validation["success"]:
                failure_reason = (
                    "target_sweep_invalid: staged model does not support a first QC return to "
                    f"N_R={target_N_R}; source model N_R={runtime_sweep_validation['start_N_R']}, "
                    f"expected {expected_source}. " + str(runtime_sweep_validation["failure_reason"])
                )
                actuation_plan["runtime_target_sweep_validation"] = runtime_sweep_validation
                log(f"stopped before jump step={step_index}: {failure_reason}")
                break
            original_delta = planned_angle_delta
            planned_angle_delta = float(runtime_sweep_validation["selected"]["angle_delta"])
            log(f"jump validation actuator={actuator_label} original_delta={original_delta:.6g} "
                f"validated_delta={planned_angle_delta:.6g} first_return_N_R={target_N_R}")
        degrees_per_substep = rotation_calibration.get(
            actuator_label,
            DEFAULT_ROTATION_DEGREES_PER_SUBSTEP
        )
        if degrees_per_substep <= 0:
            failure_reason = f"degrees_per_substep must be positive for {actuator_label}."
            break

        controller = mapping.get("controller", DEFAULT_ROTATION_CONTROLLER)
        actuator = int(mapping["actuator"])
        hardware_direction = int(np.sign(mapping.get("direction", 1)) or 1)
        move_sign = int(np.sign(planned_angle_delta) or 1)
        execution_pulse_substeps = int(abs(
            pulse_table.get("pulse_substeps", calibration_pulse_substeps)
            if isinstance(pulse_table, dict) else calibration_pulse_substeps
        ))
        pulse_angle_delta, pulse_source = _pulse_response_from_table(
            pulse_table,
            actuator_label,
            move_sign,
            execution_pulse_substeps,
            degrees_per_substep,
        )
        if pulse_source == "fallback_degrees_per_substep" and actuator_label in staging_rotation_observations:
            pulse_source = "staging_qc_fit"
        if np.sign(pulse_angle_delta) != move_sign:
            failure_reason = (
                f"Pulse calibration for {actuator_label} has wrong sign: "
                f"{pulse_angle_delta} for requested delta {planned_angle_delta}."
            )
            break
        pulse_count_abs = int(round(abs(planned_angle_delta / pulse_angle_delta)))
        if pulse_count_abs == 0 and abs(planned_angle_delta) > 0:
            pulse_count_abs = 1
        pulse_count = move_sign * pulse_count_abs
        min_reacquire_pulses = int(np.ceil(abs(pulse_count) * float(min_reacquire_pulse_fraction)))
        min_reacquire_pulses = max(0, min(abs(pulse_count), min_reacquire_pulses))
        expected_angle_moved = pulse_count_abs * pulse_angle_delta
        residual_angle = planned_angle_delta - expected_angle_moved

        before_qc = _read_quadcell_x_with_signal(
            hardware,
            dry_run=dry_run,
            dry_run_x=x_physical,
            base_mirrors=base_mirrors,
            qc_readout_sign=qc_readout_sign,
            qc_min_signal=qc_min_signal,
            qc_min_signals=qc_min_signals,
            dry_run_signal_strengths=dry_run_signal_strengths,
        )
        stage_move = bool(
            not target_jump_step and not target_center_step and (
                surface_stage_strategy or step.get("reflection_count_stage_move", False)
            )
        )
        stage_limit = surface_stage_path_qc_limit if surface_stage_strategy else stage_qc_limit
        qc_feedback_stage = bool(stage_move and surface_stage_strategy)
        if qc_feedback_stage:
            stage_limit += float(surface_stage_qc_overshoot_allowance)
        if stage_move:
            failure_reason = _staging_qc_failure(before_qc, stage_limit)
            if failure_reason is not None:
                log(f"stopped before step={step_index}: {failure_reason}")
                break
            if qc_feedback_stage:
                expected_stage_reflections = int(actuation_plan.get("stage_reflections", actuation_plan.get("start_reflections")))
                start_stage_reflections = int(S.get_reflection_count(*S.unpack_variables(x_estimate, *base_mirrors)))
                if start_stage_reflections != expected_stage_reflections:
                    requires_inverse_refresh = True
                    failure_reason = (
                        f"stage_model_invalid: before {actuator_label}, estimated N_R={start_stage_reflections} "
                        f"does not match staging N_R={expected_stage_reflections}. Inverse refresh required."
                    )
                    log(f"stopped before step={step_index}: {failure_reason}")
                    break

        remaining_pulses = pulse_count
        remaining_substeps = pulse_count * execution_pulse_substeps
        pulse_logs = []
        step_commanded_substeps = 0
        step_sim_substeps = 0
        extra_reacquire_sweep_pulses = 0
        step_t0 = time.perf_counter()
        step_reacquired = False
        last_jump_progress_t = -np.inf
        last_jump_stop_state = None
        x_before_step = x_estimate.copy()
        physical_x_before_step = x_physical.copy() if dry_run else None
        current_qc = before_qc
        stage_detail = None
        stage_assimilation = None
        if reacquire_stop_enabled and reacquire_stop_mode == "center_crossing":
            qc_center_crossed_mask[:] = False
            qc_center_crossed_at = [None, None]
            previous_center_qc_x = (
                before_qc["x"].copy()
                if before_qc["valid"] and (
                    not require_beam_departure_before_reacquire or
                    beam_departed_reacquire_window
                ) else None
            )

        if qc_feedback_stage:
            def read_stage_qc():
                return _read_quadcell_x_with_signal(
                    hardware, dry_run=dry_run, dry_run_x=x_physical, base_mirrors=base_mirrors,
                    qc_readout_sign=qc_readout_sign, qc_min_signal=qc_min_signal,
                    qc_min_signals=qc_min_signals, dry_run_signal_strengths=dry_run_signal_strengths,
                )

            def record_stage_sample(sample, qc):
                sample.update(
                    pulse=int(sample["chunk"]), elapsed_s=float(time.perf_counter() - execution_t0),
                    estimate_x_after_pulse=x_estimate.tolist(),
                    physical_x_after_pulse=x_physical.tolist() if dry_run else None,
                    qc_signal_strengths=qc["signal_strengths"], qc_signal_valid=bool(qc["valid"]),
                    qc_signal_thresholds=qc["signal_thresholds"], qc_valid_mask=qc["valid_mask"],
                    qc_read_error=qc.get("error"),
                    **_qc_model_comparison(x_estimate, base_mirrors, qc, qc_readout_sign),
                )

            stage_calibration = dict(rotation_calibration)
            stage_calibration[actuator_label] = abs(pulse_angle_delta) / execution_pulse_substeps
            feedback_step = step
            feedback_tolerance = surface_stage_qc_step_tolerance
            goal_detail = {"goal_mode": "fixed_qc"}
            if track_stage_angles:
                try:
                    feedback_step, feedback_tolerance, goal_detail = _staging_angle_feedback_step(
                        step, x_estimate, before_qc["x"], base_mirrors,
                        qc_readout_sign=qc_readout_sign,
                        degrees_per_substep=stage_calibration[actuator_label],
                        qc_step_tolerance=surface_stage_qc_step_tolerance,
                        qc_safety_limit=stage_limit, expected_reflections=expected_stage_reflections,
                    )
                except ValueError as exc:
                    failure_reason = f"stage_angle_goal_invalid: {exc}. Inverse refresh required."
                    requires_inverse_refresh = True
                    log(f"stopped before step={step_index}: {failure_reason}")
                    break
            stage_detail = _execute_rotation_step_fixed(
                feedback_step, mapping, hardware, x_estimate, x_physical, base_mirrors, stage_calibration, rng,
                dry_run=dry_run, dry_run_rotation_error=dry_run_rotation_error,
                qc_readout_sign=qc_readout_sign, qc_step_tolerance=feedback_tolerance,
                qc_safety_limit=stage_limit, fast_qc_avg=1, fast_qc_delay=0,
                max_rotation_chunks_per_step=max(0, min(
                    int(surface_stage_max_chunks_per_step), int(max_total_steps - total_execution_pulses))),
                max_rotation_chunk_substeps=max_rotation_chunk_substeps,
                min_rotation_chunk_substeps=min_rotation_chunk_substeps,
                rotation_settle_delay=rotation_settle_delay,
                qc_reader=read_stage_qc, sample_callback=record_stage_sample,
                dry_run_delta=lambda n: _dry_run_rotation_delta_for_steps(
                    actuator_label, n, rotation_calibration, rng,
                    dry_run_rotation_error=dry_run_rotation_error,
                    dry_run_pulse_response=dry_run_pulse_response,
                    pulse_substeps=execution_pulse_substeps,
                ),
                stop_on_target_crossing=True,
            )
            stage_detail.update(goal_detail)
            pulse_logs = stage_detail["chunks"]
            step_commanded_substeps = stage_detail["total_commanded_substeps"]
            step_sim_substeps = stage_detail["total_sim_substeps"]
            total_commanded_substeps += step_commanded_substeps
            total_execution_pulses += len(pulse_logs)
            failure_reason = stage_detail["failure_reason"]
            if stage_detail["stop_reason"] == "max_rotation_chunks_reached":
                failure_reason = "stage_feedback_budget: QC endpoint not reached within the chunk/pulse budget."
            chunk_qc = read_stage_qc()
            remaining_pulses = 0
            remaining_substeps = 0
            if failure_reason is None and not chunk_qc["valid"]:
                failure_reason = "stage_qc_invalid: " + _qc_invalid_details(chunk_qc) + " after staging move."
            if failure_reason is None and dry_run:
                physical_reflections = int(S.get_reflection_count(*S.unpack_variables(x_physical, *base_mirrors)))
                if physical_reflections != expected_stage_reflections:
                    failure_reason = (
                        f"stage_reflection_count_changed: simulated physical N_R={physical_reflections} "
                        f"after {actuator_label}, expected {expected_stage_reflections}."
                    )
            if failure_reason is None and chunk_qc["valid"]:
                try:
                    fitted, stage_assimilation = _assimilate_staging_rotation_from_qc(
                        x_estimate, x_before_step, feedback_step, chunk_qc["x"], base_mirrors,
                        expected_reflections=expected_stage_reflections,
                        qc_readout_sign=qc_readout_sign, max_fit_error=surface_stage_angle_fit_qc_tolerance,
                    )
                    if not stage_assimilation["accepted"]:
                        failure_reason = "stage_angle_fit_invalid: " + stage_assimilation["message"] + " Inverse refresh required."
                    else:
                        x_estimate[:] = fitted
                        if track_stage_angles:
                            stage_detail["axis_angle_error_after_deg"] = float(
                                fitted[axis_index] - goal_detail["target_angle_deg"])
                    observed_delta = float(fitted[axis_index] - x_before_step[axis_index])
                    if (
                            stage_assimilation["accepted"] and
                            surface_stage_learn_rotation_response and
                            actuator_label not in explicit_rotation_calibration and
                            pulse_source in {"fallback_degrees_per_substep", "staging_qc_fit"} and
                            abs(step_sim_substeps) >= 3 and stage_assimilation["success"] and
                            stage_assimilation["fit_error_norm"] <= surface_stage_qc_step_tolerance and
                            observed_delta * step_sim_substeps > 0 and
                            S.get_reflection_count(*S.unpack_variables(fitted, *base_mirrors)) ==
                            int(actuation_plan.get("stage_reflections", actuation_plan.get("start_reflections")))):
                        observed = abs(observed_delta / step_sim_substeps)
                        if np.isfinite(observed):
                            observations = staging_rotation_observations.setdefault(actuator_label, [])
                            observations.append(observed)
                            rotation_calibration[actuator_label] = float(np.median(observations))
                            stage_assimilation["observed_degrees_per_substep"] = observed
                            log(f"stage response actuator={actuator_label} "
                                f"degrees_per_substep={rotation_calibration[actuator_label]:.6g} source=QC_fit")
                except Exception as exc:
                    stage_assimilation = {"success": False, "message": str(exc)}
                    failure_reason = f"stage_angle_fit_invalid: {exc}. Inverse refresh required."

        while (
                not qc_feedback_stage and
                (
                    remaining_substeps != 0 or
                    (extend_target_reacquire_sweep and not step_reacquired)
                )):
            if total_execution_pulses >= max_total_steps:
                failure_reason = (
                    f"Reached max_total_steps={max_total_steps} pulses during "
                    f"step {step_index}/{len(steps)} ({actuator_label}); "
                    f"{len(pulse_logs)}/{abs(pulse_count)} pulses executed in this step."
                )
                break

            extending_reacquire_sweep = extra_reacquire_sweep_pulses > 0
            if remaining_substeps == 0:
                remaining_substeps = move_sign * execution_pulse_substeps
                extra_reacquire_sweep_pulses += 1
                extending_reacquire_sweep = True

            chunk_substeps = min(
                abs(remaining_substeps), execution_pulse_substeps, max_rotation_chunk_substeps,
            )
            sim_steps = int(np.sign(remaining_substeps) * chunk_substeps)
            hardware_steps = hardware_direction * sim_steps
            command_angle_delta = pulse_angle_delta * chunk_substeps / execution_pulse_substeps

            if dry_run:
                actual_delta = _dry_run_rotation_delta_for_steps(
                    actuator_label,
                    sim_steps,
                    rotation_calibration,
                    rng,
                    dry_run_rotation_error=dry_run_rotation_error,
                    dry_run_pulse_response=dry_run_pulse_response,
                    pulse_substeps=execution_pulse_substeps,
                )
                x_physical[axis_index] += actual_delta
            else:
                if hardware is None or getattr(hardware, "rotation_stages", None) is None:
                    failure_reason = "hardware.rotation_stages is required to execute rotation moves."
                    break
                hardware.rotation_stages.move_relative_steps(controller, actuator, hardware_steps)

            x_estimate[axis_index] += command_angle_delta
            remaining_substeps -= sim_steps
            remaining_pulses = remaining_substeps / execution_pulse_substeps
            step_commanded_substeps += hardware_steps
            step_sim_substeps += sim_steps
            total_commanded_substeps += hardware_steps
            total_execution_pulses += 1

            if rotation_settle_delay and rotation_settle_delay > 0:
                time.sleep(rotation_settle_delay)

            chunk_qc = _read_quadcell_x_with_signal(
                hardware,
                dry_run=dry_run,
                dry_run_x=x_physical,
                base_mirrors=base_mirrors,
                qc_readout_sign=qc_readout_sign,
                qc_min_signal=qc_min_signal,
                qc_min_signals=qc_min_signals,
                dry_run_signal_strengths=dry_run_signal_strengths,
            )
            estimated_mirrors = S.unpack_variables(x_estimate, *base_mirrors)
            current_qc = chunk_qc
            estimate_reflection_count = int(S.get_reflection_count(*estimated_mirrors))
            simulation_reflection_count = (
                int(S.get_reflection_count(*S.unpack_variables(x_physical, *base_mirrors)))
                if dry_run else estimate_reflection_count
            )
            pulse_log = {
                "elapsed_s": float(time.perf_counter() - execution_t0),
                "estimate_x_after_pulse": x_estimate.tolist(),
                "physical_x_after_pulse": x_physical.tolist() if dry_run else None,
                "qc_normalized_x": chunk_qc.get("normalized_x"),
                "qc_read_error": chunk_qc.get("error"),
                **_qc_model_comparison(x_estimate, base_mirrors, chunk_qc, qc_readout_sign),
                "pulse": len(pulse_logs) + 1,
                "hardware_steps": int(hardware_steps),
                "sim_steps": int(sim_steps),
                "pulse_angle_delta": float(command_angle_delta),
                "nominal_pulse_angle_delta": float(pulse_angle_delta),
                "calibration_pulses_completed": abs(step_sim_substeps) / execution_pulse_substeps,
                "extra_reacquire_sweep_pulse": bool(extending_reacquire_sweep),
                "extra_reacquire_sweep_pulse_count": int(extra_reacquire_sweep_pulses),
                "expected_angle_after_pulse": float(x_estimate[axis_index]),
                "estimated_angle": float(x_estimate[axis_index]),
                "qc_x": chunk_qc["x"].tolist(),
                "qc_y": chunk_qc["y"].tolist(),
                "qc_signal_strengths": chunk_qc["signal_strengths"],
                "qc_signal_valid": bool(chunk_qc["valid"]),
                "qc_valid_mask": chunk_qc["valid_mask"],
                "sim_reflection_count": simulation_reflection_count,
                "angle_estimate_reflection_count": estimate_reflection_count,
                "sim_OPD": float(S.OPD_from_variables(x_estimate, *base_mirrors)),
            }
            if stage_move:
                failure_reason = _staging_qc_failure(chunk_qc, stage_limit)
                if failure_reason is not None:
                    pulse_log["failure_reason"] = failure_reason
            # Loss of signal (or a clipped position) is also a departure. It
            # must not stop the target sweep before it can regain the beam.
            if target_jump_step and not chunk_qc["valid"]:
                if not beam_departed_reacquire_window:
                    beam_departed_at = {
                        "execution_step": int(step_index), "pulse": len(pulse_logs) + 1,
                        "actuator": actuator_label, "qc_x": chunk_qc["x"].tolist(),
                        "reason": "qc_invalid_or_saturated",
                    }
                beam_departed_reacquire_window = True
                pulse_log["beam_departed_reacquire_window"] = True
                pulse_log["beam_departure_seen"] = True
            if chunk_qc["valid"]:
                pulse_log["qc_reacquire_max_abs"] = float(np.max(np.abs(chunk_qc["x"])))
                pulse_log["qc_in_reacquire_window"] = (
                    pulse_log["qc_reacquire_max_abs"] <= float(qc_reacquire_limit)
                )
                pulse_log["qc_inside_safety_limit"] = (
                    pulse_log["qc_reacquire_max_abs"] <= float(final_center_qc_safety_limit)
                )
                pulse_log["target_N_R_estimate_reached"] = (
                    int(pulse_log["sim_reflection_count"]) == int(target_N_R)
                )
                pulse_log["min_reacquire_pulses"] = int(min_reacquire_pulses)
                pulse_log["step_pulses_completed"] = int(len(pulse_logs) + 1)
                pulse_log["past_min_reacquire_pulses"] = (
                    abs(step_sim_substeps) >= int(min_reacquire_pulses) * execution_pulse_substeps
                )
                pulse_log["target_jump_step"] = bool(target_jump_step)
                pulse_log["target_center_step"] = bool(target_center_step)
                pulse_log["reacquire_stop_disabled"] = bool(reacquire_stop_disabled)
                pulse_log["qc_only_reacquire_stop"] = bool(qc_only_reacquire_stop)
                if target_jump_step and not pulse_log["qc_in_reacquire_window"]:
                    pulse_log["beam_departed_reacquire_window"] = True
                    if not beam_departed_reacquire_window:
                        beam_departed_at = {
                            "execution_step": int(step_index),
                            "pulse": int(len(pulse_logs) + 1),
                            "actuator": actuator_label,
                            "qc_x": chunk_qc["x"].tolist(),
                            "qc_reacquire_max_abs": float(pulse_log["qc_reacquire_max_abs"]),
                        }
                    beam_departed_reacquire_window = True
                else:
                    pulse_log["beam_departed_reacquire_window"] = False
                pulse_log["beam_departure_seen"] = bool(beam_departed_reacquire_window)
                pulse_log["reacquire_requires_departure"] = bool(
                    require_beam_departure_before_reacquire
                )
                departure_ok = (
                    True
                    if not require_beam_departure_before_reacquire
                    else beam_departed_reacquire_window
                )
                pulse_log["reacquire_departure_ok"] = bool(departure_ok)
                target_estimate_ok = (
                    not require_target_N_R_estimate_for_reacquire_stop or
                    pulse_log["target_N_R_estimate_reached"]
                )
                pulse_log["reacquire_target_estimate_ok"] = bool(target_estimate_ok)
                if reacquire_stop_enabled and reacquire_stop_mode == "center_crossing":
                    if departure_ok:
                        centered = np.abs(chunk_qc["x"]) <= reacquire_center_tolerance
                        crossed = np.zeros(2, dtype=bool)
                        if previous_center_qc_x is not None:
                            crossed = np.sign(previous_center_qc_x) * np.sign(chunk_qc["x"]) < 0
                        if pulse_log["past_min_reacquire_pulses"]:
                            for qc_index in range(2):
                                if not qc_center_crossed_mask[qc_index] and (centered[qc_index] or crossed[qc_index]):
                                    qc_center_crossed_mask[qc_index] = True
                                    qc_center_crossed_at[qc_index] = {
                                        "execution_step": int(step_index),
                                        "pulse": int(len(pulse_logs) + 1),
                                        "actuator": actuator_label,
                                        "previous_qc_x": (
                                            None if previous_center_qc_x is None
                                            else float(previous_center_qc_x[qc_index])
                                        ),
                                        "qc_x": float(chunk_qc["x"][qc_index]),
                                        "crossed_zero": bool(crossed[qc_index]),
                                    }
                        previous_center_qc_x = chunk_qc["x"].copy()
                    else:
                        previous_center_qc_x = None
                center_stop_ok = bool(
                    reacquire_stop_mode == "window" or np.any(qc_center_crossed_mask)
                )
                pulse_log["reacquire_center_stop_ok"] = center_stop_ok
                if (
                    reacquire_stop_enabled and
                    pulse_log["qc_in_reacquire_window"] and
                    departure_ok and
                    pulse_log["past_min_reacquire_pulses"] and
                    target_estimate_ok and
                    center_stop_ok
                ):
                    pulse_log["reacquired"] = True
                    if reacquire_stop_mode == "center_crossing":
                        pulse_log["reacquired_by"] = (
                            "departed_then_qc_center_crossing"
                            if require_beam_departure_before_reacquire
                            else "qc_center_crossing"
                        )
                    elif qc_only_reacquire_stop:
                        pulse_log["reacquired_by"] = (
                            "departed_then_qc_only"
                            if require_beam_departure_before_reacquire
                            else "qc_only"
                        )
                    else:
                        pulse_log["reacquired_by"] = (
                            "qc_and_target_N_R_estimate"
                            if pulse_log["target_N_R_estimate_reached"]
                            else "departed_then_qc_only"
                        )
                    if qc_only_reacquire_stop or reacquire_stop_mode == "center_crossing":
                        requires_inverse_refresh = True
                    step_reacquired = True
                    beam_reacquired_during_sweep = True
                    reacquired_qc = chunk_qc
                    reacquired_at = {
                        "execution_step": int(step_index),
                        "pulse": int(len(pulse_logs) + 1),
                        "actuator": actuator_label,
                    }
                    last_reacquisition_axis = axis_index
                    last_reacquisition_actuator = actuator_label
                    last_reacquired_by = pulse_log["reacquired_by"]
                    if runtime_sweep_validation is not None and simulation_reflection_count != int(target_N_R):
                        if dry_run or require_target_N_R_estimate_for_reacquire_stop:
                            failure_reason = (
                                "target_not_reached_on_reacquisition: stopped on QC return, but "
                                f"{'simulated physical' if dry_run else 'estimated'} N_R={simulation_reflection_count} "
                                f"does not match target {target_N_R}. Inverse refresh required."
                            )
                            pulse_log["failure_reason"] = failure_reason
                        else:
                            warning = (
                                "target_N_R_unverified: measured QC reacquired, but pulse-based "
                                f"N_R estimate={simulation_reflection_count} differs from target {target_N_R}. "
                                "Inverse refresh required to verify reflection count."
                            )
                            requires_inverse_refresh = True
                            execution_warnings.append(warning)
                            pulse_log["reacquisition_warning"] = warning
                            log("warning " + warning)
            elif reacquire_stop_enabled and reacquire_stop_mode == "center_crossing":
                # A sign flip across lost signal is not a measured center crossing.
                previous_center_qc_x = None
                qc_center_crossed_mask[:] = False
                qc_center_crossed_at = [None, None]
            if reacquire_stop_enabled and reacquire_stop_mode == "center_crossing":
                pulse_log["qc_center_crossed_mask"] = qc_center_crossed_mask.tolist()
            if reacquire_stop_enabled:
                blockers = []
                if not chunk_qc["valid"]:
                    blockers.append("qc_signal_invalid")
                elif not pulse_log["qc_in_reacquire_window"]:
                    blockers.append("outside_qc_window")
                if require_beam_departure_before_reacquire and not beam_departed_reacquire_window:
                    blockers.append("departure_not_seen")
                if abs(step_sim_substeps) < int(min_reacquire_pulses) * execution_pulse_substeps:
                    blockers.append("minimum_motion_not_reached")
                if require_target_N_R_estimate_for_reacquire_stop and not pulse_log.get("target_N_R_estimate_reached", False):
                    blockers.append("target_N_R_estimate_not_reached")
                if reacquire_stop_mode == "center_crossing" and not np.any(qc_center_crossed_mask):
                    blockers.append("center_crossing_not_seen")
                pulse_log["reacquire_stop_blockers"] = blockers
                progress_state = (tuple(blockers), tuple(qc_center_crossed_mask))
                now = time.perf_counter()
                if profile and (step_reacquired or progress_state != last_jump_stop_state or now - last_jump_progress_t >= 1.0):
                    log(
                        f"jump step={step_index}/{len(steps)} actuator={actuator_label} "
                        f"command={len(pulse_logs) + 1} substeps={sim_steps} "
                        f"calibration_pulses={abs(step_sim_substeps) / execution_pulse_substeps:g}/{min_reacquire_pulses} minimum "
                        f"qc_x=({chunk_qc['x'][0]:.3f},{chunk_qc['x'][1]:.3f}) "
                        f"signals={chunk_qc['signal_strengths']} valid_mask={chunk_qc['valid_mask']} "
                        f"N_R_model={estimate_reflection_count} "
                        f"N_R_simulated_physical={simulation_reflection_count if dry_run else 'unavailable'} "
                        f"crossed={qc_center_crossed_mask.tolist()} "
                        f"wait={','.join(blockers) if blockers else 'none'} stopped={step_reacquired}"
                    )
                    last_jump_progress_t = now
                    last_jump_stop_state = progress_state
            pulse_logs.append(pulse_log)
            if step_reacquired or failure_reason is not None:
                break

        after_qc = (
            chunk_qc if len(pulse_logs) > 0
            else _read_quadcell_x_with_signal(
                hardware,
                dry_run=dry_run,
                dry_run_x=x_physical,
                base_mirrors=base_mirrors,
                qc_readout_sign=qc_readout_sign,
                qc_min_signal=qc_min_signal,
                qc_min_signals=qc_min_signals,
                dry_run_signal_strengths=dry_run_signal_strengths,
            )
        )
        estimate_metrics = _path_metrics_from_x(
            x_estimate,
            base_mirrors,
            include_edge_ends=False,
        )
        physical_metrics = None
        if dry_run:
            physical_metrics = _path_metrics_from_x(
                x_physical,
                base_mirrors,
                include_edge_ends=False,
            )

        command_sizes = [abs(int(pulse["hardware_steps"])) for pulse in pulse_logs]
        command_size_min = min(command_sizes, default=0)
        command_size_max = max(command_sizes, default=0)
        step_rollbacks = sum(bool(pulse.get("rolled_back")) for pulse in pulse_logs)
        entry = {
            "completed": bool(remaining_pulses == 0 and failure_reason is None),
            "stage_control_mode": "qc_endpoint_feedback" if qc_feedback_stage else None,
            "stage_feedback": stage_detail,
            "stage_assimilation": stage_assimilation,
            "stage_angle_fit_accepted": stage_assimilation.get("accepted", False) if stage_assimilation is not None else None,
            "failure_reason": failure_reason,
            "remaining_pulses": float(remaining_pulses),
            "remaining_substeps": int(remaining_substeps),
            "estimate_x_before_step": x_before_step.tolist(),
            "physical_x_before_step": physical_x_before_step.tolist() if dry_run else None,
            **_qc_model_comparison(x_estimate, base_mirrors, after_qc, qc_readout_sign),
            "execution_step": step_index,
            "planner_step": step.get("step"),
            "phase": (
                "target_reacquisition" if target_jump_step else
                "target_centering" if target_center_step else
                "stage"
            ),
            "actuator": actuator_label,
            "axis_index": axis_index,
            "target_jump_step": bool(target_jump_step),
            "target_center_step": bool(target_center_step),
            "reacquire_stop_disabled": bool(reacquire_stop_disabled),
            "qc_only_reacquire_stop": bool(qc_only_reacquire_stop),
            "reacquire_stop_mode": reacquire_stop_mode,
            "qc_center_crossed_mask": qc_center_crossed_mask.tolist(),
            "qc_center_crossed_at": qc_center_crossed_at,
            "target_jump_sequence_part": target_jump_sequence_part,
            "target_sweep_validation": runtime_sweep_validation,
            "target_reacquire_extension_enabled": bool(extend_target_reacquire_sweep),
            "extra_reacquire_sweep_pulses": int(extra_reacquire_sweep_pulses),
            "planned_angle_delta": planned_angle_delta,
            "planned_path_angle_delta": float(step["command_value"]),
            "command_value": planned_angle_delta,
            "degrees_per_substep": float(degrees_per_substep),
            "pulse_response_source": pulse_source,
            "execution_pulse_substeps": int(execution_pulse_substeps),
            "pulse_angle_delta": float(pulse_angle_delta),
            "planned_pulse_count": int(pulse_count),
            "min_reacquire_pulses": int(min_reacquire_pulses),
            "min_reacquire_pulse_fraction": float(min_reacquire_pulse_fraction),
            "pulse_count": int(pulse_count),
            "executed_pulse_count": int(len(pulse_logs)),
            "command_chunk_substeps_min": command_size_min,
            "command_chunk_substeps_max": command_size_max,
            "rollback_count": step_rollbacks,
            "executed_angle_delta_estimate": float(x_estimate[axis_index] - x_before_step[axis_index]),
            "remaining_angle_delta_estimate": float(
                planned_angle_delta - (x_estimate[axis_index] - x_before_step[axis_index])
            ),
            "expected_angle_moved": float(expected_angle_moved),
            "residual_angle_after_pulse_rounding": float(residual_angle),
            "total_commanded_substeps": int(step_commanded_substeps),
            "total_sim_substeps": int(step_sim_substeps),
            "before_qc_x": before_qc["x"].tolist(),
            "before_qc_signal_strengths": before_qc["signal_strengths"],
            "before_qc_signal_valid": bool(before_qc["valid"]),
            "after_qc_x": after_qc["x"].tolist(),
            "after_qc_signal_strengths": after_qc["signal_strengths"],
            "after_qc_signal_valid": bool(after_qc["valid"]),
            "estimate_path_metrics": estimate_metrics,
            "physical_path_metrics": physical_metrics,
            "estimate_x_after_step": x_estimate.tolist(),
            "physical_x_after_step": x_physical.tolist() if dry_run else None,
            "beam_reacquired_after_step": bool(step_reacquired),
            "beam_departure_seen": bool(beam_departed_reacquire_window),
            "beam_departed_at": beam_departed_at,
            "chunks": pulse_logs,
            "pulses": pulse_logs,
            "dt": time.perf_counter() - step_t0,
        }
        execution_log.append(entry)
        extra_sweep_text = (
            f" extra_reacquire_pulses={extra_reacquire_sweep_pulses}"
            if extra_reacquire_sweep_pulses else
            ""
        )
        log(
            f"step={step_index}/{len(steps)} actuator={actuator_label} "
            f"planned_pulses={pulse_count} executed_pulses={len(pulse_logs)} "
            f"substeps={step_commanded_substeps} "
            f"nominal_pulse_substeps={execution_pulse_substeps} "
            f"chunk_substeps={command_size_min}..{command_size_max} rollbacks={step_rollbacks} "
            f"residual_angle={residual_angle:.4g} "
            f"qc_valid={after_qc['valid']} "
            f"qc_x=({after_qc['x'][0]:.3f},{after_qc['x'][1]:.3f}) "
            f"{'N_R_simulated_physical' if dry_run else 'N_R_model'}="
            f"{(physical_metrics if dry_run else estimate_metrics)['reflection_count']} "
            f"N_R_angle_estimate={estimate_metrics['reflection_count']} "
            f"model_qc_residual={entry['qc_residual_norm']:.4g} "
            f"stage_stop={stage_detail.get('stop_reason') if stage_detail else None} "
            f"stage_fit_accepted={entry['stage_angle_fit_accepted']} "
            f"departed={beam_departed_reacquire_window} reacquired={step_reacquired} "
            f"failure={failure_reason}{extra_sweep_text}"
        )
        if failure_reason is not None:
            break
        if (
                failure_reason is None and
                last_stage_step_index > 0 and
                step_index >= last_stage_step_index and
                not target_jump_step):
            stage_reached_during_execution = True
        if step_reacquired:
            if requires_inverse_refresh:
                entry["reacquisition_assimilation_skipped"] = "requires_inverse_refresh"
            else:
                try:
                    x_fit, last_reacquisition_assimilation = assimilate_rotation_angle_from_qc(
                        x_estimate,
                        axis_index,
                        after_qc["y"],
                        *base_mirrors,
                        qc_readout_sign=qc_readout_sign,
                        angle_prior=x_estimate[axis_index],
                    )
                    x_estimate[:] = x_fit
                    if dry_run:
                        x_physical[:] = x_estimate
                    entry["reacquisition_assimilation"] = last_reacquisition_assimilation
                except Exception as exc:
                    entry["reacquisition_assimilation_error"] = str(exc)
            break

    stage_only_refresh = bool(actuation_plan.get("stage_only_refresh", False))
    if failure_reason is None and stage_only_refresh:
        requires_inverse_refresh = True
        if last_reacquired_by is None:
            last_reacquired_by = "surface_stage_qc_centered"

    if failure_reason is None:
        current_qc = (
            reacquired_qc if reacquired_qc is not None
            else _read_quadcell_x_with_signal(
                hardware,
                dry_run=dry_run,
                dry_run_x=x_physical,
                base_mirrors=base_mirrors,
                qc_readout_sign=qc_readout_sign,
                qc_min_signal=qc_min_signal,
                qc_min_signals=qc_min_signals,
                dry_run_signal_strengths=dry_run_signal_strengths,
            )
        )
        beam_reacquired_now = bool(
            current_qc["valid"] and
            float(np.max(np.abs(current_qc["x"]))) <= float(qc_reacquire_limit) and
            (not center_crossing_required or beam_reacquired_during_sweep) and
            (
                stage_only_refresh or
                not require_beam_departure_before_reacquire or
                beam_departed_reacquire_window
            ) and
            (
                requires_inverse_refresh or
                not require_target_N_R_estimate_for_reacquire_stop or
                int(S.get_reflection_count(*S.unpack_variables(x_estimate, *base_mirrors))) == int(target_N_R)
            )
        )
        if not beam_reacquired_now:
            if (
                    not stage_only_refresh and
                    require_beam_departure_before_reacquire and
                    not beam_departed_reacquire_window):
                failure_reason = (
                    "beam_not_reacquired: beam never left the "
                    f"+/-{qc_reacquire_limit} mm reacquisition window."
                )
            elif center_crossing_required and not beam_reacquired_during_sweep:
                failure_reason = (
                    "beam_not_centered: at least one quadcell must hit or cross x=0 "
                    "with valid signal and the required reacquisition conditions "
                    f"during the target sweep (crossed={qc_center_crossed_mask.tolist()})."
                )
            else:
                failure_reason = (
                    "beam_not_reacquired: QC did not become valid within "
                    f"+/-{qc_reacquire_limit} mm during reacquisition sweep."
                )
        elif (
                not requires_inverse_refresh and
                bool(actuation_plan.get("qc_only_reacquire_stop", False)) and
                any(entry.get("target_jump_step", False) for entry in execution_log)):
            requires_inverse_refresh = True
            if last_reacquired_by is None:
                last_reacquired_by = (
                    "departed_then_qc_only"
                    if require_beam_departure_before_reacquire else
                    "qc_only"
                )

    if failure_reason is None and final_center_after_reacquire and not requires_inverse_refresh:
        center_qc = _read_quadcell_x_with_signal(
            hardware,
            dry_run=dry_run,
            dry_run_x=x_physical,
            base_mirrors=base_mirrors,
            qc_readout_sign=qc_readout_sign,
            qc_min_signal=qc_min_signal,
            qc_min_signals=qc_min_signals,
            dry_run_signal_strengths=dry_run_signal_strengths,
        )
        if center_qc["valid"] and float(np.max(np.abs(center_qc["x"]))) <= float(final_qc_tolerance):
            log(
                f"final center skipped qc_x=({center_qc['x'][0]:.3f},{center_qc['x'][1]:.3f})"
            )
        else:
            try:
                x_centered, center_res = S.solve_recenter_angles(
                    x_estimate,
                    *base_mirrors,
                    target_reflections=int(target_N_R),
                    max_qc_error=max(float(qc_reacquire_limit), float(final_qc_tolerance)),
                    u_min=u_min,
                    u_max=u_max,
                    sigma_edge=sigma_edge,
                    include_edge_ends=False,
                    verbose=0,
                )
            except Exception as exc:
                x_centered = x_estimate.copy()
                center_res = SimpleNamespace(success=False, message=str(exc))

            center_plan_steps = []
            center_path_plan = None
            if getattr(center_res, "success", False):
                try:
                    _, center_path_plan = S.append_waypoint_constrained_path_steps(
                        center_plan_steps,
                        x_estimate,
                        x_centered,
                        *base_mirrors,
                        max_axis_splits=final_center_max_axis_splits,
                        max_waypoint_depth=final_center_waypoint_depth,
                        max_qc_error=max(float(final_center_qc_safety_limit), float(final_qc_tolerance)),
                        max_qc_difference=None,
                        preserve_reflection_count=True,
                        motion_samples_per_step=25,
                        fast_motion_samples_per_step=5,
                        u_min=u_min,
                        u_max=u_max,
                        enforce_edge_bounds=False,
                        include_edge_ends=False,
                        constraint_tolerance=0.0,
                    )
                    if center_path_plan.get("failure_reason") is not None:
                        center_plan_steps[:] = []
                except Exception as exc:
                    center_path_plan = {
                        "failure_reason": str(exc),
                        "search_mode": "exception",
                    }

            for center_step in center_plan_steps:
                center_step["reflection_count_final_center_move"] = True

            log(
                f"final center solve success={getattr(center_res, 'success', False)} "
                f"steps={len(center_plan_steps)} "
                f"path_failure={None if center_path_plan is None else center_path_plan.get('failure_reason')}"
            )

            for center_index, center_step in enumerate(center_plan_steps, start=1):
                if total_execution_pulses >= max_total_steps:
                    failure_reason = f"Reached max_total_steps={max_total_steps} before final centering completed."
                    break
                if center_index > 1 and step_settle_delay and step_settle_delay > 0:
                    time.sleep(step_settle_delay)

                actuator_label = center_step.get("actuator")
                mapping = actuator_map.get(actuator_label)
                if mapping is None:
                    failure_reason = f"No hardware mapping for actuator {actuator_label}."
                    break
                if mapping.get("kind") != "rotation":
                    failure_reason = f"Final centering only supports rotation actuators, got {actuator_label}."
                    break

                detail = _execute_rotation_step_fixed(
                    center_step,
                    mapping,
                    hardware,
                    x_estimate,
                    x_physical,
                    base_mirrors,
                    rotation_calibration,
                    rng,
                    dry_run=dry_run,
                    dry_run_rotation_error=dry_run_rotation_error,
                    qc_readout_sign=qc_readout_sign,
                    qc_step_tolerance=final_qc_tolerance,
                    qc_safety_limit=max(float(final_center_qc_safety_limit), float(final_qc_tolerance)),
                    fast_qc_avg=1,
                    fast_qc_delay=0.0,
                    max_rotation_chunks_per_step=max_rotation_chunk_substeps,
                    max_rotation_chunk_substeps=max_rotation_chunk_substeps,
                    min_rotation_chunk_substeps=min_rotation_chunk_substeps,
                    rotation_settle_delay=rotation_settle_delay,
                )
                total_commanded_substeps += int(detail.get("total_commanded_substeps", 0))
                total_execution_pulses += len(detail.get("chunks", []))

                center_after_qc = _read_quadcell_x_with_signal(
                    hardware,
                    dry_run=dry_run,
                    dry_run_x=x_physical,
                    base_mirrors=base_mirrors,
                    qc_readout_sign=qc_readout_sign,
                    qc_min_signal=qc_min_signal,
                    qc_min_signals=qc_min_signals,
                    dry_run_signal_strengths=dry_run_signal_strengths,
                )
                assimilation = None
                assimilation_error = None
                if center_after_qc["valid"]:
                    try:
                        x_fit, assimilation = assimilate_rotation_angle_from_qc(
                            x_estimate,
                            int(center_step["axis_index"]),
                            center_after_qc["y"],
                            *base_mirrors,
                            qc_readout_sign=qc_readout_sign,
                            angle_prior=x_estimate[int(center_step["axis_index"])],
                        )
                        x_estimate[:] = x_fit
                        if dry_run:
                            x_physical[:] = x_estimate
                    except Exception as exc:
                        assimilation_error = str(exc)

                center_entry = {
                    "execution_step": len(execution_log) + len(center_execution_log) + 1,
                    "planner_step": center_step.get("step"),
                    "phase": "final_center",
                    "actuator": actuator_label,
                    "axis_index": int(center_step["axis_index"]),
                    "planned_angle_delta": float(center_step["command_value"]),
                    "command_value": float(center_step["command_value"]),
                    "detail": detail,
                    "after_qc_x": center_after_qc["x"].tolist(),
                    "after_qc_signal_strengths": center_after_qc["signal_strengths"],
                    "after_qc_signal_valid": bool(center_after_qc["valid"]),
                    "assimilation": assimilation,
                    "assimilation_error": assimilation_error,
                }
                center_execution_log.append(center_entry)
                log(
                    f"final_center step={center_index}/{len(center_plan_steps)} "
                    f"actuator={actuator_label} stop={detail.get('stop_reason')} "
                    f"qc_x=({center_after_qc['x'][0]:.3f},{center_after_qc['x'][1]:.3f})"
                )
                if detail.get("failure_reason"):
                    final_center_path_failure = "Final centering path failed: " + str(detail["failure_reason"])
                    break
                if center_after_qc["valid"] and float(np.max(np.abs(center_after_qc["x"]))) <= float(final_qc_tolerance):
                    break

            if not getattr(center_res, "success", False):
                center_execution_log.append({
                    "phase": "final_center",
                    "success": False,
                    "failure_reason": getattr(center_res, "message", "No final center solution found."),
                })
            elif center_path_plan is not None and center_path_plan.get("failure_reason") is not None:
                final_center_path_failure = (
                    "No constrained final centering path found: " +
                    str(center_path_plan["failure_reason"])
                )
                center_execution_log.append({
                    "phase": "final_center",
                    "success": False,
                    "failure_reason": final_center_path_failure,
                    "path_plan": center_path_plan,
                })

    if failure_reason is None and final_center_after_reacquire and not requires_inverse_refresh:
        feedback_qc = _read_quadcell_x_with_signal(
            hardware,
            dry_run=dry_run,
            dry_run_x=x_physical,
            base_mirrors=base_mirrors,
            qc_readout_sign=qc_readout_sign,
            qc_min_signal=qc_min_signal,
            qc_min_signals=qc_min_signals,
            dry_run_signal_strengths=dry_run_signal_strengths,
        )
        feedback_started = bool(
            feedback_qc["valid"] and
            float(np.max(np.abs(feedback_qc["x"]))) > float(final_qc_tolerance)
        )
        feedback_pulse_substeps = int(abs(
            pulse_table.get("pulse_substeps", calibration_pulse_substeps)
            if isinstance(pulse_table, dict) else calibration_pulse_substeps
        ))
        feedback_pulse_substeps = max(1, feedback_pulse_substeps)
        feedback_labels = ("M1.dangle", "M2.dangle", "M3.dangle", "M4.dangle")

        for feedback_index in range(1, int(final_center_feedback_max_pulses) + 1):
            if not feedback_started:
                break
            if total_execution_pulses >= max_total_steps:
                failure_reason = f"Reached max_total_steps={max_total_steps} during final centering feedback."
                break
            if not feedback_qc["valid"]:
                break

            current_norm = float(np.linalg.norm(feedback_qc["x"]))
            current_max_abs = float(np.max(np.abs(feedback_qc["x"])))
            if current_max_abs <= float(final_qc_tolerance):
                break

            accepted = False
            best_attempt_log = None
            for actuator_label in feedback_labels:
                mapping = actuator_map.get(actuator_label)
                if mapping is None or mapping.get("kind") != "rotation":
                    continue
                axis_index = int({"M1.dangle": 1, "M2.dangle": 3, "M3.dangle": 5, "M4.dangle": 7}[actuator_label])
                controller = mapping.get("controller", DEFAULT_ROTATION_CONTROLLER)
                actuator = int(mapping["actuator"])
                hardware_direction = int(np.sign(mapping.get("direction", 1)) or 1)
                degrees_per_substep = rotation_calibration.get(
                    actuator_label,
                    DEFAULT_ROTATION_DEGREES_PER_SUBSTEP
                )

                for sign in (1, -1):
                    if total_execution_pulses >= max_total_steps:
                        failure_reason = f"Reached max_total_steps={max_total_steps} during final centering feedback."
                        break

                    pulse_angle_delta, pulse_source = _pulse_response_from_table(
                        pulse_table,
                        actuator_label,
                        sign,
                        feedback_pulse_substeps,
                        degrees_per_substep,
                    )
                    sim_steps = sign * feedback_pulse_substeps
                    hardware_steps = hardware_direction * sim_steps
                    x_estimate_before = x_estimate.copy()
                    x_physical_before = x_physical.copy()

                    if dry_run:
                        actual_delta = _dry_run_rotation_delta_for_steps(
                            actuator_label,
                            sim_steps,
                            rotation_calibration,
                            rng,
                            dry_run_rotation_error=dry_run_rotation_error,
                            dry_run_pulse_response=dry_run_pulse_response,
                            pulse_substeps=feedback_pulse_substeps,
                        )
                        x_physical[axis_index] += actual_delta
                    else:
                        if hardware is None or getattr(hardware, "rotation_stages", None) is None:
                            failure_reason = "hardware.rotation_stages is required to execute final centering feedback."
                            break
                        hardware.rotation_stages.move_relative_steps(controller, actuator, hardware_steps)

                    x_estimate[axis_index] += pulse_angle_delta
                    total_commanded_substeps += hardware_steps
                    total_execution_pulses += 1
                    if rotation_settle_delay and rotation_settle_delay > 0:
                        time.sleep(rotation_settle_delay)

                    trial_qc = _read_quadcell_x_with_signal(
                        hardware,
                        dry_run=dry_run,
                        dry_run_x=x_physical,
                        base_mirrors=base_mirrors,
                        qc_readout_sign=qc_readout_sign,
                        qc_min_signal=qc_min_signal,
                        qc_min_signals=qc_min_signals,
                        dry_run_signal_strengths=dry_run_signal_strengths,
                    )
                    trial_norm = (
                        float(np.linalg.norm(trial_qc["x"]))
                        if trial_qc["valid"] else float("inf")
                    )
                    trial_max_abs = (
                        float(np.max(np.abs(trial_qc["x"])))
                        if trial_qc["valid"] else float("inf")
                    )
                    improved = (
                        trial_qc["valid"] and
                        trial_max_abs <= float(final_center_qc_safety_limit) and
                        trial_norm < current_norm - float(final_center_min_improvement)
                    )
                    attempt_log = {
                        "phase": "final_center_feedback",
                        "feedback_index": int(feedback_index),
                        "actuator": actuator_label,
                        "axis_index": int(axis_index),
                        "sign": int(sign),
                        "hardware_steps": int(hardware_steps),
                        "sim_steps": int(sim_steps),
                        "pulse_angle_delta": float(pulse_angle_delta),
                        "pulse_response_source": pulse_source,
                        "before_qc_x": feedback_qc["x"].tolist(),
                        "after_qc_x": trial_qc["x"].tolist(),
                        "before_norm": float(current_norm),
                        "after_norm": float(trial_norm),
                        "after_max_abs": float(trial_max_abs),
                        "accepted": bool(improved),
                    }

                    if improved:
                        assimilation = None
                        assimilation_error = None
                        try:
                            x_fit, assimilation = assimilate_rotation_angle_from_qc(
                                x_estimate,
                                axis_index,
                                trial_qc["y"],
                                *base_mirrors,
                                qc_readout_sign=qc_readout_sign,
                                angle_prior=x_estimate[axis_index],
                            )
                            x_estimate[:] = x_fit
                            if dry_run:
                                x_physical[:] = x_estimate
                        except Exception as exc:
                            assimilation_error = str(exc)
                        attempt_log["assimilation"] = assimilation
                        attempt_log["assimilation_error"] = assimilation_error
                        center_execution_log.append(attempt_log)
                        feedback_qc = trial_qc
                        accepted = True
                        break

                    if dry_run:
                        x_physical[:] = x_physical_before
                    else:
                        hardware.rotation_stages.move_relative_steps(controller, actuator, -hardware_steps)
                        if rotation_settle_delay and rotation_settle_delay > 0:
                            time.sleep(rotation_settle_delay)
                    x_estimate[:] = x_estimate_before
                    total_commanded_substeps -= hardware_steps
                    attempt_log["rolled_back"] = True
                    center_execution_log.append(attempt_log)
                    best_attempt_log = attempt_log

                if failure_reason is not None or accepted:
                    break

            if failure_reason is not None:
                break
            if accepted:
                log(
                    f"final_center_feedback pulse={feedback_index} "
                    f"qc_x=({feedback_qc['x'][0]:.3f},{feedback_qc['x'][1]:.3f})"
                )
                continue

            center_execution_log.append({
                "phase": "final_center_feedback",
                "success": False,
                "failure_reason": "No feedback pulse improved QC centering.",
                "last_attempt": best_attempt_log,
            })
            break

    final_qc = _read_quadcell_x_with_signal(
        hardware,
        dry_run=dry_run,
        dry_run_x=x_physical,
        base_mirrors=base_mirrors,
        qc_readout_sign=qc_readout_sign,
        qc_min_signal=qc_min_signal,
        qc_min_signals=qc_min_signals,
        dry_run_signal_strengths=dry_run_signal_strengths,
    )
    final_x = x_physical if dry_run else x_estimate
    final_mirrors = S.unpack_variables(final_x, *base_mirrors)
    final_res = S.set_OPD_result_full_x(
        SimpleNamespace(success=False, message=""),
        *final_mirrors
    )
    final_OPD = S.OPD_from_variables(final_x, *base_mirrors)
    final_sim_qc = S.quadcell_errors_from_variables(final_x, *base_mirrors)
    final_sim_reflections = S.get_reflection_count(*final_mirrors)
    final_qc_max_abs = float(np.max(np.abs(final_qc["x"])))
    final_qc_centered = final_qc_max_abs <= float(final_qc_tolerance)
    beam_reacquired = bool(
        final_qc["valid"] and
        final_qc_max_abs <= float(qc_reacquire_limit)
    )
    target_reflection_sim_reached = int(final_sim_reflections) == int(target_N_R)
    stage_reached = bool(
        stage_reached_during_execution or
        (
            failure_reason is None and stage_only_refresh and
            int(final_sim_reflections) == int(actuation_plan.get("stage_reflections", final_sim_reflections)) and
            final_qc["valid"] and
            final_qc_max_abs <= float(qc_reacquire_limit)
        )
    )

    if failure_reason is None and not beam_reacquired:
        if not final_qc["valid"]:
            failure_reason = "beam_not_reacquired: QC signal below threshold."
        else:
            failure_reason = (
                f"beam_not_reacquired: final QC max abs {final_qc_max_abs:.4g} "
                f"exceeds reacquisition limit {qc_reacquire_limit}."
            )
    if failure_reason is None and not requires_inverse_refresh and not final_qc_centered:
        failure_reason = (
            f"beam_not_centered: final QC max abs {final_qc_max_abs:.4g} "
            f"exceeds {final_qc_tolerance}."
        )
    if failure_reason is None and final_center_after_reacquire and not requires_inverse_refresh and len(center_execution_log) == 0:
        if final_qc["valid"] and final_qc_max_abs > float(final_qc_tolerance):
            failure_reason = (
                "beam_not_centered: final centering did not run and final QC "
                f"max abs {final_qc_max_abs:.4g} exceeds {final_qc_tolerance}."
            )
    if failure_reason is None and (not requires_inverse_refresh or (dry_run and not stage_only_refresh)) and not target_reflection_sim_reached:
        failure_reason = (
            f"simulated reflection count {final_sim_reflections} != target {target_N_R}."
        )

    final_success = failure_reason is None
    if not final_success and (total_execution_pulses > 0 or calibration_result is not None):
        requires_inverse_refresh = True
    final_res.success = bool(final_success)
    completion_status = "failed"
    if final_success and stage_only_refresh:
        completion_status = "staged_pending_inverse_refresh"
        final_res.message = (
            "Surface staging point reached; inverse refresh required before "
            "the target reflection-count jump."
        )
    elif final_success and dry_run:
        completion_status = "completed_in_simulation"
        final_res.message = f"Dry run completed with simulated physical N_R={final_sim_reflections}."
    elif final_success and requires_inverse_refresh:
        completion_status = "reacquired_pending_inverse_refresh"
        final_res.message = (
            f"QC reacquired; target N_R={target_N_R} is unverified until inverse refresh."
        )
    elif final_success:
        completion_status = "completed"
        final_res.message = "Reflection-count execution reacquired centered QC signal."
    else:
        final_res.message = "Reflection-count execution stopped before success criteria were met."

    execution = {
        "success": bool(final_success),
        "failure_reason": failure_reason,
        "completion_status": completion_status,
        "completion_message": str(final_res.message),
        "warnings": execution_warnings,
        "reflection_count_change": True,
        "reflection_count_reacquisition": True,
        "rotation_only": True,
        "qc_path_unconstrained": True,
        "target_N_R": int(target_N_R),
        "reacquisition_strategy": reacquisition_strategy,
        "stage_only_refresh": bool(stage_only_refresh),
        "stage_reached": bool(stage_reached),
        "stage_arrival": stage_arrival,
        "staging_angle_targets": bool(track_stage_angles),
        "target_N_R_reached": bool(target_reflection_sim_reached),
        "target_N_R_reached_source": "simulated_physical" if dry_run else "angle_estimate",
        "stage_plan": actuation_plan.get("stage_plan"),
        "target_jump_step": actuation_plan.get("target_jump_step"),
        "target_center_after_jump": bool(target_center_after_jump),
        "target_centered_plan": bool(actuation_plan.get("target_centered_plan", False)),
        "target_center_u_min": actuation_plan.get("target_center_u_min"),
        "target_center_u_max": actuation_plan.get("target_center_u_max"),
        "target_center_steps": actuation_plan.get("target_center_steps", []),
        "target_center_plan": actuation_plan.get("target_center_plan"),
        "requires_inverse_refresh": bool(requires_inverse_refresh),
        "suggested_next_step": (
            "recover valid QC signal, take light/dark images and run optimize_inverse before replanning"
            if requires_inverse_refresh and not final_success else
            actuation_plan.get(
                "suggested_next_step",
                "take light/dark images and run optimize_inverse"
            )
            if requires_inverse_refresh else None
        ),
        "reacquired_by": last_reacquired_by,
        "reacquire_stop_mode": reacquire_stop_mode,
        "reacquire_center_tolerance": float(reacquire_center_tolerance),
        "qc_center_crossing_required": bool(center_crossing_required),
        "qc_center_crossed_mask": qc_center_crossed_mask.tolist(),
        "qc_center_crossed_at": qc_center_crossed_at,
        "final_sim_reflection_count": int(final_sim_reflections),
        "final_angle_estimate_reflection_count": int(S.get_reflection_count(*S.unpack_variables(x_estimate, *base_mirrors))),
        "target_reflection_sim_reached": bool(target_reflection_sim_reached),
        "reflection_count_verified": bool(dry_run),
        "beam_reacquired": bool(beam_reacquired),
        "beam_reacquired_during_sweep": bool(beam_reacquired_during_sweep),
        "require_beam_departure_before_reacquire": bool(require_beam_departure_before_reacquire),
        "beam_departed_reacquire_window": bool(beam_departed_reacquire_window),
        "beam_departed_at": beam_departed_at,
        "reacquired_at": reacquired_at,
        "reacquired_qc_x": (
            None if reacquired_qc is None else reacquired_qc["x"].tolist()
        ),
        "reacquisition_assimilation": last_reacquisition_assimilation,
        "final_qc_centered": bool(final_qc_centered),
        "final_qc_max_abs": float(final_qc_max_abs),
        "final_qc_tolerance": float(final_qc_tolerance),
        "qc_reacquire_limit": float(qc_reacquire_limit),
        "stage_qc_limit": float(stage_qc_limit),
        "final_center_qc_safety_limit": float(final_center_qc_safety_limit),
        "final_qc_x": final_qc["x"].tolist(),
        "final_qc_y": final_qc["y"].tolist(),
        "final_qc_signal_strengths": final_qc["signal_strengths"],
        "final_qc_signal_thresholds": final_qc["signal_thresholds"],
        "final_qc_signal_valid": bool(final_qc["valid"]),
        "final_qc_valid_mask": final_qc["valid_mask"],
        "final_OPD": float(final_OPD),
        "final_sim_qc": [float(final_sim_qc[0]), float(final_sim_qc[1])],
        "final_mirrors": _mirrors_to_lists(final_mirrors),
        "target_mirrors": _mirrors_to_lists(mirrors_target),
        "total_commanded_substeps": int(total_commanded_substeps),
        "total_execution_pulses": int(total_execution_pulses),
        "pulse_budget": pulse_budget,
        "execution_elapsed_s": float(time.perf_counter() - execution_t0),
        "initial_model_comparison": initial_model_comparison,
        "final_model_comparison": _qc_model_comparison(x_estimate, base_mirrors, final_qc, qc_readout_sign),
        "angle_trajectory_source": "simulated_physical" if dry_run else "calibrated_pulse_estimate",
        "final_qc_position_saturated": bool(final_qc.get("position_saturated", False)),
        "rotation_calibration": dict(rotation_calibration),
        "staging_rotation_observations": staging_rotation_observations,
        "surface_stage_validate_target_sweep": bool(surface_stage_validate_target_sweep),
        "surface_stage_learn_rotation_response": bool(surface_stage_learn_rotation_response),
        "calibrate_before": bool(calibrate_before),
        "preflight_before_calibration": bool(preflight_before_calibration),
        "calibration_pulse_substeps": int(calibration_pulse_substeps),
        "calibration_repeats": int(calibration_repeats),
        "calibration_recenter_after": bool(calibration_recenter_after),
        "calibration_recenter_after_each_actuator": bool(calibration_recenter_after_each_actuator),
        "calibration_recenter_tolerance": float(calibration_recenter_tolerance),
        "calibration_recenter_max_pulses": int(calibration_recenter_max_pulses),
        "calibration_recenter_min_improvement": float(calibration_recenter_min_improvement),
        "calibration_result": calibration_result,
        "rotation_pulse_calibration": pulse_table,
        "execution_log": execution_log,
        "center_execution_log": center_execution_log,
        "final_center_path_failure": final_center_path_failure,
        "actuation_plan": actuation_plan,
        "planner_timing": actuation_plan.get("planner_timing"),
        "planner_result": planner_res,
        "dry_run": bool(dry_run),
        "dry_run_use_settle_delays": bool(dry_run_use_settle_delays),
        "qc_min_signal": float(qc_min_signal),
        "qc_min_signals": (
            None if qc_min_signals is None
            else np.array(qc_min_signals, dtype=float).tolist()
        ),
        "qc_readout_axis": "x",
        "qc_readout_sign": float(qc_readout_sign),
        "dry_run_signal_strengths": (
            np.array(dry_run_signal_strengths, dtype=float).tolist()
            if dry_run else None
        ),
        "dry_run_pulse_response": dry_run_pulse_response if dry_run else None,
        "max_rotation_chunk_substeps": int(max_rotation_chunk_substeps),
        "execution_uses_calibrated_pulses": True,
        "execution_pulse_substeps": int(
            pulse_table.get("pulse_substeps", calibration_pulse_substeps)
            if isinstance(pulse_table, dict) else calibration_pulse_substeps
        ),
        "rotation_settle_delay": float(rotation_settle_delay),
        "step_settle_delay": float(step_settle_delay),
        "requested_rotation_settle_delay": float(requested_rotation_settle_delay),
        "requested_step_settle_delay": float(requested_step_settle_delay),
        "final_center_after_reacquire": bool(final_center_after_reacquire),
        "final_center_max_axis_splits": int(final_center_max_axis_splits),
        "final_center_waypoint_depth": int(final_center_waypoint_depth),
        "final_center_feedback_max_pulses": int(final_center_feedback_max_pulses),
        "final_center_min_improvement": float(final_center_min_improvement),
        "require_target_N_R_estimate_for_reacquire_stop": bool(require_target_N_R_estimate_for_reacquire_stop),
        "min_reacquire_pulse_fraction": float(min_reacquire_pulse_fraction),
        "target_reacquire_extend_until_reacquired": bool(target_reacquire_extend_until_reacquired),
        "reacquire_angle_scan_limit": float(reacquire_angle_scan_limit),
        "reacquire_scan_samples": int(reacquire_scan_samples),
        "reacquire_max_first_leg_candidates": int(reacquire_max_first_leg_candidates),
        "reacquire_stage_max_candidates": int(reacquire_stage_max_candidates),
        "forced_stage_samples": int(forced_stage_samples),
        "forced_stage_free_angle_regularization": float(forced_stage_free_angle_regularization),
        "forced_stage_max_nfev": int(forced_stage_max_nfev),
        "surface_stage_active_actuators": list(surface_stage_active_actuators),
        "surface_stage_sweep_actuator": surface_stage_sweep_actuator,
        "surface_stage_distance_metric": surface_stage_distance_metric,
        "surface_stage_match_target_3d": bool(surface_stage_match_target_3d),
        "surface_stage_qc_tolerance": float(surface_stage_qc_tolerance),
        "surface_stage_path_qc_limit": float(surface_stage_path_qc_limit),
        "surface_stage_qc_step_tolerance": float(surface_stage_qc_step_tolerance),
        "surface_stage_angle_fit_qc_tolerance": surface_stage_angle_fit_qc_tolerance,
        "surface_stage_qc_overshoot_allowance": float(surface_stage_qc_overshoot_allowance),
        "surface_stage_control_mode": "qc_endpoint_feedback" if surface_stage_strategy else None,
        "surface_stage_qc_safety_limit": float(surface_stage_path_qc_limit + surface_stage_qc_overshoot_allowance),
        "surface_stage_max_chunks_per_step": int(surface_stage_max_chunks_per_step),
        "surface_stage_sweep_half_span_deg": (
            None if surface_stage_sweep_half_span_deg is None
            else float(surface_stage_sweep_half_span_deg)
        ),
        "surface_stage_sweep_step_deg": float(surface_stage_sweep_step_deg),
        "surface_stage_sweep_refinement_levels": int(surface_stage_sweep_refinement_levels),
        "surface_stage_sweep_expand_to_bounds": bool(surface_stage_sweep_expand_to_bounds),
        "surface_stage_sweep_max_half_span_deg": surface_stage_sweep_max_half_span_deg,
        "surface_stage_sweep_min_step_deg": float(surface_stage_sweep_min_step_deg),
        "surface_stage_preferred_axis": surface_stage_preferred_axis,
        "surface_stage_line_max_steps_per_side": int(surface_stage_line_max_steps_per_side),
        "surface_stage_line_step_deg": float(surface_stage_line_step_deg),
        "surface_stage_projection_refinement_samples": int(surface_stage_projection_refinement_samples),
        "surface_stage_max_path_candidates": int(surface_stage_max_path_candidates),
        "surface_stage_repair_source_bounds": bool(surface_stage_repair_source_bounds),
        "surface_stage_source_repair_n_tries": (
            None if surface_stage_source_repair_n_tries is None
            else int(surface_stage_source_repair_n_tries)
        ),
        "surface_stage_source_repair_angle_perturb": (
            None if surface_stage_source_repair_angle_perturb is None
            else float(surface_stage_source_repair_angle_perturb)
        ),
        "surface_stage_stop_after_stage": bool(surface_stage_stop_after_stage),
        "surface_stage_target_jump_include_active_residuals": bool(
            surface_stage_target_jump_include_active_residuals
        ),
        "target_center_after_jump": bool(target_center_after_jump),
        "target_center_u_min": (
            None if target_center_u_min is None else float(target_center_u_min)
        ),
        "target_center_u_max": (
            None if target_center_u_max is None else float(target_center_u_max)
        ),
        "max_target_jump_center_candidates": int(max_target_jump_center_candidates),
    }

    log(
        f"done success={final_success} "
        f"status={completion_status} "
        f"{'N_R_simulated_physical' if dry_run else 'N_R_model'}={final_sim_reflections} target={int(target_N_R)} "
        f"beam_reacquired={beam_reacquired} "
        f"requires_inverse_refresh={requires_inverse_refresh} "
        f"reflection_count_verified={bool(dry_run)} warnings={len(execution_warnings)} "
        f"qc_valid={final_qc['valid']} "
        f"qc_x=({final_qc['x'][0]:.3f},{final_qc['x'][1]:.3f}) "
        f"pulses={total_execution_pulses}/{max_total_steps} failure={failure_reason}"
    )

    return final_mirrors, final_res, execution


def test_linear_stage_scale(hardware, serial, delta=0.05, settle=2.0):
    """Move a KDC linear stage out and back to verify mm readback scale."""
    if hardware is None or getattr(hardware, "stages", None) is None:
        raise ValueError("hardware.stages is required for the linear stage scale test.")

    serial = str(serial)
    delta = float(delta)
    settle = float(settle)

    p0 = float(hardware.stages.get_position(serial))
    print(f"[linear scale] {serial} before={p0:.6f} mm")

    hardware.stages.move_relative(serial, delta)
    if settle > 0:
        time.sleep(settle)
    p1 = float(hardware.stages.get_position(serial))
    print(
        f"[linear scale] {serial} after +{delta:.6f} mm: "
        f"{p1:.6f} mm readback_delta={p1 - p0:.6f} mm"
    )

    hardware.stages.move_relative(serial, -delta)
    if settle > 0:
        time.sleep(settle)
    p2 = float(hardware.stages.get_position(serial))
    print(
        f"[linear scale] {serial} after return: "
        f"{p2:.6f} mm residual={p2 - p0:.6f} mm"
    )

    return {
        "serial": serial,
        "command_delta": delta,
        "settle": settle,
        "before_position": p0,
        "after_plus_position": p1,
        "after_return_position": p2,
        "readback_delta": p1 - p0,
        "return_residual": p2 - p0,
    }


def run_fixed_plan_dry_run_trials(
        target_OPD,
        M1,
        M2,
        M3,
        M4,
        *,
        seeds=range(20),
        dry_run_rotation_error=0.10,
        qc_plan_limit=1.5,
        qc_detector_limit=3.9,
        qc_hardware_stop=3.5,
        qc_step_tolerance=0.15,
        final_qc_tolerance=0.5,
        final_OPD_tolerance=0.5,
        profile=False,
        **execute_kwargs):
    """Run repeated fixed-plan dry-runs with randomized rotation step error."""
    rows = []
    for seed in list(seeds):
        _, _, execution = execute_OPD_fixed_plan(
            target_OPD,
            M1,
            M2,
            M3,
            M4,
            dry_run=True,
            rng_seed=int(seed),
            dry_run_rotation_error=dry_run_rotation_error,
            qc_plan_limit=qc_plan_limit,
            qc_detector_limit=qc_detector_limit,
            qc_hardware_stop=qc_hardware_stop,
            qc_step_tolerance=qc_step_tolerance,
            final_qc_tolerance=final_qc_tolerance,
            final_OPD_tolerance=final_OPD_tolerance,
            profile=profile,
            **execute_kwargs,
        )
        final_qc = execution.get("final_sim_qc", [np.nan, np.nan])
        rows.append({
            "seed": int(seed),
            "success": bool(execution.get("success")),
            "failure_reason": execution.get("failure_reason"),
            "success_checks": execution.get("success_checks"),
            "needs_final_recenter": execution.get("needs_final_recenter"),
            "planner_failure_ignored": execution.get("planner_failure_ignored"),
            "planner_failure_reason": execution.get("planner_failure_reason"),
            "final_OPD": execution.get("final_OPD"),
            "final_OPD_error": execution.get("final_OPD_error"),
            "final_qc1": float(final_qc[0]),
            "final_qc2": float(final_qc[1]),
            "max_abs_final_qc": float(np.max(np.abs(final_qc))),
            "max_abs_measured_qc": execution.get("max_abs_measured_qc"),
            "max_step_target_miss": execution.get("max_step_target_miss"),
            "max_abs_OPD_divergence": execution.get("max_abs_OPD_divergence"),
            "rollback_count": execution.get("rollback_count"),
            "n_execution_steps": len(execution.get("execution_log", [])),
            "n_planner_runs": len(execution.get("planner_runs", [])),
        })

    summary = {
        "target_OPD": float(target_OPD),
        "n_trials": len(rows),
        "n_success": sum(1 for row in rows if row["success"]),
        "all_success": all(row["success"] for row in rows) if rows else False,
        "qc_plan_limit": float(qc_plan_limit),
        "qc_detector_limit": float(qc_detector_limit),
        "qc_hardware_stop": float(qc_hardware_stop),
        "qc_step_tolerance": float(qc_step_tolerance),
        "final_qc_tolerance": float(final_qc_tolerance),
        "final_OPD_tolerance": float(final_OPD_tolerance),
        "dry_run_rotation_error": float(dry_run_rotation_error),
        "max_abs_measured_qc": max(
            (row["max_abs_measured_qc"] for row in rows if row["max_abs_measured_qc"] is not None),
            default=0.0
        ),
        "max_step_target_miss": max(
            (row["max_step_target_miss"] for row in rows if row["max_step_target_miss"] is not None),
            default=0.0
        ),
        "rows": rows,
    }
    return summary


def plot_choose_OPD_quadcell_overlay(actuation_plan, show_difference=True):
    """Plot the planned quadcell offsets from a choose_OPD actuation plan."""
    return S.plot_choose_OPD_quadcell_overlay(
        actuation_plan,
        show_difference=show_difference
    )


def plot_choose_OPD_reflection_u_overlay(actuation_plan):
    """Plot planned reflection-u positions from a choose_OPD actuation plan."""
    return S.plot_choose_OPD_reflection_u_overlay(actuation_plan)


def save_choose_OPD_actuation_gif(actuation_plan, output_path="choose_OPD_actuation.gif",
                                  fps=8, **kwargs):
    """Save a simulated GIF from a choose_OPD actuation plan."""
    return S.save_choose_OPD_actuation_gif(
        actuation_plan,
        output_path=output_path,
        fps=fps,
        **kwargs
    )


def save_fixed_plan_execution(execution, output_path=None, *, overwrite=True):
    """Save an execute_OPD_fixed_plan execution dict for later plotting."""
    target = execution.get("target_OPD", "unknown") if isinstance(execution, dict) else "unknown"
    if output_path is None:
        try:
            target_text = f"{float(target):.0f}"
        except Exception:
            target_text = str(target).replace(" ", "_")
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        output_path = Path("fixed_plan_executions") / f"fixed_OPD_{target_text}_{timestamp}.pkl"
    else:
        output_path = Path(output_path)
        if output_path.suffix == "":
            output_path = output_path.with_suffix(".pkl")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} already exists.")

    payload = {
        "format": "fixed_plan_execution",
        "format_version": 1,
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "execution": execution,
    }
    with output_path.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    return str(output_path)


def load_fixed_plan_execution(path):
    """Load an execution dict saved by save_fixed_plan_execution."""
    with Path(path).open("rb") as f:
        payload = pickle.load(f)
    if isinstance(payload, dict) and payload.get("format") == "fixed_plan_execution":
        return payload["execution"]
    return payload


def _fixed_plan_overlay_records(execution, show_pulses):
    """Align QC, angle snapshots, and rollbacks on the planned step axis."""
    log = execution.get("execution_log", [])
    plan = execution.get("actuation_plan") or {}
    sign = float(execution.get("qc_readout_sign", -1.0))

    def record(position, kind, raw_qc, valid=True, estimate_x=None, physical_x=None, **extra):
        qc = _sim_qc_from_quadcell_readout(
            np.array([np.nan, np.nan] if raw_qc is None else raw_qc, dtype=float), sign,
        )
        return {
            "position": float(position), "kind": kind, "qc": qc, "valid": bool(valid),
            "estimate_x": estimate_x, "physical_x": physical_x, **extra,
        }

    if not log:
        initial_qc = execution.get("initial_qc_x")
        if initial_qc is None:
            return []
        return [record(
            0, "start", initial_qc, np.all(np.isfinite(initial_qc)),
            execution.get("initial_estimate_x", plan.get("start_x")),
            execution.get("initial_physical_x"),
        )]

    first = log[0]
    records = [record(
        0, "start", first.get("before_qc_x", first.get("before_qc_y")),
        first.get("before_qc_signal_valid", True),
        first.get("estimate_x_before_step", plan.get("start_x")),
        first.get("physical_x_before_step"),
    )]
    for index, entry in enumerate(log, start=1):
        step = int(entry.get("execution_step", index))
        samples = (
            entry.get("pulses") or entry.get("chunks") or
            (entry.get("detail") or {}).get("chunks") or []
        ) if show_pulses else []
        event_count = len(samples) + sum(bool(sample.get("rolled_back")) for sample in samples)
        event_index = 0
        previous_estimate = entry.get("estimate_x_before_step")
        previous_physical = entry.get("physical_x_before_step")
        for sample in samples:
            event_index += 1
            records.append(record(
                step - 1 + event_index / (event_count + 1), "pulse",
                sample.get("qc_x", sample.get("qc_y")), sample.get("qc_signal_valid", True),
                sample.get("estimate_x_after_pulse"), sample.get("physical_x_after_pulse"),
                unsafe=bool(sample.get("rolled_back")),
            ))
            if sample.get("rolled_back"):
                event_index += 1
                records.append(record(
                    step - 1 + event_index / (event_count + 1), "rollback",
                    sample.get("rollback_qc_x", sample.get("rollback_qc_y")),
                    sample.get("rollback_qc_signal_valid", entry.get("after_qc_signal_valid", True)),
                    previous_estimate, previous_physical,
                ))
            else:
                previous_estimate = sample.get("estimate_x_after_pulse")
                previous_physical = sample.get("physical_x_after_pulse")
        endpoint = record(
            step, "endpoint", entry.get("after_qc_x", entry.get("after_qc_y")),
            entry.get("after_qc_signal_valid", True),
            entry.get("estimate_x_after_step"), entry.get("physical_x_after_step"),
            estimate_path_metrics=entry.get("estimate_path_metrics"),
            physical_path_metrics=entry.get("physical_path_metrics"),
        )
        if entry.get("measured_sim_qc") is not None:
            endpoint["qc"] = np.array(entry["measured_sim_qc"], dtype=float)
        records.append(endpoint)
    return records


def _finish_fixed_plan_overlay(fig, ax, execution):
    import textwrap

    failure = execution.get("failure_reason")
    log = execution.get("execution_log", [])
    if failure:
        failed_entry = next((entry for entry in log if (
            entry.get("failure_reason") or (entry.get("detail") or {}).get("failure_reason")
        )), log[-1] if log else None)
        if failed_entry is not None:
            ax.axvline(
                int(failed_entry["execution_step"]), color="crimson", linestyle=":",
                linewidth=1.2, label="execution stopped",
            )
        elif execution.get("fixed_plan_substep_log"):
            ax.axvline(0, color="crimson", linestyle=":", linewidth=1.2, label="no actuator moves executed")
    detailed_log = bool(execution.get("reflection_count_change") or execution.get("fixed_plan_substep_log"))
    if detailed_log:
        fig.set_size_inches(12, 5, forward=False)
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8)
    else:
        ax.legend()
    if detailed_log or failure:
        target = execution.get("target_N_R")
        heading = f"N_R={target}: " if target is not None else ""
        if not heading and execution.get("target_OPD") is not None:
            heading = f"OPD={float(execution['target_OPD']):g} mm: "
        status = str(failure) if failure else execution.get("completion_message", "Execution completed")
        if failure and not log and execution.get("fixed_plan_substep_log"):
            status = "No actuator moves executed. " + status
        fig.suptitle(textwrap.fill(heading + status, width=90), fontsize=10)
        fig.tight_layout()
    else:
        fig.tight_layout()


def plot_fixed_plan_quadcell_overlay(execution, show_difference=True, *, show_pulses=None, xlim=None):
    """Overlay planned QC offsets and recorded readings, including failed runs.

    Reflection-count and new OPD logs include pulse/rollback samples by default.
    Readouts use the simulation sign convention to match the planned curves.
    The displayed offset range is fixed to +/-1 mm; recorded data is unchanged.
    xlim=(start_step, end_step) selects an actuator-step window, including
    fractional steps. None keeps the automatic full-range view.
    """
    actuation_plan = execution.get("actuation_plan") or {}
    fig, ax = S.plot_actuation_quadcell_offsets(
        actuation_plan,
        show_difference=show_difference
    )
    ax.set_ylim(-1.0, 1.0)
    if xlim is not None:
        ax.set_xlim(*xlim)

    if show_pulses is None:
        show_pulses = bool(execution.get("reflection_count_change") or execution.get("fixed_plan_substep_log"))
    records = _fixed_plan_overlay_records(execution, show_pulses)
    if not records:
        _finish_fixed_plan_overlay(fig, ax, execution)
        return fig, ax

    step_numbers = np.array([record["position"] for record in records])
    qc = np.array([record["qc"] for record in records])
    valid = np.array([record["valid"] for record in records])
    qc1_actual = np.where(valid, qc[:, 0], np.nan)
    qc2_actual = np.where(valid, qc[:, 1], np.nan)

    ax.plot(
        step_numbers,
        qc1_actual,
        marker="x",
        linewidth=1.3,
        linestyle="--",
        label="actual QC1 offset"
    )
    ax.plot(
        step_numbers,
        qc2_actual,
        marker="x",
        linewidth=1.3,
        linestyle="--",
        label="actual QC2 offset"
    )

    if show_difference:
        qc_diff_actual = [a - b for a, b in zip(qc1_actual, qc2_actual)]
        ax.plot(
            step_numbers,
            qc_diff_actual,
            marker="x",
            linewidth=1.0,
            linestyle=":",
            label="actual QC1 - QC2"
        )

    for mask, label, marker, color in (
        (np.array([record.get("unsafe", False) for record in records]), "safety-triggering pulse", "x", "crimson"),
        (np.array([record["kind"] == "rollback" for record in records]), "after rollback", "v", "tab:green"),
        (~valid, "invalid QC signal", "x", "gray"),
    ):
        if np.any(mask):
            ax.scatter(
                np.repeat(step_numbers[mask], 2), qc[mask].ravel(),
                marker=marker, color=color, s=45, zorder=5, label=label,
            )
    if execution.get("reflection_count_change"):
        surface_stage = execution.get("surface_stage_control_mode") == "qc_endpoint_feedback"
        stage_limit = execution.get("surface_stage_path_qc_limit" if surface_stage else "stage_qc_limit")
        safety_limit = execution.get("surface_stage_qc_safety_limit") if surface_stage else None
        for limit, label, color, style in (
            (stage_limit, "staging planned QC limit", "gray", "--"),
            (safety_limit, "staging QC safety limit", "crimson", ":"),
        ):
            if limit is not None and np.isfinite(float(limit)):
                ax.axhline(float(limit), color=color, linestyle=style, linewidth=1, label=label)
                ax.axhline(-float(limit), color=color, linestyle=style, linewidth=1)

    ax.set_title("Quadcell Beam Offset During Fixed-Plan Execution")
    _finish_fixed_plan_overlay(fig, ax, execution)
    return fig, ax


def plot_fixed_plan_reflection_u_overlay(execution, prefer_physical=True, *, show_pulses=None):
    """Plot planned reflection-u bounds with the executed model path overlaid.

    For dry-run executions, prefer_physical=True overlays the randomized physical
    dry-run path. For real hardware, reflection u is not directly measured, so
    this falls back to the estimated model path. Reflection-count and new OPD
    logs include pulse/rollback positions by default, using recorded states.
    """
    actuation_plan = execution.get("actuation_plan") or {}
    fig, ax = S.plot_actuation_reflection_u(actuation_plan)

    if show_pulses is None:
        show_pulses = bool(execution.get("reflection_count_change") or execution.get("fixed_plan_substep_log"))
    records = _fixed_plan_overlay_records(execution, show_pulses)
    if not show_pulses:
        records = [record for record in records if record["kind"] == "endpoint"]
    if not records:
        _finish_fixed_plan_overlay(fig, ax, execution)
        return fig, ax

    use_physical = (
        prefer_physical and
        any(record.get("physical_path_metrics") is not None or record.get("physical_x") is not None for record in records)
    )
    metrics_key = "physical_path_metrics" if use_physical else "estimate_path_metrics"
    label_prefix = "actual dry-run" if use_physical else "executed model"
    state_key = "physical_x" if use_physical else "estimate_x"
    base_mirrors = actuation_plan.get("start_mirrors")
    if base_mirrors is None:
        base_mirrors = execution.get("final_mirrors")
    if base_mirrors is None:
        base_mirrors = next((step["mirrors"] for step in actuation_plan.get("steps", []) if step.get("mirrors") is not None), None)

    step_numbers = []
    min_us = []
    max_us = []
    margins = []
    for record in records:
        metrics = record.get(metrics_key)
        if metrics is None and record.get(state_key) is not None and base_mirrors is not None:
            edge = S.reflection_edge_summary(
                record[state_key], *base_mirrors,
                include_ends=bool(actuation_plan.get("include_edge_ends", False)),
            )
            metrics = {
                "min_reflection_u": edge["min_u"], "max_reflection_u": edge["max_u"],
                "closest_edge_margin": edge["closest_edge_margin"],
            }
        metrics = metrics or {}
        step_numbers.append(record["position"])
        min_us.append(float(metrics.get("min_reflection_u", np.nan)))
        max_us.append(float(metrics.get("max_reflection_u", np.nan)))
        margins.append(float(metrics.get("closest_edge_margin", np.nan)))

    ax.plot(
        step_numbers,
        min_us,
        marker="x",
        linewidth=1.3,
        linestyle="--",
        label=f"{label_prefix} minimum reflection u"
    )
    ax.plot(
        step_numbers,
        max_us,
        marker="x",
        linewidth=1.3,
        linestyle="--",
        label=f"{label_prefix} maximum reflection u"
    )
    ax.plot(
        step_numbers,
        margins,
        marker="x",
        linewidth=1.0,
        linestyle=":",
        label=f"{label_prefix} closest edge margin"
    )

    ax.set_title("Reflection Positions During Fixed-Plan Execution")
    unsafe = np.array([record.get("unsafe", False) for record in records])
    if np.any(unsafe):
        ax.scatter(
            np.repeat(np.array(step_numbers)[unsafe], 2),
            np.column_stack([min_us, max_us])[unsafe].ravel(),
            marker="x", color="crimson", s=45, zorder=5, label="safety-triggering pulse",
        )
    rollback = np.array([record["kind"] == "rollback" for record in records])
    if np.any(rollback):
        ax.scatter(
            np.array(step_numbers)[rollback], np.array(min_us)[rollback],
            marker="v", color="tab:green", s=45, zorder=5, label="after rollback",
        )
    _finish_fixed_plan_overlay(fig, ax, execution)
    return fig, ax


def plot_reflection_count_execution_diagnostics(execution):
    """Compare measured QC readouts with pulse-model predictions and signals."""
    import matplotlib.pyplot as plt
    import textwrap

    entries = execution.get("execution_log", [])
    rows = []
    step_starts = []
    for entry in entries:
        step_starts.append((len(rows) + 1, entry.get("execution_step")))
        for pulse in entry.get("pulses") or entry.get("chunks") or [entry]:
            rows.append({
                "qc": pulse.get("qc_x", pulse.get("after_qc_x", [np.nan, np.nan])),
                "predicted": pulse.get("predicted_qc_x", [np.nan, np.nan]),
                "signal": pulse.get("qc_signal_strengths", pulse.get("after_qc_signal_strengths", [np.nan, np.nan])),
                "valid": pulse.get("qc_signal_valid", pulse.get("after_qc_signal_valid", False)),
            })
    initial = execution.get("initial_model_comparison", {})
    rows.insert(0, {
        "qc": initial.get("measured_qc_x", entries[0].get("before_qc_x", [np.nan, np.nan]) if entries else [np.nan, np.nan]),
        "predicted": initial.get("predicted_qc_x", [np.nan, np.nan]),
        "signal": entries[0].get("before_qc_signal_strengths", [np.nan, np.nan]) if entries else [np.nan, np.nan],
        "valid": initial.get("qc_position_valid", entries[0].get("before_qc_signal_valid", False) if entries else False),
    })
    measured = np.asarray([r["qc"] for r in rows], dtype=float)
    predicted = np.asarray([r["predicted"] for r in rows], dtype=float)
    signals = np.asarray([r["signal"] for r in rows], dtype=float)
    valid = np.asarray([r["valid"] for r in rows], dtype=bool)
    pulse_index = np.arange(len(rows))
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    for index, color in enumerate(("tab:blue", "tab:orange")):
        label = f"QC{index + 1}"
        axes[0].plot(pulse_index, measured[:, index], color=color, label=f"{label} measured")
        axes[0].plot(pulse_index, predicted[:, index], color=color, linestyle="--", label=f"{label} predicted")
        axes[0].scatter(pulse_index[~valid], measured[~valid, index], color=color, marker="x", zorder=3)
        residual = measured[:, index] - predicted[:, index]
        axes[1].plot(pulse_index, np.where(valid, residual, np.nan), color=color, label=label)
        axes[2].plot(pulse_index, signals[:, index], color=color, label=label)
        threshold = execution.get("final_qc_signal_thresholds", [execution.get("qc_min_signal", 0.04)] * 2)[index]
        axes[2].axhline(threshold, color=color, linestyle=":")
    axes[0].set_title("QC readout vs model prediction; crosses mark invalid/clipped readings")
    axes[0].set_ylabel("QC readout (converted mm)")
    axes[1].set_ylabel("Measured - predicted\n(converted mm)")
    axes[1].axhline(0, color="gray", linewidth=0.7)
    axes[2].set_ylabel("Signal sum")
    axes[2].set_xlabel("Executed pulse (0 = after calibration, before staging)")
    for ax in axes:
        for pulse, step in step_starts:
            ax.axvline(pulse, color="gray", linewidth=0.5, alpha=0.4)
        ax.grid(True, linewidth=0.3)
        ax.legend(loc="best")
    for pulse, step in step_starts:
        axes[0].text(pulse, 1.01, str(step), transform=axes[0].get_xaxis_transform(), fontsize=8)
    status = execution.get("failure_reason") or execution.get("completion_message", "Execution completed")
    fig.suptitle(textwrap.fill(str(status), width=100), fontsize=11)
    fig.tight_layout()
    return fig, axes


def plot_reflection_count_execution_summary(execution):
    """Plot timing, QC, reflection count, and pulse usage for an N_R execution."""
    import matplotlib.pyplot as plt

    log = execution.get("execution_log", [])
    plan_timing = execution.get("planner_timing") or {}
    actuation_plan = execution.get("actuation_plan", {})
    plan_steps = actuation_plan.get("steps", [])

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    ax_timing, ax_qc, ax_nr, ax_pulses = axes.ravel()

    timing_items = [
        (key, float(value))
        for key, value in plan_timing.items()
        if key != "total" and np.isfinite(float(value))
    ]
    timing_items.sort(key=lambda row: row[1], reverse=True)
    if timing_items:
        labels = [label for label, _ in timing_items]
        values = [value for _, value in timing_items]
        ax_timing.barh(labels, values, color="tab:blue", alpha=0.8)
        ax_timing.invert_yaxis()
        ax_timing.set_xlabel("seconds")
    ax_timing.set_title("Planner Time")
    total_dt = plan_timing.get("total")
    if total_dt is not None:
        ax_timing.text(
            0.98,
            0.04,
            f"total {float(total_dt):.2f}s",
            transform=ax_timing.transAxes,
            ha="right",
            va="bottom",
        )

    planned_step_numbers = [0] + [int(step.get("step", idx + 1)) for idx, step in enumerate(plan_steps)]
    planned_qc1 = [
        float(actuation_plan.get("start_qc1_error", actuation_plan.get("initial_qc1_error", np.nan)))
    ] + [float(step.get("qc1_error", np.nan)) for step in plan_steps]
    planned_qc2 = [
        float(actuation_plan.get("start_qc2_error", actuation_plan.get("initial_qc2_error", np.nan)))
    ] + [float(step.get("qc2_error", np.nan)) for step in plan_steps]
    if plan_steps:
        ax_qc.plot(
            planned_step_numbers,
            planned_qc1,
            marker=".",
            linewidth=1.2,
            linestyle=":",
            color="tab:blue",
            label="planned QC1",
        )
        ax_qc.plot(
            planned_step_numbers,
            planned_qc2,
            marker=".",
            linewidth=1.2,
            linestyle=":",
            color="tab:orange",
            label="planned QC2",
        )

    step_numbers = [0]
    qc1 = []
    qc2 = []
    if log:
        qc_readout_sign = float(execution.get("qc_readout_sign", -1.0))
        start_qc = _sim_qc_from_quadcell_readout(
            np.array(log[0].get("before_qc_x", log[0].get("before_qc_y", [np.nan, np.nan])), dtype=float),
            qc_readout_sign,
        )
        qc1.append(float(start_qc[0]))
        qc2.append(float(start_qc[1]))
        for entry in log:
            step_numbers.append(int(entry["execution_step"]))
            raw_qc = entry.get("measured_sim_qc")
            if raw_qc is None:
                raw_qc = _sim_qc_from_quadcell_readout(
                    np.array(entry.get("after_qc_x", entry.get("after_qc_y", [np.nan, np.nan])), dtype=float),
                    qc_readout_sign,
                )
            raw_qc = np.array(raw_qc, dtype=float)
            qc1.append(float(raw_qc[0]))
            qc2.append(float(raw_qc[1]))
    ax_qc.plot(step_numbers, qc1, marker="o", color="tab:blue", label="executed QC1")
    ax_qc.plot(step_numbers, qc2, marker="o", color="tab:orange", label="executed QC2")
    qc_limit = execution.get("qc_reacquire_limit")
    if qc_limit is not None:
        ax_qc.axhline(float(qc_limit), color="black", linestyle=":", linewidth=1)
        ax_qc.axhline(-float(qc_limit), color="black", linestyle=":", linewidth=1)
    ax_qc.set_title("Executed QC Offset")
    ax_qc.set_xlabel("execution step")
    ax_qc.set_ylabel("mm")
    ax_qc.grid(True, linewidth=0.3)
    ax_qc.legend()

    nr_steps = [entry["execution_step"] for entry in log]
    nr_values = [
        entry.get("estimate_path_metrics", {}).get("reflection_count", np.nan)
        for entry in log
    ]
    ax_nr.step(nr_steps, nr_values, where="post", marker="o", label="sim N_R")
    target_N_R = execution.get("target_N_R")
    if target_N_R is not None:
        ax_nr.axhline(int(target_N_R), color="black", linestyle=":", linewidth=1, label="target")
    ax_nr.set_title("Reflection Count")
    ax_nr.set_xlabel("execution step")
    ax_nr.set_ylabel("N_R")
    ax_nr.grid(True, linewidth=0.3)
    ax_nr.legend()

    pulse_labels = [
        f"{entry['execution_step']}: {entry.get('actuator', '')}"
        for entry in log
    ]
    planned_pulses = [abs(int(entry.get("planned_pulse_count", 0))) for entry in log]
    executed_pulses = [int(entry.get("executed_pulse_count", 0)) for entry in log]
    x = np.arange(len(log))
    width = 0.38
    if len(log) > 0:
        ax_pulses.bar(x - width / 2, planned_pulses, width, label="planned")
        ax_pulses.bar(x + width / 2, executed_pulses, width, label="executed")
        ax_pulses.set_xticks(x)
        ax_pulses.set_xticklabels(pulse_labels, rotation=30, ha="right")
    ax_pulses.set_title("Pulse Counts")
    ax_pulses.set_ylabel("pulses")
    ax_pulses.legend()

    status = "success" if execution.get("success") else "failed"
    fig.suptitle(f"Reflection-count execution {status}", y=0.995)
    fig.tight_layout()
    return fig, axes


def plot_reflection_count_actuation_plan(actuation_plan):
    """Plot the simulated actuation plan for an N_R reacquisition attempt."""
    import matplotlib.pyplot as plt

    steps = actuation_plan.get("steps", [])
    step_numbers = [0] + [int(step.get("step", idx + 1)) for idx, step in enumerate(steps)]
    qc1 = [
        float(actuation_plan.get("start_qc1_error", actuation_plan.get("initial_qc1_error", np.nan)))
    ] + [float(step.get("qc1_error", np.nan)) for step in steps]
    qc2 = [
        float(actuation_plan.get("start_qc2_error", actuation_plan.get("initial_qc2_error", np.nan)))
    ] + [float(step.get("qc2_error", np.nan)) for step in steps]
    reflection_counts = [
        float(actuation_plan.get("start_reflections", np.nan))
    ] + [float(step.get("reflection_count", np.nan)) for step in steps]
    min_us = [np.nan] + [float(step.get("min_reflection_u", np.nan)) for step in steps]
    max_us = [np.nan] + [float(step.get("max_reflection_u", np.nan)) for step in steps]
    margins = [np.nan] + [float(step.get("closest_edge_margin", np.nan)) for step in steps]

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    ax_qc, ax_nr, ax_u, ax_moves = axes.ravel()

    ax_qc.plot(step_numbers, qc1, marker="o", label="planned QC1")
    ax_qc.plot(step_numbers, qc2, marker="o", label="planned QC2")
    qc_limit = actuation_plan.get(
        "qc_reacquire_limit",
        actuation_plan.get("stage_qc_limit", actuation_plan.get("max_qc_error")),
    )
    if qc_limit is not None:
        ax_qc.axhline(float(qc_limit), color="black", linestyle=":", linewidth=1)
        ax_qc.axhline(-float(qc_limit), color="black", linestyle=":", linewidth=1)
    ax_qc.set_title("Planned QC Offset")
    ax_qc.set_xlabel("planned step")
    ax_qc.set_ylabel("mm")
    ax_qc.grid(True, linewidth=0.3)
    ax_qc.legend()

    ax_nr.step(step_numbers, reflection_counts, where="post", marker="o", label="planned N_R")
    target_N_R = actuation_plan.get("target_N_R")
    if target_N_R is not None:
        ax_nr.axhline(int(target_N_R), color="black", linestyle=":", linewidth=1, label="target")
    ax_nr.set_title("Planned Reflection Count")
    ax_nr.set_xlabel("planned step")
    ax_nr.set_ylabel("N_R")
    ax_nr.grid(True, linewidth=0.3)
    ax_nr.legend()

    ax_u.plot(step_numbers, min_us, marker="o", label="min u")
    ax_u.plot(step_numbers, max_us, marker="o", label="max u")
    ax_u.plot(step_numbers, margins, marker=".", linestyle="--", label="closest edge margin")
    u_min = actuation_plan.get("u_min", 0.1)
    u_max = actuation_plan.get("u_max", 0.9)
    ax_u.axhline(float(u_min), color="black", linestyle=":", linewidth=1)
    ax_u.axhline(float(u_max), color="black", linestyle=":", linewidth=1)
    ax_u.set_title("Planned Reflection Positions")
    ax_u.set_xlabel("planned step")
    ax_u.grid(True, linewidth=0.3)
    ax_u.legend()

    labels = [
        f"{int(step.get('step', idx + 1))}: {step.get('actuator', '')}"
        for idx, step in enumerate(steps)
    ]
    moves = [float(step.get("command_value", 0.0)) for step in steps]
    colors = [
        "tab:red" if step.get("target_jump_step") else
        "tab:blue" if step.get("target_center_step") else
        "tab:green"
        for step in steps
    ]
    x = np.arange(len(steps))
    if steps:
        ax_moves.bar(x, moves, color=colors, alpha=0.8)
        ax_moves.set_xticks(x)
        ax_moves.set_xticklabels(labels, rotation=30, ha="right")
    ax_moves.axhline(0.0, color="black", linewidth=0.8)
    ax_moves.set_title("Planned Angle Moves")
    ax_moves.set_ylabel("deg")

    status = "success" if actuation_plan.get("failure_reason") is None else "failed"
    search_mode = actuation_plan.get("search_mode", "unknown")
    fig.suptitle(f"Simulated N_R actuation plan {status}: {search_mode}", y=0.995)
    fig.tight_layout()
    return fig, axes


def plot_reflection_count_dangle_trajectory(execution):
    """Overlay planned and executed mirror dangle values for an N_R execution."""
    import matplotlib.pyplot as plt

    actuation_plan = execution.get("actuation_plan", {})
    plan_steps = actuation_plan.get("steps", [])
    log = execution.get("execution_log", [])
    angle_axes = np.array([1, 3, 5, 7], dtype=int)
    mirror_labels = ["M1", "M2", "M3", "M4"]

    x_start = actuation_plan.get("start_x")
    if x_start is None:
        x_start = actuation_plan.get("initial_x")
    if x_start is None and plan_steps:
        first_positions = plan_steps[0].get("positions", {})
        x_start = np.full(8, np.nan, dtype=float)
        for idx, mirror in enumerate(mirror_labels):
            pos = first_positions.get(mirror, {})
            if "angle" in pos:
                x_start[angle_axes[idx]] = float(pos["angle"]) - (
                    float(plan_steps[0].get("command_value", 0.0))
                    if int(plan_steps[0].get("axis_index", -1)) == int(angle_axes[idx])
                    else 0.0
                )
    x_start = np.array(
        np.full(8, np.nan, dtype=float) if x_start is None else x_start,
        dtype=float,
    )

    planned_step_numbers = [0]
    planned_angles = [x_start[angle_axes].astype(float)]
    for idx, step in enumerate(plan_steps, start=1):
        positions = step.get("positions", {})
        angles = planned_angles[-1].copy()
        for mirror_index, mirror in enumerate(mirror_labels):
            pos = positions.get(mirror, {})
            if "angle" in pos:
                angles[mirror_index] = float(pos["angle"])
        if not positions and step.get("axis_index") is not None:
            local = np.where(angle_axes == int(step["axis_index"]))[0]
            if len(local) == 1:
                angles[int(local[0])] += float(step.get("command_value", 0.0))
        planned_step_numbers.append(int(step.get("step", idx)))
        planned_angles.append(angles)
    planned_angles = np.array(planned_angles, dtype=float)

    executed_step_numbers = [0]
    executed_angles = [x_start[angle_axes].astype(float)]
    physical_step_numbers = [0]
    physical_angles = [x_start[angle_axes].astype(float)]
    reconstructed_x = x_start.copy()
    reconstructed_physical_x = x_start.copy()
    has_physical = False

    for entry in log:
        executed_step_numbers.append(int(entry.get("execution_step", len(executed_step_numbers))))
        if entry.get("estimate_x_after_step") is not None:
            x_est = np.array(entry["estimate_x_after_step"], dtype=float)
        else:
            x_est = reconstructed_x.copy()
            axis_index = entry.get("axis_index")
            if axis_index is not None:
                x_est[int(axis_index)] += (
                    float(entry.get("pulse_angle_delta", 0.0)) *
                    int(entry.get("executed_pulse_count", 0))
                )
        executed_angles.append(x_est[angle_axes].astype(float))
        reconstructed_x = x_est.copy()

        physical_x = entry.get("physical_x_after_step")
        if physical_x is not None:
            has_physical = True
            x_phys = np.array(physical_x, dtype=float)
        else:
            x_phys = reconstructed_physical_x.copy()
            axis_index = entry.get("axis_index")
            if axis_index is not None:
                x_phys[int(axis_index)] += (
                    float(entry.get("pulse_angle_delta", 0.0)) *
                    int(entry.get("executed_pulse_count", 0))
                )
        physical_step_numbers.append(int(entry.get("execution_step", len(physical_step_numbers))))
        physical_angles.append(x_phys[angle_axes].astype(float))
        reconstructed_physical_x = x_phys.copy()

    executed_angles = np.array(executed_angles, dtype=float)
    physical_angles = np.array(physical_angles, dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    axes = axes.ravel()
    for idx, ax in enumerate(axes):
        ax.plot(
            planned_step_numbers,
            planned_angles[:, idx],
            marker="o",
            linestyle=":",
            linewidth=1.4,
            label="planned simulated",
        )
        ax.plot(
            executed_step_numbers,
            executed_angles[:, idx],
            marker="x",
            linestyle="-",
            linewidth=1.2,
            label="executed estimate",
        )
        if has_physical:
            ax.plot(
                physical_step_numbers,
                physical_angles[:, idx],
                marker=".",
                linestyle="--",
                linewidth=1.1,
                label="dry-run physical",
            )
        ax.set_title(f"{mirror_labels[idx]}.dangle")
        ax.set_ylabel("deg")
        ax.grid(True, linewidth=0.3)
        ax.legend()

    for ax in axes[-2:]:
        ax.set_xlabel("step")

    status = "success" if execution.get("success") else "failed"
    fig.suptitle(f"Mirror dangle trajectory during N_R execution ({status})", y=0.995)
    fig.tight_layout()
    return fig, axes


def quadcell_angle_jacobian(M1, M2, M3, M4, angles=None, step_deg=1e-4, active_actuators=None):
    """Finite-difference Jacobian d(QC1, QC2) / d(M*.dangle)."""
    return S.quadcell_angle_jacobian(
        M1,
        M2,
        M3,
        M4,
        angles=angles,
        step_deg=step_deg,
        active_actuators=active_actuators,
    )


def trace_centered_quadcell_angle_curve(M1, M2, M3, M4, **kwargs):
    """Trace a centered-QC fixed-N_R curve in four mirror-angle coordinates."""
    return S.trace_centered_quadcell_angle_curve(M1, M2, M3, M4, **kwargs)


def trace_full_centered_quadcell_angle_curve(M1, M2, M3, M4, **kwargs):
    """Trace a long centered-QC fixed-N_R curve until a boundary or step limit."""
    return S.trace_full_centered_quadcell_angle_curve(M1, M2, M3, M4, **kwargs)


def solve_and_trace_centered_quadcell_angle_curve(M1, M2, M3, M4, **kwargs):
    """Find a centered fixed-N_R config, then trace its centered-QC curve."""
    return S.solve_and_trace_centered_quadcell_angle_curve(M1, M2, M3, M4, **kwargs)


def correct_centered_quadcell_angles(M1, M2, M3, M4, **kwargs):
    """Run the local fixed-actuator QC-centering corrector used by curve tracing."""
    return S.correct_centered_quadcell_angles(M1, M2, M3, M4, **kwargs)


def search_centered_quadcell_angles(M1, M2, M3, M4, **kwargs):
    """Try multiple local QC-centering starts at fixed inactive actuators."""
    return S.search_centered_quadcell_angles(M1, M2, M3, M4, **kwargs)


def trace_centered_quadcell_angle_surface(M1, M2, M3, M4, **kwargs):
    """Trace a centered-QC surface as fixed-actuator curve slices."""
    return S.trace_centered_quadcell_angle_surface(M1, M2, M3, M4, **kwargs)


def solve_and_trace_centered_quadcell_angle_surface(M1, M2, M3, M4, **kwargs):
    """Find a centered fixed-N_R config, then trace a centered-QC surface."""
    return S.solve_and_trace_centered_quadcell_angle_surface(M1, M2, M3, M4, **kwargs)


def reflection_count_targets_up_to(M1, M2, M3, M4, max_N_R, **kwargs):
    """Return reflection counts from current/start count up to max_N_R."""
    return S.reflection_count_targets_up_to(M1, M2, M3, M4, max_N_R, **kwargs)


def relaxed_u_accept_bounds_for_step(u_min, u_max, step_index, divisor=2.0):
    """Relax edge margins toward [0, 1] by divisor**step_index."""
    return S.relaxed_u_accept_bounds_for_step(u_min, u_max, step_index, divisor=divisor)


def trace_reflection_count_surface_family(M1, M2, M3, M4, **kwargs):
    """Trace centered-QC surfaces for a sequence of reflection counts."""
    return S.trace_reflection_count_surface_family(M1, M2, M3, M4, **kwargs)


def plan_reflection_count_surface_stage(M1, M2, M3, M4, **kwargs):
    """Plan a same-N_R centered staging move before an inverse refresh."""
    return S.plan_reflection_count_surface_stage(M1, M2, M3, M4, **kwargs)


def summarize_reflection_count_surface_family(family):
    """Return one status row per requested reflection-count surface."""
    return S.summarize_reflection_count_surface_family(family)


def find_nearest_surface_curve_by_projection(reference_surface, target_surface, **kwargs):
    """Find the target surface curve closest in projected angle space."""
    return S.find_nearest_surface_curve_by_projection(reference_surface, target_surface, **kwargs)


def sample_quadcell_tolerance_cloud_around_surface_curve(surface, **kwargs):
    """Sample a loose-QC cloud around selected fixed-sweep surface curves."""
    return S.sample_quadcell_tolerance_cloud_around_surface_curve(surface, **kwargs)


def sample_quadcell_tolerance_tube_around_surface_curve(surface, curve_index, **kwargs):
    """Build a visible QC-tolerance tube around one centered surface curve."""
    return S.sample_quadcell_tolerance_tube_around_surface_curve(
        surface,
        curve_index,
        **kwargs,
    )


def scan_one_actuator_target_cloud_from_surface(surface, target_reflections, **kwargs):
    """Scan one actuator from a source surface and keep valid target-N_R landings."""
    return S.scan_one_actuator_target_cloud_from_surface(
        surface,
        target_reflections,
        **kwargs,
    )


def sample_reflection_count_surface_family_tubes(family, **kwargs):
    """Sample QC-tolerance tubes around closest adjacent family curves."""
    return S.sample_reflection_count_surface_family_tubes(family, **kwargs)


def find_nearest_surface_cloud_point_by_projection(reference_surface, target_cloud, **kwargs):
    """Find the target cloud point closest in projected angle space."""
    return S.find_nearest_surface_cloud_point_by_projection(reference_surface, target_cloud, **kwargs)


def plot_centered_quadcell_angle_curve(curve, color_by="coordinate"):
    """Plot pairwise projections of a centered-QC angle curve."""
    return S.plot_centered_quadcell_angle_curve(curve, color_by=color_by)


def plot_centered_quadcell_angle_curve_3d(curve, axes=None, color_by="coordinate",
                                          marker_size=4, show=True, width="100%",
                                          height=650, renderer=None, axis_ranges=None):
    """Interactive 3D plot of a centered-QC angle curve."""
    return S.plot_centered_quadcell_angle_curve_3d(
        curve,
        axes=axes,
        color_by=color_by,
        marker_size=marker_size,
        show=show,
        width=width,
        height=height,
        renderer=renderer,
        axis_ranges=axis_ranges,
    )


def plot_centered_quadcell_angle_curves_3d(curves, labels=None, axes=None,
                                           marker_size=4, show=True, width="100%",
                                           height=650, renderer=None, title=None,
                                           show_starts=True, axis_ranges=None):
    """Interactive 3D overlay of multiple centered-QC angle curves."""
    return S.plot_centered_quadcell_angle_curves_3d(
        curves,
        labels=labels,
        axes=axes,
        marker_size=marker_size,
        show=show,
        width=width,
        height=height,
        renderer=renderer,
        title=title,
        show_starts=show_starts,
        axis_ranges=axis_ranges,
    )


def plot_centered_quadcell_angle_surface_3d(surface, label=None, axes=None,
                                            marker_size=3, show=True, width="100%",
                                            height=650, renderer=None, title=None,
                                            opacity=0.9, show_start_markers=False,
                                            show_reference_markers=True,
                                            show_empty_surface_reference_markers=True,
                                            reference_marker_size=9,
                                            axis_ranges=None,
                                            clouds=None,
                                            cloud_labels=None,
                                            cloud_marker_size=2,
                                            cloud_opacity=0.35,
                                            tubes=None,
                                            tube_labels=None,
                                            tube_opacity=0.18,
                                            tube_color="rgba(35, 170, 255, 0.55)"):
    """Interactive 3D plot of a fixed-actuator slice surface."""
    return S.plot_centered_quadcell_angle_surface_3d(
        surface,
        label=label,
        axes=axes,
        marker_size=marker_size,
        show=show,
        width=width,
        height=height,
        renderer=renderer,
        title=title,
        opacity=opacity,
        show_start_markers=show_start_markers,
        show_reference_markers=show_reference_markers,
        show_empty_surface_reference_markers=show_empty_surface_reference_markers,
        reference_marker_size=reference_marker_size,
        axis_ranges=axis_ranges,
        clouds=clouds,
        cloud_labels=cloud_labels,
        cloud_marker_size=cloud_marker_size,
        cloud_opacity=cloud_opacity,
        tubes=tubes,
        tube_labels=tube_labels,
        tube_opacity=tube_opacity,
        tube_color=tube_color,
    )


def plot_centered_quadcell_angle_surfaces_3d(surfaces, labels=None, axes=None,
                                             marker_size=3, show=True, width="100%",
                                             height=650, renderer=None, title=None,
                                             opacity=0.9, show_start_markers=False,
                                             show_reference_markers=True,
                                             show_empty_surface_reference_markers=True,
                                             reference_marker_size=9,
                                             axis_ranges=None,
                                             clouds=None,
                                             cloud_labels=None,
                                             cloud_marker_size=2,
                                             cloud_opacity=0.35,
                                             tubes=None,
                                             tube_labels=None,
                                             tube_opacity=0.18,
                                             tube_color="rgba(35, 170, 255, 0.55)",
                                             color_by="sweep", cmap="viridis",
                                             marker_symbol=None):
    """Interactive 3D overlay of multiple centered-QC slice surfaces."""
    return S.plot_centered_quadcell_angle_surfaces_3d(
        surfaces,
        labels=labels,
        axes=axes,
        marker_size=marker_size,
        show=show,
        width=width,
        height=height,
        renderer=renderer,
        title=title,
        opacity=opacity,
        show_start_markers=show_start_markers,
        show_reference_markers=show_reference_markers,
        show_empty_surface_reference_markers=show_empty_surface_reference_markers,
        reference_marker_size=reference_marker_size,
        axis_ranges=axis_ranges,
        clouds=clouds,
        cloud_labels=cloud_labels,
        cloud_marker_size=cloud_marker_size,
        cloud_opacity=cloud_opacity,
        tubes=tubes,
        tube_labels=tube_labels,
        tube_opacity=tube_opacity,
        tube_color=tube_color,
        color_by=color_by,
        cmap=cmap,
        marker_symbol=marker_symbol,
    )


def plot_reflection_count_surface_family_3d(family, **kwargs):
    """Plot a surface family with color_by="N_R" or "sweep" and a chosen cmap."""
    return S.plot_reflection_count_surface_family_3d(family, **kwargs)


def extract_reflection_count_surface_boundary(surface_or_family, **kwargs):
    """Return ordered exact states along an outer edge of one traced surface."""
    return S.extract_reflection_count_surface_boundary(surface_or_family, **kwargs)


def render_reflection_count_surface_boundary_frame_3d(boundary, **kwargs):
    """Render one angle-surface boundary frame with a red point marker."""
    return S.render_reflection_count_surface_boundary_frame_3d(boundary, **kwargs)


def render_reflection_count_surface_boundary_simulation_frame(boundary, **kwargs):
    """Render the optical simulation for one exact surface-boundary state."""
    return S.render_reflection_count_surface_boundary_simulation_frame(boundary, **kwargs)


def save_reflection_count_surface_boundary_gifs(surface_or_family, **kwargs):
    """Save synchronized angle-surface and optical-simulation boundary GIFs."""
    return S.save_reflection_count_surface_boundary_gifs(surface_or_family, **kwargs)


def plot_reflection_count_surface_stage_plan_3d(plan, **kwargs):
    """Plot a staging plan, or an execution dict including its estimated path."""
    return S.plot_reflection_count_surface_stage_plan_3d(plan, **kwargs)


def plot_centered_quadcell_curve_diagnostics(curve):
    """Plot QC and reflection-u diagnostics along a centered-QC angle curve."""
    return S.plot_centered_quadcell_curve_diagnostics(curve)


def plot_fixed_plan_overlays(execution, show_difference=True, prefer_physical=True):
    """Return both fixed-plan overlay plots: quadcell offsets and reflection u."""
    return (
        plot_fixed_plan_quadcell_overlay(execution, show_difference=show_difference),
        plot_fixed_plan_reflection_u_overlay(execution, prefer_physical=prefer_physical),
    )


def run_closed_loop_dry_run_trials(
        target_OPD,
        M1,
        M2,
        M3,
        M4,
        *,
        seeds=range(20),
        dry_run_rotation_error=0.10,
        final_qc_tolerance=0.5,
        final_OPD_relaxed_tolerance=0.5,
        qc_detector_limit=3.9,
        qc_plan_limit=1.5,
        qc_hardware_stop=3.5,
        linear_stage_guard_mm=DEFAULT_LINEAR_STAGE_GUARD_MM,
        profile=False,
        **execute_kwargs):
    """Run repeated closed-loop dry-runs with randomized rotation step error."""
    rows = []
    for seed in list(seeds):
        _, _, execution = execute_OPD_closed_loop(
            target_OPD,
            M1,
            M2,
            M3,
            M4,
            dry_run=True,
            rng_seed=int(seed),
            dry_run_rotation_error=dry_run_rotation_error,
            final_qc_tolerance=final_qc_tolerance,
            final_OPD_relaxed_tolerance=final_OPD_relaxed_tolerance,
            qc_detector_limit=qc_detector_limit,
            qc_plan_limit=qc_plan_limit,
            qc_hardware_stop=qc_hardware_stop,
            linear_stage_guard_mm=linear_stage_guard_mm,
            profile=profile,
            **execute_kwargs,
        )
        final_qc = execution.get("final_sim_qc", [np.nan, np.nan])
        rows.append({
            "seed": int(seed),
            "success": bool(execution.get("success")),
            "failure_reason": execution.get("failure_reason"),
            "final_OPD": execution.get("final_OPD"),
            "final_OPD_error": execution.get("final_OPD_error"),
            "final_qc1": float(final_qc[0]),
            "final_qc2": float(final_qc[1]),
            "max_abs_final_qc": float(np.max(np.abs(final_qc))),
            "max_abs_measured_qc": execution.get("max_abs_measured_qc"),
            "rollback_count": execution.get("rollback_count"),
            "n_execution_steps": len(execution.get("execution_log", [])),
            "n_planner_runs": len(execution.get("planner_runs", [])),
        })

    summary = {
        "target_OPD": float(target_OPD),
        "n_trials": len(rows),
        "n_success": sum(1 for row in rows if row["success"]),
        "all_success": all(row["success"] for row in rows) if rows else False,
        "qc_detector_limit": float(qc_detector_limit),
        "qc_plan_limit": float(qc_plan_limit),
        "qc_hardware_stop": float(qc_hardware_stop),
        "linear_stage_guard_mm": float(linear_stage_guard_mm),
        "final_qc_tolerance": float(final_qc_tolerance),
        "final_OPD_relaxed_tolerance": float(final_OPD_relaxed_tolerance),
        "dry_run_rotation_error": float(dry_run_rotation_error),
        "max_abs_measured_qc": max(
            (row["max_abs_measured_qc"] for row in rows if row["max_abs_measured_qc"] is not None),
            default=0.0
        ),
        "rows": rows,
    }
    return summary
