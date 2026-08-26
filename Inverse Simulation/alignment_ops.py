import time
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
    "M3": "27601694",
}

DEFAULT_ROTATION_CONTROLLER = "newport"
DEFAULT_ROTATION_DEGREES_PER_SUBSTEP = 0.0023

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
    "M3.dx": {
        "kind": "linear",
        "mirror": "M3",
        "serial": DEFAULT_LINEAR_STAGE_SERIALS["M3"],
        "direction": 1.0,
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
            "valid": bool(np.all(valid_mask)),
            "error": None,
        }

    if hardware is None or not hasattr(hardware, "quads"):
        raise ValueError("A HardwareOps-like object with .quads is required unless dry_run=True.")

    signals = np.array(hardware.quads.get_signal_strength(), dtype=float)
    if signals.size != 2:
        raise ValueError(f"Expected two quadcell signal strengths, got {signals.tolist()}.")

    valid_mask = signals >= thresholds
    values = []
    raw = []
    read_error = None
    for sn, signal_sum in zip(hardware.quads.serial_numbers, signals):
        try:
            if hasattr(hardware.quads, "get_status"):
                status_full = hardware.quads.get_status(sn)
            else:
                status_full = hardware.quads.controllers[sn].Status
            status = status_full.PositionDifference
            signal_sum = float(status_full.Sum)
            x_mm, y_mm = hardware.quads.raw_to_mm(status.X, status.Y, signal_sum)
            values.extend([float(x_mm), float(y_mm)])
            raw.extend([status.X, status.Y])
        except Exception as exc:
            read_error = str(exc)
            values.extend([np.nan, np.nan])
            raw.extend([np.nan, np.nan])

    x = np.array([values[0], values[2]], dtype=float)
    return {
        "raw": raw,
        "x": x,
        "y": x,
        "axis": "x",
        "signal_strengths": signals.tolist(),
        "signal_thresholds": thresholds.tolist(),
        "valid_mask": valid_mask.tolist(),
        "valid": bool(np.all(valid_mask) and np.all(np.isfinite(x))),
        "error": read_error,
    }


def _initial_linear_stage_locs(
        hardware,
        actuator_map,
        *,
        M1_linear_loc=None,
        M2_linear_loc=None,
        M3_linear_loc=None,
        dry_run=False):
    provided = {
        "M1": M1_linear_loc,
        "M2": M2_linear_loc,
        "M3": M3_linear_loc,
    }
    locs = {}
    midpoint = getattr(S, "LINEAR_STAGE_TRAVEL_MM", 24.0) / 2.0

    for mirror_name, provided_loc in provided.items():
        if provided_loc is not None:
            locs[mirror_name] = float(provided_loc)
            continue

        label = f"{mirror_name}.dx"
        mapping = actuator_map.get(label)
        if (
            not dry_run and
            hardware is not None and
            getattr(hardware, "stages", None) is not None and
            mapping is not None
        ):
            direction = float(mapping.get("direction", 1.0))
            locs[mirror_name] = direction * float(hardware.stages.get_position(mapping["serial"]))
        else:
            locs[mirror_name] = float(midpoint)

    return locs


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
        rotation_settle_delay=0.5,
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
        "x_physical": x_physical.copy(),
        "calibrated_mirrors": calibrated_mirrors,
        "calibrated_mirrors_list": _mirrors_to_lists(calibrated_mirrors),
        "qc_min_signal": float(qc_min_signal),
        "qc_min_signals": (
            None if qc_min_signals is None
            else np.array(qc_min_signals, dtype=float).tolist()
        ),
    }
    log(f"done success={result['success']} failure={failure_reason}")
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
        linear_settle_delay):
    axis_index = step["axis_index"]
    command_value = float(step["command_value"])
    direction = float(mapping.get("direction", 1.0))
    hardware_delta = direction * command_value
    serial = mapping["serial"]
    mirror_name = mapping.get("mirror", step["actuator"].split(".")[0])

    before_sim_position = linear_stage_locs.get(mirror_name)
    after_sim_position = None
    before_hardware_position = None
    after_hardware_position = None
    actual_sim_delta = command_value

    if dry_run:
        if before_sim_position is None:
            before_sim_position = 0.0
        after_sim_position = before_sim_position + command_value
        x_physical[axis_index] += command_value
    else:
        if hardware is None or getattr(hardware, "stages", None) is None:
            raise ValueError("hardware.stages is required to execute linear moves.")
        before_hardware_position = float(hardware.stages.get_position(serial))
        if before_sim_position is None:
            before_sim_position = direction * before_hardware_position
        hardware.stages.move_relative(serial, hardware_delta)
        if linear_settle_delay and linear_settle_delay > 0:
            time.sleep(linear_settle_delay)
        after_hardware_position = float(hardware.stages.get_position(serial))
        actual_hardware_delta = after_hardware_position - before_hardware_position
        actual_sim_delta = actual_hardware_delta / direction
        after_sim_position = before_sim_position + actual_sim_delta

    x_model[axis_index] += actual_sim_delta
    linear_stage_locs[mirror_name] = after_sim_position

    return {
        "kind": "linear",
        "serial": serial,
        "planned_sim_delta": command_value,
        "hardware_delta": hardware_delta,
        "actual_sim_delta": actual_sim_delta,
        "before_position": before_hardware_position if not dry_run else before_sim_position,
        "after_position": after_hardware_position if not dry_run else after_sim_position,
        "before_hardware_position": before_hardware_position,
        "after_hardware_position": after_hardware_position,
        "before_sim_position": before_sim_position,
        "after_sim_position": after_sim_position,
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
        rotation_settle_delay):
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
    if best_distance <= qc_step_tolerance:
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
            error_factor = 1.0 + rng.uniform(-dry_run_rotation_error, dry_run_rotation_error)
            x_physical[axis_index] += sim_steps * degrees_per_substep * error_factor
        else:
            if hardware is None or getattr(hardware, "rotation_stages", None) is None:
                raise ValueError("hardware.rotation_stages is required to execute rotation moves.")
            hardware.rotation_stages.move_relative_steps(controller, actuator, hardware_steps)

        x_estimate[axis_index] += sim_steps * degrees_per_substep
        total_commanded_substeps += hardware_steps
        total_sim_substeps += sim_steps
        if rotation_settle_delay and rotation_settle_delay > 0:
            time.sleep(rotation_settle_delay)

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
            "qc_x": current_y.tolist(),
            "qc_y": current_y.tolist(),
            "distance_to_target": distance,
            "progress": progress,
            "improved": bool(improved),
        }
        chunk_logs.append(chunk_log)

        if float(np.max(np.abs(current_y))) > qc_safety_limit:
            if dry_run:
                x_physical[:] = x_physical_before_chunk
            else:
                hardware.rotation_stages.move_relative_steps(controller, actuator, -hardware_steps)
            x_estimate[:] = x_estimate_before_chunk
            total_commanded_substeps -= hardware_steps
            total_sim_substeps -= sim_steps
            rollback_count += 1
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
    rotation_calibration = _normalized_rotation_calibration(rotation_calibration)
    rng = np.random.default_rng(rng_seed)
    choose_OPD_kwargs = dict(choose_OPD_kwargs or {})
    qc_hardware_stop = float(qc_hardware_stop)
    qc_detector_limit = float(qc_detector_limit)
    qc_plan_limit = float(qc_plan_limit)
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
        f"linear_locs={linear_stage_locs}"
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
        mirrors_opt, final_res, latest_plan = S.choose_OPD(
            target_OPD,
            *current_mirrors,
            return_actuation_plan=True,
            final_center_qc_threshold=final_qc_tolerance,
            final_qc_tolerance=final_qc_tolerance,
            target_OPD_tolerance=target_OPD_tolerance,
            M1_linear_loc=linear_stage_locs["M1"],
            M2_linear_loc=linear_stage_locs["M2"],
            M3_linear_loc=linear_stage_locs["M3"],
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
                replan_reason = (
                    f"{actuator_label} QC target miss "
                    f"{detail.get('distance_to_target'):.3f} mm"
                )
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
        qc_plan_limit=1.5,
        qc_detector_limit=3.9,
        qc_hardware_stop=3.5,
        qc_step_tolerance=0.15,
        final_qc_tolerance=0.5,
        final_OPD_tolerance=0.5,
        require_final_qc=False,
        allow_final_qc_planner_failure=True,
        fast_qc_avg=3,
        fast_qc_delay=0.3,
        final_qc_avg=5,
        final_qc_delay=0.3,
        linear_settle_delay=2.0,
        rotation_settle_delay=0.35,
        step_settle_delay=0.35,
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
        choose_OPD_kwargs=None,
        **legacy_hardware_kwargs):
    """Execute one precomputed choose_OPD plan without intermediate replanning."""
    if profile and profile_sink is None:
        profile_sink = print

    t0 = time.perf_counter()

    def log(message):
        if not profile:
            return
        profile_sink(f"[execute_fixed_OPD {time.perf_counter() - t0:.3f}s] {message}")

    actuator_map = _merged_actuator_map(actuator_map)
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
    )

    planner_kwargs = dict(choose_OPD_kwargs)
    planner_qc_limit = qc_plan_limit
    planner_kwargs.setdefault("qc_plan_limit", planner_qc_limit)
    planner_kwargs.setdefault("qc_detector_limit", qc_detector_limit)
    planner_kwargs.setdefault("qc_hardware_stop", qc_hardware_stop)
    planner_kwargs.setdefault("final_qc_tolerance", final_qc_tolerance)
    planner_kwargs.setdefault("final_center_qc_threshold", final_qc_tolerance)
    planner_kwargs.setdefault("final_OPD_relaxed_tolerance", final_OPD_tolerance)
    planner_kwargs.setdefault("final_center_qc_priority", True)

    planner_profile = []
    mirrors_opt, planner_res, actuation_plan = S.choose_OPD(
        target_OPD,
        *base_mirrors,
        return_actuation_plan=True,
        M1_linear_loc=linear_stage_locs["M1"],
        M2_linear_loc=linear_stage_locs["M2"],
        M3_linear_loc=linear_stage_locs["M3"],
        profile=bool(profile),
        profile_sink=planner_profile.append,
        **planner_kwargs,
    )

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
        f"ignored_planner_failure={planner_failure_ignored}"
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
        log(
            f"step={step_index}/{len(steps)} actuator={actuator_label} "
            f"miss={target_miss:.3f} qc_x=({after_qc['x'][0]:.3f},{after_qc['x'][1]:.3f})"
        )

        if detail.get("failure_reason") is not None:
            failure_reason = detail["failure_reason"]
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
    }

    log(
        f"done success={final_success} OPD_error={final_OPD_error:.3f} "
        f"final_qc_x=({final_qc['x'][0]:.3f},{final_qc['x'][1]:.3f})"
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
        center_n_tries=2000,
        angle_perturb=0.3,
        seed=0,
        u_min=0.1,
        u_max=0.9,
        sigma_edge=0.1,
        final_qc_tolerance=0.5,
        qc_reacquire_limit=3.0,
        stage_qc_limit=3.0,
        final_center_qc_safety_limit=3.5,
        final_center_max_axis_splits=80,
        final_center_waypoint_depth=4,
        final_center_feedback_max_pulses=80,
        final_center_min_improvement=0.01,
        require_target_N_R_estimate_for_reacquire_stop=False,
        require_beam_departure_before_reacquire=True,
        min_reacquire_pulse_fraction=0.0,
        reacquire_angle_scan_limit=1.0,
        reacquire_scan_samples=1001,
        reacquire_max_first_leg_candidates=50,
        reacquire_stage_max_candidates=24,
        reacquisition_strategy="forced_stage_one_axis",
        forced_stage_samples=41,
        forced_stage_free_angle_regularization=0.02,
        forced_stage_max_nfev=160,
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
        rotation_settle_delay=0.5,
        step_settle_delay=0.35,
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
        dry_run_rotation_error=0.10,
        dry_run_signal_strengths=(1.0, 1.0),
        dry_run_pulse_response=None,
        rng_seed=None,
        profile=True,
        profile_sink=None):
    """Execute a rotation-only reflection-count change with beam reacquisition.

    The topology transition is executed as small calibrated pulses. QC position
    is not treated as a path constraint until the beam is reacquired with valid
    signal strength. Once reacquired, a short QC-guided final centering phase is
    attempted.
    """
    if profile and profile_sink is None:
        profile_sink = print

    t0 = time.perf_counter()

    def log(message):
        if not profile:
            return
        profile_sink(f"[execute_N_R {time.perf_counter() - t0:.3f}s] {message}")

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
    x_physical = x_estimate.copy()
    calibration_result = None
    pulse_table = rotation_pulse_calibration
    reacquisition_strategy = str(reacquisition_strategy)
    valid_reacquisition_strategies = {
        "forced_stage_one_axis",
        "staged_one_axis",
        "direct_scan",
    }
    if reacquisition_strategy not in valid_reacquisition_strategies:
        raise ValueError(
            "reacquisition_strategy must be one of "
            f"{sorted(valid_reacquisition_strategies)}, got {reacquisition_strategy!r}."
        )

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

    if calibrate_before and staged_qc_only_strategy:
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
                "reflection_count_change": True,
                "reflection_count_reacquisition": True,
                "rotation_only": True,
                "target_N_R": int(target_N_R),
                "reacquisition_strategy": reacquisition_strategy,
                "requires_inverse_refresh": False,
                "suggested_next_step": None,
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
        log("calibration complete; planning from post-calibration mirror estimate")

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
            "qc_only_reacquire_stop": staged_qc_only_strategy,
            "stage_plan": None,
            "target_jump_step": None,
            "failure_reason": str(exc),
        }
        planner_failure_reason = str(exc)

    steps = actuation_plan.get("steps", [])
    failure_reason = None
    if planner_failure_reason is not None:
        failure_reason = "Planner failed: " + planner_failure_reason
    elif len(steps) == 0:
        start_reflections = S.get_reflection_count(*base_mirrors)
        if start_reflections != int(target_N_R):
            failure_reason = "Planner returned no rotation steps."

    log(
        f"start target_N_R={int(target_N_R)} dry_run={dry_run} "
        f"strategy={reacquisition_strategy} "
        f"planned_steps={len(steps)} search_mode={actuation_plan.get('search_mode')} "
        f"qc_reacquire_limit={qc_reacquire_limit} "
        f"{planner_timing_text(actuation_plan)} failure={failure_reason}"
    )

    execution_log = []
    center_execution_log = []
    total_commanded_substeps = 0
    total_execution_pulses = 0
    beam_reacquired_during_sweep = False
    beam_departed_reacquire_window = not bool(require_beam_departure_before_reacquire)
    beam_departed_at = None
    reacquired_qc = None
    reacquired_at = None
    last_reacquisition_axis = None
    last_reacquisition_actuator = None
    last_reacquisition_assimilation = None
    last_reacquired_by = None
    requires_inverse_refresh = False
    final_center_path_failure = None

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
        target_center_step = bool(step.get("target_center_step", False))
        reacquire_stop_disabled = bool(step.get("disable_reacquire_stop", False))
        qc_only_reacquire_stop = bool(
            actuation_plan.get("qc_only_reacquire_stop", False) and
            target_jump_step and
            not reacquire_stop_disabled
        )
        reacquire_stop_enabled = bool(target_jump_step and not reacquire_stop_disabled)
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

        remaining_pulses = pulse_count
        pulse_logs = []
        step_commanded_substeps = 0
        step_sim_substeps = 0
        step_t0 = time.perf_counter()
        step_reacquired = False

        while remaining_pulses != 0:
            if total_execution_pulses >= max_total_steps:
                failure_reason = f"Reached max_total_steps={max_total_steps} pulses."
                break

            sim_steps = int(np.sign(remaining_pulses) * execution_pulse_substeps)
            hardware_steps = hardware_direction * sim_steps

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

            x_estimate[axis_index] += pulse_angle_delta
            remaining_pulses -= int(np.sign(remaining_pulses))
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
            pulse_log = {
                "pulse": len(pulse_logs) + 1,
                "hardware_steps": int(hardware_steps),
                "sim_steps": int(sim_steps),
                "pulse_angle_delta": float(pulse_angle_delta),
                "expected_angle_after_pulse": float(x_estimate[axis_index]),
                "estimated_angle": float(x_estimate[axis_index]),
                "qc_x": chunk_qc["x"].tolist(),
                "qc_y": chunk_qc["y"].tolist(),
                "qc_signal_strengths": chunk_qc["signal_strengths"],
                "qc_signal_valid": bool(chunk_qc["valid"]),
                "qc_valid_mask": chunk_qc["valid_mask"],
                "sim_reflection_count": int(S.get_reflection_count(*estimated_mirrors)),
                "sim_OPD": float(S.OPD_from_variables(x_estimate, *base_mirrors)),
            }
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
                    pulse_log["step_pulses_completed"] >= int(min_reacquire_pulses)
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
                if (
                    reacquire_stop_enabled and
                    pulse_log["qc_in_reacquire_window"] and
                    departure_ok and
                    pulse_log["past_min_reacquire_pulses"] and
                    target_estimate_ok
                ):
                    pulse_log["reacquired"] = True
                    if qc_only_reacquire_stop:
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
                    if qc_only_reacquire_stop:
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
            pulse_logs.append(pulse_log)
            if step_reacquired:
                break

        if failure_reason is not None:
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

        entry = {
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
            "planned_angle_delta": planned_angle_delta,
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
        log(
            f"step={step_index}/{len(steps)} actuator={actuator_label} "
            f"planned_pulses={pulse_count} executed_pulses={len(pulse_logs)} "
            f"substeps={step_commanded_substeps} "
            f"residual_angle={residual_angle:.4g} "
            f"qc_valid={after_qc['valid']} "
            f"qc_x=({after_qc['x'][0]:.3f},{after_qc['x'][1]:.3f}) "
            f"N_R_sim={estimate_metrics['reflection_count']} "
            f"departed={beam_departed_reacquire_window} reacquired={step_reacquired}"
        )
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
            (
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
            if require_beam_departure_before_reacquire and not beam_departed_reacquire_window:
                failure_reason = (
                    "beam_not_reacquired: beam never left the "
                    f"+/-{qc_reacquire_limit} mm reacquisition window."
                )
            else:
                failure_reason = (
                    "beam_not_reacquired: QC did not become valid within "
                    f"+/-{qc_reacquire_limit} mm during reacquisition sweep."
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
    final_mirrors = S.unpack_variables(x_estimate, *base_mirrors)
    final_res = S.set_OPD_result_full_x(
        SimpleNamespace(success=False, message=""),
        *final_mirrors
    )
    final_OPD = S.OPD_from_variables(x_estimate, *base_mirrors)
    final_sim_qc = S.quadcell_errors_from_variables(x_estimate, *base_mirrors)
    final_sim_reflections = S.get_reflection_count(*final_mirrors)
    final_qc_max_abs = float(np.max(np.abs(final_qc["x"])))
    final_qc_centered = final_qc_max_abs <= float(final_qc_tolerance)
    beam_reacquired = bool(
        final_qc["valid"] and
        final_qc_max_abs <= float(qc_reacquire_limit)
    )
    target_reflection_sim_reached = int(final_sim_reflections) == int(target_N_R)

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
    if failure_reason is None and not requires_inverse_refresh and not target_reflection_sim_reached:
        failure_reason = (
            f"simulated reflection count {final_sim_reflections} != target {target_N_R}."
        )

    final_success = failure_reason is None
    final_res.success = bool(final_success)
    if final_success and requires_inverse_refresh:
        final_res.message = (
            "Reflection-count execution reacquired QC signal; inverse refresh required."
        )
    elif final_success:
        final_res.message = "Reflection-count execution reacquired centered QC signal."
    else:
        final_res.message = "Reflection-count execution stopped before success criteria were met."

    execution = {
        "success": bool(final_success),
        "failure_reason": failure_reason,
        "reflection_count_change": True,
        "reflection_count_reacquisition": True,
        "rotation_only": True,
        "qc_path_unconstrained": True,
        "target_N_R": int(target_N_R),
        "reacquisition_strategy": reacquisition_strategy,
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
            "take light/dark images and run optimize_inverse"
            if requires_inverse_refresh else None
        ),
        "reacquired_by": last_reacquired_by,
        "final_sim_reflection_count": int(final_sim_reflections),
        "target_reflection_sim_reached": bool(target_reflection_sim_reached),
        "reflection_count_verified": False,
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
        "rotation_calibration": dict(rotation_calibration),
        "calibrate_before": bool(calibrate_before),
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
        "final_center_after_reacquire": bool(final_center_after_reacquire),
        "final_center_max_axis_splits": int(final_center_max_axis_splits),
        "final_center_waypoint_depth": int(final_center_waypoint_depth),
        "final_center_feedback_max_pulses": int(final_center_feedback_max_pulses),
        "final_center_min_improvement": float(final_center_min_improvement),
        "require_target_N_R_estimate_for_reacquire_stop": bool(require_target_N_R_estimate_for_reacquire_stop),
        "min_reacquire_pulse_fraction": float(min_reacquire_pulse_fraction),
        "reacquire_angle_scan_limit": float(reacquire_angle_scan_limit),
        "reacquire_scan_samples": int(reacquire_scan_samples),
        "reacquire_max_first_leg_candidates": int(reacquire_max_first_leg_candidates),
        "reacquire_stage_max_candidates": int(reacquire_stage_max_candidates),
        "forced_stage_samples": int(forced_stage_samples),
        "forced_stage_free_angle_regularization": float(forced_stage_free_angle_regularization),
        "forced_stage_max_nfev": int(forced_stage_max_nfev),
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
        f"N_R_sim={final_sim_reflections} target={int(target_N_R)} "
        f"beam_reacquired={beam_reacquired} "
        f"requires_inverse_refresh={requires_inverse_refresh} "
        f"qc_valid={final_qc['valid']} "
        f"qc_x=({final_qc['x'][0]:.3f},{final_qc['x'][1]:.3f})"
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


def plot_fixed_plan_quadcell_overlay(execution, show_difference=True):
    """Plot planned quadcell offsets with the fixed-plan measured path overlaid."""
    actuation_plan = execution["actuation_plan"]
    fig, ax = S.plot_actuation_quadcell_offsets(
        actuation_plan,
        show_difference=show_difference
    )

    log = execution.get("execution_log", [])
    if len(log) == 0:
        return fig, ax

    qc_readout_sign = float(execution.get("qc_readout_sign", -1.0))

    def sim_qc_from_entry(entry, key):
        if key == "before":
            raw = entry.get("before_qc_x", entry.get("before_qc_y"))
        else:
            raw = entry.get("after_qc_x", entry.get("after_qc_y"))
        if raw is None:
            return None
        return _sim_qc_from_quadcell_readout(np.array(raw, dtype=float), qc_readout_sign)

    step_numbers = [0]
    start_sim_qc = sim_qc_from_entry(log[0], "before")
    if start_sim_qc is None:
        start_sim_qc = np.array([np.nan, np.nan], dtype=float)
    qc1_actual = [float(start_sim_qc[0])]
    qc2_actual = [float(start_sim_qc[1])]

    for entry in log:
        step_numbers.append(int(entry["execution_step"]))
        sim_qc = entry.get("measured_sim_qc")
        if sim_qc is None:
            sim_qc = sim_qc_from_entry(entry, "after")
        sim_qc = np.array(
            [np.nan, np.nan] if sim_qc is None else sim_qc,
            dtype=float
        )
        qc1_actual.append(float(sim_qc[0]))
        qc2_actual.append(float(sim_qc[1]))

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

    ax.set_title("Quadcell Beam Offset During Fixed-Plan Execution")
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_fixed_plan_reflection_u_overlay(execution, prefer_physical=True):
    """Plot planned reflection-u bounds with the executed model path overlaid.

    For dry-run executions, prefer_physical=True overlays the randomized physical
    dry-run path. For real hardware, reflection u is not directly measured, so
    this falls back to the command-estimated model path.
    """
    actuation_plan = execution["actuation_plan"]
    fig, ax = S.plot_actuation_reflection_u(actuation_plan)

    log = execution.get("execution_log", [])
    if len(log) == 0:
        return fig, ax

    use_physical = (
        prefer_physical and
        any(entry.get("physical_path_metrics") is not None for entry in log)
    )
    metrics_key = "physical_path_metrics" if use_physical else "estimate_path_metrics"
    label_prefix = "actual dry-run" if use_physical else "executed model"

    step_numbers = []
    min_us = []
    max_us = []
    margins = []
    for entry in log:
        metrics = entry.get(metrics_key)
        if metrics is None:
            continue
        step_numbers.append(int(entry["execution_step"]))
        min_us.append(float(metrics["min_reflection_u"]))
        max_us.append(float(metrics["max_reflection_u"]))
        margins.append(float(metrics["closest_edge_margin"]))

    if len(step_numbers) == 0:
        return fig, ax

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
    ax.legend()
    fig.tight_layout()
    return fig, ax


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


def trace_centered_quadcell_angle_surface(M1, M2, M3, M4, **kwargs):
    """Trace a centered-QC surface as fixed-actuator curve slices."""
    return S.trace_centered_quadcell_angle_surface(M1, M2, M3, M4, **kwargs)


def solve_and_trace_centered_quadcell_angle_surface(M1, M2, M3, M4, **kwargs):
    """Find a centered fixed-N_R config, then trace a centered-QC surface."""
    return S.solve_and_trace_centered_quadcell_angle_surface(M1, M2, M3, M4, **kwargs)


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
