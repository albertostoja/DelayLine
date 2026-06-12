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
        "direction": 1.0,
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
            locs[mirror_name] = float(hardware.stages.get_position(mapping["serial"]))
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

    before_position = linear_stage_locs.get(mirror_name)
    after_position = None
    actual_sim_delta = command_value

    if dry_run:
        if before_position is None:
            before_position = 0.0
        after_position = before_position + hardware_delta
        x_physical[axis_index] += command_value
    else:
        if hardware is None or getattr(hardware, "stages", None) is None:
            raise ValueError("hardware.stages is required to execute linear moves.")
        before_position = float(hardware.stages.get_position(serial))
        hardware.stages.move_relative(serial, hardware_delta)
        if linear_settle_delay and linear_settle_delay > 0:
            time.sleep(linear_settle_delay)
        after_position = float(hardware.stages.get_position(serial))
        actual_hardware_delta = after_position - before_position
        actual_sim_delta = actual_hardware_delta / direction

    x_model[axis_index] += actual_sim_delta
    linear_stage_locs[mirror_name] = after_position

    return {
        "kind": "linear",
        "serial": serial,
        "planned_sim_delta": command_value,
        "hardware_delta": hardware_delta,
        "actual_sim_delta": actual_sim_delta,
        "before_position": before_position,
        "after_position": after_position,
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
    edge_summary = S.reflection_edge_summary(
        x, *base_mirrors,
        include_ends=include_edge_ends
    )
    return {
        "OPD": float(S.OPD_from_variables(x, *base_mirrors)),
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
        planned_sim_qc = np.array([step["qc1_error"], step["qc2_error"]], dtype=float)
        planned_qc_y = _planned_step_qc_readout(step, qc_readout_sign)
        target_miss = float(np.linalg.norm(planned_qc_y - after_qc["y"]))
        estimate_sim_qc = np.array(
            S.quadcell_errors_from_variables(x_estimate, *base_mirrors),
            dtype=float
        )
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
    final_x_for_report = x_physical if dry_run else x_estimate
    final_mirrors = _x_to_mirrors(final_x_for_report, base_mirrors)
    final_OPD = S.OPD_from_variables(final_x_for_report, *base_mirrors)
    final_sim_qc = S.quadcell_errors_from_variables(final_x_for_report, *base_mirrors)
    final_OPD_error = float(final_OPD - target_OPD)

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
    }

    log(
        f"done success={final_success} OPD_error={final_OPD_error:.3f} "
        f"final_qc_x=({final_qc['x'][0]:.3f},{final_qc['x'][1]:.3f})"
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
    step_numbers = [0]
    first_before = np.array(log[0]["before_qc_y"], dtype=float)
    start_sim_qc = _sim_qc_from_quadcell_readout(first_before, qc_readout_sign)
    qc1_actual = [float(start_sim_qc[0])]
    qc2_actual = [float(start_sim_qc[1])]

    for entry in log:
        step_numbers.append(int(entry["execution_step"]))
        sim_qc = np.array(entry.get("measured_sim_qc"), dtype=float)
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
