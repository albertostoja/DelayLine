from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Literal, Sequence

import numpy as np
import numpy.typing as npt


ArrayLike = npt.ArrayLike


def _require_positive(name: str, value: float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value!r}")


def _require_nonnegative(name: str, value: float) -> None:
    if value < 0:
        raise ValueError(f"{name} must be nonnegative, got {value!r}")


def _require_unit_interval(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1, got {value!r}")


def _as_float_array(name: str, value: ArrayLike) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1D array")
    return array


@dataclass(frozen=True)
class BeamConfig:
    """Gaussian beam parameters.

    The most bench-like configuration is to set `radius_at_output_aperture_m`,
    `distance_output_aperture_to_fsm_m`, and `divergence_half_angle_rad`. Then
    the beam radius is computed from the total distance downstream of the laser
    output aperture:

        w(z) = sqrt(w_output^2 + (theta_div * z)^2)

    If `radius_at_output_aperture_m` is left as None, the legacy
    `radius_at_fsm_m` value is used as the 1/e^2 intensity radius at the FSM:

        w(z) = sqrt(w_fsm^2 + (theta_div * z)^2)
    """

    wavelength_m: float = 633e-9
    radius_at_fsm_m: float = 0.5e-3
    divergence_half_angle_rad: float = 0.0
    radius_at_output_aperture_m: float | None = None
    distance_output_aperture_to_fsm_m: float = 0.0

    def __post_init__(self) -> None:
        _require_positive("wavelength_m", self.wavelength_m)
        _require_positive("radius_at_fsm_m", self.radius_at_fsm_m)
        _require_nonnegative(
            "divergence_half_angle_rad", self.divergence_half_angle_rad
        )
        if self.radius_at_output_aperture_m is not None:
            _require_positive(
                "radius_at_output_aperture_m", self.radius_at_output_aperture_m
            )
        _require_nonnegative(
            "distance_output_aperture_to_fsm_m",
            self.distance_output_aperture_to_fsm_m,
        )

    @property
    def wave_number_rad_per_m(self) -> float:
        return 2.0 * math.pi / self.wavelength_m

    @property
    def radius_at_fsm_effective_m(self) -> float:
        """1/e^2 intensity radius at the FSM after applying the chosen beam model."""
        if self.radius_at_output_aperture_m is None:
            return self.radius_at_fsm_m
        return float(self.radius_from_output_aperture(self.distance_output_aperture_to_fsm_m))

    def radius_from_output_aperture(
        self, distance_m: float | np.ndarray
    ) -> float | np.ndarray:
        """Beam radius a distance downstream of the laser output aperture."""
        if self.radius_at_output_aperture_m is None:
            raise ValueError(
                "radius_from_output_aperture requires "
                "radius_at_output_aperture_m to be set"
            )
        distance = np.asarray(distance_m)
        return np.sqrt(
            self.radius_at_output_aperture_m**2
            + (self.divergence_half_angle_rad * distance) ** 2
        )

    def radius_at(self, distance_m: float | np.ndarray) -> float | np.ndarray:
        """Beam radius a distance downstream of the FSM."""
        distance = np.asarray(distance_m)
        if self.radius_at_output_aperture_m is not None:
            return self.radius_from_output_aperture(
                self.distance_output_aperture_to_fsm_m + distance
            )
        return np.sqrt(
            self.radius_at_fsm_m**2
            + (self.divergence_half_angle_rad * distance) ** 2
        )


@dataclass(frozen=True)
class InterferometerConfig:
    """Mach-Zehnder interferometer geometry and static phase settings."""

    distance_fsm_to_combiner_m: float = 0.25
    fsm_incidence_angle_rad: float = math.pi / 4.0
    input_power_w: float = 1.0
    baseline_visibility: float = 0.98
    nominal_phase_rad: float = 0.0
    static_opd_m: float = 0.0
    quadrature_phase_rad: float = math.pi / 2.0
    fsm_vertical_pivot_offset_m: float = 0.0
    fsm_horizontal_pivot_offset_m: float = 0.0
    piston_shear_axis: Literal["horizontal", "vertical"] = "horizontal"
    include_piston_opd: bool = True
    include_piston_shear: bool = True
    include_angular_path_length: bool = True

    def __post_init__(self) -> None:
        _require_positive("distance_fsm_to_combiner_m", self.distance_fsm_to_combiner_m)
        _require_nonnegative("fsm_incidence_angle_rad", self.fsm_incidence_angle_rad)
        if self.fsm_incidence_angle_rad >= math.pi / 2.0:
            raise ValueError("fsm_incidence_angle_rad must be less than pi/2")
        _require_positive("input_power_w", self.input_power_w)
        _require_unit_interval("baseline_visibility", self.baseline_visibility)
        if self.piston_shear_axis not in ("horizontal", "vertical"):
            raise ValueError("piston_shear_axis must be 'horizontal' or 'vertical'")


@dataclass(frozen=True)
class DetectorConfig:
    """Photodiode channel calibration and noise settings."""

    vertical_gain: float = 1.0
    horizontal_gain: float = 1.0
    vertical_offset_w: float = 0.0
    horizontal_offset_w: float = 0.0
    vertical_contrast: float = 1.0
    horizontal_contrast: float = 1.0
    read_noise_std_w: float = 0.0
    relative_intensity_noise_std: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "vertical_gain",
            "horizontal_gain",
            "read_noise_std_w",
            "relative_intensity_noise_std",
        ):
            _require_nonnegative(name, float(getattr(self, name)))
        _require_unit_interval("vertical_contrast", self.vertical_contrast)
        _require_unit_interval("horizontal_contrast", self.horizontal_contrast)


@dataclass(frozen=True)
class PolarizationConfig:
    """Jones-vector polarization optics for the two interferometer arms.

    Angles are fast-axis angles measured from vertical toward horizontal.
    The default assumes a vertically polarized laser, a HWP in the reference
    arm that produces 45 degree diagonal linear polarization, and a QWP in the
    FSM arm that produces circular polarization.
    """

    input_linear_angle_rad: float = 0.0
    reference_hwp_fast_axis_angle_rad: float = math.radians(22.5)
    reference_hwp_retardance_rad: float = math.pi
    fsm_qwp_fast_axis_angle_rad: float = math.radians(45.0)
    fsm_qwp_retardance_rad: float = math.pi / 2.0
    reference_arm_power_fraction: float = 0.5
    fsm_arm_power_fraction: float = 0.5

    def __post_init__(self) -> None:
        _require_nonnegative(
            "reference_arm_power_fraction", self.reference_arm_power_fraction
        )
        _require_nonnegative("fsm_arm_power_fraction", self.fsm_arm_power_fraction)
        total_fraction = self.reference_arm_power_fraction + self.fsm_arm_power_fraction
        _require_positive(
            "reference_arm_power_fraction + fsm_arm_power_fraction", total_fraction
        )
        if total_fraction > 1.0:
            raise ValueError(
                "reference_arm_power_fraction + fsm_arm_power_fraction must be <= 1"
            )


@dataclass(frozen=True)
class FSMTone:
    """Sinusoidal FSM disturbance component."""

    frequency_hz: float
    vertical_steering_amplitude_rad: float = 0.0
    horizontal_steering_amplitude_rad: float = 0.0
    piston_amplitude_m: float = 0.0
    phase_rad: float = 0.0
    horizontal_steering_phase_rad: float = 0.0
    piston_phase_rad: float = 0.0

    def __post_init__(self) -> None:
        _require_positive("frequency_hz", self.frequency_hz)
        for name in (
            "vertical_steering_amplitude_rad",
            "horizontal_steering_amplitude_rad",
            "piston_amplitude_m",
        ):
            _require_nonnegative(name, abs(float(getattr(self, name))))


@dataclass(frozen=True)
class FSMJitterConfig:
    """Synthetic FSM motion generator settings.

    Vertical steering is the FSM mirror rotation that sends the reflected beam
    out of the optical-table plane. Horizontal steering sends it left/right
    while staying in the optical-table plane. These are mirror angles; reflected
    beam deflection is twice as large.
    """

    sample_rate_hz: float = 20_000.0
    duration_s: float = 0.05
    vertical_steering_rms_rad: float = 1.0e-6
    horizontal_steering_rms_rad: float = 1.0e-6
    piston_rms_m: float = 15e-9
    correlation_time_s: float = 0.0
    vertical_steering_drift_rad_per_s: float = 0.0
    horizontal_steering_drift_rad_per_s: float = 0.0
    piston_drift_m_per_s: float = 0.0
    tones: Sequence[FSMTone] = field(default_factory=tuple)
    seed: int | None = 7

    def __post_init__(self) -> None:
        _require_positive("sample_rate_hz", self.sample_rate_hz)
        _require_positive("duration_s", self.duration_s)
        for name in (
            "vertical_steering_rms_rad",
            "horizontal_steering_rms_rad",
            "piston_rms_m",
            "correlation_time_s",
        ):
            _require_nonnegative(name, float(getattr(self, name)))


@dataclass(frozen=True)
class FSMMotion:
    """FSM time-series motion.

    `vertical_steering_angle_rad` and `horizontal_steering_angle_rad` are FSM
    mirror angle changes. The reflected beam deflects by twice these angles.
    `piston_m` is physical mirror-normal surface motion, not already-doubled
    optical path difference.
    """

    time_s: np.ndarray
    vertical_steering_angle_rad: np.ndarray
    horizontal_steering_angle_rad: np.ndarray
    piston_m: np.ndarray

    def __post_init__(self) -> None:
        time_s = _as_float_array("time_s", self.time_s)
        vertical_steering_angle_rad = _as_float_array(
            "vertical_steering_angle_rad", self.vertical_steering_angle_rad
        )
        horizontal_steering_angle_rad = _as_float_array(
            "horizontal_steering_angle_rad", self.horizontal_steering_angle_rad
        )
        piston_m = _as_float_array("piston_m", self.piston_m)
        shape = time_s.shape
        for name, array in (
            ("vertical_steering_angle_rad", vertical_steering_angle_rad),
            ("horizontal_steering_angle_rad", horizontal_steering_angle_rad),
            ("piston_m", piston_m),
        ):
            if array.shape != shape:
                raise ValueError(f"{name} shape {array.shape} does not match time {shape}")
        object.__setattr__(self, "time_s", time_s)
        object.__setattr__(
            self, "vertical_steering_angle_rad", vertical_steering_angle_rad
        )
        object.__setattr__(
            self, "horizontal_steering_angle_rad", horizontal_steering_angle_rad
        )
        object.__setattr__(self, "piston_m", piston_m)


@dataclass(frozen=True)
class ReferenceMirrorTapConfig:
    """Sinusoidal tapped-mirror motion in the other MZI arm.

    Positive displacement is along the nominal direction toward the combiner by
    default. `opd_coupling` converts the longitudinal displacement into optical
    path difference. The default `1.0` is a simple one-way arm-length change.
    Use `2.0` for a normal-incidence reflected piston or `sqrt(2)` for a 45 deg
    fold-normal piston, depending on the actual mirror motion being modeled.
    Small horizontal/vertical direction angles add beam walk at the combiner.
    """

    sample_rate_hz: float = 20_000.0
    duration_s: float = 0.05
    amplitude_m: float = 150e-9
    frequency_hz: float = 250.0
    phase_rad: float = 0.0
    horizontal_direction_angle_rad: float = 0.0
    vertical_direction_angle_rad: float = 0.0
    opd_coupling: float = 1.0
    positive_toward_combiner_shortens_opd: bool = True

    def __post_init__(self) -> None:
        _require_positive("sample_rate_hz", self.sample_rate_hz)
        _require_positive("duration_s", self.duration_s)
        _require_nonnegative("amplitude_m", abs(self.amplitude_m))
        _require_positive("frequency_hz", self.frequency_hz)
        _require_nonnegative("opd_coupling", self.opd_coupling)


@dataclass(frozen=True)
class ReferenceMirrorMotion:
    """Time-series motion for the non-FSM mirror used to sweep OPD."""

    time_s: np.ndarray
    displacement_m: np.ndarray
    opd_m: np.ndarray
    horizontal_shear_m: np.ndarray
    vertical_shear_m: np.ndarray

    def __post_init__(self) -> None:
        time_s = _as_float_array("time_s", self.time_s)
        displacement_m = _as_float_array("displacement_m", self.displacement_m)
        opd_m = _as_float_array("opd_m", self.opd_m)
        horizontal_shear_m = _as_float_array(
            "horizontal_shear_m", self.horizontal_shear_m
        )
        vertical_shear_m = _as_float_array("vertical_shear_m", self.vertical_shear_m)
        shape = time_s.shape
        for name, array in (
            ("displacement_m", displacement_m),
            ("opd_m", opd_m),
            ("horizontal_shear_m", horizontal_shear_m),
            ("vertical_shear_m", vertical_shear_m),
        ):
            if array.shape != shape:
                raise ValueError(f"{name} shape {array.shape} does not match time {shape}")
        object.__setattr__(self, "time_s", time_s)
        object.__setattr__(self, "displacement_m", displacement_m)
        object.__setattr__(self, "opd_m", opd_m)
        object.__setattr__(self, "horizontal_shear_m", horizontal_shear_m)
        object.__setattr__(self, "vertical_shear_m", vertical_shear_m)


@dataclass(frozen=True)
class SimulationResult:
    """Interferometer simulation output arrays."""

    time_s: np.ndarray
    vertical_w: np.ndarray
    horizontal_w: np.ndarray
    phase_rad: np.ndarray
    visibility: np.ndarray
    gaussian_overlap: np.ndarray
    opd_m: np.ndarray
    mirror_vertical_steering_angle_rad: np.ndarray
    mirror_horizontal_steering_angle_rad: np.ndarray
    piston_m: np.ndarray
    beam_vertical_deflection_rad: np.ndarray
    beam_horizontal_deflection_rad: np.ndarray
    steering_vertical_shear_m: np.ndarray
    steering_horizontal_shear_m: np.ndarray
    piston_induced_shear_m: np.ndarray
    piston_induced_opd_m: np.ndarray
    reference_mirror_displacement_m: np.ndarray
    reference_mirror_opd_m: np.ndarray
    reference_mirror_vertical_shear_m: np.ndarray
    reference_mirror_horizontal_shear_m: np.ndarray
    vertical_shear_m: np.ndarray
    horizontal_shear_m: np.ndarray
    beam_radius_at_combiner_m: float

    @property
    def wrapped_phase_rad(self) -> np.ndarray:
        return np.angle(np.exp(1j * self.phase_rad))

    def to_dataframe(self):
        """Return the simulation as a pandas DataFrame."""
        import pandas as pd

        return pd.DataFrame(
            {
                "time_s": self.time_s,
                "vertical_w": self.vertical_w,
                "horizontal_w": self.horizontal_w,
                "phase_rad": self.phase_rad,
                "wrapped_phase_rad": self.wrapped_phase_rad,
                "visibility": self.visibility,
                "gaussian_overlap": self.gaussian_overlap,
                "opd_m": self.opd_m,
                "mirror_vertical_steering_angle_rad": self.mirror_vertical_steering_angle_rad,
                "mirror_horizontal_steering_angle_rad": self.mirror_horizontal_steering_angle_rad,
                "piston_m": self.piston_m,
                "beam_vertical_deflection_rad": self.beam_vertical_deflection_rad,
                "beam_horizontal_deflection_rad": self.beam_horizontal_deflection_rad,
                "steering_vertical_shear_m": self.steering_vertical_shear_m,
                "steering_horizontal_shear_m": self.steering_horizontal_shear_m,
                "piston_induced_shear_m": self.piston_induced_shear_m,
                "piston_induced_opd_m": self.piston_induced_opd_m,
                "reference_mirror_displacement_m": self.reference_mirror_displacement_m,
                "reference_mirror_opd_m": self.reference_mirror_opd_m,
                "reference_mirror_vertical_shear_m": self.reference_mirror_vertical_shear_m,
                "reference_mirror_horizontal_shear_m": self.reference_mirror_horizontal_shear_m,
                "vertical_shear_m": self.vertical_shear_m,
                "horizontal_shear_m": self.horizontal_shear_m,
            }
        )


def simulate_fsm_motion(config: FSMJitterConfig = FSMJitterConfig()) -> FSMMotion:
    """Generate synthetic FSM vertical steering, horizontal steering, and piston motion."""

    n_samples = max(2, int(round(config.duration_s * config.sample_rate_hz)))
    time_s = np.arange(n_samples, dtype=float) / config.sample_rate_hz
    rng = np.random.default_rng(config.seed)
    dt = 1.0 / config.sample_rate_hz

    vertical_steering_angle = _noise_with_correlation(
        rng, n_samples, config.vertical_steering_rms_rad, dt, config.correlation_time_s
    )
    horizontal_steering_angle = _noise_with_correlation(
        rng, n_samples, config.horizontal_steering_rms_rad, dt, config.correlation_time_s
    )
    piston = _noise_with_correlation(
        rng, n_samples, config.piston_rms_m, dt, config.correlation_time_s
    )

    vertical_steering_angle += config.vertical_steering_drift_rad_per_s * time_s
    horizontal_steering_angle += config.horizontal_steering_drift_rad_per_s * time_s
    piston += config.piston_drift_m_per_s * time_s

    for tone in config.tones:
        omega_t = 2.0 * math.pi * tone.frequency_hz * time_s
        vertical_steering_angle += tone.vertical_steering_amplitude_rad * np.sin(
            omega_t + tone.phase_rad
        )
        horizontal_steering_angle += tone.horizontal_steering_amplitude_rad * np.sin(
            omega_t + tone.phase_rad + tone.horizontal_steering_phase_rad
        )
        piston += tone.piston_amplitude_m * np.sin(
            omega_t + tone.phase_rad + tone.piston_phase_rad
        )

    return FSMMotion(
        time_s=time_s,
        vertical_steering_angle_rad=vertical_steering_angle,
        horizontal_steering_angle_rad=horizontal_steering_angle,
        piston_m=piston,
    )


def simulate_reference_mirror_tap(
    config: ReferenceMirrorTapConfig = ReferenceMirrorTapConfig(),
) -> ReferenceMirrorMotion:
    """Generate a sinusoidal tapped-mirror displacement in the non-FSM arm."""

    n_samples = max(2, int(round(config.duration_s * config.sample_rate_hz)))
    time_s = np.arange(n_samples, dtype=float) / config.sample_rate_hz
    displacement = config.amplitude_m * np.sin(
        2.0 * math.pi * config.frequency_hz * time_s + config.phase_rad
    )
    return reference_mirror_motion_from_displacement(
        time_s=time_s,
        displacement_m=displacement,
        horizontal_direction_angle_rad=config.horizontal_direction_angle_rad,
        vertical_direction_angle_rad=config.vertical_direction_angle_rad,
        opd_coupling=config.opd_coupling,
        positive_toward_combiner_shortens_opd=config.positive_toward_combiner_shortens_opd,
    )


def reference_mirror_motion_from_displacement(
    time_s: ArrayLike,
    displacement_m: ArrayLike,
    horizontal_direction_angle_rad: float = 0.0,
    vertical_direction_angle_rad: float = 0.0,
    opd_coupling: float = 1.0,
    positive_toward_combiner_shortens_opd: bool = True,
) -> ReferenceMirrorMotion:
    """Convert tapped-mirror displacement into OPD and combiner beam walk."""

    _require_nonnegative("opd_coupling", opd_coupling)
    time = _as_float_array("time_s", time_s)
    displacement = _as_float_array("displacement_m", displacement_m)
    if displacement.shape != time.shape:
        raise ValueError(
            f"displacement_m shape {displacement.shape} does not match time {time.shape}"
        )

    longitudinal = (
        displacement
        * math.cos(horizontal_direction_angle_rad)
        * math.cos(vertical_direction_angle_rad)
    )
    horizontal_shear = displacement * math.sin(horizontal_direction_angle_rad)
    vertical_shear = displacement * math.sin(vertical_direction_angle_rad)
    sign = -1.0 if positive_toward_combiner_shortens_opd else 1.0
    opd = sign * opd_coupling * longitudinal

    return ReferenceMirrorMotion(
        time_s=time,
        displacement_m=displacement,
        opd_m=opd,
        horizontal_shear_m=horizontal_shear,
        vertical_shear_m=vertical_shear,
    )


def simulate_quadrature(
    motion: FSMMotion,
    beam: BeamConfig = BeamConfig(),
    interferometer: InterferometerConfig = InterferometerConfig(),
    detector: DetectorConfig = DetectorConfig(),
    reference_mirror_motion: ReferenceMirrorMotion | None = None,
    polarization: PolarizationConfig | None = None,
    noise_seed: int | None = None,
) -> SimulationResult:
    """Simulate quadrature photodiode channels for a moving FSM."""

    distance_m = interferometer.distance_fsm_to_combiner_m
    k = beam.wave_number_rad_per_m
    beam_radius_m = float(beam.radius_at(distance_m))

    vertical_deflection = 2.0 * motion.vertical_steering_angle_rad
    horizontal_deflection = 2.0 * motion.horizontal_steering_angle_rad
    deflection_mag = np.hypot(vertical_deflection, horizontal_deflection)

    steering_vertical_shear = distance_m * np.tan(vertical_deflection)
    steering_horizontal_shear = distance_m * np.tan(horizontal_deflection)

    piston_induced_shear = _piston_induced_shear(
        motion.piston_m,
        interferometer.fsm_incidence_angle_rad,
        interferometer.include_piston_shear,
    )
    piston_vertical_shear, piston_horizontal_shear = _piston_shear_components(
        piston_induced_shear,
        interferometer.piston_shear_axis,
    )

    (
        reference_displacement,
        reference_opd,
        reference_vertical_shear,
        reference_horizontal_shear,
    ) = _reference_mirror_arrays(reference_mirror_motion, motion.time_s)

    vertical_shear = (
        steering_vertical_shear + piston_vertical_shear - reference_vertical_shear
    )
    horizontal_shear = (
        steering_horizontal_shear + piston_horizontal_shear - reference_horizontal_shear
    )
    shear_mag = np.hypot(vertical_shear, horizontal_shear)

    gaussian_overlap = np.exp(
        -(shear_mag**2) / (2.0 * beam_radius_m**2)
        - ((k * beam_radius_m * deflection_mag) ** 2) / 8.0
    )
    gaussian_overlap = np.clip(gaussian_overlap, 0.0, 1.0)

    mode_overlap_phase = 0.5 * k * (
        vertical_deflection * vertical_shear
        + horizontal_deflection * horizontal_shear
    )
    piston_induced_opd = _piston_induced_opd(
        motion.piston_m,
        interferometer.fsm_incidence_angle_rad,
        interferometer.include_piston_opd,
    )
    opd_m = (
        interferometer.static_opd_m
        + piston_induced_opd
        + reference_opd
        + 2.0
        * (
            interferometer.fsm_vertical_pivot_offset_m
            * motion.vertical_steering_angle_rad
            + interferometer.fsm_horizontal_pivot_offset_m
            * motion.horizontal_steering_angle_rad
        )
    )

    if interferometer.include_angular_path_length:
        clipped_deflection = np.minimum(deflection_mag, math.radians(85.0))
        opd_m = opd_m + distance_m * (1.0 / np.cos(clipped_deflection) - 1.0)

    phase = interferometer.nominal_phase_rad + k * opd_m + mode_overlap_phase
    visibility = np.clip(interferometer.baseline_visibility * gaussian_overlap, 0.0, None)

    if polarization is None:
        vertical_clean = detector.vertical_offset_w + detector.vertical_gain * (
            0.5
            * interferometer.input_power_w
            * (1.0 + detector.vertical_contrast * visibility * np.cos(phase))
        )
        horizontal_clean = detector.horizontal_offset_w + detector.horizontal_gain * (
            0.5
            * interferometer.input_power_w
            * (
                1.0
                + detector.horizontal_contrast
                * visibility
                * np.cos(phase + interferometer.quadrature_phase_rad)
            )
        )
    else:
        vertical_clean, horizontal_clean = _polarized_detector_signals(
            phase,
            visibility,
            interferometer.input_power_w,
            detector,
            polarization,
        )

    vertical, horizontal = _apply_detector_noise(
        vertical_clean,
        horizontal_clean,
        detector.read_noise_std_w,
        detector.relative_intensity_noise_std,
        noise_seed,
    )

    return SimulationResult(
        time_s=motion.time_s,
        vertical_w=vertical,
        horizontal_w=horizontal,
        phase_rad=phase,
        visibility=visibility,
        gaussian_overlap=gaussian_overlap,
        opd_m=opd_m,
        mirror_vertical_steering_angle_rad=motion.vertical_steering_angle_rad,
        mirror_horizontal_steering_angle_rad=motion.horizontal_steering_angle_rad,
        piston_m=motion.piston_m,
        beam_vertical_deflection_rad=vertical_deflection,
        beam_horizontal_deflection_rad=horizontal_deflection,
        steering_vertical_shear_m=steering_vertical_shear,
        steering_horizontal_shear_m=steering_horizontal_shear,
        piston_induced_shear_m=piston_induced_shear,
        piston_induced_opd_m=piston_induced_opd,
        reference_mirror_displacement_m=reference_displacement,
        reference_mirror_opd_m=reference_opd,
        reference_mirror_vertical_shear_m=reference_vertical_shear,
        reference_mirror_horizontal_shear_m=reference_horizontal_shear,
        vertical_shear_m=vertical_shear,
        horizontal_shear_m=horizontal_shear,
        beam_radius_at_combiner_m=beam_radius_m,
    )


def plot_quadrature(
    result: SimulationResult,
    ax=None,
    color_by: Literal["time", "phase", "visibility", "overlap"] | None = "time",
    point_size: float = 8.0,
    alpha: float = 0.85,
):
    """Plot horizontal channel intensity versus vertical channel intensity."""

    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 5.5))

    color_values, color_label = _color_values(result, color_by)
    if color_values is None:
        ax.scatter(result.horizontal_w, result.vertical_w, s=point_size, alpha=alpha)
    else:
        scatter = ax.scatter(
            result.horizontal_w,
            result.vertical_w,
            c=color_values,
            s=point_size,
            alpha=alpha,
            cmap="viridis",
        )
        ax.figure.colorbar(scatter, ax=ax, label=color_label)

    ax.set_xlabel("Horizontal polarization photodiode intensity")
    ax.set_ylabel("Vertical polarization photodiode intensity")
    ax.set_title("Quadrature channel plot")
    ax.grid(True, alpha=0.25)
    ax.set_aspect("equal", adjustable="box")
    return ax


def plot_beam_cross_section(
    result: SimulationResult,
    ax=None,
    color_by: Literal["time", "visibility", "overlap"] | None = "time",
    point_size: float = 8.0,
    alpha: float = 0.85,
    show_beam_radius: bool = True,
    zoom: Literal["path", "beam"] = "path",
):
    """Plot beam-center walk on a stationary cross-section at the combiner.

    The plotted coordinates are the relative beam-center displacement between
    the two arms at the recombining beamsplitter plane. Horizontal displacement
    stays in the optical-table plane; vertical displacement is out of the
    optical-table plane.
    """

    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 5.5))

    x_um = result.horizontal_shear_m * 1e6
    y_um = result.vertical_shear_m * 1e6
    color_values, color_label = _beam_cross_section_color_values(result, color_by)

    ax.axhline(0.0, color="0.35", linewidth=1.0, alpha=0.75)
    ax.axvline(0.0, color="0.35", linewidth=1.0, alpha=0.75)

    if show_beam_radius:
        radius_um = result.beam_radius_at_combiner_m * 1e6
        beam_circle = Circle(
            (0.0, 0.0),
            radius_um,
            fill=False,
            linestyle="--",
            linewidth=1.2,
            edgecolor="0.45",
            alpha=0.7,
            label="1/e^2 beam radius",
        )
        ax.add_patch(beam_circle)

    if color_values is None:
        ax.scatter(x_um, y_um, s=point_size, alpha=alpha)
    else:
        scatter = ax.scatter(
            x_um,
            y_um,
            c=color_values,
            s=point_size,
            alpha=alpha,
            cmap="viridis",
        )
        ax.figure.colorbar(scatter, ax=ax, label=color_label)

    ax.scatter([0.0], [0.0], marker="+", s=100, color="black", linewidths=1.5)
    ax.set_xlabel("Rel. horizontal displacement at combiner (um)")
    ax.set_ylabel("Rel. vertical displacement at combiner (um)")
    ax.set_title("Relative beam walk at stationary combiner cross-section")
    ax.grid(True, alpha=0.25)
    ax.set_aspect("equal", adjustable="box")
    _set_beam_cross_section_limits(ax, x_um, y_um, result.beam_radius_at_combiner_m, zoom)
    if show_beam_radius:
        ax.legend(loc="best")
    return ax


def estimate_opd_from_quadrature(
    result: SimulationResult,
    wavelength_m: float,
    reference_result: SimulationResult | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate unwrapped OPD from the quadrature plot angle.

    The horizontal/vertical channels are centered and scaled, then their angle
    around the quadrature ring is unwrapped. One full revolution corresponds to
    one wavelength of optical path difference. When `reference_result` is
    supplied, its ring center and scale are used as the normalization reference.
    """

    _require_positive("wavelength_m", wavelength_m)
    norm_source = reference_result if reference_result is not None else result
    horizontal_center, horizontal_scale = _center_and_scale(norm_source.horizontal_w)
    vertical_center, vertical_scale = _center_and_scale(norm_source.vertical_w)

    horizontal_norm = (result.horizontal_w - horizontal_center) / horizontal_scale
    vertical_norm = (result.vertical_w - vertical_center) / vertical_scale
    phase_rad = np.unwrap(np.arctan2(vertical_norm, horizontal_norm))
    opd_m = wavelength_m * phase_rad / (2.0 * math.pi)
    return phase_rad, opd_m


def animate_example_evolution(
    result: SimulationResult,
    beam: BeamConfig,
    gif_path,
    reference_result: SimulationResult | None = None,
    nominal_reference_result: SimulationResult | None = None,
    title: str = "FSM example",
    fps: int = 20,
    max_frames: int = 160,
    dpi: int = 120,
):
    """Create a GIF of beam walk, quadrature trace, and inferred OPD over time."""

    from pathlib import Path

    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    _require_positive("fps", fps)
    _require_positive("max_frames", max_frames)
    _require_positive("dpi", dpi)

    gif_path = Path(gif_path)
    gif_path.parent.mkdir(parents=True, exist_ok=True)

    frame_indices = _animation_frame_indices(result.time_s.size, int(max_frames))
    time_ms = result.time_s * 1e3
    x_um = result.horizontal_shear_m * 1e6
    y_um = result.vertical_shear_m * 1e6
    _, opd_m = estimate_opd_from_quadrature(
        result,
        beam.wavelength_m,
        reference_result=reference_result,
    )
    opd_nm = opd_m * 1e9

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    fig.suptitle(title)

    axes[0].axhline(0.0, color="0.35", linewidth=1.0, alpha=0.75)
    axes[0].axvline(0.0, color="0.35", linewidth=1.0, alpha=0.75)
    beam_path_line, = axes[0].plot([], [], color="tab:blue", linewidth=1.6)
    beam_path_point, = axes[0].plot([], [], "o", color="tab:blue", markersize=5)
    axes[0].set_xlabel("Rel. horizontal displacement (um)")
    axes[0].set_ylabel("Rel. vertical displacement (um)")
    axes[0].set_title("Beam walk")
    axes[0].grid(True, alpha=0.25)
    axes[0].set_aspect("equal", adjustable="box")
    _set_xy_limits_from_data(axes[0], x_um, y_um)

    if nominal_reference_result is not None:
        axes[1].plot(
            nominal_reference_result.horizontal_w,
            nominal_reference_result.vertical_w,
            color="0.35",
            linewidth=1.5,
            alpha=0.75,
            label="nominal zero-error ring",
        )
    if reference_result is not None:
        axes[1].scatter(
            reference_result.horizontal_w,
            reference_result.vertical_w,
            s=12,
            color="0.72",
            alpha=0.45,
            label="current waveplate tap ring",
        )
    if reference_result is not None or nominal_reference_result is not None:
        axes[1].legend(loc="best")
    quad_line, = axes[1].plot([], [], color="tab:blue", linewidth=1.6)
    quad_point, = axes[1].plot([], [], "o", color="tab:blue", markersize=5)
    axes[1].set_xlabel("Horizontal polarization intensity")
    axes[1].set_ylabel("Vertical polarization intensity")
    axes[1].set_title("Quadrature")
    axes[1].grid(True, alpha=0.25)
    axes[1].set_aspect("equal", adjustable="box")
    reference_x = []
    reference_y = []
    if reference_result is not None:
        reference_x.append(reference_result.horizontal_w)
        reference_y.append(reference_result.vertical_w)
    if nominal_reference_result is not None:
        reference_x.append(nominal_reference_result.horizontal_w)
        reference_y.append(nominal_reference_result.vertical_w)
    _set_xy_limits_from_data(
        axes[1],
        result.horizontal_w,
        result.vertical_w,
        None if not reference_x else np.concatenate(reference_x),
        None if not reference_y else np.concatenate(reference_y),
    )

    opd_line, = axes[2].plot([], [], color="tab:blue", linewidth=1.6)
    opd_point, = axes[2].plot([], [], "o", color="tab:blue", markersize=5)
    axes[2].set_xlabel("Time (ms)")
    axes[2].set_ylabel("Unwrapped quadrature OPD (nm)")
    axes[2].set_title("OPD")
    axes[2].grid(True, alpha=0.25)
    _set_line_limits(axes[2], time_ms, opd_nm)

    status = axes[2].text(
        0.02,
        0.96,
        "",
        transform=axes[2].transAxes,
        va="top",
        ha="left",
    )

    def update(frame_idx: int):
        idx = int(frame_indices[frame_idx])
        sl = slice(0, idx + 1)
        beam_path_line.set_data(x_um[sl], y_um[sl])
        beam_path_point.set_data([x_um[idx]], [y_um[idx]])
        quad_line.set_data(result.horizontal_w[sl], result.vertical_w[sl])
        quad_point.set_data([result.horizontal_w[idx]], [result.vertical_w[idx]])
        opd_line.set_data(time_ms[sl], opd_nm[sl])
        opd_point.set_data([time_ms[idx]], [opd_nm[idx]])
        status.set_text(f"t = {time_ms[idx]:.2f} ms")
        return (
            beam_path_line,
            beam_path_point,
            quad_line,
            quad_point,
            opd_line,
            opd_point,
            status,
        )

    animation = FuncAnimation(
        fig,
        update,
        frames=len(frame_indices),
        interval=1000 / fps,
        blit=True,
    )
    fig.tight_layout()
    animation.save(gif_path, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    return gif_path


def plot_diagnostics(result: SimulationResult, axes=None):
    """Plot motion, OPD, visibility, and channel time traces."""

    import matplotlib.pyplot as plt

    if axes is None:
        _, axes = plt.subplots(4, 1, figsize=(9.0, 8.5), sharex=True)

    axes = np.asarray(axes)
    time_ms = result.time_s * 1e3
    axes[0].plot(
        time_ms,
        result.mirror_vertical_steering_angle_rad * 1e6,
        label="vertical steering",
    )
    axes[0].plot(
        time_ms,
        result.mirror_horizontal_steering_angle_rad * 1e6,
        label="horizontal steering",
    )
    axes[0].set_ylabel("Mirror angle (urad)")
    axes[0].legend(loc="best")

    axes[1].plot(time_ms, result.opd_m * 1e9)
    axes[1].set_ylabel("OPD (nm)")

    axes[2].plot(time_ms, result.visibility)
    axes[2].set_ylabel("Visibility")
    axes[2].set_ylim(bottom=0.0)

    axes[3].plot(time_ms, result.vertical_w, label="vertical")
    axes[3].plot(time_ms, result.horizontal_w, label="horizontal")
    axes[3].set_ylabel("Intensity")
    axes[3].set_xlabel("Time (ms)")
    axes[3].legend(loc="best")

    for ax in axes:
        ax.grid(True, alpha=0.25)

    return axes


def _noise_with_correlation(
    rng: np.random.Generator,
    n_samples: int,
    rms: float,
    dt_s: float,
    correlation_time_s: float,
) -> np.ndarray:
    if rms == 0:
        return np.zeros(n_samples, dtype=float)

    white = rng.normal(loc=0.0, scale=rms, size=n_samples)
    if correlation_time_s <= 0:
        return white

    alpha = math.exp(-dt_s / correlation_time_s)
    innovation_scale = math.sqrt(max(0.0, 1.0 - alpha**2))
    filtered = np.empty_like(white)
    filtered[0] = white[0]
    for idx in range(1, n_samples):
        filtered[idx] = alpha * filtered[idx - 1] + innovation_scale * white[idx]
    return filtered


def _piston_induced_opd(
    piston_m: np.ndarray,
    incidence_angle_rad: float,
    include_piston_opd: bool,
) -> np.ndarray:
    if not include_piston_opd:
        return np.zeros_like(piston_m)
    return 2.0 * piston_m * math.cos(incidence_angle_rad)


def _piston_induced_shear(
    piston_m: np.ndarray,
    incidence_angle_rad: float,
    include_piston_shear: bool,
) -> np.ndarray:
    if not include_piston_shear:
        return np.zeros_like(piston_m)
    return 2.0 * piston_m * math.sin(incidence_angle_rad)


def _piston_shear_components(
    piston_induced_shear_m: np.ndarray,
    axis: Literal["horizontal", "vertical"],
) -> tuple[np.ndarray, np.ndarray]:
    if axis == "horizontal":
        return np.zeros_like(piston_induced_shear_m), piston_induced_shear_m
    return piston_induced_shear_m, np.zeros_like(piston_induced_shear_m)


def _reference_mirror_arrays(
    reference_mirror_motion: ReferenceMirrorMotion | None,
    time_s: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if reference_mirror_motion is None:
        zeros = np.zeros_like(time_s)
        return zeros, zeros, zeros, zeros

    if reference_mirror_motion.time_s.shape != time_s.shape:
        raise ValueError(
            "reference_mirror_motion must have the same sample count as the FSM motion"
        )
    if not np.allclose(reference_mirror_motion.time_s, time_s):
        raise ValueError("reference_mirror_motion time_s must match FSM motion time_s")

    return (
        reference_mirror_motion.displacement_m,
        reference_mirror_motion.opd_m,
        reference_mirror_motion.vertical_shear_m,
        reference_mirror_motion.horizontal_shear_m,
    )


def _linear_polarization_jones(angle_rad: float) -> np.ndarray:
    """Unit Jones vector in the vertical/horizontal basis."""
    return np.array([math.cos(angle_rad), math.sin(angle_rad)], dtype=complex)


def _retarder_jones(fast_axis_angle_rad: float, retardance_rad: float) -> np.ndarray:
    """Jones matrix for an ideal linear retarder.

    The fast axis has zero added phase. The orthogonal slow axis has phase
    `retardance_rad`.
    """
    fast_axis = np.array(
        [math.cos(fast_axis_angle_rad), math.sin(fast_axis_angle_rad)],
        dtype=complex,
    )
    slow_axis = np.array(
        [-math.sin(fast_axis_angle_rad), math.cos(fast_axis_angle_rad)],
        dtype=complex,
    )
    return np.outer(fast_axis, fast_axis.conjugate()) + np.exp(
        1j * retardance_rad
    ) * np.outer(slow_axis, slow_axis.conjugate())


def _polarized_detector_signals(
    phase_rad: np.ndarray,
    visibility: np.ndarray,
    input_power_w: float,
    detector: DetectorConfig,
    polarization: PolarizationConfig,
) -> tuple[np.ndarray, np.ndarray]:
    input_jones = _linear_polarization_jones(polarization.input_linear_angle_rad)
    reference_jones = (
        _retarder_jones(
            polarization.reference_hwp_fast_axis_angle_rad,
            polarization.reference_hwp_retardance_rad,
        )
        @ input_jones
    )
    fsm_jones = (
        _retarder_jones(
            polarization.fsm_qwp_fast_axis_angle_rad,
            polarization.fsm_qwp_retardance_rad,
        )
        @ input_jones
    )

    reference_power = input_power_w * polarization.reference_arm_power_fraction
    fsm_power = input_power_w * polarization.fsm_arm_power_fraction
    coherent_scale = 2.0 * math.sqrt(reference_power * fsm_power)

    def channel_power(index: int, contrast: float) -> np.ndarray:
        reference_field = reference_jones[index]
        fsm_field = fsm_jones[index]
        dc_power = (
            reference_power * abs(reference_field) ** 2
            + fsm_power * abs(fsm_field) ** 2
        )
        interference = coherent_scale * contrast * visibility * np.real(
            reference_field * fsm_field.conjugate() * np.exp(-1j * phase_rad)
        )
        return dc_power + interference

    vertical_power = channel_power(0, detector.vertical_contrast)
    horizontal_power = channel_power(1, detector.horizontal_contrast)
    vertical = detector.vertical_offset_w + detector.vertical_gain * vertical_power
    horizontal = detector.horizontal_offset_w + detector.horizontal_gain * horizontal_power
    return vertical, horizontal


def _apply_detector_noise(
    vertical_clean: np.ndarray,
    horizontal_clean: np.ndarray,
    read_noise_std_w: float,
    relative_intensity_noise_std: float,
    seed: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    if read_noise_std_w == 0 and relative_intensity_noise_std == 0:
        return vertical_clean, horizontal_clean

    rng = np.random.default_rng(seed)
    vertical = vertical_clean.copy()
    horizontal = horizontal_clean.copy()

    if relative_intensity_noise_std:
        vertical *= 1.0 + rng.normal(0.0, relative_intensity_noise_std, vertical.shape)
        horizontal *= 1.0 + rng.normal(0.0, relative_intensity_noise_std, horizontal.shape)
    if read_noise_std_w:
        vertical += rng.normal(0.0, read_noise_std_w, vertical.shape)
        horizontal += rng.normal(0.0, read_noise_std_w, horizontal.shape)

    return vertical, horizontal


def _color_values(
    result: SimulationResult,
    color_by: Literal["time", "phase", "visibility", "overlap"] | None,
) -> tuple[np.ndarray | None, str | None]:
    if color_by is None:
        return None, None
    if color_by == "time":
        return result.time_s * 1e3, "Time (ms)"
    if color_by == "phase":
        return result.wrapped_phase_rad, "Wrapped phase (rad)"
    if color_by == "visibility":
        return result.visibility, "Visibility"
    if color_by == "overlap":
        return result.gaussian_overlap, "Gaussian overlap"
    raise ValueError(f"Unsupported color_by value: {color_by!r}")


def _center_and_scale(values: np.ndarray) -> tuple[float, float]:
    center = 0.5 * (float(np.nanmin(values)) + float(np.nanmax(values)))
    scale = 0.5 * (float(np.nanmax(values)) - float(np.nanmin(values)))
    if scale <= 0.0:
        scale = 1.0
    return center, scale


def _beam_cross_section_color_values(
    result: SimulationResult,
    color_by: Literal["time", "visibility", "overlap"] | None,
) -> tuple[np.ndarray | None, str | None]:
    if color_by is None:
        return None, None
    if color_by == "time":
        return result.time_s * 1e3, "Time (ms)"
    if color_by == "visibility":
        return result.visibility, "Visibility"
    if color_by == "overlap":
        return result.gaussian_overlap, "Gaussian overlap"
    raise ValueError(f"Unsupported color_by value: {color_by!r}")


def _set_beam_cross_section_limits(
    ax,
    x_um: np.ndarray,
    y_um: np.ndarray,
    beam_radius_m: float,
    zoom: Literal["path", "beam"],
) -> None:
    if zoom == "beam":
        limit_um = beam_radius_m * 1e6 * 1.15
    elif zoom == "path":
        max_abs_um = max(float(np.max(np.abs(x_um))), float(np.max(np.abs(y_um))), 1e-6)
        limit_um = max_abs_um * 1.25
    else:
        raise ValueError("zoom must be 'path' or 'beam'")

    ax.set_xlim(-limit_um, limit_um)
    ax.set_ylim(-limit_um, limit_um)


def _animation_frame_indices(n_samples: int, max_frames: int) -> np.ndarray:
    if n_samples <= max_frames:
        return np.arange(n_samples, dtype=int)
    return np.unique(np.linspace(0, n_samples - 1, max_frames, dtype=int))


def _set_xy_limits_from_data(ax, x: np.ndarray, y: np.ndarray, x_ref=None, y_ref=None) -> None:
    if x_ref is not None:
        x = np.concatenate([np.asarray(x), np.asarray(x_ref)])
    if y_ref is not None:
        y = np.concatenate([np.asarray(y), np.asarray(y_ref)])
    x_min, x_max = float(np.nanmin(x)), float(np.nanmax(x))
    y_min, y_max = float(np.nanmin(y)), float(np.nanmax(y))
    x_pad = max(0.05 * (x_max - x_min), 1e-9)
    y_pad = max(0.05 * (y_max - y_min), 1e-9)
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)


def _set_line_limits(ax, x: np.ndarray, y: np.ndarray) -> None:
    x_min, x_max = float(np.nanmin(x)), float(np.nanmax(x))
    y_min, y_max = float(np.nanmin(y)), float(np.nanmax(y))
    x_pad = max(0.02 * (x_max - x_min), 1e-9)
    y_pad = max(0.08 * (y_max - y_min), 1e-9)
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
