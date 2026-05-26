"""Reusable helpers for the Etalon investigation notebook.

This module pulls the exploratory logic out of the exported HTML notebook so
the same analysis and plots can be rerun on any compatible data file.
Expected dataframe columns are usually I1, I2, I3, and I4.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_STYLE = {
    "figure.figsize": (12, 5),
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
}


@dataclass
class FlatRegionResult:
    """Detected flat regions and their supporting phase data."""

    flat_dfs: list[pd.DataFrame]
    flat_regions: list[tuple[int, int]]
    unwrapped_angles: np.ndarray
    phase_smooth: np.ndarray
    slope: np.ndarray


@dataclass
class PhaseJumpResult:
    """Phase jump measured between flat plateaus in one piezo dataset."""

    path: Path
    start_percent: float
    end_percent: float
    x_percent: float
    phase_start: float
    phase_end: float
    phase_jump: float
    flat_regions: list[tuple[int, int]]
    flat_phase_values: list[float]


def set_plot_style(style: Mapping[str, object] | None = None) -> None:
    """Apply a compact plotting style for investigation figures."""

    plt.rcParams.update(DEFAULT_STYLE)
    if style:
        plt.rcParams.update(style)


def load_data(path: str | Path, **read_csv_kwargs) -> pd.DataFrame:
    """Load an Etalon CSV file into a dataframe."""

    return pd.read_csv(Path(path), **read_csv_kwargs)


def require_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    """Raise a helpful error if expected signal columns are missing."""

    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise KeyError(f"Missing required column(s): {', '.join(missing)}")


def slice_series(series: pd.Series, start: int | None = None, end: int | None = None, step: int = 1) -> pd.Series:
    """Return a positional slice while preserving the original index."""

    return series.iloc[slice(start, end, step)]


def compute_unwrapped_phase(
    df: pd.DataFrame,
    i1_col: str = "I1",
    i2_col: str = "I2",
    start: int | None = None,
    end: int | None = None,
) -> np.ndarray:
    """Compute unwrapped quadrature phase from two intensity columns."""

    require_columns(df, [i1_col, i2_col])
    i1 = slice_series(df[i1_col], start, end).to_numpy()
    i2 = slice_series(df[i2_col], start, end).to_numpy()
    i1_centered = i1 - np.mean(i1)
    i2_centered = i2 - np.mean(i2)
    return np.unwrap(np.arctan2(i2_centered, i1_centered))


def phase_summary(unwrapped_angles: Sequence[float]) -> dict[str, float]:
    """Summarize total phase change and revolutions."""

    angles = np.asarray(unwrapped_angles)
    total_phase_change = float(angles[-1] - angles[0])
    revolutions = float(total_phase_change / (2 * np.pi))
    return {
        "total_phase_change": total_phase_change,
        "revolutions": revolutions,
        "absolute_revolutions": abs(revolutions),
    }


def find_signal_midpoint(
    df: pd.DataFrame,
    start_idx: float,
    threshold: float = 0.02,
    signal_col: str = "I3",
    min_fall_offset: int = 10,
) -> tuple[float | None, int | None]:
    """Find the midpoint between a rising and falling threshold crossing."""

    require_columns(df, [signal_col])
    start_idx = int(round(start_idx))
    search_after_start = df[signal_col].loc[start_idx:]
    if not (search_after_start >= threshold).any():
        return None, None

    rising_edge_idx = int((search_after_start >= threshold).idxmax())
    search_for_fall = df[signal_col].loc[rising_edge_idx:].iloc[min_fall_offset:]
    if search_for_fall.empty or not (search_for_fall < threshold).any():
        return None, None

    falling_edge_idx = int((search_for_fall < threshold).idxmax())
    midpoint = (rising_edge_idx + falling_edge_idx) / 2
    return midpoint, falling_edge_idx


def find_all_peak_centers(
    df: pd.DataFrame,
    initial_start: float,
    jump_distance: int = 100,
    threshold: float = 0.02,
    signal_col: str = "I3",
) -> list[float]:
    """Repeatedly find signal-center points after an initial start index."""

    centers: list[float] = []
    current_start = int(round(initial_start))
    max_limit = int(df.index[-1])

    while current_start < max_limit:
        center, edge = find_signal_midpoint(
            df,
            current_start,
            threshold=threshold,
            signal_col=signal_col,
        )
        if center is None or edge is None:
            break
        if centers and center <= centers[-1]:
            break

        centers.append(center)
        current_start = edge + jump_distance

    return centers


def find_voltage_peak(
    df: pd.DataFrame,
    column_name: str,
    start_range: int,
    end_range: int,
    width: int = 300,
    gap: int = 20,
    return_details: bool = False,
) -> float | dict[str, float]:
    """Refine a peak by intersecting linear fits on the left and right slopes."""

    require_columns(df, [column_name])
    search_range = df.loc[start_range:end_range]
    if search_range.empty:
        raise ValueError("Search range is empty.")

    initial_guess_idx = int(search_range[column_name].idxmax())
    left_side = df.loc[initial_guess_idx - width : initial_guess_idx - gap]
    right_side = df.loc[initial_guess_idx + gap : initial_guess_idx + width]
    if left_side.empty or right_side.empty:
        raise ValueError("Peak-fit windows are empty. Try reducing width or gap.")

    m1, b1 = np.polyfit(left_side.index, left_side[column_name], 1)
    m2, b2 = np.polyfit(right_side.index, right_side[column_name], 1)
    refined_peak_x = float((b2 - b1) / (m1 - m2))
    refined_peak_y = float(m1 * refined_peak_x + b1)

    if return_details:
        return {
            "peak_x": refined_peak_x,
            "peak_y": refined_peak_y,
            "initial_guess_idx": float(initial_guess_idx),
            "left_slope": float(m1),
            "left_intercept": float(b1),
            "right_slope": float(m2),
            "right_intercept": float(b2),
        }
    return refined_peak_x


def detect_flat_regions(
    df: pd.DataFrame,
    i1_col: str = "I1",
    i2_col: str = "I2",
    window: int = 2000,
    slope_threshold: float = 2e-3,
    min_length: int = 50000,
    buffer: int = 0,
    plot: bool = False,
    ax=None,
) -> FlatRegionResult:
    """Detect flat/plateau regions in the unwrapped phase signal."""

    unwrapped_angles = compute_unwrapped_phase(df, i1_col=i1_col, i2_col=i2_col)
    phase_smooth = (
        pd.Series(unwrapped_angles)
        .rolling(window=window, center=True)
        .mean()
        .bfill()
        .ffill()
        .to_numpy()
    )
    slope = np.gradient(phase_smooth)
    flat_mask = np.abs(slope) < slope_threshold

    raw_regions: list[tuple[int, int]] = []
    in_region = False
    start = 0
    for idx, is_flat in enumerate(flat_mask):
        if is_flat and not in_region:
            start = idx
            in_region = True
        elif not is_flat and in_region:
            raw_regions.append((start, idx))
            in_region = False
    if in_region:
        raw_regions.append((start, len(flat_mask)))

    flat_regions = [
        (start + buffer, end - buffer)
        for start, end in raw_regions
        if end - start >= min_length and end - buffer > start + buffer
    ]
    flat_dfs = [df.iloc[start:end].reset_index(drop=True) for start, end in flat_regions]

    result = FlatRegionResult(
        flat_dfs=flat_dfs,
        flat_regions=flat_regions,
        unwrapped_angles=unwrapped_angles,
        phase_smooth=phase_smooth,
        slope=slope,
    )
    if plot:
        plot_flat_regions(result, ax=ax)
    return result


def find_voltage_peaks(
    flat_dfs: Sequence[pd.DataFrame],
    column_name: str = "I4",
    start_range: int = 0,
    end_range: int = 50000,
    width: int = 10000,
) -> list[float]:
    """Find a refined voltage peak in each flat-region dataframe."""

    return [
        float(find_voltage_peak(df, column_name, start_range, end_range, width=width))
        for df in flat_dfs
    ]


def find_peak_centers_by_region(
    flat_dfs: Sequence[pd.DataFrame],
    voltage_peaks: Sequence[float],
    jump_distance: int = 100,
    threshold: float = 0.02,
    signal_col: str = "I3",
) -> list[list[float]]:
    """Find all I3 peak centers for each flat dataframe."""

    return [
        find_all_peak_centers(
            df,
            peak,
            jump_distance=jump_distance,
            threshold=threshold,
            signal_col=signal_col,
        )
        for df, peak in zip(flat_dfs, voltage_peaks)
    ]


def distances_from_voltage(
    peak_centers: Sequence[Sequence[float]],
    voltage_peaks: Sequence[float],
) -> list[np.ndarray]:
    """Return peak-center distances from each corresponding voltage peak."""

    return [
        np.asarray(centers, dtype=float) - float(voltage_peak)
        for centers, voltage_peak in zip(peak_centers, voltage_peaks)
    ]


def plot_signals(
    df: pd.DataFrame,
    columns: Sequence[str] = ("I1", "I2"),
    start: int | None = None,
    end: int | None = None,
    title: str | None = None,
    ax=None,
):
    """Plot one or more signal columns over index."""

    require_columns(df, columns)
    ax = ax or plt.subplots(figsize=(12, 5))[1]
    for column in columns:
        sliced = slice_series(df[column], start, end)
        ax.plot(sliced.index, sliced.to_numpy(), label=column, linewidth=1.2)
    ax.set(title=title or "Signal Traces", xlabel="Index", ylabel="Value")
    ax.legend()
    return ax


def plot_i1_i2_scatter(
    df: pd.DataFrame,
    i1_col: str = "I1",
    i2_col: str = "I2",
    start: int | None = None,
    end: int | None = None,
    step: int = 100,
    cmap: str = "PuBu",
    ax=None,
):
    """Plot I1 vs I2 with color encoding sample order."""

    require_columns(df, [i1_col, i2_col])
    ax = ax or plt.subplots(figsize=(8, 6))[1]
    i1 = slice_series(df[i1_col], start, end, step)
    i2 = slice_series(df[i2_col], start, end, step)
    t = np.arange(len(i1))
    sc = ax.scatter(i1, i2, c=t, cmap=cmap, s=12)
    ax.set(title=f"{i1_col} vs {i2_col} Over Time", xlabel=i1_col, ylabel=i2_col)
    ax.axis("equal")
    ax.figure.colorbar(sc, ax=ax, label="Time Index")
    return ax


def plot_unwrapped_phase(
    df: pd.DataFrame,
    i1_col: str = "I1",
    i2_col: str = "I2",
    start: int | None = None,
    end: int | None = None,
    ax=None,
):
    """Plot unwrapped phase progression."""

    angles = compute_unwrapped_phase(df, i1_col=i1_col, i2_col=i2_col, start=start, end=end)
    ax = ax or plt.subplots(figsize=(12, 4))[1]
    ax.plot(np.arange(len(angles)), angles, color="darkblue", linewidth=1)
    summary = phase_summary(angles)
    ax.set(
        title=f"Unwrapped Phase ({summary['absolute_revolutions']:.2f} revolutions)",
        xlabel="Time Index",
        ylabel="Cumulative Angle (radians)",
    )
    return ax


def plot_flat_regions(result: FlatRegionResult, ax=None):
    """Plot unwrapped phase with detected flat regions highlighted."""

    ax = ax or plt.subplots(figsize=(12, 5))[1]
    ax.plot(result.unwrapped_angles, color="darkblue", label="Unwrapped phase", linewidth=1)
    for idx, (start, end) in enumerate(result.flat_regions):
        label = "Flat region" if idx == 0 else None
        ax.axvspan(start, end, alpha=0.25, color="orange", label=label)
    ax.set(
        title="Automatically Detected Flat Regions",
        xlabel="Time Index",
        ylabel="Cumulative Angle (radians)",
    )
    ax.legend()
    return ax


def plot_voltage_peak(
    df: pd.DataFrame,
    peak_x: float,
    column_name: str = "I4",
    window: int = 100,
    title: str | None = None,
    ax=None,
):
    """Plot a zoomed signal window around a refined voltage peak."""

    require_columns(df, [column_name])
    peak_idx = int(round(peak_x))
    start = max(0, peak_idx - window)
    end = min(int(df.index[-1]), peak_idx + window)
    ax = ax or plt.subplots(figsize=(12, 5))[1]
    series = df[column_name].loc[start:end]
    ax.plot(series.index, series.to_numpy(), label=column_name, linewidth=1.2)
    ax.scatter([peak_idx], [df[column_name].loc[peak_idx]], c="red", s=50, zorder=5, label="Peak")
    ax.set(title=title or f"{column_name} Voltage Peak", xlabel="Index", ylabel=column_name)
    ax.legend()
    return ax


def plot_peak_centers(
    df: pd.DataFrame,
    peak_centers: Sequence[float],
    signal_col: str = "I3",
    xlim: tuple[float, float] | None = None,
    title: str | None = None,
    ax=None,
):
    """Plot a signal and mark detected peak centers."""

    require_columns(df, [signal_col])
    rounded_peaks = [int(round(point)) for point in peak_centers]
    peak_heights = df[signal_col].loc[rounded_peaks]
    ax = ax or plt.subplots(figsize=(12, 5))[1]
    ax.plot(df.index, df[signal_col].to_numpy(), label=signal_col, linewidth=1)
    ax.scatter(rounded_peaks, peak_heights, color="red", label="Detected centers", zorder=5)
    if xlim:
        ax.set_xlim(*xlim)
    ax.set(title=title or f"{signal_col} Detected Centers", xlabel="Index", ylabel=signal_col)
    ax.legend()
    return ax


def plot_interpeak_distances(
    distance_arrays: Sequence[Sequence[float]],
    labels: Sequence[str] | None = None,
    first_n: int | None = 8,
    ax=None,
):
    """Plot consecutive interpeak distances vs distance from voltage peak."""

    labels = labels or [str(i) for i in range(len(distance_arrays))]
    ax = ax or plt.subplots(figsize=(12, 6))[1]
    for label, distances in zip(labels, distance_arrays):
        data = np.asarray(distances, dtype=float)
        if first_n is not None:
            data = data[:first_n]
        if len(data) < 2:
            continue
        interpeak_dist = np.diff(data)
        ax.plot(data[:-1], interpeak_dist, marker="o", linestyle="-", label=f"Dist {label}")
    ax.set(
        title="Interpeak Distances vs. Distance from Volt",
        xlabel="Distance from Volt (Index/Time)",
        ylabel="Interpeak Distance",
    )
    ax.legend()
    return ax


def parse_piezo_range(path: str | Path) -> tuple[float, float]:
    """Parse the trailing start/end piezo percentages from a filename."""

    stem = Path(path).stem
    match = re.search(r"_(\d+(?:\.\d+)?)_(\d+(?:\.\d+)?)$", stem)
    if not match:
        raise ValueError(f"Could not parse piezo range from filename: {Path(path).name}")
    return float(match.group(1)), float(match.group(2))


def discover_piezo_files(
    data_dir: str | Path = "Etalon Data",
    pattern: str = "piezo_test_5_18_*.xls",
) -> list[Path]:
    """Find piezo files and sort them by the parsed percentage range."""

    paths = list(Path(data_dir).glob(pattern))
    return sorted(paths, key=lambda path: parse_piezo_range(path))


def measure_phase_jump_between_flat_regions(
    df: pd.DataFrame,
    flat_kwargs: Mapping[str, object] | None = None,
    phase_stat: str = "median",
) -> tuple[float, float, float, FlatRegionResult, list[float]]:
    """Measure the cumulative-angle jump between the first and last flat regions."""

    flat_kwargs = {
        "window": 2000,
        "slope_threshold": 5e-3,
        "min_length": 50000,
        "buffer": 10000,
        "plot": False,
        **dict(flat_kwargs or {}),
    }
    flat_result = detect_flat_regions(df, **flat_kwargs)
    if len(flat_result.flat_regions) < 2:
        raise ValueError(
            "Expected at least two flat regions. Try adjusting slope_threshold, min_length, or buffer."
        )

    stat_func = {
        "median": np.median,
        "mean": np.mean,
    }.get(phase_stat)
    if stat_func is None:
        raise ValueError("phase_stat must be 'median' or 'mean'.")

    flat_phase_values = [
        float(stat_func(flat_result.unwrapped_angles[start:end]))
        for start, end in flat_result.flat_regions
    ]
    phase_start = flat_phase_values[0]
    phase_end = flat_phase_values[-1]
    return phase_start, phase_end, phase_end - phase_start, flat_result, flat_phase_values


def analyze_piezo_phase_jumps(
    data_dir: str | Path = "Etalon Data",
    pattern: str = "piezo_test_5_18_*.xls",
    flat_kwargs: Mapping[str, object] | None = None,
    phase_stat: str = "median",
    x_value: str = "end",
    read_csv_kwargs: Mapping[str, object] | None = None,
) -> pd.DataFrame:
    """Analyze all matching piezo datasets and return one phase-jump row per file."""

    rows: list[dict[str, object]] = []
    for path in discover_piezo_files(data_dir=data_dir, pattern=pattern):
        start_percent, end_percent = parse_piezo_range(path)
        if x_value == "start":
            x_percent = start_percent
        elif x_value == "mid":
            x_percent = (start_percent + end_percent) / 2
        elif x_value == "end":
            x_percent = end_percent
        else:
            raise ValueError("x_value must be 'start', 'mid', or 'end'.")

        df = load_data(path, **dict(read_csv_kwargs or {}))
        phase_start, phase_end, phase_jump, flat_result, flat_phase_values = (
            measure_phase_jump_between_flat_regions(
                df,
                flat_kwargs=flat_kwargs,
                phase_stat=phase_stat,
            )
        )
        rows.append(
            {
                "file": path.name,
                "path": path,
                "start_percent": start_percent,
                "end_percent": end_percent,
                "x_percent": x_percent,
                "phase_start": phase_start,
                "phase_end": phase_end,
                "phase_jump": phase_jump,
                "abs_phase_jump": abs(phase_jump),
                "flat_region_count": len(flat_result.flat_regions),
                "flat_regions": flat_result.flat_regions,
                "flat_phase_values": flat_phase_values,
            }
        )

    if not rows:
        raise FileNotFoundError(f"No files matched {Path(data_dir) / pattern}")
    return pd.DataFrame(rows).sort_values("x_percent").reset_index(drop=True)


def plot_piezo_phase_jumps(
    phase_jump_df: pd.DataFrame,
    y_col: str = "phase_jump",
    ax=None,
):
    """Plot phase jump versus piezo percentage."""

    ax = ax or plt.subplots(figsize=(12, 5))[1]
    ax.plot(
        phase_jump_df["x_percent"],
        phase_jump_df[y_col],
        marker="o",
        linewidth=1.5,
    )
    ax.set(
        title="Cumulative Angle Jump vs. Piezo Percentage",
        xlabel="Piezo percentage",
        ylabel="Cumulative angle jump (radians)",
    )
    return ax


def interpolate_local_fsr(
    distance_array: Sequence[float],
    distance_from_volt: float,
    first_n: int | None = None,
) -> float:
    """Interpolate local FSR from peak-to-peak spacing at a distance from volt."""

    distances = np.asarray(distance_array, dtype=float)
    if first_n is not None:
        distances = distances[:first_n]
    if len(distances) < 2:
        raise ValueError("Need at least two transmission peaks to estimate FSR.")

    x_values = distances[:-1]
    fsr_values = np.diff(distances)
    order = np.argsort(x_values)
    x_values = x_values[order]
    fsr_values = fsr_values[order]

    return float(np.interp(distance_from_volt, x_values, fsr_values))


def measure_transmission_shift_fraction(
    df: pd.DataFrame,
    reference_peak_number: int = 6,
    flat_kwargs: Mapping[str, object] | None = None,
    fsr_source: str = "mean",
    first_n_for_fsr: int | None = None,
) -> dict[str, object]:
    """Measure a transmission shift as a fraction of the local FSR.

    The reference is the Nth I3 transmission peak in the first flat region.
    The shifted peak is the first I3 peak in the second flat region whose
    distance from its I4 voltage peak is directly left of that reference
    distance. The returned fraction can be multiplied by a known FSR, such as
    10 GHz, to convert it to frequency shift.
    """

    if reference_peak_number < 1:
        raise ValueError("reference_peak_number is 1-based and must be >= 1.")

    results = run_initial_workflow(df, flat_kwargs=flat_kwargs)
    distances = results["distances_from_voltage"]
    if len(distances) < 2:
        raise ValueError("Expected at least two flat regions.")

    first_distances = np.asarray(distances[0], dtype=float)
    second_distances = np.asarray(distances[1], dtype=float)
    reference_idx = reference_peak_number - 1
    if len(first_distances) <= reference_idx:
        raise ValueError(f"First flat region has fewer than {reference_peak_number} peaks.")

    reference_distance = float(first_distances[reference_idx])
    left_candidates = second_distances[second_distances < reference_distance]
    if len(left_candidates) == 0:
        raise ValueError("No second-region transmission peak was left of the reference distance.")

    shifted_distance = float(left_candidates.max())
    distance_difference = reference_distance - shifted_distance
    mean_distance = (reference_distance + shifted_distance) / 2

    if fsr_source == "first":
        local_fsr = interpolate_local_fsr(first_distances, mean_distance, first_n=first_n_for_fsr)
    elif fsr_source == "second":
        local_fsr = interpolate_local_fsr(second_distances, mean_distance, first_n=first_n_for_fsr)
    elif fsr_source == "mean":
        fsr_first = interpolate_local_fsr(first_distances, mean_distance, first_n=first_n_for_fsr)
        fsr_second = interpolate_local_fsr(second_distances, mean_distance, first_n=first_n_for_fsr)
        local_fsr = float(np.mean([fsr_first, fsr_second]))
    else:
        raise ValueError("fsr_source must be 'first', 'second', or 'mean'.")

    fsr_fraction = distance_difference / local_fsr
    return {
        "reference_distance": reference_distance,
        "shifted_distance": shifted_distance,
        "distance_difference": distance_difference,
        "mean_distance": mean_distance,
        "local_fsr": local_fsr,
        "fsr_fraction": fsr_fraction,
        "flat_regions": results["flat_regions"],
        "voltage_peaks": results["voltage_peaks"],
        "peak_centers": results["peak_centers"],
        "distances_from_voltage": distances,
    }


def analyze_transmission_shift_fractions(
    data_dir: str | Path = "Etalon Data",
    pattern: str = "piezo_test_5_18_*.xls",
    reference_peak_number: int = 6,
    flat_kwargs: Mapping[str, object] | None = None,
    fsr_source: str = "mean",
    first_n_for_fsr: int | None = None,
    known_fsr_ghz: float = 10.0,
) -> pd.DataFrame:
    """Analyze transmission shifts for all matching piezo datasets."""

    rows: list[dict[str, object]] = []
    for path in discover_piezo_files(data_dir=data_dir, pattern=pattern):
        start_percent, end_percent = parse_piezo_range(path)
        df = load_data(path)
        measurement = measure_transmission_shift_fraction(
            df,
            reference_peak_number=reference_peak_number,
            flat_kwargs=flat_kwargs,
            fsr_source=fsr_source,
            first_n_for_fsr=first_n_for_fsr,
        )
        rows.append(
            {
                "file": path.name,
                "path": path,
                "start_percent": start_percent,
                "end_percent": end_percent,
                "x_percent": end_percent,
                "fsr_fraction": measurement["fsr_fraction"],
                "frequency_shift_ghz": measurement["fsr_fraction"] * known_fsr_ghz,
                "reference_distance": measurement["reference_distance"],
                "shifted_distance": measurement["shifted_distance"],
                "distance_difference": measurement["distance_difference"],
                "mean_distance": measurement["mean_distance"],
                "local_fsr": measurement["local_fsr"],
                "flat_regions": measurement["flat_regions"],
                "voltage_peaks": measurement["voltage_peaks"],
            }
        )

    if not rows:
        raise FileNotFoundError(f"No files matched {Path(data_dir) / pattern}")
    return pd.DataFrame(rows).sort_values("x_percent").reset_index(drop=True)


def plot_transmission_shift_fractions(
    shift_df: pd.DataFrame,
    y_col: str = "fsr_fraction",
    ax=None,
):
    """Plot transmission shift as a fraction of FSR versus piezo percentage."""

    ax = ax or plt.subplots(figsize=(12, 5))[1]
    ax.plot(shift_df["x_percent"], shift_df[y_col], marker="o", linewidth=1.5)
    ylabel = "Transmission shift / FSR"
    if y_col == "frequency_shift_ghz":
        ylabel = "Transmission shift (GHz)"
    ax.set(
        title="Transmission Shift vs. Piezo Percentage",
        xlabel="Piezo percentage",
        ylabel=ylabel,
    )
    return ax


def run_initial_workflow(
    df: pd.DataFrame,
    flat_kwargs: Mapping[str, object] | None = None,
    voltage_labels: Sequence[str] | None = None,
) -> dict[str, object]:
    """Run the main workflow from the HTML notebook and return reusable results."""

    flat_kwargs = {
        "window": 2000,
        "slope_threshold": 5e-3,
        "min_length": 50000,
        "buffer": 10000,
        "plot": False,
        **dict(flat_kwargs or {}),
    }
    flat_result = detect_flat_regions(df, **flat_kwargs)
    voltage_peaks = find_voltage_peaks(flat_result.flat_dfs)
    peak_centers = find_peak_centers_by_region(flat_result.flat_dfs, voltage_peaks)
    distances = distances_from_voltage(peak_centers, voltage_peaks)
    labels = voltage_labels or [str(i) for i in range(len(flat_result.flat_dfs))]
    return {
        "flat_result": flat_result,
        "flat_dfs": flat_result.flat_dfs,
        "flat_regions": flat_result.flat_regions,
        "voltage_peaks": voltage_peaks,
        "peak_centers": peak_centers,
        "distances_from_voltage": distances,
        "labels": list(labels),
    }
