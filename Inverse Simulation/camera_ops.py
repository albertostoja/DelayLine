from __future__ import annotations

import importlib
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2 as cv


MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_IMAGE_DIR = MODULE_DIR / "images"
DEFAULT_LAMP_PORT = None
DEFAULT_LAMP_BAUDRATE = 9600
DEFAULT_LAMP_TIMEOUT = 1
LAMP_RELAY_VID = 0x1A86
LAMP_RELAY_PID = 0x7523
LAMP_RELAY_DESCRIPTION = "CH340"
LAMP_ON_COMMAND = bytes.fromhex("A0 01 01 A2")
LAMP_OFF_COMMAND = bytes.fromhex("A0 01 00 A1")

_lamp_serial = None
_lamp_serial_config = None


def _import_serial():
    try:
        import serial
    except ImportError as exc:
        raise ImportError("pyserial is required to control the lamp relay.") from exc
    return serial


def _import_list_ports():
    try:
        import serial.tools.list_ports as list_ports
    except ImportError as exc:
        raise ImportError("pyserial is required to detect the lamp relay port.") from exc
    return list_ports


def _usb_id_matches(value, expected):
    if value is None:
        return False
    try:
        return int(value) == int(expected)
    except (TypeError, ValueError):
        try:
            return int(str(value).lower().removeprefix("0x"), 16) == int(expected)
        except ValueError:
            return False


def _format_port_info(port_info):
    device = getattr(port_info, "device", None) or str(port_info)
    description = getattr(port_info, "description", "") or ""
    hwid = getattr(port_info, "hwid", "") or ""
    return f"{device} ({description}; {hwid})"


def _is_lamp_relay_port(port_info):
    vid = getattr(port_info, "vid", None)
    pid = getattr(port_info, "pid", None)
    if _usb_id_matches(vid, LAMP_RELAY_VID) and _usb_id_matches(pid, LAMP_RELAY_PID):
        return True

    needle = LAMP_RELAY_DESCRIPTION.lower()
    description = str(getattr(port_info, "description", "") or "").lower()
    hwid = str(getattr(port_info, "hwid", "") or "").lower()
    return needle in description or needle in hwid


def find_lamp_port():
    """Return the detected COM port for the CH340 USB lamp relay."""
    list_ports = _import_list_ports()
    ports = list(list_ports.comports())
    candidates = [port for port in ports if _is_lamp_relay_port(port)]

    if len(candidates) == 1:
        return candidates[0].device

    if len(candidates) == 0:
        available = ", ".join(_format_port_info(port) for port in ports)
        if not available:
            available = "no serial ports found"
        raise RuntimeError(
            "Could not find the CH340 USB lamp relay. "
            f"Available ports: {available}."
        )

    candidate_list = ", ".join(_format_port_info(port) for port in candidates)
    raise RuntimeError(
        "Found multiple possible CH340 lamp relays. "
        f"Pass lamp_port explicitly. Candidates: {candidate_list}."
    )


def get_lamp_serial(
    port=DEFAULT_LAMP_PORT,
    baudrate=DEFAULT_LAMP_BAUDRATE,
    timeout=DEFAULT_LAMP_TIMEOUT,
):
    """Open or reuse the USB relay serial connection for the lamp."""
    global _lamp_serial, _lamp_serial_config

    resolved_port = find_lamp_port() if port is None else str(port)
    config = (resolved_port, int(baudrate), float(timeout))
    if _lamp_serial is not None and _lamp_serial_config == config:
        if getattr(_lamp_serial, "is_open", True):
            return _lamp_serial
        _lamp_serial.open()
        return _lamp_serial

    if _lamp_serial is not None:
        close_lamp_serial()

    serial = _import_serial()

    _lamp_serial = serial.Serial(config[0], config[1], timeout=config[2])
    _lamp_serial_config = config
    return _lamp_serial


def close_lamp_serial():
    """Close the cached lamp relay serial connection, if one is open."""
    global _lamp_serial, _lamp_serial_config

    if _lamp_serial is not None:
        _lamp_serial.close()
    _lamp_serial = None
    _lamp_serial_config = None


def _write_lamp_command(
    command,
    *,
    port=DEFAULT_LAMP_PORT,
    baudrate=DEFAULT_LAMP_BAUDRATE,
    timeout=DEFAULT_LAMP_TIMEOUT,
    serial_connection=None,
):
    relay = serial_connection or get_lamp_serial(
        port=port,
        baudrate=baudrate,
        timeout=timeout,
    )
    bytes_written = relay.write(command)
    if bytes_written is not None and bytes_written != len(command):
        raise RuntimeError(
            f"Lamp relay wrote {bytes_written} bytes; expected {len(command)}."
        )


def lamp_on(
    *,
    port=DEFAULT_LAMP_PORT,
    baudrate=DEFAULT_LAMP_BAUDRATE,
    timeout=DEFAULT_LAMP_TIMEOUT,
    serial_connection=None,
):
    """Turn the lamp relay on."""
    _write_lamp_command(
        LAMP_ON_COMMAND,
        port=port,
        baudrate=baudrate,
        timeout=timeout,
        serial_connection=serial_connection,
    )


def lamp_off(
    *,
    port=DEFAULT_LAMP_PORT,
    baudrate=DEFAULT_LAMP_BAUDRATE,
    timeout=DEFAULT_LAMP_TIMEOUT,
    serial_connection=None,
):
    """Turn the lamp relay off."""
    _write_lamp_command(
        LAMP_OFF_COMMAND,
        port=port,
        baudrate=baudrate,
        timeout=timeout,
        serial_connection=serial_connection,
    )


def _load_simulation_module():
    module_dir = str(MODULE_DIR)
    added_to_path = False
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
        added_to_path = True

    cwd = os.getcwd()
    try:
        # Simulation.py loads JSON files with paths relative to the repo root.
        os.chdir(MODULE_DIR)
        return importlib.import_module("Simulation")
    finally:
        os.chdir(cwd)
        if added_to_path:
            try:
                sys.path.remove(module_dir)
            except ValueError:
                pass


def _default_image_path(image_dir, suffix):
    image_dir = Path(image_dir)
    base = datetime.now().strftime(f"%Y%m%d-%H%M%S-{suffix}")
    path = image_dir / f"{base}.jpg"
    if not path.exists():
        return path

    for counter in range(1, 1000):
        candidate = image_dir / f"{base}-{counter:03d}.jpg"
        if not candidate.exists():
            return candidate

    raise RuntimeError(f"Could not create a unique image name in {image_dir}")


def _resolve_output_path(output_path, image_dir, suffix):
    if output_path is not None and image_dir is not None:
        raise ValueError("Pass either output_path or image_dir, not both.")

    if output_path is not None:
        return Path(output_path)

    return _default_image_path(DEFAULT_IMAGE_DIR if image_dir is None else image_dir, suffix)


def _reflection_detection_kwargs(eps, min_samples, show):
    kwargs = {
        "min_samples": min_samples,
        "show": show,
    }
    if eps is not None:
        kwargs["eps"] = eps
    return kwargs


def _capture_camera_frame(camera_index, width, height, warmup_seconds):
    camera = cv.VideoCapture(camera_index)
    try:
        camera.set(cv.CAP_PROP_FRAME_WIDTH, int(width))
        camera.set(cv.CAP_PROP_FRAME_HEIGHT, int(height))

        if not camera.isOpened():
            raise RuntimeError(f"Could not open camera index {camera_index}.")

        warmup_seconds = float(warmup_seconds)
        if warmup_seconds > 0:
            warmup_until = time.monotonic() + warmup_seconds
            while time.monotonic() < warmup_until:
                camera.read()

        ret, image_bgr = camera.read()
        if not ret or image_bgr is None:
            raise RuntimeError(f"Failed to capture image from camera index {camera_index}.")

        return image_bgr
    finally:
        camera.release()


def _save_image(path, image_bgr):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv.imwrite(str(path), image_bgr):
        raise RuntimeError(f"Failed to save image to {path}.")
    return path


def _detect_reflection_points(path, image_bgr, eps, min_samples, show, return_images):
    gray = cv.cvtColor(image_bgr, cv.COLOR_BGR2GRAY)
    simulation = _load_simulation_module()

    detect_kwargs = _reflection_detection_kwargs(eps, min_samples, show)
    raw_detect_kwargs = dict(detect_kwargs)
    raw_detect_kwargs["show"] = False

    points = simulation.reflec_pts_cam(gray, **detect_kwargs)
    raw_points, raw_counts = simulation.reflec_pts_cam_num_reflec(gray, **raw_detect_kwargs)

    return {
        "path": str(path),
        "points": points,
        "raw_points": raw_points,
        "raw_counts": raw_counts,
        "detected_total": int(sum(raw_counts.values())),
        "image_bgr": image_bgr if return_images else None,
        "gray": gray if return_images else None,
    }


def _show_aruco_points(image_bgr, aruco_points):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 10))
    plt.imshow(cv.cvtColor(image_bgr, cv.COLOR_BGR2RGB))
    if aruco_points:
        xs = [point[0] for point in aruco_points]
        ys = [point[1] for point in aruco_points]
        plt.scatter(xs, ys, c="lime", s=50, marker="x")
    plt.title(f"Light image ArUco points: {len(aruco_points)}")
    plt.axis("off")
    plt.show()


def _detect_aruco_points(path, image_bgr, show, return_images):
    gray = cv.cvtColor(image_bgr, cv.COLOR_BGR2GRAY)
    simulation = _load_simulation_module()
    aruco_points = simulation.camera_arucos(str(path))
    if show:
        _show_aruco_points(image_bgr, aruco_points)

    return {
        "path": str(path),
        "points": aruco_points,
        "aruco_points": aruco_points,
        "aruco_count": int(len(aruco_points)),
        "image_bgr": image_bgr if return_images else None,
        "gray": gray if return_images else None,
    }


def _validate_lamp_callbacks(lamp_on, lamp_off):
    if (lamp_on is None) != (lamp_off is None):
        raise ValueError("Pass both lamp_on and lamp_off, or neither.")
    if lamp_on is not None and not callable(lamp_on):
        raise TypeError("lamp_on must be callable.")
    if lamp_off is not None and not callable(lamp_off):
        raise TypeError("lamp_off must be callable.")


def _turn_lamp_off(lamp_off, prior_error):
    try:
        lamp_off()
    except Exception as off_exc:
        if prior_error is not None:
            raise RuntimeError(
                f"Failed while taking light image ({prior_error}); "
                f"additionally failed to turn lamp off ({off_exc})."
            ) from off_exc
        raise RuntimeError(f"Failed to turn lamp off after light image capture: {off_exc}") from off_exc


def _resolve_light_lamp_callbacks(
    lamp_on_callback,
    lamp_off_callback,
    use_lamp,
    lamp_port,
    lamp_baudrate,
    lamp_timeout,
):
    if lamp_on_callback is None and lamp_off_callback is None:
        if not use_lamp:
            return None, None
        return (
            lambda: globals()["lamp_on"](
                port=lamp_port,
                baudrate=lamp_baudrate,
                timeout=lamp_timeout,
            ),
            lambda: globals()["lamp_off"](
                port=lamp_port,
                baudrate=lamp_baudrate,
                timeout=lamp_timeout,
            ),
        )

    _validate_lamp_callbacks(lamp_on_callback, lamp_off_callback)
    return lamp_on_callback, lamp_off_callback


def take_dark_image(
    output_path=None,
    *,
    image_dir=None,
    camera_index=0,
    width=1920,
    height=1080,
    warmup_seconds=3.0,
    eps=None,
    min_samples=35,
    show=False,
    return_images=True,
):
    """Capture a dark setup image and detect laser reflection points."""
    resolved_path = _resolve_output_path(output_path, image_dir, "D")
    image_bgr = _capture_camera_frame(camera_index, width, height, warmup_seconds)
    saved_path = _save_image(resolved_path, image_bgr)
    return _detect_reflection_points(saved_path, image_bgr, eps, min_samples, show, return_images)


def take_light_image(
    output_path=None,
    *,
    image_dir=None,
    camera_index=0,
    width=1920,
    height=1080,
    warmup_seconds=4.0,
    lamp_on=None,
    lamp_off=None,
    lamp_settle_seconds=0.5,
    use_lamp=True,
    lamp_port=DEFAULT_LAMP_PORT,
    lamp_baudrate=DEFAULT_LAMP_BAUDRATE,
    lamp_timeout=DEFAULT_LAMP_TIMEOUT,
    show=False,
    return_images=True,
):
    """Turn the lamp on, capture a light setup image, then turn the lamp off."""
    lamp_on_callback, lamp_off_callback = _resolve_light_lamp_callbacks(
        lamp_on,
        lamp_off,
        use_lamp,
        lamp_port,
        lamp_baudrate,
        lamp_timeout,
    )
    resolved_path = _resolve_output_path(output_path, image_dir, "L")

    should_turn_lamp_off = lamp_on_callback is not None
    capture_error = None
    try:
        if lamp_on_callback is not None:
            lamp_on_callback()
            lamp_settle_seconds = float(lamp_settle_seconds)
            if lamp_settle_seconds > 0:
                time.sleep(lamp_settle_seconds)

        image_bgr = _capture_camera_frame(camera_index, width, height, warmup_seconds)
        saved_path = _save_image(resolved_path, image_bgr)
    except BaseException as exc:
        capture_error = exc
        raise
    finally:
        if should_turn_lamp_off:
            _turn_lamp_off(lamp_off_callback, capture_error)

    result = _detect_aruco_points(saved_path, image_bgr, show, return_images)
    result["lamp_controlled"] = lamp_on_callback is not None
    return result
