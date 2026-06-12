import clr
import time
import numpy as np
import json
try:
    import serial
except ImportError:
    serial = None

clr.AddReference("C:\\Program Files\\Thorlabs\\Kinesis\\Thorlabs.MotionControl.DeviceManagerCLI.dll")
clr.AddReference("C:\\Program Files\\Thorlabs\\Kinesis\\Thorlabs.MotionControl.GenericMotorCLI.dll")
clr.AddReference("C:\\Program Files\\Thorlabs\\Kinesis\\ThorLabs.MotionControl.KCube.DCServoCLI.dll")
clr.AddReference("C:\\Program Files\\Thorlabs\\Kinesis\\ThorLabs.MotionControl.KCube.PiezoCLI.dll")
clr.AddReference("C:\\Program Files\\Thorlabs\\Kinesis\\Thorlabs.MotionControl.GenericPiezoCLI.dll")
clr.AddReference("C:\\Program Files\\Thorlabs\\Kinesis\\ThorLabs.MotionControl.KCube.InertialMotorCLI.dll")
clr.AddReference("C:\\Program Files\\Thorlabs\\Kinesis\\Thorlabs.MotionControl.KCube.PositionAlignerCLI.dll.")
clr.AddReference("C:\\Program Files\\Thorlabs\\Kinesis\\ThorLabs.MotionControl.KCube.PiezoStrainGaugeCLI.dll")

from Thorlabs.MotionControl.DeviceManagerCLI import *
from Thorlabs.MotionControl.GenericMotorCLI import *
from Thorlabs.MotionControl.KCube.DCServoCLI import *
from Thorlabs.MotionControl.KCube.PiezoCLI import *
from Thorlabs.MotionControl.KCube.PositionAlignerCLI import *
from Thorlabs.MotionControl.KCube.InertialMotorCLI import *
from Thorlabs.MotionControl.KCube.PiezoStrainGaugeCLI import *
from Thorlabs.MotionControl.GenericPiezoCLI import *
from System import Decimal
from time import sleep

class PiezoController:

    def __init__(self, serial_dict):
        self.serial_dict = serial_dict
        self.controllers = {}
        self.device_types = {}
        self.initialize_devices()
    
    def initialize_devices(self):
        DeviceManagerCLI.BuildDeviceList()
    
        for sn, dev_type in self.serial_dict.items():
            if dev_type == "kpz":
                piezo = KCubePiezo.CreateKCubePiezo(sn)
            elif dev_type == "kpc":
                piezo = KCubePiezoStrainGauge.CreateKCubePiezoStrainGauge(sn)
            else:
                raise ValueError(f"Unknown piezo device type for {sn}: {dev_type}")

            piezo.Connect(sn)
            piezo.WaitForSettingsInitialized(10000)
            piezo.StartPolling(250)
            piezo.EnableDevice()

            piezo.SetMaxOutputVoltage(Decimal(150.0))

            self.controllers[sn] = piezo
            self.device_types[sn] = dev_type

    def get_voltage(self, sn):
        device = self.controllers[sn]
        return float(str(device.GetOutputVoltage()))
        
    def get_all_voltages(self):
        values = [self.get_voltage(sn) for sn in self.controllers]
        return values

    def set_voltage(self, sn, target_voltage, step=1.0, delay=0.25):
        if not (0.0 <= target_voltage <= 150.0):
            raise ValueError(f"target_voltage must be between 0 and 100. Got {target_voltage}.")
        
        device = self.controllers[sn]

        current_value = float(str(device.GetOutputVoltage()))

        # Prepare ramping
        target_voltage = float(target_voltage)
        step = abs(step)
        direction = 1 if target_voltage > current_value else -1

        # Build step list
        voltages = []
        v = current_value
        while (direction == 1 and v < target_voltage) or (direction == -1 and v > target_voltage):
            voltages.append(v)
            v += direction * step

        voltages.append(target_voltage)  # final target

        # Apply steps
        for v in voltages:
            device.SetOutputVoltage(Decimal(v))
            print(f"{sn} set to {v}")
            sleep(delay)

    def set_to_voltage_zero_value(self, piezo_list, voltage):
        for i in range(len(piezo_list)):
            self.set_voltage(piezo_list[i], voltage)

    def shutdown(self):
        for sn, piezo in self.controllers.items():
            current_val = self.get_voltage(sn)
            if abs(current_val) >= 0.05:
                print(f"[INFO] Setting {sn} to 0 V/position before shutdown.")
                self.set_voltage(sn, 0.0)
            else:
                print(f"[INFO] {sn} already near 0 (value = {current_val:.3f}).")

            piezo.StopPolling()
            piezo.Disconnect()
            print(f"[OK] Shutdown piezo with serial {sn}")


class QuadCellController:
    def __init__(self, serial_numbers):
        self.serial_numbers = serial_numbers
        self.controllers = {}
        self.initialize_devices()

    def initialize_devices(self):
        DeviceManagerCLI.BuildDeviceList()

        for sn in self.serial_numbers:
            aligner = KCubePositionAligner.CreateKCubePositionAligner(sn)
            time.sleep(0.2)
            aligner.Connect(sn)
            aligner.WaitForSettingsInitialized(10000)
            aligner.StartPolling(250)
            aligner.EnableDevice()
            self.controllers[sn] = aligner

    def get_position_error(self, sn):
        status = self.controllers[sn].Status.PositionDifference
        return status.PositionError

    def raw_to_mm(self, x_diff, y_diff, signal_sum):
        k = 1.500
        signal_sum = np.asarray(signal_sum, dtype=float)
        signal_sum = np.where(signal_sum == 0, np.finfo(float).eps, signal_sum)

        x_norm = np.asarray(x_diff, dtype=float) / signal_sum
        y_norm = np.asarray(y_diff, dtype=float) / signal_sum

        clip_limit = 0.9999
        x_norm = np.clip(x_norm, -clip_limit, clip_limit)
        y_norm = np.clip(y_norm, -clip_limit, clip_limit)

        # Detector.m comments mention erfinv, but the current MATLAB code uses
        # this linear k * normalized-difference conversion.
        return k * x_norm, k * y_norm

    def get_xy_position_raw(self, sig_strength=0.04):
        values = []
        sums = self.get_sum()

        for sn, signal_sum in zip(self.serial_numbers, sums):
            if signal_sum > sig_strength:
                status = self.controllers[sn].Status.PositionDifference
                values.extend([status.X, status.Y])
            else:
                raise ValueError(f"Signal too low for {sn}: {signal_sum}")
        return values

    def get_xy_position(self, sig_strength=0.04): #qc1x, qc1y, qc2x, qc2y
        values = []
        sums = self.get_sum()

        for sn, signal_sum in zip(self.serial_numbers, sums):
            if signal_sum > sig_strength:
                status = self.controllers[sn].Status.PositionDifference
                x_mm, y_mm = self.raw_to_mm(status.X, status.Y, signal_sum)
                values.extend([float(x_mm), float(y_mm)])
            else:
                raise ValueError(f"Signal too low for {sn}: {signal_sum}")
        return values

    def get_xy_position_tavg(self, times=50, delay=0.04):
        all_results = []

        for _ in range(times):
            sums = self.get_sum()

            for sn, signal_sum in zip(self.serial_numbers, sums):
                if signal_sum <= 0.1:
                    raise ValueError(f"Signal strength too low for device {sn}: {signal_sum}")

            res = self.get_xy_position()  # e.g., [x1, y1, x2, y2]
            all_results.append(res)
            time.sleep(delay)

        all_results = np.array(all_results)  # shape: (times, 4)
        avg_result = np.mean(all_results, axis=0)  # shape: (4,)
        return avg_result.tolist()

    def get_sum(self):
        sums = []
        for sn in self.serial_numbers:
            status = self.controllers[sn].Status.Sum
            sums.append(status)
        return sums

    def get_signal_strength(self):
        return self.get_sum()

    def shutdown(self):
        for sn, aligner in self.controllers.items():
            aligner.StopPolling()
            aligner.Disconnect()
            print(f"Shutdown aligner with serial {sn}")

class LinearStageControllerKIM:

    Channel_map = {
        "chan1": InertialMotorStatus.MotorChannels.Channel1,
        "chan2": InertialMotorStatus.MotorChannels.Channel2,
        "chan3": InertialMotorStatus.MotorChannels.Channel3,
        "chan4": InertialMotorStatus.MotorChannels.Channel4
    }

    def __init__(self, serial_numbers, StepRate = 500, StepAcceleration = 100000):
        self.serial_numbers = serial_numbers
        self.controllers = {}
        self.initialize_devices(StepRate, StepAcceleration)

    def initialize_devices(self, StepRate, StepAcceleration):
        DeviceManagerCLI.BuildDeviceList()
        for sn in self.serial_numbers:
            stage = KCubeInertialMotor.CreateKCubeInertialMotor(sn)
            stage.Connect(sn)
            stage.WaitForSettingsInitialized(10000)
            stage.StartPolling(250)
            stage.EnableDevice()
            self.controllers[sn] = stage

            inertial_motor_config = stage.GetInertialMotorConfiguration(sn)
            device_settings = ThorlabsInertialMotorSettings.GetSettings(inertial_motor_config)
            chan1 = InertialMotorStatus.MotorChannels.Channel1
            chan2 = InertialMotorStatus.MotorChannels.Channel2
            chan3 = InertialMotorStatus.MotorChannels.Channel3
            chan4 = InertialMotorStatus.MotorChannels.Channel4
            device_settings.Drive.Channel(chan1).StepRate = StepRate
            device_settings.Drive.Channel(chan1).StepAcceleration = StepAcceleration
            device_settings.Drive.Channel(chan2).StepRate = StepRate
            device_settings.Drive.Channel(chan2).StepAcceleration = StepAcceleration
            device_settings.Drive.Channel(chan3).StepRate = StepRate
            device_settings.Drive.Channel(chan3).StepAcceleration = StepAcceleration
            device_settings.Drive.Channel(chan4).StepRate = StepRate
            device_settings.Drive.Channel(chan4).StepAcceleration = StepAcceleration

            stage.SetSettings(device_settings, True, True)

    def move_absolute(self, sn, channel, distance):
        device = self.controllers[sn]
        move_distance = int(distance)
        print(f'Moving to position {move_distance}')
        device.MoveTo(self.Channel_map[str(channel)], move_distance, 10000)  # 10 second timeout
        print("Move Complete")

    def get_position(self, sn, channel):
        device = self.controllers[sn]
        ch = self.Channel_map[str(channel)]
        pos = device.GetPosition(ch)
        return float(pos)

    def get_all_positions(self):
        values = []
        for sn in self.controllers:
            for ch in ['chan1', 'chan2', 'chan3', 'chan4']:
                try:
                    pos = self.get_position(sn, ch)
                    values.append(pos)
                except Exception as e:
                    print(f"Channel {ch} on device {sn} failed: {e}")
                    values.append(None)
        return values

    def shutdown(self):
        for sn, stage in self.controllers.items():
            stage.StopPolling()
            stage.Disconnect()
            print(f"Shutdown KIM stage with serial {sn}")

class LinearStageController:

    def __init__(self, serial_numbers, stage_settings_names=None,
                 polling_interval=250, home_on_start=False):
        if isinstance(serial_numbers, dict):
            self.serial_numbers = list(serial_numbers.keys())
            self.stage_settings_names = serial_numbers
        else:
            self.serial_numbers = serial_numbers
            self.stage_settings_names = stage_settings_names or {}

        self.polling_interval = polling_interval
        self.controllers = {}
        self.initialize_devices(home_on_start=home_on_start)

    def initialize_devices(self, home_on_start=False):
        DeviceManagerCLI.BuildDeviceList()

        for sn in self.serial_numbers:
            stage = KCubeDCServo.CreateKCubeDCServo(sn)
            stage.Connect(sn)
            stage.WaitForSettingsInitialized(10000)

            motor_config = stage.LoadMotorConfiguration(
                sn,
                DeviceConfiguration.DeviceSettingsUseOptionType.UseFileSettings
            )
            settings_name = self.stage_settings_names.get(sn)
            if settings_name is not None:
                motor_config.DeviceSettingsName = settings_name
            else:
                print(
                    f"[WARN] No stage settings name provided for {sn}. "
                    "KDC101 mm conversion may be wrong."
                )
            motor_config.UpdateCurrentConfiguration()
            stage.SetSettings(stage.MotorDeviceSettings, True, False)

            stage.StartPolling(self.polling_interval)
            stage.EnableDevice()
            time.sleep(0.5)

            self.controllers[sn] = stage

            if home_on_start:
                self.home(sn)

    def home(self, sn, timeout_ms=60000):
        device = self.controllers[sn]
        print(f"Homing KDC101 stage {sn}")
        device.Home(timeout_ms)
        print(f"Home complete for {sn}")

    def move_absolute(self, sn, position_mm, timeout_ms=60000):
        device = self.controllers[sn]
        target = Decimal(float(position_mm))
        print(f"Moving KDC101 stage {sn} to {position_mm} mm")
        device.MoveTo(target, timeout_ms)
        print(f"Move complete for {sn}")

    def move_relative(self, sn, delta_mm, timeout_ms=60000):
        current = self.get_position(sn)
        self.move_absolute(sn, current + float(delta_mm), timeout_ms=timeout_ms)

    def get_position(self, sn):
        device = self.controllers[sn]
        try:
            return float(str(device.GetPosition()))
        except AttributeError:
            return float(str(device.Position))

    def get_all_positions(self):
        return [self.get_position(sn) for sn in self.controllers]

    def shutdown(self):
        for sn, stage in self.controllers.items():
            stage.StopPolling()
            stage.Disconnect()
            print(f"Shutdown KDC101 linear stage with serial {sn}")

class RotationStageController:

    def __init__(self, ports, baudrate=921600, timeout=1.0, terminator="\r\n"):
        if serial is None:
            raise ImportError("pyserial is required for RotationStageController.")

        self.ports = ports
        self.terminator = terminator
        self.controllers = {}

        for name, port in self._iter_ports(ports):
            self.controllers[name] = serial.Serial(
                port=port,
                baudrate=baudrate,
                timeout=timeout,
                write_timeout=timeout
            )
            self.set_remote(name)

    def _iter_ports(self, ports):
        if isinstance(ports, dict):
            return ports.items()
        return [(str(port), port) for port in ports]

    def command(self, name, cmd, read=True):
        device = self.controllers[name]
        msg = f"{cmd}{self.terminator}".encode("ascii")
        device.write(msg)
        if not read:
            return None
        return device.readline().decode("ascii", errors="replace").strip()

    def set_remote(self, name):
        return self.command(name, "MR", read=False)

    def set_local(self, name):
        return self.command(name, "ML", read=False)

    def select_channel(self, name, channel):
        channel = int(channel)
        if not (1 <= channel <= 4):
            raise ValueError("AG-UC8 channel must be between 1 and 4.")
        return self.command(name, f"CC{channel}", read=False)

    def _axis_in_channel(self, actuator):
        actuator = int(actuator)
        if not (1 <= actuator <= 8):
            raise ValueError("AG-UC8 actuator must be between 1 and 8.")

        channel = (actuator - 1) // 2 + 1
        axis = 1 if actuator % 2 == 1 else 2
        return channel, axis

    def move_relative_steps(self, name, actuator, steps):
        channel, axis = self._axis_in_channel(actuator)
        self.select_channel(name, channel)
        return self.command(name, f"{axis}PR{int(steps)}", read=False)

    def move_relative(self, name, actuator, steps):
        return self.move_relative_steps(name, actuator, steps)

    def jog(self, name, actuator, mode):
        channel, axis = self._axis_in_channel(actuator)
        mode = int(mode)
        if mode < -4 or mode > 4:
            raise ValueError("AG-UC8 jog mode must be between -4 and 4.")
        self.select_channel(name, channel)
        return self.command(name, f"{axis}JA{mode}", read=False)

    def stop(self, name, actuator=None):
        if actuator is None:
            return self.command(name, "ST", read=False)

        channel, axis = self._axis_in_channel(actuator)
        self.select_channel(name, channel)
        return self.command(name, f"{axis}ST", read=False)

    def zero_position(self, name, actuator):
        channel, axis = self._axis_in_channel(actuator)
        self.select_channel(name, channel)
        return self.command(name, f"{axis}ZP", read=False)

    def set_step_amplitude(self, name, actuator, amplitude):
        channel, axis = self._axis_in_channel(actuator)
        self.select_channel(name, channel)
        return self.command(name, f"{axis}SU{int(amplitude)}", read=False)

    def get_position_steps(self, name, actuator):
        channel, axis = self._axis_in_channel(actuator)
        self.select_channel(name, channel)
        response = self.command(name, f"{axis}TP?")
        if response.startswith(f"{axis}TP"):
            response = response[len(f"{axis}TP"):]
        try:
            return int(response)
        except ValueError:
            return response

    def get_position(self, name, actuator):
        return self.get_position_steps(name, actuator)

    def get_all_positions(self, actuators=range(1, 9)):
        positions = {}
        for name in self.controllers:
            positions[name] = {
                actuator: self.get_position_steps(name, actuator)
                for actuator in actuators
            }
        return positions

    def shutdown(self):
        for name, device in self.controllers.items():
            self.set_local(name)
            device.close()
            print(f"Shutdown Newport rotation controller {name}")

class HardwareOps:
    def __init__(self, config_filename):
        with open(config_filename, 'r') as f:
            config = json.load(f)

        self.piezo_serials = config["piezo_serials"]
        self.piezo_serials_list = list(self.piezo_serials.keys())
        self.quad_serials = config["quad_serials"]
        self.linear_stage_serials = config.get("linear_stage_serials", [])
        self.linear_stage_settings = config.get("linear_stage_settings", {})
        self.kim_stage_serials = config.get("kim_stage_serials", config.get("stage_serials", []))
        self.rotation_stage_ports = config.get("rotation_stage_ports", {})

        self.piezos = PiezoController(self.piezo_serials)
        self.quads = QuadCellController(self.quad_serials)
        self.stages = (
            LinearStageController(
                self.linear_stage_serials,
                stage_settings_names=self.linear_stage_settings
            )
            if self.linear_stage_serials else None
        )
        self.kim_stages = (
            LinearStageControllerKIM(self.kim_stage_serials)
            if self.kim_stage_serials else None
        )
        self.rotation_stages = (
            RotationStageController(self.rotation_stage_ports)
            if self.rotation_stage_ports else None
        )

    def get_all_actuator_values(self):
        voltages_piezo = self.piezos.get_all_voltages()
        positions_linear = self.stages.get_all_positions() if self.stages else []
        positions_kim = self.kim_stages.get_all_positions() if self.kim_stages else []
        positions = self.quads.get_xy_position()

        combined_values = (
            list(voltages_piezo)
            + list(positions_linear)
            + list(positions_kim)
            + list(positions)
        )
        return combined_values
    
    def shutdown(self):
        print("Shutting down piezos...")
        self.piezos.shutdown()
        print("Shutting down quadcells...")
        self.quads.shutdown()
        if self.stages:
            print("Shutting down KDC101 linear stages...")
            self.stages.shutdown()
        if self.kim_stages:
            print("Shutting down KIM stages...")
            self.kim_stages.shutdown()
        if self.rotation_stages:
            print("Shutting down rotation stages...")
            self.rotation_stages.shutdown()
        print("All hardware safely shut down.")
