import clr
import time
import numpy as np
import json

clr.AddReference("C:\\Program Files\\Thorlabs\\Kinesis\\Thorlabs.MotionControl.DeviceManagerCLI.dll")
clr.AddReference("C:\\Program Files\\Thorlabs\\Kinesis\\Thorlabs.MotionControl.GenericMotorCLI.dll")
clr.AddReference("C:\\Program Files\\Thorlabs\\Kinesis\\ThorLabs.MotionControl.KCube.PiezoCLI.dll")
clr.AddReference("C:\\Program Files\\Thorlabs\\Kinesis\\Thorlabs.MotionControl.GenericPiezoCLI.dll")
clr.AddReference("C:\\Program Files\\Thorlabs\\Kinesis\\ThorLabs.MotionControl.KCube.InertialMotorCLI.dll")
clr.AddReference("C:\\Program Files\\Thorlabs\\Kinesis\\Thorlabs.MotionControl.KCube.PositionAlignerCLI.dll.")
clr.AddReference("C:\\Program Files\\Thorlabs\\Kinesis\\ThorLabs.MotionControl.KCube.PiezoStrainGaugeCLI.dll")

from Thorlabs.MotionControl.DeviceManagerCLI import *
from Thorlabs.MotionControl.GenericMotorCLI import *
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

    def get_xy_position(self, sig_strength = 0.08):
        values = []
        strengths = self.get_signal_strength()  # Now returns a list or tuple

        for sn, strength in zip(self.serial_numbers, strengths):
            if strength > sig_strength:
                status = self.controllers[sn].Status.PositionDifference
                values.extend([status.X, status.Y])
            else:
                raise ValueError(f"Signal too low for {sn}: {strength}")
        return values

    def get_xy_position_tavg(self, times=50, delay=0.04):
        all_results = []

        for _ in range(times):
            strengths = self.get_signal_strength()  # Returns a list/tuple of strengths

            for sn, strength in zip(self.serial_numbers, strengths):
                if strength <= 0.1:
                    raise ValueError(f"Signal strength too low for device {sn}: {strength}")

            res = self.get_xy_position()  # e.g., [x1, y1, x2, y2]
            all_results.append(res)
            time.sleep(delay)

        all_results = np.array(all_results)  # shape: (times, 4)
        avg_result = np.mean(all_results, axis=0)  # shape: (4,)
        return avg_result.tolist()

    def get_signal_strength(self):
        strengths = []
        for sn in self.serial_numbers:
            status = self.controllers[sn].Status.Sum
            strengths.append(status)
        return strengths

    def shutdown(self):
        for sn, aligner in self.controllers.items():
            aligner.StopPolling()
            aligner.Disconnect()
            print(f"Shutdown aligner with serial {sn}")

class LinearStageController:

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
            print(f"Shutdown linear stage with serial {sn}")

class HardwareOps:
    def __init__(self, config_filename):
        with open(config_filename, 'r') as f:
            config = json.load(f)

        self.piezo_serials = config["piezo_serials"]
        self.piezo_serials_list = list(self.piezo_serials.keys())
        self.quad_serials = config["quad_serials"]
        self.stage_serials = config["stage_serials"]

        # Now you can pass these to your controllers
        self.piezos = PiezoController(self.piezo_serials)
        self.quads = QuadCellController(self.quad_serials)
        self.stages = LinearStageController(self.stage_serials)

    def get_all_actuator_values(self):
        voltages_piezo = self.piezos.get_all_voltages()
        voltages_linear = self.stages.get_all_positions()
        positions = self.quads.get_xy_position()

        combined_values = list(voltages_piezo) + list(voltages_linear) + list(positions)
        return combined_values
    
    def shutdown(self):
        print("Shutting down piezos...")
        self.piezos.shutdown()
        print("Shutting down quadcells...")
        self.quads.shutdown()
        print("Shutting down stages...")
        self.stages.shutdown()
        print("All hardware safely shut down.")
