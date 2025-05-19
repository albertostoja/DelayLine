import clr
import time
import numpy as np

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

            piezo.SetMaxOutputVoltage(Decimal(100.0))

            self.controllers[sn] = piezo
            self.device_types[sn] = dev_type

    def get_voltage(self, sn):
        device = self.controllers[sn]
        return float(str(device.GetOutputVoltage()))
        
    def get_all_voltages(self):
        values = [self.get_voltage(sn) for sn in self.controllers]
        matrix = np.array(values).reshape(2, 2)  # Adjust shape as needed
        return matrix

    def set_voltage(self, sn, target_voltage, step=1.0, delay=0.25):
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

    def get_xy_position(self):
        ordered_keys = sorted(self.controllers.keys(), key=int)
        values = []
        for sn in ordered_keys:
            status = self.controllers[sn].Status.PositionDifference
            values.extend([status.X, status.Y])
        return np.array(values).reshape(2, 2)  # Rows: [X1, Y1], [X2, Y2]

    def get_signal_strength(self, sn):
        status = self.controllers[sn].GetStatus()
        return status.TotalSignal

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

    def move_absolute(self, serial_number, channel, distance):
        device = self.controllers[serial_number]
        move_distance = int(distance)
        print(f'Moving to position {move_distance}')
        device.MoveTo(self.Channel_map[str(channel)], move_distance, 1000)  # 1 second timeout
        print("Move Complete")

    def get_all_positions(self):
        values = []
        for sn in sorted(self.controllers.keys(), key=int):  # sort for consistent order
            device = self.controllers[sn]
            for ch_name in ["chan1", "chan2", "chan3", "chan4"]:
                ch_enum = self.Channel_map[ch_name]
                try:
                    pos = float(str(device.positions(ch_enum)))
                except Exception:
                    pos = np.nan  # use NaN for unavailable channels
                values.append(pos)

    def get_position(self, sn):
        return float(str(self.controllers[sn].Position))

    def shutdown(self):
        for sn, stage in self.controllers.items():
            stage.StopPolling()
            stage.Disconnect()
            print(f"Shutdown linear stage with serial {sn}")